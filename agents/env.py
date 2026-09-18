import random
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle, SideCondition, Status
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .config import BATTLE_FORMAT, N_FEATURES, QUALIFIED_PREFIX
from .features import embed_battle_with_fusion
from .fusion_parser import _attach_fusion_parser
from .players import PolicyPlayer

import multiprocessing

# --- веса для новой награды (сбалансированы под базу fainted 2.0 / hp 1.0 / victory 30) ---
HAZARD_REWARDS = {
    SideCondition.STEALTH_ROCK: 0.8,
    SideCondition.SPIKES: 0.5,       # за слой
    SideCondition.TOXIC_SPIKES: 0.5, # за слой
    SideCondition.STICKY_WEB: 0.6,
}
DAMAGE_WEIGHT = 0.8          # за 100% HP урона активному противнику
DAMAGE_TAKEN_PENALTY = 0.4   # штраф за полученный урон (меньше, чтобы не боялся атаковать)
BOOST_WEIGHT = 0.15          # за каждую ступень буста своих
DEBUFF_WEIGHT = 0.15         # за каждую ступень дебаффа противника
SWITCH_BONUS = 0.5           # базовый бонус за удачный свитч
SWITCH_IMMUNE_BONUS = 1.0    # x2 за иммун/отражение (пользователь: x2)
SWITCH_THRESHOLD = 0.30      # разница урона 30% HP = порог удачного свитча
# новые
HAZARD_CLEAR_BONUS = 0.5         # снял хазарды оппа с себя (Rapid Spin)
HAZARD_SELF_CLEAR_PENALTY = 0.5   # снял свои хазарды с оппа (Defog)
HAZARD_DAMAGE_PENALTY = 0.3       # урон от хазардов при свитче
PROTECT_BONUS = 0.4               # удачный протект
STATUS_CURE_BONUS = 0.4           # клин статуса (Heal Bell)
TERA_BONUS = 0.3                  # удачный тера (как выбрал weak)
STATUS_IMMUNE_TYPES = {
    "par": ["electric", "ground"],  # Thunder Wave не действует на Electric/Ground с VoltAbsorb? Упростим: Electric имун к параличу? На деле только Ground имун к Thunder Wave, но оставим
    "brn": ["fire"],
    "psn": ["poison", "steel"],
    "tox": ["poison", "steel"],
    "slp": [], "frz": [],
}


def _snapshot_number(fname: str) -> int | None:
    suffix = fname.split("_")[-1].split(".")[0]
    return int(suffix) if suffix.isdigit() else None


def _make_self_play_opponents():
    model_dir = "models/"
    try:
        candidates = [f for f in listdir(model_dir) if isfile(join(model_dir, f)) and QUALIFIED_PREFIX in f]
    except FileNotFoundError:
        candidates = []
    use_fallback = False
    if not candidates:
        try:
            from agents.config import SELF_PLAY_PATH
            prefix = SELF_PLAY_PATH.split("/")[-1] + "_"
            candidates = [f for f in listdir(model_dir) if isfile(join(model_dir, f)) and f.startswith(prefix) and QUALIFIED_PREFIX not in f]
            if candidates:
                use_fallback = True
        except Exception:
            pass
    numbered = [(f, _snapshot_number(f)) for f in candidates]
    numbered = [(f, n) for f, n in numbered if n is not None]
    files = [f for f, _ in sorted(numbered, key=lambda pair: pair[1])][-3:]
    # лог только 1 раз из главного процесса, иначе 8 воркеров спамят (на Windows spawn _MAIN_PID не работает)
    if use_fallback and files and multiprocessing.current_process().name == "MainProcess":
        try:
            from agents.config import MIN_WINRATE_TO_QUALIFY
            print(f"self_play fallback: нет qualified (порог {MIN_WINRATE_TO_QUALIFY}), беру последние {len(files)} обычных снапшотов: {files}")
        except Exception:
            pass

    players = []
    for fname in files:
        try:
            snap = PPO.load(join(model_dir, fname), device="cpu")
            # проверка совместимости: если снапшот был обучен на другом N_FEATURES, пропускаем
            # (иначе ppo.policy будет падать на mismatch observation_space)
            try:
                obs_dim = snap.observation_space["observation"].shape[0]  # type: ignore
                if obs_dim != N_FEATURES:
                    print(f"Skip qualified snapshot {fname}: obs {obs_dim} != {N_FEATURES}")
                    continue
            except Exception:
                pass
            players.append(PolicyPlayer(policy=snap.policy, battle_format=BATTLE_FORMAT, start_listening=False))
        except Exception as e:
            print(f"Failed to load qualified snapshot {fname}: {e}")
    return players


class ExampleEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32) for agent in self.possible_agents
        }
        _attach_fusion_parser(self.agent1)
        _attach_fusion_parser(self.agent2)
        # для dense награды: храним предыдущее состояние по battle_tag
        self._reward_state: dict[str, dict] = {}

    @classmethod
    def create_env(cls, opponent_weights: dict[str, float] | None = None) -> Monitor:
        env = cls(battle_format=BATTLE_FORMAT, log_level=40, open_timeout=None)
        # Тренируем только против сильного соперника: Random/Max слишком легкие,
        # агент находит читерскую стратегию против них и забывает эвристику (31% -> 13%).
        # Оставляем их только в evaluate_win_rates для проверки, что не деградировал.
        heuristics = [
            SimpleHeuristicsPlayer(start_listening=False),
        ]
        self_play_opp = _make_self_play_opponents()
        # self-play оппоненты с start_listening=False не получают _handle_battle_message —
        # их _fusion_stats/_protect_state всегда пустые. Шэрим готовые словари с agent1
        # (только чтение), но НЕ шэрим _pending_stats_side — это промежуточное состояние
        # которое ломалось при общем dict. Агент (agent1) единственный парсит поток.
        for opp in self_play_opp:
            try:
                opp._fusion_stats = env.agent1._fusion_stats  # type: ignore
                opp._protect_state = env.agent1._protect_state  # type: ignore
            except Exception:
                pass

        all_opponents = heuristics + self_play_opp

        if not all_opponents:
            # fallback если пусто
            opponent = SimpleHeuristicsPlayer(start_listening=False)
            return Monitor(SingleAgentWrapper(env, opponent))

        if opponent_weights:
            # opponent_weights приходит из training._get_opponent_weights
            # ключи: "RandomPlayer","MaxBasePowerPlayer","SimpleHeuristicsPlayer","self_play"
            # Для self_play вес делится поровну между всеми снапшотами.
            weights = []
            # суммарный вес self_play
            self_play_total = opponent_weights.get("self_play", None)
            # если self_play не в словаре (старые чекпоинты) - считаем его как среднее
            if self_play_total is None:
                # равномерное распределение если ключа нет
                self_play_total = 0.25  # fallback 25%
            n_sp = len(self_play_opp)
            for opp in all_opponents:
                if opp in self_play_opp:
                    w = self_play_total / max(n_sp, 1) if n_sp else 0
                else:
                    w = opponent_weights.get(type(opp).__name__, 1.0 / len(all_opponents))
                weights.append(max(w, 1e-6))
            # нормализуем (random.choices делает это сам, но делаем явно для стабильности)
            total = sum(weights)
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(all_opponents)] * len(all_opponents)

        opponent = random.choices(all_opponents, weights=weights, k=1)[0]
        return Monitor(SingleAgentWrapper(env, opponent))

    def _hazard_score(self, side_conditions: dict) -> float:
        """Суммарный скор хазардов 0..~2.5"""
        s = 0.0
        for cond, per_layer in HAZARD_REWARDS.items():
            v = side_conditions.get(cond, 0)
            if isinstance(v, bool):
                v = 1 if v else 0
            try:
                v = int(v)
            except Exception:
                v = 1 if v else 0
            if v:
                # SPIKES 3 слоя, TOXIC 2 слоя — per_layer уже за слой
                if cond in (SideCondition.SPIKES, SideCondition.TOXIC_SPIKES):
                    s += v * per_layer
                else:
                    s += per_layer if v > 0 else 0.0
        return s

    def _boost_score(self, boosts: dict, positive: bool = True) -> float:
        """Сумма бустов. positive=True -> только >0, иначе только <0 (по модулю)"""
        if not boosts:
            return 0.0
        total = 0.0
        for k, v in boosts.items():
            if k not in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
                continue
            if positive and v > 0:
                total += v
            elif not positive and v < 0:
                total += abs(v)
        return total

    def _estimate_damage_mult(self, attacker, defender) -> float:
        """Устарел: оставлен для совместимости, теперь используй _estimate_max_damage"""
        return self._estimate_max_damage(attacker, defender, use_mult_only=True)

    def _boost_multiplier(self, stage: int) -> float:
        if stage >= 0:
            return (2 + stage) / 2
        else:
            return 2 / (2 - stage)

    def _estimate_max_damage(self, attacker, defender, use_mult_only: bool = False, attacker_fusion: dict | None = None, defender_fusion: dict | None = None) -> float:
        """
        Оценка макс урона атакующего по защитнику 0..~4 (нормирована).
        Если use_mult_only=True — только по типам (старый путь, для совместимости).
        Иначе: base_power * mult / defense_stat * attack_stat с учётом категории и бустов.
        Для статуса (base_power 0) — не считаем.
        Статы берутся из таблицы в чате (fusion) если найдена, иначе стандартные — как просил.
        """
        if attacker is None or defender is None:
            return 1.0
        try:
            # fallback по типам если нет мувов или просили только мульт
            if use_mult_only or not getattr(attacker, "moves", {}):
                atk_types = [t for t in (attacker.type_1, attacker.type_2) if t is not None]
                if not atk_types:
                    return 1.0
                max_mult = 0.0
                for atk in atk_types:
                    try:
                        mult = atk.damage_multiplier(defender.type_1, defender.type_2)
                    except Exception:
                        try:
                            from poke_env.data import GenData as GD
                            chart = GD.from_gen(9).type_chart
                            mult = atk.damage_multiplier(defender.type_1, defender.type_2, type_chart=chart)
                        except Exception:
                            mult = 1.0
                    max_mult = max(max_mult, mult)
                return float(max_mult) if max_mult else 1.0

            # основной путь: перебираем реальные мувы атакующего
            max_dmg = 0.0
            has_damaging = False
            for move in list(getattr(attacker, "moves", {}).values())[:8]:
                try:
                    bp = getattr(move, "base_power", 0) or 0
                    # poke-env иногда 0 для статуса, но в entry может быть power
                    if bp == 0:
                        entry = getattr(move, "entry", {}) or {}
                        bp = entry.get("basePower", 0) or entry.get("base_power", 0) or 0
                    if not bp or bp < 10:
                        continue  # статус — пропускаем
                    has_damaging = True
                    # категория
                    cat = getattr(move, "category", None)
                    cat_name = ""
                    if cat is not None:
                        cat_name = getattr(cat, "name", str(cat)).upper()
                    else:
                        entry = getattr(move, "entry", {}) or {}
                        cat_name = str(entry.get("category", "")).upper()
                    is_physical = cat_name == "PHYSICAL"
                    is_special = cat_name == "SPECIAL"
                    if not is_physical and not is_special:
                        # fallback: считаем физическим если не статус
                        is_physical = True

                    # типовая эффективность
                    mtype = getattr(move, "type", None)
                    if mtype is None:
                        entry = getattr(move, "entry", {}) or {}
                        tname = entry.get("type", "")
                        if tname:
                            try:
                                from poke_env.battle import PokemonType
                                mtype = PokemonType.from_name(tname)
                            except Exception:
                                mtype = None
                    mult = 1.0
                    if mtype is not None:
                        try:
                            mult = mtype.damage_multiplier(defender.type_1, defender.type_2)
                        except Exception:
                            try:
                                from poke_env.data import GenData as GD
                                chart = GD.from_gen(9).type_chart
                                mult = mtype.damage_multiplier(defender.type_1, defender.type_2, type_chart=chart)
                            except Exception:
                                mult = 1.0
                    if mult == 0:
                        # иммун — урон 0
                        continue

                    # статы + бусты
                    # base_stats берём из таблицы в чате (fusion) если найдена, иначе стандартные — как просил
                    def _get_base(mon, stat, fusion):
                        try:
                            if fusion and "base_stats" in fusion:
                                bs = fusion["base_stats"]
                                v = bs.get(stat, bs.get(stat.upper(), None))
                                if v is not None:
                                    return float(v)
                            bs = getattr(mon, "base_stats", {}) or {}
                            v = bs.get(stat, bs.get(stat.upper(), 80)) or 80
                            return float(v)
                        except Exception:
                            return 80.0
                    if is_physical:
                        atk_stat = _get_base(attacker, "atk", attacker_fusion)
                        def_stat = _get_base(defender, "def", defender_fusion)
                        atk_boost = getattr(attacker, "boosts", {}).get("atk", 0) if getattr(attacker, "boosts", None) else 0
                        def_boost = getattr(defender, "boosts", {}).get("def", 0) if getattr(defender, "boosts", None) else 0
                    else:
                        atk_stat = _get_base(attacker, "spa", attacker_fusion)
                        def_stat = _get_base(defender, "spd", defender_fusion)
                        atk_boost = getattr(attacker, "boosts", {}).get("spa", 0) if getattr(attacker, "boosts", None) else 0
                        def_boost = getattr(defender, "boosts", {}).get("spd", 0) if getattr(defender, "boosts", None) else 0

                    # бусты в множитель
                    atk_stat *= self._boost_multiplier(int(atk_boost))
                    def_stat *= self._boost_multiplier(int(def_boost))
                    # защита не может быть 0
                    def_stat = max(def_stat, 1.0)

                    # прокси урона: power * mult * atk / def  (нормируем /100 чтобы влезло в 0..4)
                    dmg = (float(bp) * float(mult) * float(atk_stat) / float(def_stat)) / 100.0
                    # клип 0..4
                    dmg = float(np.clip(dmg, 0, 4))
                    max_dmg = max(max_dmg, dmg)
                except Exception:
                    continue

            if not has_damaging:
                # нет дамажных мувов — fallback к типам
                return self._estimate_max_damage(attacker, defender, use_mult_only=True)
            return float(max_dmg) if max_dmg > 0 else 0.0
        except Exception:
            return 1.0

    def calc_reward(self, battle) -> float:
        base = self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, status_value=0.5, victory_value=30.0,
        )
        extra = 0.0
        tag = getattr(battle, "battle_tag", None) or getattr(battle, "battle_tag", "unknown")
        try:
            # достаём текущие значения
            opp_haz = self._hazard_score(battle.opponent_side_conditions)
            own_haz = self._hazard_score(battle.side_conditions)
            # активные HP
            curr_own_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
            curr_opp_hp = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0.0
            # бусты
            curr_own_boost = self._boost_score(getattr(battle.active_pokemon, "boosts", {}) if battle.active_pokemon else {}, positive=True)
            curr_opp_debuff = self._boost_score(getattr(battle.opponent_active_pokemon, "boosts", {}) if battle.opponent_active_pokemon else {}, positive=False)
            curr_active_species = getattr(getattr(battle.active_pokemon, "species", None), "lower", lambda: "")() if battle.active_pokemon else ""
            if not curr_active_species and battle.active_pokemon:
                curr_active_species = str(getattr(battle.active_pokemon, "species", ""))[:40]
            curr_status = getattr(battle.active_pokemon, "status", None) if battle.active_pokemon else None
            curr_opp_status = getattr(battle.opponent_active_pokemon, "status", None) if battle.opponent_active_pokemon else None
            curr_is_tera = bool(getattr(battle.active_pokemon, "is_terastallized", False)) if battle.active_pokemon else False
            curr_protected = False
            try:
                if self.battle1 is not None and getattr(battle, "battle_tag", None) == getattr(self.battle1, "battle_tag", None):
                    src_tmp = self.agent1
                elif self.battle2 is not None and getattr(battle, "battle_tag", None) == getattr(self.battle2, "battle_tag", None):
                    src_tmp = self.agent2
                else:
                    src_tmp = self.agent1 if getattr(battle, "player_role", "p1") == "p1" else self.agent2
                tag_tmp = getattr(battle, "battle_tag", "")
                prot_state = src_tmp._protect_state.get(tag_tmp, {}) if hasattr(src_tmp, "_protect_state") else {}
                our_side_tmp = getattr(battle, "player_role", "p1")
                curr_protected = bool(prot_state.get(f"last_{our_side_tmp}", False))
                if not curr_protected and battle.active_pokemon and any("protect" in str(k).lower() for k in getattr(battle.active_pokemon, "effects", {}).keys()):
                    curr_protected = True
            except Exception:
                curr_protected = False
            # предыдущее состояние
            prev = self._reward_state.get(tag)
            if prev is None:
                # первый вызов для этого боя — инициализируем и не даём extra (иначе награда за стартовые хазарды 0)
                self._reward_state[tag] = {
                    "opp_haz": opp_haz,
                    "own_haz": own_haz,
                    "own_hp": curr_own_hp,
                    "opp_hp": curr_opp_hp,
                    "own_boost": curr_own_boost,
                    "opp_debuff": curr_opp_debuff,
                    "active_species": curr_active_species,
                    "active_hp": curr_own_hp,
                    "team_hp": {k: v.current_hp_fraction for k, v in battle.team.items()},
                    "opp_team_hp": {k: v.current_hp_fraction for k, v in battle.opponent_team.items()},
                    "turn": getattr(battle, "turn", 0),
                    "status": curr_status,
                    "opp_status": curr_opp_status,
                    "is_tera": curr_is_tera,
                    "protected": curr_protected,
                }
                return base - 0.02

            # 1) Hazards: размещение/снятие + штраф за урон от них
            d_opp_haz = opp_haz - prev["opp_haz"]
            d_own_haz = own_haz - prev["own_haz"]
            if d_opp_haz > 0:
                extra += d_opp_haz * 1.0
            elif d_opp_haz < 0:
                # сняли свои хазарды с оппа (Defog) — штраф, как просил
                extra -= abs(d_opp_haz) * HAZARD_SELF_CLEAR_PENALTY
            if d_own_haz > 0:
                extra -= d_own_haz * 0.6
            elif d_own_haz < 0:
                # сняли хазарды оппа с себя (Rapid Spin/Defog) — бонус
                extra += abs(d_own_haz) * HAZARD_CLEAR_BONUS

            # 2) Урон: по активному противнику (учитывает DEF/SPD уже через HP дельту)
            prev_opp_active_species = prev.get("opp_active_species", "")
            curr_opp_species = getattr(getattr(battle.opponent_active_pokemon, "species", None), "lower", lambda: "")() if battle.opponent_active_pokemon else ""
            opp_switched = prev_opp_active_species and curr_opp_species and prev_opp_active_species != curr_opp_species
            if not opp_switched:
                dmg_dealt = prev["opp_hp"] - curr_opp_hp
                if dmg_dealt > 1e-6:
                    extra += dmg_dealt * DAMAGE_WEIGHT
                dmg_taken = prev["own_hp"] - curr_own_hp
                if dmg_taken > 1e-6:
                    extra -= dmg_taken * DAMAGE_TAKEN_PENALTY
            # 3) Бусты своих
            d_own_boost = curr_own_boost - prev["own_boost"]
            if d_own_boost > 0:
                extra += d_own_boost * BOOST_WEIGHT
            # 4) Дебаффы противника
            d_opp_debuff = curr_opp_debuff - prev["opp_debuff"]
            if d_opp_debuff > 0:
                extra += d_opp_debuff * DEBUFF_WEIGHT

            # 5) Удачный свитч (с DEF/SPD)
            prev_species = prev.get("active_species", "")
            was_switch = False
            if prev_species and curr_active_species and prev_species != curr_active_species:
                prev_active_hp = prev.get("active_hp", 1.0)
                curr_turn = getattr(battle, "turn", 0)
                prev_turn = prev.get("turn", 0)
                if prev_active_hp > 0.05 and curr_turn > prev_turn:
                    was_switch = True
            if was_switch:
                # оцениваем урон который бы получил старый покемон vs новый от текущего активного противника
                prev_mon = None
                # ищем prev_mon в team по species
                for mon in battle.team.values():
                    try:
                        if str(getattr(mon, "species", "")).lower() == prev_species.lower():
                            prev_mon = mon
                            break
                    except Exception:
                        pass
                new_mon = battle.active_pokemon
                opp_mon = battle.opponent_active_pokemon
                if prev_mon is not None and new_mon is not None and opp_mon is not None:
                    # берём таблицы статов из чата (fusion) если есть, иначе стандартные — как просил
                    try:
                        if self.battle1 is not None and getattr(battle, "battle_tag", None) == getattr(self.battle1, "battle_tag", None):
                            src = self.agent1
                        elif self.battle2 is not None and getattr(battle, "battle_tag", None) == getattr(self.battle2, "battle_tag", None):
                            src = self.agent2
                        else:
                            src = self.agent1 if getattr(battle, "player_role", "p1") == "p1" else self.agent2
                        tag_fs = getattr(battle, "battle_tag", "")
                        our_side_fs = getattr(battle, "player_role", "p1")
                        opp_side_fs = "p2" if our_side_fs == "p1" else "p1"
                        our_by_species = src._fusion_stats.get(tag_fs, {}).get(f"{our_side_fs}_by_species", {}) if hasattr(src, "_fusion_stats") else {}
                        opp_by_species = src._fusion_stats.get(tag_fs, {}).get(f"{opp_side_fs}_by_species", {}) if hasattr(src, "_fusion_stats") else {}
                        our_active_fusion = src._fusion_stats.get(tag_fs, {}).get(our_side_fs) if hasattr(src, "_fusion_stats") else None
                        opp_active_fusion = src._fusion_stats.get(tag_fs, {}).get(opp_side_fs) if hasattr(src, "_fusion_stats") else None
                        def _fusion_for(mon, is_opp):
                            try:
                                from poke_env.data.normalize import to_id_str
                                sid = to_id_str(getattr(mon, "species", "") or getattr(mon, "base_species", ""))
                                m = opp_by_species if is_opp else our_by_species
                                f = m.get(sid) if m else None
                                if f:
                                    return f
                                if mon is opp_mon and opp_active_fusion:
                                    return opp_active_fusion
                                if (mon is new_mon or mon is prev_mon) and mon is not None and getattr(mon, "active", False) and our_active_fusion:
                                    return our_active_fusion
                                return f
                            except Exception:
                                return None
                        prev_fusion = _fusion_for(prev_mon, False)
                        new_fusion = _fusion_for(new_mon, False)
                        opp_fusion = _fusion_for(opp_mon, True)
                    except Exception:
                        prev_fusion = new_fusion = opp_fusion = None
                    # считаем урон с учётом DEF/SPD и категории приёма (а не только типы)
                    prev_dmg = self._estimate_max_damage(opp_mon, prev_mon, attacker_fusion=opp_fusion, defender_fusion=prev_fusion)
                    new_dmg = self._estimate_max_damage(opp_mon, new_mon, attacker_fusion=opp_fusion, defender_fusion=new_fusion)
                    # для логов/порогов оставим и чистый mult (для иммун проверки)
                    prev_mult = self._estimate_max_damage(opp_mon, prev_mon, use_mult_only=True)
                    new_mult = self._estimate_max_damage(opp_mon, new_mon, use_mult_only=True)
                    delta_dmg = prev_dmg - new_dmg
                    # также считаем фактический урон полученный новым моном в этот ход
                    # prev_new_hp — HP нового мона до свитча (из bench)
                    prev_new_hp = prev.get("team_hp", {}).get(next((k for k, v in battle.team.items() if str(getattr(v, "species","")).lower()==curr_active_species.lower()), ""), None)
                    # fallback: ищем в prev team_hp по ключу
                    if prev_new_hp is None:
                        for k, v in prev.get("team_hp", {}).items():
                            try:
                                if str(getattr(battle.team.get(k, None), "species","")).lower() == curr_active_species.lower():
                                    prev_new_hp = v
                                    break
                            except Exception:
                                pass
                    dmg_taken_new = 0.0
                    if prev_new_hp is not None:
                        try:
                            dmg_taken_new = max(0.0, float(prev_new_hp) - float(curr_own_hp))
                        except Exception:
                            dmg_taken_new = 0.0
                    # порог: если новый получил на SWITCH_THRESHOLD меньше урона чем ожидалось для старого
                    is_success = False
                    is_immune_or_reflect = False
                    # проверка иммун по типам (0) или по урону 0 (защита+иммун)
                    if new_dmg == 0 and prev_dmg > 0.1:
                        is_success = True
                        is_immune_or_reflect = True
                    elif new_mult == 0 and prev_mult > 0:
                        is_success = True
                        is_immune_or_reflect = True
                    elif delta_dmg >= 0.8:  # значимое снижение с учётом DEF/SPD (напр 2.2->1.0)
                        is_success = True
                    elif dmg_taken_new < SWITCH_THRESHOLD and prev_dmg >= 1.2:
                        # новый получил <30% HP урона, а старый бы получил >=1.2 (сильный хит) — успех
                        is_success = True
                    # статус: если новый не застатуслен, а старый был бы уязвим
                    # проверяем: если у нового после свитча статус None, а ход противника был статусным и теперь без эффекта
                    try:
                        new_status = getattr(new_mon, "status", None)
                        # проверяем отражение: у противника появился статус после нашего свитча на Magic Bounce
                        opp_status = getattr(opp_mon, "status", None)
                        new_ability = str(getattr(new_mon, "ability", "") or "").lower()
                        if new_status is None and opp_status is not None and "magicbounce" in new_ability:
                            is_success = True
                            is_immune_or_reflect = True
                        # также если новый имунен к типу статуса (яд/сталь к токсику, огонь к ожогу)
                        # считаем что если новый тип имунен — это тоже отражение/иммун
                        if new_status is None and new_mult == 0:
                            # уже учтено выше
                            pass
                    except Exception:
                        pass

                    if is_success:
                        if is_immune_or_reflect:
                            extra += SWITCH_IMMUNE_BONUS  # x2
                        else:
                            extra += SWITCH_BONUS
                    # также небольшой бонус если вообще сменили на более толстый резист (с учётом DEF/SPD)
                    elif delta_dmg > 0.4:
                        extra += 0.2

            # штраф за урон от хазардов при свитче (дополнительно к dmg_taken, который уже учтён)
            if was_switch and own_haz > 0.01:
                # Stealth Rock 0.8 -> -0.14, 3 слоя Spikes 1.5 -> -0.27
                extra -= own_haz * HAZARD_DAMAGE_PENALTY * 0.6

            # удачный протект (был в protect прошлом ходом и не получил урона, опп бил)
            try:
                prev_protected = bool(prev.get("protected", False))
                # protect длится 1 ход: prev_protected True означает в прошлом ходу нажали Protect
                # сейчас уже нет, но в прошлом ходу урона не было и опп не свитчил
                if prev_protected and not curr_protected:
                    dmg_taken_protect = prev.get("own_hp", curr_own_hp) - curr_own_hp
                    if abs(dmg_taken_protect) < 1e-6 and not opp_switched and abs(prev.get("opp_hp", curr_opp_hp) - curr_opp_hp) < 1e-6:
                        # опп пытался атаковать но мы в протект — бонус
                        extra += PROTECT_BONUS
            except Exception:
                pass

            # клин статуса (Heal Bell / Natural Cure)
            try:
                prev_status = prev.get("status", None)
                if prev_status is not None and curr_status is None and curr_own_hp > 0.05:
                    extra += STATUS_CURE_BONUS
            except Exception:
                pass

            # удачный тера (смена типа дала резист или добила)
            try:
                prev_is_tera = bool(prev.get("is_tera", False))
                if not prev_is_tera and curr_is_tera:
                    opp_fainted_now = curr_opp_hp == 0 and prev.get("opp_hp", 0) > 0.2
                    if opp_fainted_now:
                        dmg_before = prev_dmg if 'prev_dmg' in locals() else 1.0
                        # если до теры урон был бы не гарантированно летальным (<1.5) — полный бонус, иначе малый
                        if dmg_before < 1.5:
                            extra += TERA_BONUS
                        else:
                            extra += 0.15
                    else:
                        # без убийства — проверяем снижение входящего урона от теры
                        # delta_dmg уже есть если был свитч+тера, иначе считаем заново
                        has_delta = 'delta_dmg' in locals() and delta_dmg > 0.6
                        has_immune = 'new_mult' in locals() and new_mult == 0 and prev_mult != 0
                        if has_delta or has_immune:
                            extra += TERA_BONUS
            except Exception:
                pass

            # обновляем состояние
            self._reward_state[tag] = {
                "opp_haz": opp_haz,
                "own_haz": own_haz,
                "own_hp": curr_own_hp,
                "opp_hp": curr_opp_hp,
                "own_boost": curr_own_boost,
                "opp_debuff": curr_opp_debuff,
                "active_species": curr_active_species,
                "active_hp": curr_own_hp,
                "team_hp": {k: v.current_hp_fraction for k, v in battle.team.items()},
                "opp_team_hp": {k: v.current_hp_fraction for k, v in battle.opponent_team.items()},
                "opp_active_species": curr_opp_species,
                "turn": getattr(battle, "turn", 0),
                "status": curr_status,
                "opp_status": curr_opp_status,
                "is_tera": curr_is_tera,
                "protected": curr_protected,
            }
            # чистим завершённые бои (победа/поражение)
            if getattr(battle, "finished", False):
                self._reward_state.pop(tag, None)

        except Exception as e:
            # не роняем шаг из-за награды
            # print(f"reward shaping warn {e}")
            pass

        # клип extra чтобы не взорвать PPO (VecNormalize нормализует, но всё равно)
        extra = float(np.clip(extra, -2.0, 2.0))
        return base + extra - 0.02

    def action_to_order(self, action, battle, fake=False, strict=True):
        mask = SinglesEnv.get_action_mask(battle)
        if sum(mask) == 0:
            from poke_env.player import DefaultBattleOrder
            return DefaultBattleOrder()
        return super().action_to_order(action, battle, fake=fake, strict=strict)

    def embed_battle(self, battle: AbstractBattle):
        # Надёжный выбор источника по battle_tag (а не по is и не по player_role).
        # is ломался: SinglesEnv передаёт копию battle, p1!=agent1 по идентичности.
        # player_role==p1 -> agent1 — эвристика, может сломаться при реконнекте.
        # battle_tag — стабильный id боя, единственный надёжный ключ.
        try:
            # сначала пробуем по тегу (работает даже если player_role переприсвоили)
            if self.battle1 is not None and getattr(battle, "battle_tag", None) == getattr(self.battle1, "battle_tag", None):
                source = self.agent1
            elif self.battle2 is not None and getattr(battle, "battle_tag", None) == getattr(self.battle2, "battle_tag", None):
                source = self.agent2
            else:
                # fallback: player_role (покрывает случай когда battle1/2 ещё None в начале боя)
                source = self.agent1 if getattr(battle, "player_role", "p1") == "p1" else self.agent2
        except Exception:
            try:
                source = self.agent1 if getattr(battle, "player_role", "p1") == "p1" else self.agent2
            except Exception:
                source = self.agent1 if battle is self.battle1 else self.agent2
        fusion_entry = lambda is_ours: (
            source._fusion_stats.get(battle.battle_tag, {})
            .get(battle.player_role if is_ours else ("p2" if battle.player_role == "p1" else "p1"))
        )
        protect_state = source._protect_state.get(battle.battle_tag, {})
        our_side = battle.player_role
        opp_side = "p2" if our_side == "p1" else "p1"
        our_protect = 1.0 if protect_state.get(f"last_{our_side}", False) else 0.0
        opp_protect = 1.0 if protect_state.get(f"last_{opp_side}", False) else 0.0
        # per-species карты для скамейки (было None -> заниженные дексовые статы резерва)
        our_team_fusions = source._fusion_stats.get(battle.battle_tag, {}).get(f"{our_side}_by_species")
        opp_team_fusions = source._fusion_stats.get(battle.battle_tag, {}).get(f"{opp_side}_by_species")
        return embed_battle_with_fusion(
            battle,
            our_fusion=fusion_entry(True),
            opp_fusion=fusion_entry(False),
            our_protected_last_turn=our_protect,
            opp_protected_last_turn=opp_protect,
            our_team_fusions=our_team_fusions,
            opp_team_fusions=opp_team_fusions,
        )
