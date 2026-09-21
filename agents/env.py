import random
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle, SideCondition, Status, Weather, Field, Effect
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .config import BATTLE_FORMAT, N_FEATURES, QUALIFIED_PREFIX
from .features import _move_wasted_flag, embed_battle_with_fusion, move_slots_for_action

try:
    from .type_utils import damage_multiplier_safe, summary_line
except ImportError:  # запуск модуля вне пакета
    from type_utils import damage_multiplier_safe, summary_line
from .fusion_parser import _attach_fusion_parser
from .players import PolicyPlayer

import multiprocessing

# Типы мон для награды: на поле сервер уже сообщил тип (тера/typechange), а у скамейки
# poke-env чистит `_temporary_types` -> там типы дексовые «головы». Считаем фьюжн-тип сами
# (см. fusion_types), чтобы награда видела ту же картину, что и признаки.
_EFFECTIVE_TYPES = None


def _eff_types(mon):
    global _EFFECTIVE_TYPES
    if mon is None:
        return None, None
    if _EFFECTIVE_TYPES is None:
        try:
            from .fusion_types import effective_types as _et
        except ImportError:  # запуск модуля вне пакета
            from fusion_types import effective_types as _et
        _EFFECTIVE_TYPES = _et
    try:
        t1, t2, _ = _EFFECTIVE_TYPES(mon)
        return t1, t2
    except Exception:
        return getattr(mon, "type_1", None), getattr(mon, "type_2", None)


# --- веса для новой награды (сбалансированы под базу fainted 2.0 / hp 1.0 / victory 30)
# ИСПРАВЛЕНИЕ: суммарный shaping за бой раньше мог быть 40-80 > victory 30 → доминировал над победой.
# Сжали в 4 раза + per-episode бюджет 12, чтобы победа оставалась главным сигналом.
HAZARD_REWARDS = {
    SideCondition.STEALTH_ROCK: 0.20,   # было 0.8
    SideCondition.SPIKES: 0.12,         # было 0.5
    SideCondition.TOXIC_SPIKES: 0.12,   # было 0.5
    SideCondition.STICKY_WEB: 0.15,     # было 0.6
}
DAMAGE_WEIGHT = 0.20          # было 0.8
DAMAGE_TAKEN_PENALTY = 0.10   # было 0.4
BOOST_WEIGHT = 0.04           # было 0.15
DEBUFF_WEIGHT = 0.04          # было 0.15
SWITCH_BONUS = 0.12           # было 0.5
SWITCH_IMMUNE_BONUS = 0.25    # было 1.0 (x2)
SWITCH_THRESHOLD = 0.30
# новые — тоже сжаты
HAZARD_CLEAR_BONUS = 0.12         # было 0.5
HAZARD_SELF_CLEAR_PENALTY = 0.12   # было 0.5
HAZARD_DAMAGE_PENALTY = 0.08       # было 0.3
PROTECT_BONUS = 0.10               # было 0.4
STATUS_CURE_BONUS = 0.10           # было 0.4
TERA_BONUS = 0.08                  # было 0.3 (weak)
# расширение пула — всё понемногу 0.06, с cap 12
WEATHER_BONUS = 0.08
TERRAIN_BONUS = 0.06
SUBSTITUTE_BONUS = 0.08
SUBSTITUTE_WASTED_PENALTY = 0.06
LEECH_SEED_BONUS = 0.08
LEECH_WASTED_PENALTY = 0.06
HEAL_BONUS = 0.08
HEAL_WASTED_PENALTY = 0.06
SHAPING_EPISODE_CAP = 12.0         # макс суммарный shaping за бой (меньше victory 30)
SHAPING_STEP_CLIP = 0.50           # клип на ход (было 2.0) — ещё сильнее жмём одиночный всплеск
WASTED_MOVE_PENALTY = 0.06       # универсальный штраф за любой wasted приём (если ещё не наказан спецификой)
TIME_PENALTY = 0.02             # штраф за шаг времени (начисляется РОВНО один на решение — см. DecisionWrapper)
WASTED_MOVE_IDS_SKIP = {"leechseed"}  # только leech уже имеет спец-штраф; погода/терен/саб теперь идут через generic wasted (фикс: раньше не штрафовались когда уже активны)
WEATHER_MOVE_IDS = {"sunnyday","raindance","sandstorm","snowscape","chillyreception"}
TERRAIN_MOVE_IDS = {"electricterrain","grassyterrain","mistyterrain","psychicterrain"}
STATUS_IMMUNE_TYPES = {
    "par": ["electric", "ground"],
    "brn": ["fire"],
    "psn": ["poison", "steel"],
    "tox": ["poison", "steel"],
    "slp": [], "frz": [],
}


def _snapshot_number(fname: str) -> int | None:
    suffix = fname.split("_")[-1].split(".")[0]
    return int(suffix) if suffix.isdigit() else None


_SELF_PLAY_LOAD_ERRORS_REPORTED: set = set()


_SELF_PLAY_NORM_CACHE: dict = {}


def _self_play_obs_normalizer():
    """Статистика нормализации obs для self-play оппонентов (None — если нет/выключено).

    Оппоненты — те же обученные снапшоты, а обучение идёт с `norm_obs=True`; без той же
    нормализации снапшот видит сдвинутый obs и играет заметно хуже, из-за чего self-play
    награда/винрейт в ratchet измеряются по «сломанному» сопернику.
    Отключается переменной окружения `PYBOT_SELF_PLAY_NORM=0` (если обучение идёт без
    VecNormalize — `--no-normalize-bc`).
    """
    import os
    if os.environ.get("PYBOT_SELF_PLAY_NORM", "1") == "0":
        return None
    if "value" in _SELF_PLAY_NORM_CACHE:
        return _SELF_PLAY_NORM_CACHE["value"]
    stats = None
    try:
        from .config import VECNORM_PATH
        from .vecnorm_utils import load_vecnorm_stats
        if os.path.isfile(VECNORM_PATH):
            stats = load_vecnorm_stats(VECNORM_PATH, N_FEATURES)
            if stats is not None and multiprocessing.current_process().name == "MainProcess":
                print(f"self-play оппоненты: obs нормализуются ({stats.describe()})")
                warn = stats.stale_warning()
                if warn:
                    print(warn)
    except Exception:
        stats = None
    _SELF_PLAY_NORM_CACHE["value"] = stats
    return stats


def _make_self_play_opponents(model_dir: str = "models/", cache_dir: str | None = None):
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
    # если есть meta — сортируем qualified по winrate, иначе по номеру
    meta_path = join(model_dir, "qualified_meta.json")
    winrate_map = {}
    try:
        import json, os
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                for h in meta.get("history", []):
                    fn = h.get("file", "")
                    # candidates включает расширение? в listdir без пути, сравниваем basename
                    winrate_map[fn] = float(h.get("winrate_vs_heuristics", 0))
                    # также без расширения zip?
                    if fn.endswith(".zip"):
                        winrate_map[fn] = float(h.get("winrate_vs_heuristics", 0))
                    else:
                        winrate_map[fn + ".zip"] = float(h.get("winrate_vs_heuristics", 0))
                        winrate_map[fn] = float(h.get("winrate_vs_heuristics", 0))
    except Exception:
        winrate_map = {}
    numbered = [(f, _snapshot_number(f)) for f in candidates]
    numbered = [(f, n) for f, n in numbered if n is not None]
    if winrate_map and any(f in winrate_map for f,_ in numbered):
        # сортируем по winrate desc, затем по номеру desc (свежие сильнее при равных)
        def _key(pair):
            f,n = pair
            return (winrate_map.get(f, -1), n or -1)
        files = [f for f,_ in sorted(numbered, key=_key)][-3:]
        # но если winrate_map содержит не все файлы (старые без meta), fallback к номеру для них
    else:
        files = [f for f, _ in sorted(numbered, key=lambda pair: pair[1])][-3:]
    # лог только 1 раз из главного процесса, иначе 8 воркеров спамят (на Windows spawn _MAIN_PID не работает)
    if use_fallback and files and multiprocessing.current_process().name == "MainProcess":
        try:
            from agents.config import MIN_WINRATE_TO_QUALIFY
            # динамический порог если есть meta
            thr = MIN_WINRATE_TO_QUALIFY
            try:
                import json, os
                if os.path.isfile(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                        thr = int(meta.get("threshold", thr))
            except Exception:
                pass
            print(f"self_play fallback: нет qualified (порог {thr}), беру последние {len(files)} обычных снапшотов: {files}")
        except Exception:
            pass

    players = []
    for fname in files:
        try:
            from .checkpoint_utils import load_policy_compat

            snap, info = load_policy_compat(join(model_dir, fname), N_FEATURES, cache_dir=cache_dir)
            if snap is None:
                # раньше здесь падал PPO.load на 715-снапшотах и спамил на каждой фазе
                if fname not in _SELF_PLAY_LOAD_ERRORS_REPORTED:
                    _SELF_PLAY_LOAD_ERRORS_REPORTED.add(fname)
                    print(f"Failed to load qualified snapshot {fname}: {info.get('error')}")
                continue
            # Режим self-play снапшота обязан совпасть с режимом процесса: иначе он выбирает
            # индекс в своей семантике (напр. indices-модель — индекс в порядке team), а
            # маска/ордера в этом процессе — в текущей. Молча это деградирует обучение
            # (соперник играет шумом), поэтому предупреждаем один раз на файл.
            try:
                from .checkpoint_utils import action_mode_from_checkpoint
                from .action_space import get_action_mode

                snap_mode = action_mode_from_checkpoint(join(model_dir, fname))
                cur_mode = get_action_mode()
                if snap_mode and snap_mode != cur_mode:
                    key = f"mode:{fname}"
                    if key not in _SELF_PLAY_LOAD_ERRORS_REPORTED:
                        _SELF_PLAY_LOAD_ERRORS_REPORTED.add(key)
                        print(f"self-play {fname}: режим {snap_mode}, а прогон в {cur_mode} — "
                              f"этот соперник будет играть некорректно (пересоберите снапшоты "
                              f"в текущем режиме или исключите файл)")
            except Exception:
                pass
            players.append(PolicyPlayer(policy=snap.policy, battle_format=BATTLE_FORMAT,
                                        start_listening=False,
                                        obs_normalizer=_self_play_obs_normalizer()))
        except Exception as e:
            if fname not in _SELF_PLAY_LOAD_ERRORS_REPORTED:
                _SELF_PLAY_LOAD_ERRORS_REPORTED.add(fname)
                print(f"Failed to load qualified snapshot {fname}: {e}")
    return players


class DecisionWrapper(SingleAgentWrapper):
    """Один шаг наружу = одно НАСТОЯЩЕЕ решение агента.

    Зачем. poke-env отдаёт агенту не только состояния «наш ход»: сервер присылает
    `|request|` с `wait: true` (мы уже выбрали, ждём соперника), и такой кадр тоже
    превращается в шаг RL. В этот момент:

      * `PokeEnv.step` вовсе НЕ зовёт `action_to_order` для нашей стороны
        (`agent1_to_move == False`) — выбранное действие просто выбрасывается;
      * obs приходит с маской `[1, 0, 0, ...]` (`get_action_mask`: `if battle._wait: actions = [0]`),
        то есть разрешено ровно одно действие — индекс 0, который в раскладке SinglesEnv
        означает СВИТЧ. Политика с additive-маской обязана его выбрать.

    Итог без этой обёртки: в роллаут-буфер попадают шаги-пустышки (награда ≈ −штраф времени,
    действие игнорируется), а диагностика `[mix]` показывает свитчи, которых в бою не было
    (замер на живом сервере: 10–12% всех «решений» и до половины «свитчей» — ровно такие шаги).

    Обёртка прокручивает состояния ожидания внутри себя и наружу отдаёт только то состояние,
    на котором действие будет применено. Награды суммируются (учёт внутри `calc_reward`
    дельта-базированный, поэтому сумма корректна), а лишние штрафы времени возвращаются:
    сколько раз сервер заставил ждать — не подконтрольно агенту и не должно менять награду.
    """

    MAX_SPIN = 50   # страховка от зависания (недоступный соперник, замороженный бой)

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        total = float(reward)
        n_calls = 1
        spins = 0
        # agent1_to_move == True означает: пришедшее состояние — НАШ настоящий ход,
        # следующее действие будет применено. Если False — это состояние ожидания,
        # наружу его отдавать нельзя (иначе SB3 сделает шаг, которого в бою нет).
        while (not (term or trunc) and not getattr(self.env, "agent1_to_move", False)
               and spins < self.MAX_SPIN):
            obs, r, term, trunc, info = super().step(action)
            total += float(r)
            n_calls += 1
            spins += 1
        if n_calls > 1:
            # на решение должен приходиться ровно один штраф времени: сколько раз сервер
            # заставил ждать — не подконтрольно агенту и не должно менять награду
            total += TIME_PENALTY * (n_calls - 1)
        return obs, total, term, trunc, info


def _apply_worker_action_mode(action_mode: str | None) -> str | None:
    """Выставить режим действий внутри процесса-воркера (см. ExampleEnv.create_env).

    Возвращает фактический режим (None, если ничего не меняли). Импорт локальный: agents.env
    не должен тянуть action_space при обычном импорте (там настраивается poke-env).
    """
    if action_mode is None:
        return None
    try:
        from .action_space import set_action_mode, get_action_mode
    except ImportError:  # pragma: no cover — запуск модуля вне пакета
        from action_space import set_action_mode, get_action_mode  # type: ignore
    set_action_mode(action_mode)
    return get_action_mode()


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
        # для generic wasted — помним последний ход (move id) чтобы штрафовать любой wasted (если нет специфики)
        self._last_move_id: dict[str, str] = {}
        self._last_wasted: dict[str, bool] = {}
        self._last_was_switch: dict[str, bool] = {}
        self._last_action_kind: dict[str, str] = {}
        # диагностика: сколько свитчей/приёмов/тер выбрал агент (видно, есть ли «спам атаками»)
        self._action_counts: dict[str, int] = {"switch": 0, "move": 0, "tera": 0, "unknown": 0}

    @classmethod
    def create_env(cls, opponent_weights: dict[str, float] | None = None,
                   action_mode: str | None = None) -> Monitor:
        """Фабрика env для SubprocVecEnv. `action_mode` применяется В ВОРКЕРЕ первой строкой.

        Зачем явный аргумент, если режим и так экспортируется в PYBOT_ACTION_MODE: env-воркеры
        создаются ДО того, как в главном процессе выставится режим (и со spawn переимпортируют
        модули), поэтому полагаться только на переменную окружения нельзя. Режим, переданный
        аргументом, доезжает до воркера через pickle и не зависит от порядка инициализации.
        """
        _apply_worker_action_mode(action_mode)
        return cls._build_env(opponent_weights)

    @classmethod
    def _build_env(cls, opponent_weights: dict[str, float] | None = None) -> Monitor:
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
            return Monitor(DecisionWrapper(env, opponent))

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
        return Monitor(DecisionWrapper(env, opponent))

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
                atk_types = [t for t in _eff_types(attacker) if t is not None]
                if not atk_types:
                    return 1.0
                dfn_t1, dfn_t2 = _eff_types(defender)
                max_mult = 0.0
                for atk in atk_types:
                    # безопасный расчёт: неизвестный тип защиты не маскирует иммунитет
                    mult = damage_multiplier_safe(atk, dfn_t1, dfn_t2)
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
                        # было: try damage_multiplier -> except -> chart -> except -> 1.0
                        # (маскировало иммунитет при неизвестном втором типе)
                        dfn_t1, dfn_t2 = _eff_types(defender)
                        mult = damage_multiplier_safe(mtype, dfn_t1, dfn_t2)
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

    def _team_has_weather_synergy(self, team: dict, weather) -> bool:
        """Есть ли в команде абилка/приём зависящий от погоды — только тогда награждаем за погоду"""
        if weather is None:
            return False
        try:
            wname = getattr(weather, "name", str(weather)).lower()
            for mon in team.values():
                if mon.fainted:
                    continue
                ab = str(getattr(mon, "ability", "") or "").lower().replace(" ", "").replace("-", "")
                # Sunny: chlorophyll, flowergift, solar power, Solar Beam, Growth, Synthesis
                if "sunnyday" in wname:
                    if ab in ("chlorophyll", "flowergift", "solarpower", "orichalcumpulse"):
                        return True
                if "raindance" in wname:
                    if ab in ("swiftswim", "raindance", "hydration", "dryskin"):
                        return True
                if "sandstorm" in wname:
                    if ab in ("sandrush", "sandforce", "sandveil"):
                        return True
                if "snow" in wname:
                    if ab in ("slushrush", "icebody", "snowcloak"):
                        return True
                for mv in list(getattr(mon, "moves", {}).values()):
                    mid = str(getattr(mv, "id", "") or "").lower()
                    mname = mid.replace("-", "").replace(" ", "")
                    if "sunnyday" in wname and mname in ("solarbeam", "solarblade", "growth", "synthesis", "morningsun", "moonlight", "weatherball"):
                        return True
                    if "raindance" in wname and mname in ("thunder", "hurricane", "weatherball", "hydropump"):
                        return True
                    if "sandstorm" in wname and mname in ("weatherball",):
                        return True
                    if "snow" in wname and mname in ("blizzard", "auroraveil", "weatherball"):
                        return True
        except Exception:
            pass
        return False

    def _team_has_terrain_synergy(self, team: dict, field) -> bool:
        if field is None:
            return False
        try:
            fname = getattr(field, "name", str(field)).lower().replace("_","").replace(" ","").replace("-","")
            for mon in team.values():
                if mon.fainted:
                    continue
                ab = str(getattr(mon, "ability", "") or "").lower()
                if "electricterrain" in fname and ab in ("surgesurfer", "hadronengine", "electricsurge"):
                    return True
                if "grassyterrain" in fname and ab in ("grassysurge", "grassyterrain"):
                    return True
                if "psychicterrain" in fname and ab in ("psychicsurge",):
                    return True
                if "mistyterrain" in fname and ab in ("mistysurge",):
                    return True
                for mv in list(getattr(mon, "moves", {}).values()):
                    mid = str(getattr(mv, "id", "") or "").lower()
                    if "terrain" in fname and "terrainpulse" in mid:
                        return True
                    if "grassyterrain" in fname and mid in ("grassyglide",):
                        return True
                    if "electricterrain" in fname and mid in ("risingvoltage",):
                        return True
        except Exception:
            pass
        return False

    def _has_substitute(self, pokemon) -> bool:
        if pokemon is None:
            return False
        try:
            if Effect.SUBSTITUTE in getattr(pokemon, "effects", {}):
                return True
            for k in getattr(pokemon, "effects", {}).keys():
                if "substitute" in str(getattr(k, "name", str(k))).lower():
                    return True
        except Exception:
            pass
        return False

    def _has_leech_seed(self, pokemon) -> bool:
        if pokemon is None:
            return False
        try:
            if Effect.LEECH_SEED in getattr(pokemon, "effects", {}):
                return True
            for k in getattr(pokemon, "effects", {}).keys():
                if "leechseed" in str(getattr(k, "name", str(k))).lower().replace("_", ""):
                    return True
        except Exception:
            pass
        return False

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
            curr_weather = next(iter(battle.weather), None) if getattr(battle, "weather", None) else None
            curr_field = next(iter(battle.fields), None) if getattr(battle, "fields", None) else None
            curr_has_sub = self._has_substitute(battle.active_pokemon)
            curr_opp_has_leech = self._has_leech_seed(battle.opponent_active_pokemon)
            curr_has_leech = self._has_leech_seed(battle.active_pokemon)
            # предыдущее состояние
            prev = self._reward_state.get(tag)
            if prev is None:
                # первый вызов для этого боя — инициализируем и не даём extra
                # BUGFIX: раньше не было opp_active_species → второй ход ошибочно считал урон между разными покемонами
                curr_opp_species_init = getattr(getattr(battle.opponent_active_pokemon, "species", None), "lower", lambda: "")() if battle.opponent_active_pokemon else ""
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
                    "opp_active_species": curr_opp_species_init,
                    "turn": getattr(battle, "turn", 0),
                    "status": curr_status,
                    "opp_status": curr_opp_status,
                    "is_tera": curr_is_tera,
                    "protected": curr_protected,
                    "weather": curr_weather,
                    "field": curr_field,
                    "has_sub": curr_has_sub,
                    "opp_has_leech": curr_opp_has_leech,
                    "has_leech": curr_has_leech,
                    "_last_switch_delta_dmg": None,
                    "_last_switch_new_mult": None,
                    "_last_switch_prev_mult": None,
                    "_last_switch_prev_dmg": None,
                    "extra_accum": 0.0,  # для per-episode бюджета
                }
                return base - TIME_PENALTY

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
            # FIX: dmg_taken раньше считался даже при своём свитче (prev hp vs new mon hp) — теперь скипаем если был свитч
            prev_opp_active_species = prev.get("opp_active_species", "")
            curr_opp_species = getattr(getattr(battle.opponent_active_pokemon, "species", None), "lower", lambda: "")() if battle.opponent_active_pokemon else ""
            opp_switched = prev_opp_active_species and curr_opp_species and prev_opp_active_species != curr_opp_species
            # считаем was_switch заранее чтобы не путать урон при свитче (баг: раньше dmg_taken считался по разным покемонам)
            prev_species_for_dmg = prev.get("active_species", "")
            was_switch_for_dmg = False
            if prev_species_for_dmg and curr_active_species and prev_species_for_dmg != curr_active_species:
                try:
                    _prev_hp_for_dmg = prev.get("active_hp", 1.0)
                    _curr_turn_for_dmg = getattr(battle, "turn", 0)
                    _prev_turn_for_dmg = prev.get("turn", 0)
                    if _prev_hp_for_dmg > 0.05 and _curr_turn_for_dmg > _prev_turn_for_dmg:
                        was_switch_for_dmg = True
                except Exception:
                    was_switch_for_dmg = False
            if not opp_switched and not was_switch_for_dmg:
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
                        extra += 0.05  # было 0.2 -> сжали до 0.05 чтобы не ломать бюджет 12
                    # сохраняем дельту свитча в локальные переменные для последующего tera-блока (избегаем locals() хак)
                    try:
                        _last_switch_delta_dmg = float(delta_dmg)
                        _last_switch_new_mult = float(new_mult)
                        _last_switch_prev_mult = float(prev_mult)
                        _last_switch_prev_dmg = float(prev_dmg)
                    except Exception:
                        _last_switch_delta_dmg = None
                        _last_switch_new_mult = None
                        _last_switch_prev_mult = None
                        _last_switch_prev_dmg = None

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
                # FIX: не давать бонус при свитче на здорового (false positive), но разрешаем Natural Cure / Shed Skin абилки
                _was_switch_for_cure = False
                try:
                    if "was_switch_for_dmg" in locals() and was_switch_for_dmg:
                        _was_switch_for_cure = True
                    if "was_switch" in locals() and was_switch:
                        _was_switch_for_cure = True
                except Exception:
                    pass
                # проверяем Natural Cure / Healer etc. — тогда свитч-лечение засчитываем
                _is_natural_cure = False
                if _was_switch_for_cure:
                    try:
                        # prev мон — тот что был со статусом, его абилка
                        prev_mon_species = prev.get("active_species","")
                        # ищем prev_mon в team
                        for mon in battle.team.values():
                            if str(getattr(mon, "species","")).lower() == prev_mon_species.lower():
                                ab = str(getattr(mon, "ability","") or "").lower().replace(" ","").replace("-","")
                                if ab in ("naturalcure",):
                                    _is_natural_cure = True
                                    break
                        # Hydration только если дождь — но упростим: засчитаем
                    except Exception:
                        pass
                if prev_status is not None and curr_own_hp > 0.05:
                    if not _was_switch_for_cure and curr_status is None and prev.get("active_species","") == curr_active_species:
                        # Heal Bell / Aromatherapy на том же покемоне — вылечили
                        extra += STATUS_CURE_BONUS
                    elif _was_switch_for_cure and _is_natural_cure:
                        # Natural Cure: не важно на кого меняемся, статус снимается при уходе (даже если новый тоже статуснутый — бонус всё равно)
                        extra += STATUS_CURE_BONUS
            except Exception:
                pass

            # удачный тера (смена типа дала резист или добила) — без locals() хака
            try:
                prev_is_tera = bool(prev.get("is_tera", False))
                if not prev_is_tera and curr_is_tera:
                    opp_fainted_now = curr_opp_hp == 0 and prev.get("opp_hp", 0) > 0.2
                    # достаём сохранённые значения свитча если были, иначе считаем заново для теры
                    # проверяем дельту текущего свитча (если свитч+тера в один ход) иначе берём из prev
                    cur_delta = locals().get("_last_switch_delta_dmg", None)
                    if isinstance(cur_delta, (int,float)):
                        stored_delta = cur_delta
                        stored_new_mult = locals().get("_last_switch_new_mult", None)
                        stored_prev_mult = locals().get("_last_switch_prev_mult", None)
                        stored_prev_dmg = locals().get("_last_switch_prev_dmg", None)
                    else:
                        stored_delta = prev.get("_last_switch_delta_dmg") if isinstance(prev.get("_last_switch_delta_dmg"), (int,float)) else None
                        stored_new_mult = prev.get("_last_switch_new_mult")
                        stored_prev_mult = prev.get("_last_switch_prev_mult")
                        stored_prev_dmg = prev.get("_last_switch_prev_dmg")
                    # если свитча не было — считаем оценку урона теры отдельно (без свитча: до теры vs после)
                    # упрощённо: если после теры мы получили меньше урона чем до — бонус, но без сложной оценки просто проверяем смену имуна
                    if opp_fainted_now:
                        # пытаемся оценить был ли килл возможен без теры — если prev_dmg <1.5, то тера критична
                        try:
                            # оцениваем урон оппа по нам до/после — если нет stored, считаем по текущему оппу
                            if stored_delta is None:
                                dmg_before = 1.0
                            else:
                                dmg_before = float(stored_prev_dmg or 1.0) if stored_prev_dmg is not None else 1.0
                        except Exception:
                            dmg_before = 1.0
                        if dmg_before < 1.5:
                            extra += TERA_BONUS
                        else:
                            extra += 0.15
                    else:
                        # без убийства — проверяем снижение входящего урона от теры (иммун/резист)
                        has_delta = stored_delta is not None and stored_delta > 0.6
                        has_immune = stored_new_mult == 0 and stored_prev_mult not in (None, 0)
                        if has_delta or has_immune:
                            extra += TERA_BONUS
            except Exception:
                pass

            # погода — только если в команде есть синергия + только если погоду ставили МЫ (last move == weather), иначе игнор (может поставил опп)
            try:
                prev_weather = prev.get("weather", None)
                if prev_weather is None and curr_weather is not None:
                    last_id = self._last_move_id.get(tag, "")
                    is_our_weather = last_id in WEATHER_MOVE_IDS
                    # также считаем погоду от абилки при свитче (Drought/Drizzle и т.п.) — если свитч и синергия есть, тоже наградим
                    if not is_our_weather and self._last_was_switch.get(tag, False):
                        try:
                            new_active = battle.active_pokemon
                            ab = str(getattr(new_active, "ability", "") or "").lower().replace(" ","").replace("-","")
                            if ab in ("drought","drizzle","sandstream","snowwarning","orichalcumpulse","hadronengine"):
                                is_our_weather = True
                        except Exception:
                            pass
                    if is_our_weather:
                        if self._team_has_weather_synergy(battle.team, curr_weather):
                            extra += WEATHER_BONUS
                        else:
                            extra -= 0.04
                    else:
                        # погоду поставил опп — не трогаем (не наша заслуга/вина)
                        pass
            except Exception:
                pass
            try:
                prev_field = prev.get("field", None)
                if prev_field is None and curr_field is not None:
                    last_id = self._last_move_id.get(tag, "")
                    is_our_terrain = last_id in TERRAIN_MOVE_IDS
                    if not is_our_terrain and self._last_was_switch.get(tag, False):
                        try:
                            new_active = battle.active_pokemon
                            ab = str(getattr(new_active, "ability", "") or "").lower().replace(" ","").replace("-","")
                            if ab in ("electricsurge","grassysurge","psychicsurge","mistysurge","hadronengine"):
                                is_our_terrain = True
                        except Exception:
                            pass
                    if is_our_terrain:
                        if self._team_has_terrain_synergy(battle.team, curr_field):
                            extra += TERRAIN_BONUS
                        else:
                            extra -= 0.03
            except Exception:
                pass
            try:
                prev_has_sub = bool(prev.get("has_sub", False))
                if not prev_has_sub and curr_has_sub and prev.get("own_hp", 1.0) > 0.26:
                    extra += SUBSTITUTE_BONUS
                elif not prev_has_sub and curr_has_sub and prev.get("own_hp", 1.0) <= 0.26:
                    extra -= SUBSTITUTE_WASTED_PENALTY
            except Exception:
                pass
            try:
                prev_opp_leech = bool(prev.get("opp_has_leech", False))
                if not prev_opp_leech and curr_opp_has_leech:
                    extra += LEECH_SEED_BONUS
                elif prev_opp_leech and curr_opp_has_leech and not opp_switched:
                    extra -= LEECH_WASTED_PENALTY * 0.5
                prev_has_leech = bool(prev.get("has_leech", False))
                if not prev_has_leech and curr_has_leech:
                    extra -= LEECH_WASTED_PENALTY
            except Exception:
                pass
            try:
                # heal только если последний ход был хилом — иначе это пассивы (Leftovers/Grassy) и не даём бонуса/штрафа
                last_id = self._last_move_id.get(tag, "")
                # считаем хил-приёмами те что дают heal>0 (покрываем Recover, Roost, Slack Off, Synthesis и т.д.)
                is_heal_move = False
                try:
                    # быстрый чек по id (расширен)
                    if last_id in ("recover","roost","softboiled","morningsun","moonlight","synthesis","healorder","slackoff","milkdrink","swallow","rest","shoreup","strengthsap","wish","healingwish","lunardance","purify","lifedew","junglehealing","shoreup","rest","healbell","aromatherapy"):
                        is_heal_move = True
                    # также чекаем через move entry heal если вдруг другой хил (например Floral Healing)
                    if not is_heal_move:
                        try:
                            # пытаемся достать последний move объект если есть
                            # fallback: если heal_amount >0 и не было свитча и не было урона оппа — считаем хилом
                            pass
                        except Exception:
                            pass
                except Exception:
                    pass
                heal_amount = curr_own_hp - prev.get("own_hp", curr_own_hp)
                if is_heal_move:
                    if heal_amount > 0.12 and prev.get("own_hp", 1.0) < 0.90:
                        extra += HEAL_BONUS
                    elif heal_amount <= 0.02 and prev.get("own_hp", 1.0) < 0.95:
                        # хил-приём не схилял (heal_amount ~0) — wasted (например заблокирован Heal Block)
                        extra -= HEAL_WASTED_PENALTY
                    elif heal_amount > 0.05 and prev.get("own_hp", 1.0) >= 0.95:
                        extra -= HEAL_WASTED_PENALTY
                else:
                    # не хил-приём — игнор пассивного хила
                    pass
            except Exception:
                pass

            # универсальный штраф за любой wasted приём, если ещё не наказан спецификой (просьба: "штраф за wasted для любого приёма в принципе, если его нет")
            try:
                # диагностика неизвестных типов: одна строка на закончившийся бой (только если были события)
                try:
                    from .type_utils import debug_enabled, summary
                    if debug_enabled():
                        _s = summary()
                        if _s:
                            print(f"[type-debug] конец боя {tag}: {_s}")
                        try:
                            from .fusion_parser import _RAW_TYPECHANGE
                            for _sd, _raw in (_RAW_TYPECHANGE.get(tag, {}) or {}).items():
                                print(f"[type-debug] {tag} тип-сообщение {_sd}: {_raw}")
                        except Exception:
                            pass
                except Exception:
                    pass
                was_wasted = bool(self._last_wasted.get(tag, False))
                last_id = self._last_move_id.get(tag, "")
                # если уже есть специфика для этого id — не дублируем (иначе double penalty)
                is_specific = last_id in WASTED_MOVE_IDS_SKIP
                # heal тоже специфика, но мы её уже через is_heal_move отделили — если хил, то is_specific считаем True чтобы не дублить
                try:
                    if last_id in ("recover","roost","softboiled","morningsun","moonlight","synthesis","healorder","slackoff","milkdrink","swallow","rest","shoreup","strengthsap","wish"):
                        is_specific = True
                except Exception:
                    pass
                # также статус/скрин/хазард wasted уже ловятся _move_wasted_flag, но для них специфика внутри flag, а отдельной награды нет -> generic должен сработать
                # поэтому для хазардов/скринов/статуса is_specific=False и generic применится — это и нужно
                if was_wasted and not is_specific and not self._last_was_switch.get(tag, False):
                    extra -= WASTED_MOVE_PENALTY
                # чистим флаг чтобы не наказывать повторно за тот же ход (если calc_reward вызовется дважды без action_to_order)
                # не чистим сразу, а оставим до следующего action_to_order — но чтобы не двойной штраф при повторном calc без нового хода, сбрасываем после применения
                if was_wasted:
                    # Одноразовый штраф: после применения сбрасываем, чтобы следующий calc без нового хода не штрафовал снова
                    self._last_wasted[tag] = False
            except Exception:
                pass

            # per-step клип (был 2.0 → 0.5) + per-episode бюджет 12 < victory 30
            extra = float(np.clip(extra, -SHAPING_STEP_CLIP, SHAPING_STEP_CLIP))
            # per-episode бюджет — чтобы сумма за бой не перевесила победу
            try:
                prev_accum = float(prev.get("extra_accum", 0.0)) if prev else 0.0
                new_accum = prev_accum + extra
                if new_accum > SHAPING_EPISODE_CAP:
                    extra = SHAPING_EPISODE_CAP - prev_accum
                    new_accum = SHAPING_EPISODE_CAP
                elif new_accum < -SHAPING_EPISODE_CAP:
                    extra = -SHAPING_EPISODE_CAP - prev_accum
                    new_accum = -SHAPING_EPISODE_CAP
            except Exception:
                new_accum = float(prev.get("extra_accum", 0.0)) if prev else 0.0
                try:
                    new_accum += extra
                except Exception:
                    pass

            # обновляем состояние (с аккумулятором)
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
                "weather": curr_weather,
                "field": curr_field,
                "has_sub": curr_has_sub,
                "opp_has_leech": curr_opp_has_leech,
                "has_leech": curr_has_leech,
                "_last_switch_delta_dmg": locals().get("_last_switch_delta_dmg", None),
                "_last_switch_new_mult": locals().get("_last_switch_new_mult", None),
                "_last_switch_prev_mult": locals().get("_last_switch_prev_mult", None),
                "_last_switch_prev_dmg": locals().get("_last_switch_prev_dmg", None),
                "extra_accum": float(new_accum),
            }

        except Exception as e:
            # не роняем шаг из-за награды
            pass
        finally:
            # чистим завершённые бои даже если выше был exception (фикс утечки) — также чистим wasted-кэш
            try:
                if getattr(battle, "finished", False):
                    self._reward_state.pop(tag, None)
                    self._last_wasted.pop(tag, None)
                    self._last_move_id.pop(tag, None)
                    self._last_was_switch.pop(tag, None)
                    self._last_action_kind.pop(tag, None)
            except Exception:
                pass

        return base + extra - TIME_PENALTY

    def action_to_order(self, action, battle, fake=False, strict=True):
        mask = SinglesEnv.get_action_mask(battle)
        if sum(mask) == 0:
            from poke_env.player import DefaultBattleOrder
            return DefaultBattleOrder()
        # Книжка действий: помним, приём это был или свитч, и wasted ли приём.
        #
        # ВАЖНО (исправлено): раньше здесь применялась конвенция DoublesEnv
        # (`action < len(available_moves)` -> приём, иначе свитч). В poke-env SinglesEnv
        # раскладка другая: 0..5 — свитч (индекс в battle.team), 6..9 — приём ((action-6) % 4),
        # 10..13 мега, 14..17 z, 18..21 динамакс, 22..25 тера. Что из-за этого ломалось:
        #   * настоящий приём (6..9) записывался как «switch»: _last_wasted всегда False ->
        #     generic WASTED_MOVE_PENALTY (0.06) для приёмов НИКОГДА не применялся
        #     (бесполезная атака/хил/погода не стоили ничего), а ветки награды по
        #     _last_move_id (хил, погода, террейн, статус) были мертвы, потому что id = "switch";
        #   * свитч (0..3 при четырёх приёмах) записывался как «приём» с чужим id из своего же
        #     мувсета и получал WASTED_MOVE_PENALTY, если тот приём был wasted -> штраф за свитч.
        # Суммарно градиент систематически толкал политику в «спам атаками».
        # Сам ордер всегда строился корректно (super().action_to_order), ломалась только
        # бухгалтерия награды.
        try:
            tag = getattr(battle, "battle_tag", "") or "unknown"
            act_int = int(action.item()) if hasattr(action, "item") else int(action)
            # счётчики «состава действий» — только по НАШЕЙ стороне: PokeEnv зовёт action_to_order
            # и для боя оппонента (его ордер -> индекс), иначе метрика считала бы чужие ходы
            # (книжку по тегу ведём для любого боя, но счётчики пополняем только по нашей стороне)
            is_learner = battle is getattr(self, "battle1", None)
            if act_int < 0:
                # default (-2) / forfeit (-1): это не наш «ход» по приёму и не свитч
                self._last_move_id[tag] = "default"
                self._last_was_switch[tag] = False
                self._last_wasted[tag] = False
                self._last_action_kind[tag] = "default"
                if is_learner:
                    self._action_counts["unknown"] += 1
            elif act_int < 6:
                # свитч: индекс в battle.team (как в SinglesEnv.action_to_order)
                self._last_move_id[tag] = "switch"
                self._last_was_switch[tag] = True
                self._last_wasted[tag] = False
                self._last_action_kind[tag] = "switch"
                if is_learner:
                    self._action_counts["switch"] += 1
            else:
                idx = (act_int - 6) % 4
                mvs = self._moves_for_action(battle)
                move = mvs[idx] if idx < len(mvs) else None
                if move is None:
                    self._last_move_id[tag] = "unknown"
                    self._last_was_switch[tag] = False
                    self._last_wasted[tag] = False
                    self._last_action_kind[tag] = "unknown"
                    if is_learner:
                        self._action_counts["unknown"] += 1
                else:
                    mid = str(getattr(move, "id", "") or "").lower()
                    self._last_move_id[tag] = mid
                    self._last_was_switch[tag] = False
                    self._last_action_kind[tag] = "tera" if act_int >= 22 else "move"
                    if is_learner:
                        self._action_counts[self._last_action_kind[tag]] += 1
                    try:
                        flag = bool(_move_wasted_flag(move, battle))
                        # дополнительно: типовая иммунность (0 урона) тоже wasted — _move_wasted_flag её не ловит (только абилки)
                        if not flag and battle.opponent_active_pokemon is not None and getattr(move, "type", None) is not None:
                            try:
                                # статус-приёмы не считаем (base_power 0)
                                bp = getattr(move, "base_power", 0) or 0
                                if bp == 0:
                                    entry = getattr(move, "entry", {}) or {}
                                    bp = entry.get("basePower", 0) or entry.get("base_power", 0) or 0
                                if bp and bp >= 10:
                                    mtype = getattr(move, "type", None)
                                    opp = battle.opponent_active_pokemon
                                    opp_t1, opp_t2 = _eff_types(opp)
                                    # безопасный расчёт (раньше каскад с fallback 1.0 прятал иммунитет)
                                    if damage_multiplier_safe(mtype, opp_t1, opp_t2) == 0:
                                        flag = True
                            except Exception:
                                pass
                    except Exception:
                        flag = False
                    self._last_wasted[tag] = flag
        except Exception:
            pass
        return super().action_to_order(action, battle, fake=fake, strict=strict)

    @staticmethod
    def _moves_for_action(battle) -> list:
        """Приёмы в том же порядке, что использует SinglesEnv.action_to_order.

        Тонкая обёртка над features.move_slots_for_action: логика раскладки слотов живёт в
        ОДНОМ месте, потому что её используют и бухгалтерия награды, и признаки (иначе
        wasted-штраф и признаки в слотах разъедутся с реально выбранным приёмом).
        """
        return move_slots_for_action(battle)

    def action_mix(self) -> dict:
        """Счётчик типов действий НАШЕЙ стороны: switch/move/tera/unknown (диагностика «только атаки»).

        В обучении метрика считается в родительском процессе (`[mix]` из StepCounterCallback) —
        она точнее, потому что видит ровно те действия, что выбрала политика.
        """
        total = sum(self._action_counts.values())
        if not total:
            return dict(self._action_counts)
        out = {k: v for k, v in self._action_counts.items()}
        out["_total"] = total
        out["_switch_share"] = round(self._action_counts["switch"] / total, 3)
        return out

    def reset_action_mix(self) -> None:
        for k in list(self._action_counts):
            self._action_counts[k] = 0

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
