"""Живая диагностика: спамит ли модель приёмы с нулевой эффективностью (0x).

Играет N боёв выбранным снапшотом и на каждом ходу пишет:
  * типы активного оппонента (видно, есть ли "???")
  * эффективность каждого доступного приёма (через damage_multiplier_safe)
  * какой приём выбрала политика и его эффективность
  * был ли ход помечен как wasted

В конце — сводка: сколько раз выбран 0x-приём, какие матчапы, сколько было
"неизвестных типов" и сколько раз старый патч (KeyError -> 1.0) спрятал иммунитет.

Запуск (нужен поднятый Showdown-сервер, как для обучения):
    PYBOT_DEBUG_TYPES=1 python diagnose_type_spam.py --model models/self_play_snapshot_6.zip --battles 10

Скрипт повторяет схему оценки из agents/training.evaluate_win_rates (battle_against).
"""
import argparse
import asyncio
import os
import sys
from collections import Counter

os.environ.setdefault("PYBOT_DEBUG_TYPES", "1")  # подробный лог типов — до импорта агентов

import numpy as np
import torch

from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
from poke_env.player import DefaultBattleOrder, Player, SimpleHeuristicsPlayer, RandomPlayer, MaxBasePowerPlayer

from agents.config import BATTLE_FORMAT
from agents.players import PolicyPlayer
from agents.type_utils import (
    damage_multiplier_safe, damage_multiplier_safe_ex, is_unknown_type,
    summary_line, counter_snapshot,
)
from agents.fusion_parser import _RAW_TYPECHANGE


def _norm_type_name(x) -> str:
    n = str(x or "").strip()
    if n == "???":
        return "THREE_QUESTION_MARKS"
    return n.upper()


def _to_pokemon_type(name: str):
    """Строка из лога сервера -> PokemonType (умеет '???' и 'Stellar')."""
    try:
        from poke_env.battle import PokemonType
        return PokemonType.from_name(str(name).strip())
    except Exception:
        return None


def _server_types_for(battle, side: str):
    """(ident, (type1, type2)|None) из ПОСЛЕДНЕГО typechange сервера по стороне."""
    raw = _RAW_TYPECHANGE.get(getattr(battle, "battle_tag", ""), {}).get(side)
    if not raw:
        return None
    parts = raw.split("|")
    if len(parts) < 5:
        return None
    tnames = [_norm_type_name(x) for x in parts[4].split("/") if x.strip()]
    return parts[2], (tuple(tnames) if tnames else None)


def _move_bp(move) -> int:
    bp = getattr(move, "base_power", 0) or 0
    if bp == 0:
        entry = getattr(move, "entry", {}) or {}
        bp = entry.get("basePower", 0) or entry.get("base_power", 0) or 0
    return int(bp or 0)


def _ident_matches_mon(mon, ident: str) -> bool:
    """Относится ли typechange-сообщение с этим ident к этому покемону.

    Обычный случай: ident — имя/вид, и он встречается в species мон. В фьюжн-формате мод
    пишет ident как "+Тело" (вид-«голова» при этом в `details`/`species`), поэтому дополнительно
    сверяем тело фьюжна (см. fusion_types) — иначе сообщения про свою же монку считались бы
    «сообщением про другого покемона».
    """
    if mon is None:
        return False
    ident_key = str(ident or "").split(": ", 1)[-1].lstrip("+").strip().lower()
    if not ident_key:
        return False
    species = str(getattr(mon, "species", "") or "").lower()
    if ident_key in species:
        return True
    try:
        from agents.fusion_types import fusion_pair
        head, body = fusion_pair(mon)
        return bool(body) and ident_key in {str(head or "").lower(), str(body).lower()}
    except Exception:
        return False


def _type_pair(mon):
    if mon is None:
        return ("None", "None")
    return (str(getattr(getattr(mon, "type_1", None), "name", None)),
            str(getattr(getattr(mon, "type_2", None), "name", None)))


class TypeDiagPlayer(PolicyPlayer):
    """PolicyPlayer + логирование решений (эффективность выбранного приёма)."""

    def __init__(self, *args, deterministic: bool = False, check_obs: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.deterministic = deterministic
        self.check_obs = check_obs
        self.stats = {
            "turns": 0,
            "immune_chosen": 0,
            "damaging_chosen": 0,
            "wasted_chosen": 0,
            "chosen_moves": Counter(),
            "immune_matchups": Counter(),
            "unknown_type_turns": 0,
            "unknown_pairs": Counter(),
            "best_missed": 0,
            "would_mask": 0,
            # сверка с логом сервера
            "type_checks": 0,
            "type_mismatch": 0,
            "server_msg_other_mon": 0,
            "obs_checks": 0,
            "obs_stale": 0,
            "mismatch_examples": [],
        }
        self._logged_battles = set()

    # --- повторяет PolicyPlayer.choose_move, но с логом ---
    def choose_move(self, battle: AbstractBattle):
        if battle.wait or self.policy is None:
            return super().choose_move(battle)
        obs = self.embed_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle))
        if mask.sum() == 0:
            return DefaultBattleOrder()
        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(obs, device=self.policy.device).unsqueeze(0),
                "action_mask": torch.as_tensor(mask, device=self.policy.device).unsqueeze(0),
            }
            action, _, _ = self.policy.forward(obs_dict, deterministic=self.deterministic)
        action = int(action.cpu().numpy()[0])
        try:
            self._log_decision(battle, action)
        except Exception as e:  # диагностика не должна ломать бой
            print(f"[diag] ошибка логирования: {e}")
        return SinglesEnv.action_to_order(action, battle)

    def _log_decision(self, battle, action: int):
        # СЛОТЫ действий (known_moves[:4]), а не available_moves: у SinglesEnv действие >= 6 —
        # это приём со слотом (action-6)%4, а 0..5 — свитч. Прежняя логика
        # `if action < len(available_moves)` и нумерация effs по available_moves путали
        # выбранный приём (диагностика приписывала ходу чужой приём).
        from agents.features import move_slots_for_action
        moves = move_slots_for_action(battle)
        opp = battle.opponent_active_pokemon
        t1, t2 = _type_pair(opp)
        pair = f"{t1}/{t2}"
        self.stats["turns"] += 1
        if is_unknown_type(getattr(opp, "type_1", None)) or is_unknown_type(getattr(opp, "type_2", None)):
            self.stats["unknown_type_turns"] += 1
            self.stats["unknown_pairs"][pair] += 1

        # сверка: тип в бою (то, что видит модель) против последнего typechange сервера
        self._check_server_vs_battle(battle)
        # сверка: obs-мультипликаторы против серверных типов (ловит obs, посчитанный до применения типа)
        if self.check_obs:
            try:
                self._check_obs_against_server(battle, moves)
            except Exception as e:
                print(f"[diag] ошибка obs-сверки: {e}")

        # эффективность всех приёмов — по тем же типам, что уходят в obs (у фьюжнов тип на скамейке
        # poke-env не знает, см. fusion_types.py; для активного мон это серверные типы, как и раньше)
        effs = []
        try:
            from agents.fusion_types import effective_types as _effective_types
            _ot1, _ot2, _ = _effective_types(opp)
        except Exception:
            _ot1, _ot2 = getattr(opp, "type_1", None), getattr(opp, "type_2", None)
        for m in moves:
            mult, unknown = damage_multiplier_safe_ex(getattr(m, "type", None), _ot1, _ot2)
            effs.append((m, float(mult), bool(unknown), _move_bp(m) >= 10))

        # выбранный приём
        chosen = None
        if action >= 6:
            slot = (action - 6) % 4
            if slot < len(effs):
                chosen = effs[slot]
                m, mult, unknown, damaging = chosen
                self.stats["chosen_moves"][str(getattr(m, "id", "?"))] += 1
                if damaging:
                    self.stats["damaging_chosen"] += 1
                    if mult == 0:
                        self.stats["immune_chosen"] += 1
                        self.stats["immune_matchups"][f"{getattr(m, 'id', '?')} vs {pair}"] += 1
                    best = max([e[1] for e in effs if e[3]] or [0.0])
                    if mult < best:
                        self.stats["best_missed"] += 1
                if unknown and mult == 0:
                    # именно этот случай старый патч превращал в 1.0
                    self.stats["would_mask"] += 1

        # раз в бой печатаем типы (видно "???")
        tag = getattr(battle, "battle_tag", "?")
        if tag not in self._logged_battles:
            self._logged_battles.add(tag)
            raw = _RAW_TYPECHANGE.get(tag, {})
            print(f"[diag] бой {tag}: оппонент {getattr(opp, 'species', '?')} типы {pair}"
                  + (f" | сырые typechange: {raw}" if raw else " | typechange от сервера не приходил"))

        if self.stats["turns"] <= 40:  # первые ходы — подробно
            moves_str = ", ".join(
                f"{getattr(m, 'id', '?')}({mult:g}{'?' if unk else ''})" for m, mult, unk, dmg in effs
            )
            if chosen:
                ch = f"{getattr(chosen[0], 'id', '?')}={chosen[1]:g}"
            elif action < 6:
                ch = f"switch(idx {action})"
            else:
                ch = f"move вне слотов(idx {action})"     # маска такое действие не разрешает
            print(f"[diag] ход {self.stats['turns']:3d} vs {pair:28s} выбрано {ch:22s} | доступно: {moves_str}")

    # --- сверки с логом сервера -------------------------------------------------
    def _check_server_vs_battle(self, battle):
        sides = []
        try:
            role = getattr(battle, "player_role", None)
            opp = "p2" if role == "p1" else "p1"
            sides = [(role, getattr(battle, "active_pokemon", None), "наш"),
                     (opp, getattr(battle, "opponent_active_pokemon", None), "чужой")]
        except Exception:
            return
        for side, mon, label in sides:
            if side not in ("p1", "p2") or mon is None:
                continue
            srv = _server_types_for(battle, side)
            if srv is None:
                continue
            ident, srv_types = srv
            if srv_types is None:
                continue
            # сообщение может относиться к другому покемону (свитч уже был, typechange ещё нет).
            # Для фьюжнов ident = "+Тело", поэтому сверяем и тело (см. _ident_matches_mon)
            if not _ident_matches_mon(mon, ident):
                self.stats["server_msg_other_mon"] += 1
                continue
            actual = tuple(str(getattr(t, "name", t)) for t in (mon.type_1, mon.type_2) if t is not None)
            self.stats["type_checks"] += 1
            if actual != srv_types:
                self.stats["type_mismatch"] += 1
                if len(self.stats["mismatch_examples"]) < 5:
                    self.stats["mismatch_examples"].append(
                        f"{label} {ident}: сервер={srv_types} в бою={actual}")
                    print(f"[diag] РАСХОЖДЕНИЕ типа: {label} {ident}: "
                          f"сервер прислал {srv_types}, а в бою {actual}")

    def _check_obs_against_server(self, battle, moves):
        """obs (то, что уходит в сеть) посчитан по СЕРВЕРНЫМ типам оппонента?"""
        opp_side = "p2" if getattr(battle, "player_role", "p1") == "p1" else "p1"
        srv = _server_types_for(battle, opp_side)
        if srv is None or srv[1] is None:
            return
        opp_mon = getattr(battle, "opponent_active_pokemon", None)
        if not _ident_matches_mon(opp_mon, srv[0]):
            # сервер сообщал тип про ДРУГОГО мон -> сравнивать с ним obs нельзя
            self.stats["server_msg_other_mon"] += 1
            return
        s_types = [_to_pokemon_type(x) for x in srv[1]]
        if not s_types or s_types[0] is None:
            return
        s1 = s_types[0]
        s2 = s_types[1] if len(s_types) > 1 else None
        from agents.features import embed_battle_with_fusion
        raw = embed_battle_with_fusion(
            battle,
            self.get_fusion_entry(battle, is_ours=True),
            self.get_fusion_entry(battle, is_ours=False),
            our_protected_last_turn=self.get_protected_last_turn(battle, is_ours=True),
            opp_protected_last_turn=self.get_protected_last_turn(battle, is_ours=False),
            our_team_fusions=self.get_team_fusion_map(battle, is_ours=True),
            opp_team_fusions=self.get_team_fusion_map(battle, is_ours=False),
        )
        from agents.features import move_slot_index
        for m in moves:
            # индекс слота именно ЭТОГО приёма (не порядковый номер в available_moves)
            i = move_slot_index(battle, getattr(m, "id", ""))
            if i is None or 4 + i >= len(raw):
                continue
            obs_mult = float(raw[4 + i])           # moves_dmg_multiplier[слот]
            srv_mult = float(damage_multiplier_safe(getattr(m, "type", None), s1, s2))
            self.stats["obs_checks"] += 1
            if abs(obs_mult - srv_mult) > 1e-6:
                self.stats["obs_stale"] += 1
                if self.stats["obs_stale"] <= 5:
                    print(f"[diag] obs НЕ по серверному типу: {getattr(m, 'id', '?')} "
                          f"obs={obs_mult:g} ожидалось {srv_mult:g} при серверных {srv[1]}")

    def print_summary(self):
        s = self.stats
        print("\n" + "=" * 78)
        print("ИТОГ ДИАГНОСТИКИ ТИПОВ")
        print("=" * 78)
        print(f"ходов: {s['turns']}, из них дамажных приёмов выбрано: {s['damaging_chosen']}")
        if s["damaging_chosen"]:
            print(f"выбрано 0x-приёмов (иммун): {s['immune_chosen']} "
                  f"({100 * s['immune_chosen'] / max(1, s['damaging_chosen']):.1f}% от дамажных)")
            print(f"выбран приём хуже лучшего по типу: {s['best_missed']} раз")
        else:
            print("дамажных приёмов не выбиралось вообще — проверь маску/политику")
        print(f"ходов с '???'/STELLAR типом у оппонента: {s['unknown_type_turns']} "
              f"(матчапы: {dict(s['unknown_pairs'].most_common(5))})")
        if s["would_mask"]:
            print(f"! случаев, где старый патч (KeyError -> 1.0) прятал иммунитет: {s['would_mask']}")
        else:
            print("случаев маскировки иммунитета не встретилось")
        print(f"распределение выбранных приёмов: {dict(s['chosen_moves'].most_common(10))}")
        if s["immune_matchups"]:
            print(f"иммунные матчапы: {dict(s['immune_matchups'].most_common(10))}")

        print("-" * 78)
        print("СВЕРКА С ЛОГОМ СЕРВЕРА (типы)")
        if s["type_checks"]:
            print(f"проверок 'тип в бою == последний typechange сервера': {s['type_checks']}, "
                  f"расхождений: {s['type_mismatch']}"
                  f"{'  ✓' if s['type_mismatch'] == 0 else '  ✗ см. примеры выше'}")
        else:
            print("проверок типов не было (нет сырых typechange — сервер не присылал?)")
        if s["server_msg_other_mon"]:
            print(f"пропущено (typechange был про другого покемона, ещё не применён): {s['server_msg_other_mon']}")
        if s["obs_checks"]:
            print(f"проверок 'obs посчитан по серверному типу': {s['obs_checks']}, "
                  f"устаревших: {s['obs_stale']}"
                  f"{'  ✓' if s['obs_stale'] == 0 else '  ✗ см. примеры выше'}")
        for ex in s["mismatch_examples"]:
            print(f"   пример: {ex}")

        # интерпретация порядка кадров из логов
        frames = counter_snapshot("frame_sequence")
        if frames:
            only_request = sum(v for k, v in frames.items() if set(k.split(">")) == {"request"})
            total = sum(frames.values())
            if only_request == total:
                print("порядок кадров: сервер присылает |request| отдельным кадром -> "
                      "typechange всегда применяется ДО решения, гонки нет")
            else:
                print(f"кадры с |request|: {frames} (есть кадры со смешанными сообщениями — "
                      f"перестановка typechange актуальна)")
        print(summary_line() or "[type-debug] счётчики неизвестных типов пусты")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/self_play_snapshot_6.zip")
    ap.add_argument("--battles", type=int, default=10, help="боёв против каждого бота")
    ap.add_argument("--opponents", default="SimpleHeuristicsPlayer,RandomPlayer",
                    help="список классов через запятую")
    ap.add_argument("--deterministic", action="store_true", help="жадный выбор действия")
    ap.add_argument("--no-check-obs", dest="check_obs", action="store_false",
                    help="не пересчитывать obs для сверки с серверным типом (быстрее, но без этой проверки)")
    ap.set_defaults(check_obs=True)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    ppo = PPO.load(args.model)
    vec_norm = ppo.get_vec_normalize_env() if hasattr(ppo, "get_vec_normalize_env") else None
    agent = TypeDiagPlayer(policy=ppo.policy, battle_format=BATTLE_FORMAT,
                           max_concurrent_battles=30, deterministic=args.deterministic,
                           check_obs=args.check_obs)
    if vec_norm is not None:
        orig_embed = agent.embed_battle

        def norm_embed(battle):
            raw = orig_embed(battle)
            return vec_norm.normalize_obs({"observation": raw[None, :]})["observation"][0]

        agent.embed_battle = norm_embed  # type: ignore

    known = {"SimpleHeuristicsPlayer": SimpleHeuristicsPlayer, "RandomPlayer": RandomPlayer,
             "MaxBasePowerPlayer": MaxBasePowerPlayer}
    opponents: list[Player] = []
    for name in [x.strip() for x in args.opponents.split(",") if x.strip()]:
        cls = known.get(name)
        if cls is None:
            print(f"неизвестный оппонент {name}, пропускаю")
            continue
        opponents.append(cls(battle_format=BATTLE_FORMAT, max_concurrent_battles=30))
    if not opponents:
        print("нет оппонентов")
        return 2

    print(f"Боёв: {args.battles} x {[type(o).__name__ for o in opponents]} | модель: {args.model}")
    res = agent.battle_against(*opponents, n_battles=args.battles)
    if asyncio.iscoroutine(res):  # старые версии poke-env — корутина
        asyncio.run(res)

    agent.print_summary()
    print(f"finish: {agent.n_finished_battles}, wins: {agent.n_won_battles}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
