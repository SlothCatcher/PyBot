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
from agents.type_utils import damage_multiplier_safe_ex, is_unknown_type, summary_line
from agents.fusion_parser import _RAW_TYPECHANGE


def _move_bp(move) -> int:
    bp = getattr(move, "base_power", 0) or 0
    if bp == 0:
        entry = getattr(move, "entry", {}) or {}
        bp = entry.get("basePower", 0) or entry.get("base_power", 0) or 0
    return int(bp or 0)


def _type_pair(mon):
    if mon is None:
        return ("None", "None")
    return (str(getattr(getattr(mon, "type_1", None), "name", None)),
            str(getattr(getattr(mon, "type_2", None), "name", None)))


class TypeDiagPlayer(PolicyPlayer):
    """PolicyPlayer + логирование решений (эффективность выбранного приёма)."""

    def __init__(self, *args, deterministic: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.deterministic = deterministic
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
        moves = list(getattr(battle, "available_moves", []) or [])
        opp = battle.opponent_active_pokemon
        t1, t2 = _type_pair(opp)
        pair = f"{t1}/{t2}"
        self.stats["turns"] += 1
        if is_unknown_type(getattr(opp, "type_1", None)) or is_unknown_type(getattr(opp, "type_2", None)):
            self.stats["unknown_type_turns"] += 1
            self.stats["unknown_pairs"][pair] += 1

        # эффективность всех приёмов
        effs = []
        for m in moves:
            mult, unknown = damage_multiplier_safe_ex(getattr(m, "type", None), getattr(opp, "type_1", None), getattr(opp, "type_2", None))
            effs.append((m, float(mult), bool(unknown), _move_bp(m) >= 10))

        # выбранный приём
        chosen = None
        if action < len(moves):
            chosen = effs[action]
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
            ch = f"{getattr(chosen[0], 'id', '?')}={chosen[1]:g}" if chosen else f"switch(idx {action})"
            print(f"[diag] ход {self.stats['turns']:3d} vs {pair:28s} выбрано {ch:22s} | доступно: {moves_str}")

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
        print(summary_line() or "[type-debug] счётчики неизвестных типов пусты")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/self_play_snapshot_6.zip")
    ap.add_argument("--battles", type=int, default=10, help="боёв против каждого бота")
    ap.add_argument("--opponents", default="SimpleHeuristicsPlayer,RandomPlayer",
                    help="список классов через запятую")
    ap.add_argument("--deterministic", action="store_true", help="жадный выбор действия")
    args = ap.parse_args()

    from stable_baselines3 import PPO

    ppo = PPO.load(args.model)
    vec_norm = ppo.get_vec_normalize_env() if hasattr(ppo, "get_vec_normalize_env") else None
    agent = TypeDiagPlayer(policy=ppo.policy, battle_format=BATTLE_FORMAT,
                           max_concurrent_battles=30, deterministic=args.deterministic)
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
