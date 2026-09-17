import random
from os import listdir
from os.path import isfile, join

import numpy as np
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .config import BATTLE_FORMAT, N_FEATURES, QUALIFIED_PREFIX
from .features import embed_battle_with_fusion
from .fusion_parser import _attach_fusion_parser
from .players import PolicyPlayer


def _snapshot_number(fname: str) -> int | None:
    suffix = fname.split("_")[-1].split(".")[0]
    return int(suffix) if suffix.isdigit() else None


def _make_self_play_opponents():
    model_dir = "models/"
    # сначала пробуем qualified (прошли порог)
    try:
        candidates = [f for f in listdir(model_dir) if isfile(join(model_dir, f)) and QUALIFIED_PREFIX in f]
    except FileNotFoundError:
        candidates = []
    # FALLBACK: если qualified пусто (у тебя 31.7% BC есть, но PPO упал до 25% и 30 не пробивает -> deadlock),
    # берем обычные self_play_snapshot_* чтобы не висеть 8M на 100% heuristic
    if not candidates:
        try:
            # берем последние 3 обычных снапшота как self_play, даже если они <порога
            from agents.config import SELF_PLAY_PATH
            prefix = SELF_PLAY_PATH.split("/")[-1] + "_"
            candidates = [f for f in listdir(model_dir) if isfile(join(model_dir, f)) and f.startswith(prefix) and QUALIFIED_PREFIX not in f]
            if candidates:
                print(f"self_play fallback: нет qualified (порог 30), беру последние {len(candidates)} обычных снапшотов")
        except Exception:
            pass
    numbered = [(f, _snapshot_number(f)) for f in candidates]
    numbered = [(f, n) for f, n in numbered if n is not None]
    files = [f for f, _ in sorted(numbered, key=lambda pair: pair[1])][-3:]

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

        # Убрали хак с подменой _fusion_stats:
        # каждый игрок парсит сообщения самостоятельно, общий dict ломал pending и protect.
        # Для наблюдения агента (env.agent1) достаточно его собственного парсера,
        # он видит обе стороны боя.

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

    def calc_reward(self, battle) -> float:
        base = self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, status_value=0.5, victory_value=30.0,
        )
        return base - 0.02

    def action_to_order(self, action, battle, fake=False, strict=True):
        mask = SinglesEnv.get_action_mask(battle)
        if sum(mask) == 0:
            from poke_env.player import DefaultBattleOrder
            return DefaultBattleOrder()
        return super().action_to_order(action, battle, fake=fake, strict=strict)

    def embed_battle(self, battle: AbstractBattle):
        # FIX: 'battle is self.battle1' был сломан т.к. SinglesEnv передаёт копию, а не тот же объект
        # (battle is battle1) -> False даже для p1, всегда выбирался agent2 -> терялись fusion/protect для p1
        # Используем player_role как в players.py / poke_env : p1 -> agent1, p2 -> agent2
        try:
            # battle.player_role существует у AbstractBattle; 'p1' => agent1, иначе agent2
            source = self.agent1 if getattr(battle, "player_role", "p1") == "p1" else self.agent2
        except Exception:
            # fallback на старый 'is' если player_role по какой-то причине отсутствует
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
        return embed_battle_with_fusion(
            battle,
            our_fusion=fusion_entry(True),
            opp_fusion=fusion_entry(False),
            our_protected_last_turn=our_protect,
            opp_protected_last_turn=opp_protect,
        )
