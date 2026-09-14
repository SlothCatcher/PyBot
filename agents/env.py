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
    candidates = [f for f in listdir(model_dir) if isfile(join(model_dir, f)) and QUALIFIED_PREFIX in f]
    numbered = [(f, _snapshot_number(f)) for f in candidates]
    numbered = [(f, n) for f, n in numbered if n is not None]
    files = [f for f, _ in sorted(numbered, key=lambda pair: pair[1])][-3:]

    players = []
    for fname in files:
        try:
            snap = PPO.load(join(model_dir, fname), device="cpu")
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
        heuristics = [SimpleHeuristicsPlayer(start_listening=False)]
        self_play_opp = _make_self_play_opponents()
        
        for opp in self_play_opp:
            opp._fusion_stats = env.agent1._fusion_stats
        
        all_opponents = heuristics + self_play_opp

        if opponent_weights:
            names = [type(o).__name__ if o not in self_play_opp else "self_play" for o in all_opponents]
            weights = [opponent_weights.get(n, 1.0 / len(all_opponents)) for n in names]
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