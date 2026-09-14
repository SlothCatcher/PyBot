from typing import Any, Awaitable

import numpy as np
import torch
from poke_env.battle import AbstractBattle
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, Player, SimpleHeuristicsPlayer
from stable_baselines3.common.policies import ActorCriticPolicy

from .features import embed_battle_with_fusion
from .fusion_parser import FusionInfoParser



class PolicyPlayer(FusionInfoParser, Player):
    policy: ActorCriticPolicy | None

    def __init__(self, policy: ActorCriticPolicy | None = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def choose_move(self, battle: AbstractBattle) -> BattleOrder | Awaitable[BattleOrder]:
        if battle.wait:
            return DefaultBattleOrder()
        if self.policy is None:
            return DefaultBattleOrder()
        obs = self.embed_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle))
        # FIX: если маска пустая, сразу возвращаем Default, а не лезем в политику (избегаем -inf логитов)
        if mask.sum() == 0:
            return DefaultBattleOrder()
        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(obs, device=self.policy.device).unsqueeze(0),
                "action_mask": torch.as_tensor(mask, device=self.policy.device).unsqueeze(0),
            }
            # Во время battle_against лучше детерминированно (меньше дисперсии оценки),
            # но для обучения стохастичность важна. Здесь используем deterministic=False
            # чтобы не расходиться с поведением во время тренировки; для финальной оценки
            # можно переопределить вызов с deterministic=True.
            action, _, _ = self.policy.forward(obs_dict, deterministic=False)
        action = int(action.cpu().numpy()[0])
        return SinglesEnv.action_to_order(action, battle)

    def embed_battle(self, battle: AbstractBattle):
        our_fusion = self.get_fusion_entry(battle, is_ours=True)
        opp_fusion = self.get_fusion_entry(battle, is_ours=False)
        our_protect = self.get_protected_last_turn(battle, is_ours=True)
        opp_protect = self.get_protected_last_turn(battle, is_ours=False)
        return embed_battle_with_fusion(
            battle, our_fusion, opp_fusion,
            our_protected_last_turn=our_protect,
            opp_protected_last_turn=opp_protect,
        )


class HeuristicRecorder(FusionInfoParser, SimpleHeuristicsPlayer):
    """Играет как SimpleHeuristicsPlayer, попутно записывая (obs, mask, action) для BC."""

    def __init__(self, *args, dataset: list, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset

    def choose_move(self, battle: AbstractBattle):
        order = super().choose_move(battle)
        if battle.wait or not isinstance(order, BattleOrder):
            return order
        try:
            our_fusion = self.get_fusion_entry(battle, is_ours=True)
            opp_fusion = self.get_fusion_entry(battle, is_ours=False)
            our_protect = self.get_protected_last_turn(battle, is_ours=True)
            opp_protect = self.get_protected_last_turn(battle, is_ours=False)
            obs = embed_battle_with_fusion(
                battle, our_fusion, opp_fusion,
                our_protected_last_turn=our_protect,
                opp_protected_last_turn=opp_protect,
            )
            mask = np.array(SinglesEnv.get_action_mask(battle))
            # order_to_action может бросить если order невалиден (например Forfeit)
            action = SinglesEnv.order_to_action(order, battle, fake=False, strict=False)
            # action может быть -1/-2 (forfeit/default) — такие переходы не учим
            if action is not None and action >= 0:
                self.dataset.append((obs, mask, action, battle.battle_tag))
        except Exception as e:
            # не падаем из-за одного битого перехода
            pass
        return order
