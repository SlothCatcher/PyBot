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
        our_team_fusions = self.get_team_fusion_map(battle, is_ours=True) if hasattr(self, "get_team_fusion_map") else None
        opp_team_fusions = self.get_team_fusion_map(battle, is_ours=False) if hasattr(self, "get_team_fusion_map") else None
        return embed_battle_with_fusion(
            battle, our_fusion, opp_fusion,
            our_protected_last_turn=our_protect,
            opp_protected_last_turn=opp_protect,
            our_team_fusions=our_team_fusions,
            opp_team_fusions=opp_team_fusions,
        )


class HeuristicRecorder(FusionInfoParser, SimpleHeuristicsPlayer):
    """Играет как SimpleHeuristicsPlayer, попутно записывая (obs, mask, action) для BC.
    Также сохраняет сырые снапшоты битв для кэша: raw_dataset = (battle_copy, mask, action, tag, fusion, protect)
    чтобы при смене N_FEATURES пересобирать датасет без новых боёв.
    """

    def __init__(self, *args, dataset: list, raw_dataset: list | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.raw_dataset = raw_dataset if raw_dataset is not None else []

    def choose_move(self, battle: AbstractBattle):
        order = super().choose_move(battle)
        if battle.wait or not isinstance(order, BattleOrder):
            return order
        try:
            our_fusion = self.get_fusion_entry(battle, is_ours=True)
            opp_fusion = self.get_fusion_entry(battle, is_ours=False)
            our_protect = self.get_protected_last_turn(battle, is_ours=True)
            opp_protect = self.get_protected_last_turn(battle, is_ours=False)
            our_team_fusions = self.get_team_fusion_map(battle, is_ours=True) if hasattr(self, "get_team_fusion_map") else None
            opp_team_fusions = self.get_team_fusion_map(battle, is_ours=False) if hasattr(self, "get_team_fusion_map") else None
            mask = np.array(SinglesEnv.get_action_mask(battle))
            action = SinglesEnv.order_to_action(order, battle, fake=False, strict=False)
            if action is not None and action >= 0:
                # сырой кэш: deepcopy battle для последующей перегенерации obs при смене признаков
                try:
                    import copy
                    battle_copy = copy.deepcopy(battle)
                except Exception:
                    battle_copy = battle  # fallback: shallow (лучше чем ничего)
                self.raw_dataset.append((battle_copy, mask, action, battle.battle_tag, our_fusion, opp_fusion, our_protect, opp_protect))
                # сразу считаем obs для текущей сессии (чтобы не пересчитывать)
                obs = embed_battle_with_fusion(
                    battle, our_fusion, opp_fusion,
                    our_protected_last_turn=our_protect,
                    opp_protected_last_turn=opp_protect,
                    our_team_fusions=our_team_fusions,
                    opp_team_fusions=opp_team_fusions,
                )
                self.dataset.append((obs, mask, action, battle.battle_tag))
        except Exception as e:
            # на один тестовый прогон можно раскомментировать для отладки маскирующих try/except
            # print(f"WARN HeuristicRecorder.choose_move: {e}")
            pass
        return order
