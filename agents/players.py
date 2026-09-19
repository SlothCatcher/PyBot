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
    """Player, который ходит выученной политикой.

    `obs_normalizer` — объект с методом `normalize(obs) -> obs` (обычно `VecNormStats` или
    `LiveVecNormalizeAdapter`). Нужен потому, что во время обучения obs нормализуется
    `VecNormalize` (norm_obs=True), и политика видит уже нормализованные признаки. Любой
    инференс без той же нормализации (живые боты в index.py, self-play оппоненты в
    обучении) скармливает сети сдвинутый по масштабу obs — решения становятся хуже, чем
    в обучении. Если статистики нет/не подходит, нормализация не применяется и об этом
    один раз пишется предупреждение.
    """
    policy: ActorCriticPolicy | None

    def __init__(self, policy: ActorCriticPolicy | None = None, *args: Any,
                 obs_normalizer: Any = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.obs_normalizer = obs_normalizer
        self._norm_warned = False
        # необязательный счётчик {switch, move, tera, other}: сколько раз игрок выбрал каждый
        # класс действий. Нужен, чтобы в логе оценки винрейта было видно РЕАЛЬНОЕ поведение
        # модели в оценочных боях (там нет wait-шагов, в отличие от метрики [mix] по роллаутам).
        self.action_counter: dict[str, int] | None = None

    def _apply_obs_norm(self, obs):
        norm = getattr(self, "obs_normalizer", None)
        if norm is None:
            return obs
        try:
            return norm.normalize(obs)
        except Exception as e:
            if not getattr(self, "_norm_warned", False):
                self._norm_warned = True
                print(f"[PolicyPlayer] нормализация obs не применилась ({e}) — играю на сырых признаках")
            return obs

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
        cnt = getattr(self, "action_counter", None)
        if cnt is not None:
            # раскладка poke-env SinglesEnv: 0..5 свитч, 6..9 приём, 22..25 тера
            key = ("switch" if action < 6 else ("tera" if action >= 22 else ("move" if action <= 9 else "other")))
            cnt[key] = cnt.get(key, 0) + 1
        return SinglesEnv.action_to_order(action, battle)

    def embed_battle(self, battle: AbstractBattle):
        our_fusion = self.get_fusion_entry(battle, is_ours=True)
        opp_fusion = self.get_fusion_entry(battle, is_ours=False)
        # диагностика: если в бою уже были typechange (значит фьюжн), а статов оппонента ещё нет —
        # этот ход сыгран по дексовому фолбэку. Видно, приходят ли статы в том же кадре, что и решение.
        try:
            try:
                from .fusion_parser import _RAW_TYPECHANGE
                from .type_utils import note_stats_missing
            except ImportError:
                from fusion_parser import _RAW_TYPECHANGE
                from type_utils import note_stats_missing
            if opp_fusion is None and _RAW_TYPECHANGE.get(getattr(battle, "battle_tag", "")):
                note_stats_missing(f"turn {getattr(battle, 'turn', '?')}, оппонент "
                                   f"{getattr(getattr(battle, 'opponent_active_pokemon', None), 'species', '?')}")
        except Exception:
            pass
        our_protect = self.get_protected_last_turn(battle, is_ours=True)
        opp_protect = self.get_protected_last_turn(battle, is_ours=False)
        our_team_fusions = self.get_team_fusion_map(battle, is_ours=True) if hasattr(self, "get_team_fusion_map") else None
        opp_team_fusions = self.get_team_fusion_map(battle, is_ours=False) if hasattr(self, "get_team_fusion_map") else None
        obs = embed_battle_with_fusion(
            battle, our_fusion, opp_fusion,
            our_protected_last_turn=our_protect,
            opp_protected_last_turn=opp_protect,
            our_team_fusions=our_team_fusions,
            opp_team_fusions=opp_team_fusions,
        )
        return self._apply_obs_norm(obs)


class HeuristicRecorder(FusionInfoParser, SimpleHeuristicsPlayer):
    """Играет как SimpleHeuristicsPlayer, попутно записывая (obs, mask, action) для BC.
    Также сохраняет сырые снапшоты битв для кэша:
    raw_dataset = (battle_copy, mask, action, tag, fusion, protect, team_fusions),
    где team_fusions = (our_team_fusions, opp_team_fusions) — командные фьюжн-карты по species.
    Без них пересчёт из кэша считал bench/зеркало по дексу, хотя в живом бою (и в исходной
    записи obs) они передаются: пересобранный датасет расходился с тем, что видит модель
    на инференсе (это 155-мерный блок урона). Записи старого формата читаются как (None, None).
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
                # защита от cannot pickle '_thread.lock' (если battle держит ссылку на Player/websocket)
                try:
                    import copy, pickle
                    battle_copy = copy.deepcopy(battle)
                    # быстрый тест пиклибельности — если падает, делаем stripped stub
                    try:
                        pickle.dumps(battle_copy, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception:
                        # fallback: минимальный объект только с нужными для embed полями
                        import types
                        stub = types.SimpleNamespace()
                        for attr in ["battle_tag","gen","weather","fields","side_conditions","opponent_side_conditions","available_moves","team","opponent_team","active_pokemon","opponent_active_pokemon","player_role","can_tera","won"]:
                            if hasattr(battle, attr):
                                try:
                                    stub.__dict__[attr] = copy.deepcopy(getattr(battle, attr))
                                except Exception:
                                    try:
                                        stub.__dict__[attr] = getattr(battle, attr)
                                    except Exception:
                                        pass
                        battle_copy = stub
                except Exception:
                    try:
                        battle_copy = battle  # последний fallback: shallow
                    except Exception:
                        battle_copy = None
                if battle_copy is not None:
                    self.raw_dataset.append((battle_copy, mask, action, battle.battle_tag, our_fusion,
                                             opp_fusion, our_protect, opp_protect,
                                             (our_team_fusions, opp_team_fusions)))
                else:
                    # не удалось получить копию — пропускаем сырой кэш, но всё равно пишем obs
                    pass
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
