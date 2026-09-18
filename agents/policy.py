import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .config import N_FEATURES

# N_FEATURES вырос 594 -> 629 -> 641 -> 653 -> 713 -> 715 -> 802 (+87 урона) -> 870 (+68 зеркало и флаги).
# При 870 признаках первый слой (870->512) = 445k параметров, это ~40% сети; весь блок урона
# (155 признаков) стоит +6% параметров сети, сжатие первого слоя 870->512 = 1.70x.
# Размер экстрактора параметризован: `--features-dim` в policy_player (по умолчанию 512).
# Смена размера не ломает warm start — веса старого чекпоинта паддятся (см.
# _migrate_checkpoint_dim: новые нейроны входят с нулевыми весами, LayerNorm weight=1/bias=0).
# Замер на CPU (batch 256): 512 -> 640 даёт +16% времени шага оптимизации, 768 -> +37%,
# то есть на фоне времени боёв это дёшево; мерить пользу стоит по TB-метрике arch/*.
# Старая голова [512,256,128] shared была узким местом: 715*512=366k в первом слое,
# но shared pi/vf конфликтовали (value масштаб ±30 vs policy), а bench 400/715 (56%)
# доминировал и забивал градиенты остальных 315 признаков.
# Новая архитектура:
#  - FeaturesExtractor: 713 -> 512 + LayerNorm + ReLU (+ Dropout 0.1) — нормализует bench-спарсность
#    и изолирует нормализацию от VecNormalize (который тоже нормализует, но глобально).
#  - Раздельные головы pi/vf dict(pi=[512,256], vf=[512,256]) вместо shared [512,256,128]:
#    +33% параметров (~528k -> ~700k), но без интерференции; value не тянет policy.
#  - Если нужен быстрый откат / совместимость со старыми zip — см. LegacyFeaturesExtractor ниже.
#  - Для совсем больших 50k датасетов можно попробовать pi/vf=[640,320,160] (+50%), но на CPU
#    [512,256] уже оптимален по скорости/качеству (2M шагов ~ 4-6ч на 8 envs).


class FeaturesExtractor(BaseFeaturesExtractor):
    """Нормализует N_FEATURES признаков (сейчас 870) перед pi/vf головами. features_dim=512."""
    def __init__(self, observation_space, features_dim: int = 512, dropout: float = 0.1):
        super().__init__(observation_space, features_dim=features_dim)
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs):
        return self.net(obs["observation"])


class LegacyFeaturesExtractor(BaseFeaturesExtractor):
    """Старый identity-экстрактор (features_dim=N_FEATURES) — для загрузки старых zip без переобучения."""
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = dict(pi=[512, 256], vf=[512, 256])
        if "activation_fn" not in kwargs:
            kwargs["activation_fn"] = nn.ReLU
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = FeaturesExtractor
        if "ortho_init" not in kwargs:
            kwargs["ortho_init"] = True
        super().__init__(*args, **kwargs)

    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        mask = self._mask

        no_valid_action = mask.sum(dim=-1) == 0
        if no_valid_action.any():
            print(f"WARNING: empty action mask for {no_valid_action.sum().item()} batch element(s)")
            mask = mask.clone()
            mask[no_valid_action] = 1

        additive_mask = torch.where(mask == 1, 0.0, float("-inf"))
        return self.action_dist.proba_distribution(action_logits + additive_mask)
