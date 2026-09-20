import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .config import N_FEATURES

try:
    from .optim import SplitAdam
except ImportError:  # запуск модуля вне пакета
    from optim import SplitAdam

# N_FEATURES вырос 594 -> 629 -> 641 -> 653 -> 713 -> 715 -> 802 (+87 урона) -> 870 (+68 зеркало и флаги)
# -> 991 (+121 типы соперника x наша команда, см. features.TYPE_MATCHUP_BLOCK_SIZE).
# При 991 признаке первый слой (991->512) = 507k параметров, это ~40% сети; блок урона
# (155 признаков) стоит +6% параметров сети, сжатие первого слоя 991->512 = 1.94x.
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
    """Нормализует N_FEATURES признаков (сейчас 991) перед pi/vf головами. features_dim=512."""
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
    """Политика с маскированием действий и раздельным Adam для policy/value/shared.

    SB3 строит оптимизатор как `optimizer_class(self.parameters())`, без имён, поэтому первый
    (плоский) SplitAdam сразу пересобирается по `named_parameters()` — только так pi-голова,
    vf-голова и экстрактор попадают в свои группы со своими lr/eps (см. agents/optim.py).
    Настройки оптимизатора едут в чекпоинте внутри `policy_kwargs`, поэтому загруженная модель
    восстанавливает ту же схему групп.
    """

    def __init__(self, *args, **kwargs):
        self._mask = None
        self._mask_shape_warned = False
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = dict(pi=[512, 256], vf=[512, 256])
        if "activation_fn" not in kwargs:
            kwargs["activation_fn"] = nn.ReLU
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = FeaturesExtractor
        if "ortho_init" not in kwargs:
            kwargs["ortho_init"] = True
        if "optimizer_class" not in kwargs:
            kwargs["optimizer_class"] = SplitAdam
        super().__init__(*args, **kwargs)
        self._install_split_optimizer()

    def _install_split_optimizer(self) -> None:
        """Пересобрать оптимизатор по именам параметров (группы pi / vf / shared).

        Моменты на этом шаге пустые (политика только что создана), так что пересборка бесплатна.
        Если пользователь передал чужой optimizer_class — не трогаем вовсе.
        """
        opt = getattr(self, "optimizer", None)
        if not isinstance(opt, SplitAdam):
            return
        try:
            from .optim import optimizer_settings
        except ImportError:  # pragma: no cover
            from optim import optimizer_settings
        # базовый lr: берём из расписания SB3 (а не из первого param_group — в плоском
        # оптимизаторе туда попадает единственная группа, и её lr не равен lr policy)
        try:
            base_lr = float(self.lr_schedule(1.0))
        except Exception:
            base_lr = float(opt.param_groups[0]["lr"]) if opt.param_groups else 2e-4
        kwargs = optimizer_settings(getattr(self, "optimizer_kwargs", None))
        kwargs.pop("lr", None)          # базовый lr уже посчитан выше
        try:
            self.optimizer = SplitAdam(list(self.named_parameters()), lr=base_lr, **kwargs)
        except Exception as e:  # noqa: BLE001 — не роняем обучение из-за настроек оптимизатора
            print(f"WARNING: не удалось собрать SplitAdam по именам параметров ({e}); "
                  f"остаётся один оптимизатор на всю сеть")

    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        mask = getattr(self, "_mask", None)
        if mask is None:
            # get_distribution() можно вызвать без forward (внешние инструменты/тесты): маски
            # нет — считаем допустимыми все действия. Раньше здесь был AttributeError.
            return self.action_dist.proba_distribution(action_logits)

        # защита от маски чужой длины (например, 9 действий у DoublesEnv против 26 у gen9-фьюжна):
        # без неё `action_logits + additive_mask` падает на несовместимых формах прямо в обучении
        if mask.shape[-1] != action_logits.shape[-1]:
            if not getattr(self, "_mask_shape_warned", False):
                self._mask_shape_warned = True
                print(f"WARNING: маска действий длины {mask.shape[-1]} не подходит к голове "
                      f"({action_logits.shape[-1]} действий) — маскирование пропущено")
            return self.action_dist.proba_distribution(action_logits)

        no_valid_action = mask.sum(dim=-1) == 0
        if no_valid_action.any():
            print(f"WARNING: empty action mask for {no_valid_action.sum().item()} batch element(s)")
            mask = mask.clone()
            mask[no_valid_action] = 1

        additive_mask = torch.where(mask == 1, 0.0, float("-inf"))
        return self.action_dist.proba_distribution(action_logits + additive_mask)
