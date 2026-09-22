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


def masked_action_distribution(action_dist, action_logits, mask, owner=None):
    """Дискретное распределение с additive-маской (-inf запрещённым действиям).

    Вынесено из политики, потому что этим пользуются оба режима: indices (логиты из головы) и
    embed (логиты из скоров кандидатов). Поведение 1:1 прежнее, включая предупреждения.
    """
    if mask is None:
        # get_distribution() можно вызвать без forward (внешние инструменты/тесты): маски
        # нет — считаем допустимыми все действия. Раньше здесь был AttributeError.
        return action_dist.proba_distribution(action_logits)

    # защита от маски чужой длины (например, 9 действий у DoublesEnv против 26 у gen9-фьюжна):
    # без неё `action_logits + additive_mask` падает на несовместимых формах прямо в обучении
    if mask.shape[-1] != action_logits.shape[-1]:
        if owner is not None and not getattr(owner, "_mask_shape_warned", False):
            owner._mask_shape_warned = True
            print(f"WARNING: маска действий длины {mask.shape[-1]} не подходит к голове "
                  f"({action_logits.shape[-1]} действий) — маскирование пропущено")
        return action_dist.proba_distribution(action_logits)

    no_valid_action = mask.sum(dim=-1) == 0
    if no_valid_action.any():
        print(f"WARNING: empty action mask for {no_valid_action.sum().item()} batch element(s)")
        mask = mask.clone()
        mask[no_valid_action] = 1

    additive_mask = torch.where(mask == 1, 0.0, float("-inf"))
    return action_dist.proba_distribution(action_logits + additive_mask)


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

    # ---- API, общий для всех политик (режимы indices и embed) --------------------------------
    def logits_and_values(self, obs):
        """(logits (B, N), values (B, 1)) — единая точка для BC/PPO вместо ручного разбора
        extract_features -> mlp_extractor -> action_net/value_net.

        Нужна, чтобы режим embed (кандидатные скоры) переиспользовал тот же код обучения: BC
        не обязан знать, как устроена политика внутри.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        return self.action_net(latent_pi), self.value_net(latent_vf)

    def distribution_for_logits(self, action_logits, mask=None):
        """Распределение по логитам с additive-маскированием запрещённых действий."""
        if mask is None:
            mask = getattr(self, "_mask", None)
        return masked_action_distribution(self.action_dist, action_logits, mask, self)

    def _get_action_dist_from_latent(self, latent_pi):
        return self.distribution_for_logits(self.action_net(latent_pi))


# =====================================================================================
# Режим embed: политика оценивает КАНДИДАТОВ по их признакам, а не по номеру слота
# =====================================================================================
# Проблема режима indices: действие 6+i — это «i-я позиция в списке», а не «вот этот приём».
# Сеть вынуждена выучить 4 разные головы под 4 позиции, хотя позиция ничего не значит: она
# зависит от порядка known_moves у покемона в конкретном бою. Со свитчами хуже: действие
# j — это индекс в порядке `battle.team`, который вообще не связан с bench-блоком признаков
# (тот отсортирован канонически), поэтому «свитч в слот 2» в одном бою — Garchomp, в другом —
# Ferrothorn, и одна и та же пара (obs, action) ведёт к разным последствиям.
#
# Здесь политика считает СКОР СОВМЕСТИМОСТИ между контекстом (HP, команда, погода, поле,
# урон, типы) и признаками каждого доступного кандидата:
#
#   context -> ctx_encoder -> ctx_vec ─┐
#                                      ├─> scorer([ctx_vec; cand_vec]) -> logit кандидата
#   кандидат -> *_encoder -> cand_vec ─┘
#
# Кандидаты: 4 приёма (33 признака: 30 из головы + оценка урона), те же 4 как теровые
# (флаг is_tera=1) и 5 резервов (52 признака: bench-слот + множители угрозы по типам
# соперника). Скоры раскладываются по каноническим индексам действий poke-env:
# 0..4 — свитчи (в embed-режиме действие j = j-й резерв в каноническом порядке, см.
# agents/action_space.py), 6..9 — приёмы, 22..25 — тера. Прочие 13 действий всегда
# замаскированы (мега/z/динамакс в gen9-фьюжне недоступны).
#
# Почему действие остаётся числом 0..25 (а не «выбор из переменного числа кандидатов»):
# у нас фиксированный набор из 13 возможных кандидатов, поэтому softmax по 26 логитам с
# маской — ровно та же математика, что и «softmax только по доступным кандидатам», но PPO
# остаётся на стандартном дискретном распределении с фиксированной формой батча. Это убирает
# целый класс тонких баг: variable-length Categorical + паддинг + importance ratio.
#
# Инвариантность к перестановкам (главное свойство режима): если поменять местами признаки
# двух кандидатов, скоры (и логиты) меняются местами так же — проверяется в
# test_slot_free_mode.py. В режиме indices такого свойства нет по построению.

class EmbeddedActorCriticPolicy(MaskedActorCriticPolicy):
    """Политика, оценивающая кандидатов по признакам (режим embed, см. AUDIT 32).

    Легаси-ветки (features_extractor/mlp_extractor/action_net/value_net) сохраняются, потому
    что на них завязаны механики SB3 (state_dict, predict, миграция размерностей), но в forward
    они НЕ используются и создаются крошечными (features_dim/net_arch задаёт policy_player).
    Свои головы названы так, чтобы группы SplitAdam определились правильно:
    `action_net_move*`/`action_net_switch*` -> policy, `value_net_embed*` -> value,
    энкодеры -> shared.
    """

    ACTION_MODE = "embed"
    N_CANDIDATES_MOVE = 4          # слоты приёмов 6..9
    N_CANDIDATES_TERA = 4          # слоты 22..25
    N_CANDIDATES_SWITCH = 5        # слоты 0..4 (канонический порядок резервов)
    ACTION_DIM = 26

    def __init__(self, *args, embed_dim: int = 128, ctx_dim: int = 256, scorer_hidden: int = 256,
                 candidate_dropout: float = 0.1, action_mode: str | None = None, **kwargs):
        # action_mode приезжает из policy_kwargs чекпоинта (SB3 кладёт туда всё, что мы передали);
        # сам класс — уже доказательство режима, поэтому значение только валидируем
        if action_mode not in (None, "embed"):
            raise ValueError(f"EmbeddedActorCriticPolicy не умеет режим действия {action_mode!r}")
        # гиперпараметры ДО super().__init__: он вызывает _build, которому они нужны
        self._embed_dim = int(embed_dim)
        self._ctx_dim = int(ctx_dim)
        self._scorer_hidden = int(scorer_hidden)
        self._candidate_dropout = float(candidate_dropout)
        super().__init__(*args, **kwargs)

    # --- сборка ---------------------------------------------------------------------------
    def _build(self, lr_schedule):
        super()._build(lr_schedule)          # легаси-ветки (см. docstring), крошечные

        try:
            from .features import context_dim, MOVE_FEATURES_DIM, SWITCH_CAND_DIM
        except ImportError:  # pragma: no cover — запуск вне пакета
            from features import context_dim, MOVE_FEATURES_DIM, SWITCH_CAND_DIM  # type: ignore
        # вход move_encoder: 33 признака приёма + флаг is_tera (тера-кандидат отличается от
        # обычного приёма тем же вектором, но флагом 1) => 34. can_tera доступна из контекста.
        d_ctx, d_move, d_switch, d_emb = int(context_dim()), int(MOVE_FEATURES_DIM) + 1, \
            int(SWITCH_CAND_DIM), self._embed_dim
        self._obs_dim_ctx = int(d_ctx)

        def enc(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, int(out_dim)),
                nn.LayerNorm(int(out_dim)),
                nn.ReLU(),
                nn.Dropout(self._candidate_dropout) if self._candidate_dropout > 0 else nn.Identity(),
            )

        # индексы срезов — буферы, чтобы модель могла ехать на любой device вместе с политикой
        from .features import context_feature_index, move_feature_index, switch_feature_index
        self.register_buffer("_ctx_idx", torch.as_tensor(context_feature_index(), dtype=torch.long),
                             persistent=False)
        self.register_buffer("_move_idx", torch.as_tensor(move_feature_index(), dtype=torch.long),
                             persistent=False)
        self.register_buffer("_switch_idx", torch.as_tensor(switch_feature_index(), dtype=torch.long),
                             persistent=False)
        try:
            from .features import can_tera_index as _cti
        except ImportError:  # pragma: no cover
            from features import can_tera_index as _cti  # type: ignore
        self._can_tera_idx = int(_cti())

        self.ctx_encoder = enc(d_ctx, self._ctx_dim)     # контекст -> ctx_dim
        self.move_encoder = enc(d_move, d_emb)           # 33+1 признак приёма -> embed_dim
        self.switch_encoder = enc(d_switch, d_emb)       # 52 признака резерва -> embed_dim
        score_in = int(self._ctx_dim) + int(d_emb)      # [ctx_vec; cand_vec]
        self.action_net_move = nn.Sequential(
            nn.Linear(score_in, self._scorer_hidden), nn.ReLU(),
            nn.Linear(self._scorer_hidden, 1))
        self.action_net_switch = nn.Sequential(
            nn.Linear(score_in, self._scorer_hidden), nn.ReLU(),
            nn.Linear(self._scorer_hidden, 1))
        # критика кормим контекстом и агрегатом по кандидатам (max/mean скоры + can_tera)
        self.value_net_embed = nn.Sequential(
            nn.Linear(int(self._ctx_dim) + 5, self._scorer_hidden), nn.ReLU(),
            nn.Linear(self._scorer_hidden, 1))

        for module in (self.ctx_encoder, self.move_encoder, self.switch_encoder):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
                    nn.init.zeros_(layer.bias)
        for head in (self.action_net_move, self.action_net_switch, self.value_net_embed):
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.zeros_(layer.bias)

    # --- прямой проход --------------------------------------------------------------------
    def _candidate_tensors(self, obs_flat: torch.Tensor):
        """(ctx (B,d), move (B,4,dm), tera (B,4,dm), switch (B,5,ds), can_tera (B,))."""
        ctx = obs_flat.index_select(1, self._ctx_idx)
        move = obs_flat.index_select(1, self._move_idx.reshape(-1)).reshape(
            obs_flat.shape[0], self.N_CANDIDATES_MOVE, -1)
        can_tera = obs_flat[:, self._can_tera_idx]
        is_tera = torch.ones_like(move[..., :1])
        tera = torch.cat([move, is_tera], dim=-1)
        move = torch.cat([move, torch.zeros_like(move[..., :1])], dim=-1)
        switch = obs_flat.index_select(1, self._switch_idx.reshape(-1)).reshape(
            obs_flat.shape[0], self.N_CANDIDATES_SWITCH, -1)
        return ctx, move, tera, switch, can_tera

    def logits_and_values(self, obs):
        obs_flat = obs["observation"] if isinstance(obs, dict) else obs
        obs_flat = obs_flat.reshape(obs_flat.shape[0], -1)
        ctx, move_c, tera_c, switch_c, can_tera = self._candidate_tensors(obs_flat)
        ctx_vec = self.ctx_encoder(ctx)                                    # (B, d)

        def score(encoder, cand, head):
            cand_vec = encoder(cand)                                       # (B, k, d)
            ctx_rep = ctx_vec.unsqueeze(1).expand(-1, cand_vec.shape[1], -1)
            return head(torch.cat([ctx_rep, cand_vec], dim=-1)).squeeze(-1)  # (B, k)

        move_scores = score(self.move_encoder, move_c, self.action_net_move)     # (B,4)
        tera_scores = score(self.move_encoder, tera_c, self.action_net_move)     # (B,4)
        switch_scores = score(self.switch_encoder, switch_c, self.action_net_switch)  # (B,5)

        batch = obs_flat.shape[0]
        # недоступные в gen9-фьюжне действия (5, 10..21: мега/z/динамакс/шестой свитч) — очень
        # большое отрицательное ЧИСЛО, не -inf: если маска когда-нибудь разрешит такой слот,
        # softmax не получит NaN (с -inf все -inf дали бы NaN в логитах)
        logits = torch.full((batch, self.ACTION_DIM), -1e9,
                            device=obs_flat.device, dtype=obs_flat.dtype)
        logits[:, 0:self.N_CANDIDATES_SWITCH] = switch_scores
        logits[:, 6:10] = move_scores
        logits[:, 22:26] = tera_scores

        # критика: контекст + агрегат по кандидатам (доступность кандидатов важна сама по себе)
        pooled = torch.cat([
            move_scores.max(dim=-1).values.reshape(-1, 1), move_scores.mean(dim=-1).reshape(-1, 1),
            switch_scores.max(dim=-1).values.reshape(-1, 1), switch_scores.mean(dim=-1).reshape(-1, 1),
            can_tera.reshape(-1, 1),
        ], dim=-1)
        values = self.value_net_embed(torch.cat([ctx_vec, pooled], dim=-1))
        return logits, values

    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"] if isinstance(obs, dict) else None
        logits, values = self.logits_and_values(obs)
        distribution = self.distribution_for_logits(logits, self._mask)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, values, distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"] if isinstance(obs, dict) else None
        logits, values = self.logits_and_values(obs)
        distribution = self.distribution_for_logits(logits, self._mask)
        return values, distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        self._mask = obs["action_mask"] if isinstance(obs, dict) else None
        logits, _ = self.logits_and_values(obs)
        return self.distribution_for_logits(logits, self._mask)

    def _predict(self, obs, deterministic: bool = False):
        return self.get_distribution(obs).get_actions(deterministic=deterministic)

    def predict_values(self, obs):
        _, values = self.logits_and_values(obs)
        return values

    def output_modules(self):
        """Модули-выходы для правила «ortho только для новых выходов» (см. _output_modules)."""
        return [("action_net_move", self.action_net_move, 0.01),
                ("action_net_switch", self.action_net_switch, 0.01),
                ("value_net_embed", self.value_net_embed, 0.01)]

# ---------------------------------------------------------------------------------------------
# Инициализация НОВЫХ выходов (правило «ortho только для новых выходов» при BC -> PPO / resume)
# ---------------------------------------------------------------------------------------------
# BC и PPO живут на одной архитектуре (миграция паддит только ВХОДНЫЕ колонки нулями), поэтому
# штатно новых выходов не появляется: чекпоинт загружается бит-в-бит и ничего не переинициализируется.
# Но если выходы всё-таки изменились (голова шире/уже числа действий, отсутствующие веса в
# чекпоинте, другая net_arch), новые строки нельзя оставлять случайными или нулевыми:
#   * нули в action_net = мёртвая голова (все логиты равны, политика не выбирает новые действия);
#   * случайный init без масштаба ломает уже обученные логиты на первых шагах.
# Для новых выходов берём стандарт PPO: ортогональная матрица с маленьким gain (0.01 для голов,
# sqrt(2) для скрытых слоёв), bias новых строк = 0. У дискретных действий log_std нет,
# так что «новых действий» касается только action_net.

def ortho_init_new_rows_(weight: torch.Tensor, old_rows: int, gain: float = 0.01,
                         bias: torch.Tensor | None = None) -> int:
    """Ортогонально инициализирует ТОЛЬКО новые строки [old_rows:] матрицы веса.

    Старые строки остаются бит-в-бит: это ключевое условие «ortho только для новых выходов».
    Возвращает число новых строк (0 — если новых нет).
    """
    if weight is None or weight.dim() != 2:
        return 0
    total = int(weight.shape[0])
    old = max(int(old_rows), 0)
    new = total - old
    if new <= 0:
        return 0
    with torch.no_grad():
        block = weight.data[old:].clone()
        nn.init.orthogonal_(block, gain=float(gain))
        weight.data[old:] = block.to(dtype=weight.dtype, device=weight.device)
        if bias is not None and bias.dim() == 1 and bias.shape[0] >= total:
            bias.data[old:total] = 0.0
    return new


def _output_modules(policy):
    """Список (ключ_префикс, модуль, gain) для слоёв, у которых строки веса = «выходы».

    Ключи берём РЕАЛЬНЫЕ (как в state_dict), иначе нельзя сравнить с чекпоинтом:
    например последний Linear pi-блока — это `mlp_extractor.policy_net.2`, а не «last».
    """
    out = []
    for name, gain in (("action_net", 0.01), ("value_net", 0.01)):
        module = getattr(policy, name, None)
        if isinstance(module, nn.Linear):
            out.append((name, module, gain))
    extra = getattr(policy, "output_modules", None)
    if callable(extra):
        for name, module, gain in extra():
            if isinstance(module, nn.Sequential):
                # у Sequential-головы инициализировать надо ПОСЛЕДНИЙ Linear (он и даёт логит)
                last = None
                for sub in module.modules():
                    if isinstance(sub, nn.Linear):
                        last = sub
                if last is not None:
                    out.append((name, last, gain))
            elif isinstance(module, nn.Linear):
                out.append((name, module, gain))
    mlp = getattr(policy, "mlp_extractor", None)
    hidden_gain = nn.init.calculate_gain("relu")
    for branch in ("policy_net", "value_net"):
        seq = getattr(mlp, branch, None) if mlp is not None else None
        if seq is None:
            continue
        key, module = None, None
        for sub_name, sub in seq.named_modules():
            if isinstance(sub, nn.Linear):
                key, module = (f"{branch}.{sub_name}" if sub_name else branch), sub
        if module is not None:
            out.append((f"mlp_extractor.{key}", module, hidden_gain))
    return out


def reinit_new_outputs_(policy, old_state: dict | None = None, missing_keys=None,
                        verbose: bool = True) -> list[str]:
    """Ортогональная инициализация ТОЛЬКО новых выходов политики.

    Новыми считаются выходы, о которых есть ЯВНОЕ доказательство:
      * ключ попал в `missing_keys` (его не было в чекпоинте) — модуль инициализируется целиком;
      * в `old_state` есть тот же ключ, но строк МЕНЬШЕ, чем сейчас (голова выросла) — тогда
        ортогонально инициализируются только новые строки, старые остаются бит-в-бит.

    Если про модуль ничего не известно (ключа нет ни в `old_state`, ни в `missing_keys`),
    он НЕ трогается: «на всякий случай переинициализировать» обученные выходы нельзя.
    Возвращает список сообщений (для лога/тестов); пустой список = ничего не менялось.
    """
    old_state = dict(old_state or {})
    missing = set(missing_keys or [])
    notes: list[str] = []

    def _is_missing(key: str) -> bool:
        return any(k == key or k.startswith(key.rsplit(".", 1)[0] + ".") for k in missing) if missing else False

    for prefix, module, gain in _output_modules(policy):
        if module is None or not hasattr(module, "weight") or module.weight.dim() != 2:
            continue
        wkey = f"{prefix}.weight"
        target = int(module.weight.shape[0])
        ref = old_state.get(wkey)
        if ref is None:
            if not (wkey in missing or _is_missing(wkey)):
                continue                      # новизна не доказана — не трогаем
            with torch.no_grad():
                nn.init.orthogonal_(module.weight, gain=float(gain))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            notes.append(f"{prefix}: выходов в чекпоинте не было ({target}) — ortho(gain={gain:g})")
            continue
        old_rows = int(ref.shape[0]) if ref.dim() == 2 else 0
        if old_rows <= 0 or old_rows >= target:
            continue
        n = ortho_init_new_rows_(module.weight, old_rows, gain=float(gain),
                                 bias=getattr(module, "bias", None))
        if n:
            notes.append(f"{prefix}: {n} новых выходов из {target} — ortho(gain={gain:g}), "
                         f"старые {old_rows} сохранены")

    if verbose:
        for note in notes:
            print(f"[init] {note}")
    return notes
