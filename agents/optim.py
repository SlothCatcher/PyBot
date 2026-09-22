"""Adam с раздельным управлением learning rate для Policy / Value / Shared.

ЗАЧЕМ
-----
Один `torch.optim.Adam` на всю сеть — это один lr, один eps и одна «память» моментов для
экстрактора признаков, policy-головы и value-головы. Но у них разные задачи и разный масштаб
градиента: value учится на MSE с таргетами ±30 (масштаб большой), policy — на clipped
surrogate и log-probs (масштаб маленький). Общий lr заставляет выбирать компромисс, обычно
плохой для обеих частей.

Плюс важное правило перехода BC -> PPO (рекомендации, по которым сделаны дефолты):

  1. BC: `Adam(lr=1e-3, eps=1e-8)` с плавным спадом lr до 1e-4 (cosine/linear) к концу датасета;
  2. точка перехода: **новый объект оптимизатора** для PPO — моменты BC не переносятся
     (динамика градиентов в RL другая, «память» ломает первые шаги);
  3. PPO: `Adam(lr=1e-4 или 3e-4, eps=1e-5)`.

ЧТО ЗДЕСЬ
---------
* `SplitAdam` — подкласс `torch.optim.Adam`, который раскладывает параметры политики по группам
  `policy` (`mlp_extractor.policy_net`, `action_net`, `log_std`), `value`
  (`mlp_extractor.value_net`, `value_net`) и `shared` (экстрактор признаков и прочее).
  У каждой группы свои `lr`, `eps`, `weight_decay`, `betas` — Adam поддерживает это из коробки,
  вопрос только в том, чтобы разложить параметры по группам и не дать SB3 перезаписать lr одним
  скаляром.
* `set_lrs(policy_lr)` — единственная точка применения lr: SB3 задаёт lr policy-группы (её
  считает `ppo.lr_schedule`), остальные группы получают lr пропорционально своим настройкам
  (`lr_value`/`lr_policy`, `lr_shared`/`lr_policy`), а сверху применяется адаптивный множитель.
* Адаптивное управление lr (по запросу «adam для управления learning rate»):
    - `adapt="gnorm"` — самокалибрующийся по норме градиента (AdaScale-подобный):
      `scale = clip((target/ema_gnorm)**0.5, adapt_min, adapt_max)`, где `target` — EMA нормы
      на первых `adapt_gnorm_warmup` шагах. Большой градиент -> lr снижается, маленький ->
      повышается (в пределах коридора);
    - `adapt="plateau"` — снижение lr, когда метрика (например, `train/policy_loss`) перестаёт
      улучшаться: `scale *= adapt_factor` после `adapt_patience` неудачных наблюдений, не ниже
      `adapt_min`. Метрику подаёт `StepCounterCallback` (`observe_metric`);
    - `adapt="off"` — только расписания.
* `schedule_factor(progress_done, kind, final_ratio, warmup_frac)` — чистая функция расписания
  (`constant` / `linear` / `cosine`, с прогревом и конечным уровнем лr). Используется и в BC, и в
  `training.make_lr_schedule` для RL.
* `load_state_dict` **терпим к чужим чекпоинтам**: старые zip содержат один-единственный
  param_group (плоский Adam). Совместимость важнее, чем моменты: при несовпадении структуры
  групп моменты отбрасываются, а настройки lr/eps/расписания остаются текущими (это и есть
  «оптимизатор сбрасывается при смене раскладки»).
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import torch

try:  # SB3 нужен только для SplitLRPPO; математика расписаний работает и без него
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import update_learning_rate as _sb3_update_lr
except Exception:  # pragma: no cover - окружение без SB3 (например, юнит-тесты расписаний)
    PPO = None  # type: ignore[assignment]

    def _sb3_update_lr(optimizer, learning_rate):  # type: ignore[misc]
        for group in optimizer.param_groups:
            group["lr"] = learning_rate


GROUP_POLICY = "policy"
GROUP_VALUE = "value"
GROUP_SHARED = "shared"
GROUPS = (GROUP_POLICY, GROUP_VALUE, GROUP_SHARED)

# По этим подстрокам в именах параметров определяется группа. Порядок важен: policy проверяем
# раньше value, потому что у некоторых голов встречается и то и другое в общем префиксе.
_POLICY_MARKERS = ("mlp_extractor.policy_net", "policy_net", "action_net", "log_std",
                   "pi_features_extractor")
_VALUE_MARKERS = ("mlp_extractor.value_net", "value_net", "vf_features_extractor")


def classify_param_name(name: str) -> str:
    """Группа параметра по его имени в `policy.named_parameters()`."""
    n = str(name)
    for marker in _VALUE_MARKERS:
        if marker in n:
            return GROUP_VALUE
    for marker in _POLICY_MARKERS:
        if marker in n:
            return GROUP_POLICY
    return GROUP_SHARED


def schedule_factor(progress_done: float, kind: str = "constant", final_ratio: float = 1.0,
                    warmup_frac: float = 0.0) -> float:
    """Множитель lr: `progress_done` — доля ПРОЙДЕННОГО пути (0 — старт, 1 — конец).

    kind: `constant` (множитель 1), `linear` (1 -> final_ratio), `cosine` (косинусный спад,
    как в рекомендации для BC: 1e-3 -> 1e-4). `warmup_frac` — доля пути на линейный прогрев
    от 0 до 1 (внутри прогрева множитель растёт, дальше идёт выбранный спад).
    """
    p = float(min(max(progress_done, 0.0), 1.0))
    kind = str(kind or "constant").lower()
    final_ratio = float(final_ratio)
    warmup_frac = float(warmup_frac or 0.0)
    if warmup_frac > 0.0:
        w = min(warmup_frac, 1.0)
        if p < w:
            return p / w
        q = (p - w) / max(1e-12, 1.0 - w)
    else:
        q = p
    if kind in ("constant", "const", "none", "off"):
        return 1.0
    if kind == "linear":
        return 1.0 + (final_ratio - 1.0) * q
    if kind == "cosine":
        return final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * q))
    raise ValueError(f"неизвестное расписание {kind!r} (constant/linear/cosine)")


def _as_groups(params) -> tuple[list[tuple[str, list[Any]]], bool]:
    """Приводит вход к списку (имя_группы, параметры) + признак «группы определены по именам».

    Принимает: dict {group: params}, список пар (name, tensor) или плоский список тензоров.
    Плоский список — это путь SB3 (`optimizer_class(self.parameters())`, без имён): там групп
    ещё нет, и настройки lr_value/lr_shared применять НЕЛЬЗЯ (нечему), поэтому политика после
    создания оптимизатора пересобирает его по `named_parameters()` (см. policy.py).
    """
    if isinstance(params, dict):
        out = []
        for key in GROUPS:
            if key in params:
                out.append((key, list(params[key])))
        for key, val in params.items():
            if key not in GROUPS:
                out.append((classify_param_name(str(key)), list(val)))
        return out, True
    items = list(params)
    if not items:
        return [(GROUP_SHARED, [])], bool(params) if isinstance(params, list) else False
    if all(isinstance(x, (tuple, list)) and len(x) == 2 and isinstance(x[0], str) for x in items):
        buckets: dict[str, list[Any]] = {g: [] for g in GROUPS}
        for name, param in items:
            buckets[classify_param_name(name)].append(param)
        return [(g, v) for g, v in buckets.items() if v], True
    return [(GROUP_SHARED, items)], False


class SplitAdam(torch.optim.Adam):
    """Adam с группами policy/value/shared и адаптивным управлением lr."""

    def __init__(self, params, lr: float = 2e-4, *, lr_policy: Optional[float] = None,
                 lr_value: Optional[float] = None, lr_shared: Optional[float] = None,
                 eps: float = 1e-5, eps_policy: Optional[float] = None,
                 eps_value: Optional[float] = None, eps_shared: Optional[float] = None,
                 betas: Sequence[float] = (0.9, 0.999), weight_decay: float = 0.0,
                 amsgrad: bool = False,
                 adapt: str = "off", adapt_factor: float = 0.5, adapt_patience: int = 2,
                 adapt_min: float = 0.1, adapt_max: float = 1.0,
                 adapt_gnorm_warmup: int = 50, adapt_ema: float = 0.95):
        lr = float(lr)
        base_lr = {GROUP_POLICY: float(lr_policy) if lr_policy else lr,
                   GROUP_VALUE: float(lr_value) if lr_value else lr,
                   GROUP_SHARED: float(lr_shared) if lr_shared else lr}
        base_eps = {GROUP_POLICY: float(eps_policy) if eps_policy else float(eps),
                    GROUP_VALUE: float(eps_value) if eps_value else float(eps),
                    GROUP_SHARED: float(eps_shared) if eps_shared else float(eps)}

        raw_groups, classified = _as_groups(params)
        param_groups = []
        for name, plist in raw_groups:
            if not plist:
                continue
            group_lr = base_lr.get(name, lr) if classified else lr
            group_eps = base_eps.get(name, float(eps)) if classified else float(eps)
            param_groups.append({"params": plist, "name": name,
                                 "lr": group_lr, "eps": group_eps,
                                 "lr_base": group_lr, "eps_base": group_eps})
        if not param_groups:
            raise ValueError("SplitAdam: пустой список параметров")

        self.adapt = str(adapt or "off").lower()
        self.adapt_factor = float(adapt_factor)
        self.adapt_patience = max(int(adapt_patience), 1)
        self.adapt_min = float(adapt_min)
        self.adapt_max = float(adapt_max)
        self.adapt_gnorm_warmup = max(int(adapt_gnorm_warmup), 1)
        self.adapt_ema = float(adapt_ema)
        self.adapt_scale = 1.0
        self._gnorm_ema: Optional[float] = None
        self._gnorm_target: Optional[float] = None
        self._gnorm_steps = 0
        self._plateau_best: Optional[float] = None
        self._plateau_bad = 0
        self._plateau_seen = 0
        self._policy_lr_target: Optional[float] = None
        self._legacy_state_warned = False
        # ratio: во сколько раз группа отличается от policy — считаем один раз из базовых lr.
        # Для неклассифицированного (плоского) оптимизатора групп ещё нет: все ratio 1.0
        pol = base_lr[GROUP_POLICY] if base_lr[GROUP_POLICY] > 0 else lr
        self._lr_ratio = ({g: (base_lr[g] / pol if pol else 1.0) for g in GROUPS}
                          if classified else {g: 1.0 for g in GROUPS})

        super().__init__(param_groups, lr=lr, betas=tuple(betas), eps=float(eps),
                         weight_decay=float(weight_decay), amsgrad=bool(amsgrad))
        # torch перекладывает наш список в свои группы: вернём ratio/base на место уже после super
        for group in self.param_groups:
            name = group.get("name", GROUP_SHARED)
            base = base_lr.get(name, lr)
            epsv = base_eps.get(name, float(eps))
            group.setdefault("lr_base", base)
            group.setdefault("eps_base", epsv)
            group["eps"] = epsv
        self.set_lrs(policy_lr=self._policy_lr_target)

    # ------------------------------------------------------------------ lr ---
    def lr_by_group(self) -> dict:
        """Текущий lr по группам (после всех множителей) — для логов/TB."""
        return {str(g.get("name", GROUP_SHARED)): float(g["lr"]) for g in self.param_groups}

    def eps_by_group(self) -> dict:
        return {str(g.get("name", GROUP_SHARED)): float(g["eps"]) for g in self.param_groups}

    def set_lrs(self, policy_lr: Optional[float] = None, progress_done: Optional[float] = None) -> dict:
        """Применить lr: `policy_lr` — целевой lr policy-группы (его считает расписание).

        Остальные группы = `policy_lr * ratio_группы`, сверху — `adapt_scale`. Если `policy_lr`
        не задан, берётся сохранённый целевой (последний) — так `observe_metric`/gnorm-адаптация
        могут менять масштаб, не зная прогресса.
        """
        if policy_lr is not None:
            self._policy_lr_target = float(policy_lr)
        base_policy = self._policy_lr_target
        if base_policy is None:
            base_policy = float(self.param_groups[0].get("lr_base", self.param_groups[0]["lr"]))
            self._policy_lr_target = base_policy
        for group in self.param_groups:
            name = str(group.get("name", GROUP_SHARED))
            ratio = self._lr_ratio.get(name, 1.0)
            group["lr"] = base_policy * ratio * self.adapt_scale
            if progress_done is not None:
                group["progress_done"] = float(progress_done)
        return self.lr_by_group()

    def apply_settings(self, lr_policy: Optional[float] = None, lr_value: Optional[float] = None,
                       lr_shared: Optional[float] = None, eps_policy: Optional[float] = None,
                       eps_value: Optional[float] = None, eps_shared: Optional[float] = None,
                       betas: Optional[Sequence[float]] = None, weight_decay: Optional[float] = None,
                       adapt: Optional[str] = None, adapt_factor: Optional[float] = None,
                       adapt_patience: Optional[int] = None, adapt_min: Optional[float] = None,
                       reset_adapt_state: bool = True) -> dict:
        """Применить настройки lr/eps к УЖЕ существующему оптимизатору, сохраняя моменты Adam.

        Зачем: при `--resume` оптимизатор приходит из чекпоинта со своими lr/eps, а явные флаги
        CLI (--lr/--lr-policy/--lr-value/--lr-shared/--eps-*) должны побеждать — иначе дообучение
        молча идёт на старых lr, а раздельные значения теряются. `lr_value`/`lr_shared` задаются
        АБСОЛЮТНЫМИ значениями; кто не задан — сохраняет прежнее отношение к policy (ratio).
        """
        groups = {str(g.get("name", GROUP_SHARED)): g for g in self.param_groups}
        ratio = dict(self._lr_ratio)
        ratio[GROUP_POLICY] = 1.0
        base = float(lr_policy) if lr_policy is not None else float(
            groups.get(GROUP_POLICY, {}).get("lr_base", self._policy_lr_target or 0.0))
        if base <= 0:
            base = float(groups.get(GROUP_POLICY, {}).get("lr", 2e-4))
        if lr_value is not None and base:
            ratio[GROUP_VALUE] = float(lr_value) / base
        if lr_shared is not None and base:
            ratio[GROUP_SHARED] = float(lr_shared) / base
        self._lr_ratio = ratio
        for name, group in groups.items():
            group["lr_base"] = base * ratio.get(name, 1.0)

        eps_map = {GROUP_POLICY: eps_policy, GROUP_VALUE: eps_value, GROUP_SHARED: eps_shared}
        for name, group in groups.items():
            val = eps_map.get(name)
            if val is not None:
                group["eps_base"] = float(val)
                group["eps"] = float(val)
        if betas is not None:
            b1, b2 = (float(x) for x in betas)
            for group in self.param_groups:
                group["betas"] = (b1, b2)
        if weight_decay is not None:
            for group in self.param_groups:
                group["weight_decay"] = float(weight_decay)
        if adapt is not None:
            new_adapt = str(adapt).lower()
            if new_adapt != self.adapt and reset_adapt_state:
                self._plateau_best = None
                self._plateau_bad = 0
                self._gnorm_ema = None
                self._gnorm_target = None
                self._gnorm_steps = 0
                self.adapt_scale = 1.0
            self.adapt = new_adapt
        if adapt_factor is not None:
            self.adapt_factor = float(adapt_factor)
        if adapt_patience is not None:
            self.adapt_patience = max(int(adapt_patience), 1)
        if adapt_min is not None:
            self.adapt_min = float(adapt_min)
        self.set_lrs(policy_lr=base)
        return self.lr_by_group()

    # ------------------------------------------------- адаптивное управление ---
    def _clamp_scale(self, value: float) -> float:
        return float(min(max(value, self.adapt_min), self.adapt_max))

    def _grad_norm(self) -> float:
        sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                g = getattr(p, "grad", None)
                if g is not None:
                    sq += float(g.detach().pow(2).sum())
        return math.sqrt(sq)

    def _adapt_by_gnorm(self) -> None:
        gnorm = self._grad_norm()
        if not math.isfinite(gnorm) or gnorm <= 0.0:
            return
        if self._gnorm_ema is None:
            self._gnorm_ema = gnorm
        else:
            self._gnorm_ema = self.adapt_ema * self._gnorm_ema + (1.0 - self.adapt_ema) * gnorm
        self._gnorm_steps += 1
        if self._gnorm_target is None:
            if self._gnorm_steps >= self.adapt_gnorm_warmup:
                self._gnorm_target = self._gnorm_ema
            return
        if self._gnorm_ema > 0 and self._gnorm_target > 0:
            self.adapt_scale = self._clamp_scale((self._gnorm_target / self._gnorm_ema) ** 0.5)
            self.set_lrs()

    def observe_metric(self, value: float, higher_is_better: bool = False) -> float:
        """Метрика для `adapt="plateau"` (например, train/policy_loss). Возвращает adapt_scale."""
        if self.adapt != "plateau":
            return self.adapt_scale
        v = float(value)
        if not math.isfinite(v):
            return self.adapt_scale
        self._plateau_seen += 1
        score = -v if higher_is_better else v          # меньше — лучше
        if self._plateau_best is None or score < self._plateau_best - 1e-6:
            self._plateau_best = score
            self._plateau_bad = 0
            return self.adapt_scale
        self._plateau_bad += 1
        if self._plateau_bad >= self.adapt_patience:
            self._plateau_bad = 0
            self.adapt_scale = self._clamp_scale(self.adapt_scale * self.adapt_factor)
            self.set_lrs()
        return self.adapt_scale

    def observe_metric_value(self, value: float, higher_is_better: bool = False) -> float:
        """Алиас observe_metric для читаемости в вызовах (см. StepCounterCallback)."""
        return self.observe_metric(value, higher_is_better=higher_is_better)

    # ------------------------------------------------------------- torch API ---
    def step(self, closure: Optional[Callable] = None):
        if self.adapt == "gnorm":
            try:
                self._adapt_by_gnorm()
            except Exception:
                pass
        return super().step(closure)

    def clear_moments(self) -> None:
        """Сброс моментов (exp_avg/exp_avg_sq), настройки групп и адаптации остаются."""
        self.state.clear()

    def schedule_state(self) -> dict:
        return {"adapt_scale": float(self.adapt_scale), "adapt": self.adapt,
                "gnorm_ema": self._gnorm_ema, "gnorm_target": self._gnorm_target,
                "plateau_seen": self._plateau_seen, "plateau_bad": self._plateau_bad,
                "lr": self.lr_by_group(), "eps": self.eps_by_group(),
                "lr_ratio": dict(self._lr_ratio)}

    def describe(self) -> str:
        lr = ", ".join(f"{k}={v:.2e}" for k, v in self.lr_by_group().items())
        eps = ", ".join(f"{k}={v:.1e}" for k, v in self.eps_by_group().items())
        s = f"SplitAdam(lr: {lr}; eps: {eps}; adapt={self.adapt}"
        if self.adapt != "off":
            s += f", scale={self.adapt_scale:.3f}"
        return s + ")"

    # ------------------------------------------------ терпимая загрузка state ---
    def load_state_dict(self, state_dict: dict) -> None:
        """Загрузка состояния из чекпоинта.

        Структура групп могла измениться (старые zip: один плоский param_group; или сменился
        набор голов). Моменты Adam в таком случае неприменимы — отбрасываем их, но НЕ падаем:
        обучение продолжается на текущих настройках lr/eps. Это тот же принцип, что «при смене
        раскладки признаков оптимизатор сбрасывается».
        """
        try:
            super().load_state_dict(state_dict)
            # torch копирует param_groups ЦЕЛИКОМ из файла, поэтому восстанавливаем инварианты:
            #  * eps каждой группы берём из её же базового значения (чужой чекпоинт мог принести
            #    eps другой схемы — например, 1e-5 вместо BC-шного 1e-8);
            #  * ratio value/shared к policy пересчитываем от загруженных lr_base, иначе
            #    соотношение «policy : value» осталось бы от прежних настроек
            pol_base = None
            for group in self.param_groups:
                group["eps"] = float(group.get("eps_base", group.get("eps", 1e-5)))
                group.setdefault("label", group.get("name", GROUP_SHARED))
                base = float(group.get("lr_base", group["lr"]))
                group["lr_base"] = base
                if str(group.get("name", GROUP_SHARED)) == GROUP_POLICY:
                    pol_base = base
            if pol_base:
                for group in self.param_groups:
                    name = str(group.get("name", GROUP_SHARED))
                    self._lr_ratio[name] = float(group.get("lr_base", pol_base)) / pol_base
            self.set_lrs()
            return
        except Exception as e:
            saved_groups = []
            try:
                saved_groups = state_dict.get("param_groups", [])
            except Exception:
                pass
            if not getattr(self, "_legacy_state_warned", False):
                self._legacy_state_warned = True
                warnings.warn(
                    f"SplitAdam: состояние оптимизатора из чекпоинта несовместимо "
                    f"({len(saved_groups)} групп(ы) в файле против {len(self.param_groups)} сейчас): "
                    f"моменты отброшены, lr/eps/расписание остаются текущими "
                    f"({type(e).__name__}: {e})", RuntimeWarning, stacklevel=2)

    # ----------------------------------------------------------- конструктор ---
    @classmethod
    def for_policy(cls, policy, **kwargs) -> "SplitAdam":
        """Создать оптимизатор по `policy.named_parameters()` (с именами -> группы pi/vf/shared).

        База для группы policy — `lr_policy`, если он задан, иначе `lr`; lr value/shared задаются
        абсолютными значениями (`lr_value`/`lr_shared`) и хранятся как отношение к policy.
        """
        kwargs = dict(kwargs)
        lr = float(kwargs.pop("lr", 2e-4))
        lr_policy = kwargs.pop("lr_policy", None)
        base = float(lr_policy) if lr_policy else lr
        opt = cls(list(policy.named_parameters()), lr=lr, lr_policy=lr_policy, **kwargs)
        try:
            opt.set_lrs(policy_lr=base)
        except Exception:
            pass
        return opt


def optimizer_settings(optimizer_kwargs: Optional[dict]) -> dict:
    """Выделяет из `optimizer_kwargs` те ключи, которые понимает SplitAdam (остальные — чужие)."""
    allowed = ("lr", "lr_policy", "lr_value", "lr_shared", "eps", "eps_policy", "eps_value",
               "eps_shared", "betas", "weight_decay", "amsgrad", "adapt", "adapt_factor",
               "adapt_patience", "adapt_min", "adapt_max", "adapt_gnorm_warmup", "adapt_ema")
    src = dict(optimizer_kwargs or {})
    return {k: v for k, v in src.items() if k in allowed and v is not None}


def build_policy_optimizer(policy, lr: float = 2e-4, **kwargs) -> SplitAdam:
    """Пересобрать оптимизатор политики с именованными группами и вернуть его.

    Моменты при этом теряются (создаётся новый объект) — это осознанное поведение: так
    реализовано правило «BC и PPO используют разные оптимизаторы».
    """
    kwargs = dict(kwargs)
    kwargs.pop("params", None)
    kwargs["lr"] = float(lr)
    opt = SplitAdam.for_policy(policy, **kwargs)
    policy.optimizer = opt
    return opt


def reset_policy_optimizer(policy, lr: float = 2e-4, reason: str = "", verbose: bool = True,
                           **kwargs) -> SplitAdam:
    """Новый оптимизатор для policy (правило сброса при BC -> PPO и при смене раскладки)."""
    had_state = False
    try:
        old = getattr(policy, "optimizer", None)
        had_state = bool(old is not None and getattr(old, "state", None))
    except Exception:
        pass
    opt = build_policy_optimizer(policy, lr=lr, **kwargs)
    if verbose:
        why = f" ({reason})" if reason else ""
        print(f"Оптимизатор создан заново{why}: моменты Adam "
              f"{'сброшены' if had_state else 'не было'} -> {opt.describe()}")
    return opt


def make_lr_schedule(initial_lr: float, total_timesteps: int, steps_holder: dict,
                     schedule: str = "linear", final_ratio: float = 0.0,
                     warmup_frac: float = 0.0) -> Callable[[float], float]:
    """Расписание lr для SB3: функция от `progress_remaining` (1 -> 0), как ждёт SB3.

    Прогресс берём из `steps_holder["value"]` (глобальный счётчик шагов прогона), чтобы
    расписание не сбивалось при нескольких фазах `learn()`.
    """
    initial_lr = float(initial_lr)
    if total_timesteps is None or total_timesteps <= 0:
        return lambda progress_remaining: initial_lr

    def lr_schedule(progress_remaining: float) -> float:
        global_done = min(max(steps_holder.get("value", 0) / float(total_timesteps), 0.0), 1.0)
        return float(initial_lr * schedule_factor(global_done, schedule, final_ratio, warmup_frac))

    return lr_schedule


def bc_lr_at(epoch_progress: float, lr: float, lr_final: float, schedule: str = "cosine",
             warmup_frac: float = 0.0) -> float:
    """lr для эпохи/батча BC: спад `lr -> lr_final` по выбранному расписанию."""
    lr = float(lr)
    lr_final = float(lr_final)
    if lr_final >= lr:
        return lr
    ratio = lr_final / lr
    return float(lr * schedule_factor(epoch_progress, schedule, final_ratio=ratio,
                                      warmup_frac=warmup_frac))


if PPO is not None:
    class SplitLRPPO(PPO):  # type: ignore[misc,valid-type]
        """PPO, который обновляет lr по группам (policy/value/shared), а не одним скаляром.

        SB3 штатно делает `update_learning_rate(optimizer, lr)` — то есть ПЕРЕЗАПИСЫВАЕТ lr
        всем группам одинаково. Здесь lr policy-группы берётся из `ppo.lr_schedule`, а группы
        value/shared получают свой lr из настроек SplitAdam, плюс применяется адаптивный
        множитель. Для обычного Adam поведение прежнее.
        """

        def _update_learning_rate(self, optimizers) -> None:
            opts = optimizers if isinstance(optimizers, list) else [optimizers]
            progress_remaining = float(getattr(self, "_current_progress_remaining", 1.0))
            try:
                base_lr = float(self.lr_schedule(progress_remaining))
            except Exception:
                base_lr = float(getattr(self, "learning_rate", 2e-4))
            try:
                self.logger.record("train/learning_rate", base_lr)
            except Exception:
                pass
            for opt in opts:
                if opt is None:
                    continue
                if isinstance(opt, SplitAdam):
                    opt.set_lrs(policy_lr=base_lr, progress_done=1.0 - progress_remaining)
                    try:
                        for name, lr in opt.lr_by_group().items():
                            self.logger.record(f"train/lr_{name}", float(lr))
                        self.logger.record("train/lr_adapt_scale", float(opt.adapt_scale))
                    except Exception:
                        pass
                else:
                    _sb3_update_lr(opt, base_lr)


def ensure_split_ppo(ppo):
    """Приводит загруженный из чекпоинта PPO к SplitLRPPO.

    `PPO.load()` возвращает БАЗОВЫЙ PPO: без этого метода `_update_learning_rate` берётся из SB3
    и перезаписывает lr всем группам одним значением — раздельные lr policy/value/shared молча
    теряются ровно на том маршруте, где они нужны (дообучение после BC через `--resume`).
    Класс меняем на месте: объект/моменты/веса не трогаем.
    """
    if ppo is None:
        return ppo
    if not isinstance(ppo, SplitLRPPO):
        try:
            ppo.__class__ = SplitLRPPO
        except TypeError:
            return ppo
    try:
        adopt_optimizer(ppo.policy)
    except Exception:
        pass
    return ppo


def adopt_optimizer(policy, verbose: bool = False):
    """Плоский torch.Adam -> SplitAdam с ПЕРЕНОСОМ моментов (state у Adam привязан к тензору).

    Нужно для старых чекпоинтов: там optimizer_class = torch.optim.Adam, и без конверсии
    раздельные lr на дообучении не работают (SB3 перезаписывает lr всем группам одним числом).
    Моменты экспоненциальных средних переносятся как есть — параметры те же самые объекты.
    """
    opt = getattr(policy, "optimizer", None)
    if opt is None or isinstance(opt, SplitAdam):
        return opt
    group = (getattr(opt, "param_groups", None) or [{}])[0]
    lr = float(group.get("lr", 2e-4))
    eps = float(group.get("eps", 1e-5))
    betas = tuple(group.get("betas", (0.9, 0.999)))
    wd = float(group.get("weight_decay", 0.0))
    new = SplitAdam(list(policy.named_parameters()), lr=lr, eps=eps, betas=betas, weight_decay=wd)
    moved, kept = 0, 0
    try:
        for param, state in list(getattr(opt, "state", {}).items()):
            if param in new.state or any(param is q for g in new.param_groups for q in g["params"]):
                new.state[param] = state
                moved += 1
            else:
                kept += 1
    except Exception:
        pass
    policy.optimizer = new
    if verbose and (moved or kept):
        print(f"Оптимизатор: плоский Adam -> SplitAdam (группы policy/value/shared), "
              f"моменты перенесены: {moved}, потеряно: {kept}")
    return new


def lr_state(ppo) -> dict:
    """Снимок lr/eps/адаптации для логов и тестов (работает и со старым плоским Adam)."""
    try:
        opt = ppo.policy.optimizer
    except Exception:
        return {}
    if isinstance(opt, SplitAdam):
        return opt.schedule_state()
    try:
        return {"lr": {"shared": float(opt.param_groups[0]["lr"])},
                "eps": {"shared": float(opt.param_groups[0].get("eps", 1e-8))},
                "adapt_scale": 1.0, "adapt": "off"}
    except Exception:
        return {}
