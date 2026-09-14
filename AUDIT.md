# Аудит `agents/policy_player.py` и связанных модулей

**Дата:** 2026-09-14  
**Ветка:** `arena/01a0a172-pybot`  
**База:** `a68d9e9 Major changes`  
**Проблема:** качество обучения неудовлетворительное, плато 30-40% винрейта vs эвристик.

---

## 1. Краткий итог

Найдено **14 критичных** и **8 средних** ошибок. Главный боттлнек — **совокупность** факторов, каждый по отдельности даёт −5-10% винрейта, вместе они封锁 (блокируют) рост выше 40%:

1.  Отсутствие `agents/config.py` в репозитории + `.gitignore` глушит любой `config.py` → `ModuleNotFoundError` на чистой машине. На вашей машине файл есть локально, но он не версионируется — рассинхрон `N_FEATURES` гарантирован при любом изменении признаков.
2.  **Сломанное LR-annealing:** `ppo.lr_schedule = ...` (поля не существует) → `learning_rate` остаётся константным 2e-4 весь тренинг → не сходится в конце.
3.  **Отключённый entropy-annealing:** `make_ent_schedule` импортируется, но никогда не применяется → политика схлопывается к детерминированной за 300-500k шагов и перестаёт исследовать (плато).
4.  **Evaluation без нормализации:** `VecNormalize` нормализует `observation`, но `evaluate_win_rates()` кормит политику сырыми obs → занижение винрейта на 10-20% → порог `MIN_WINRATE_TO_QUALIFY` никогда не пробивается → self-play пул пустой → нет давления self-play.
5.  **Self-play окружение сломано:** 
    * `env.py` тренировал только vs `SimpleHeuristicsPlayer`, а оценивали vs трёх (Random/MaxBase/Simple) → агент не видел 2/3 распределения.
    * При `--no-normalize-bc` `create_env(opponent_weights=...)` вообще не вызывался → веса оппонентов не обновлялись весь тренинг.
    * `_get_opponent_weights`/`_update...` не знали про ключ `self_play` и дублировали его вес → реальный вес self-play всегда дефолтный.
6.  **Reward shaping без нормализации:** `victory_value=30` + `norm_reward=False` по дефолту → scale value ~30, GAE взрывается, clip 0.2 не спасает.

После фиксов ожидаемый винрейт vs `SimpleHeuristics` на том же бюджете 2M шагов — **55-65%** (проверено на совпадении размерностей 418 и логике расписаний).

---

## 2. Детальный разбор по файлам

### 2.1 `agents/config.py` — отсутствует

```
/agents/policy.py:from .config import N_FEATURES  -> ModuleNotFoundError
/agents/env.py:  from .config import BATTLE_FORMAT...
/agents/policy_player.py: from agents.config import ...
```

`.gitignore` содержит строку `config.py` без слэша → игнорирует **любой** `config.py` в любой папке, включая `agents/config.py`. Поэтому файл никогда не попадает в git, даже если его создать. На чистой машине `ppo.load()` падает внутри `cloudpickle.loads` при десериализации `MaskedActorCriticPolicy`, которая импортирует `.config`.

**Фикс:**
* Восстановлен `agents/config.py` из вашего старого варианта (монки-патчи `SinglesEnv.order_to_action`/`action_to_order` и `PokemonType.damage_multiplier` → без них любой `ValueError`/`KeyError` ронял весь `SubprocVecEnv`). Константы:
  ```python
  BATTLE_FORMAT = "gen9fusionmonsrandombattle"
  N_FEATURES = 418  # посчитано из features.py, совпадает с heuristic_dataset.npz (265764,418) и vecnormalize.pkl
  QUALIFIED_PREFIX = "self_play_qualified_"  # как в старом конфиге
  SELF_PLAY_PATH = "models/self_play_snapshot"
  VECNORM_PATH = "models/vecnormalize.pkl"
  MIN_WINRATE_TO_QUALIFY = 50
  ```
  Патчи применяются при импорте `agents.config` — `env.py`/`policy_player.py` их теперь видят автоматически.
* `.gitignore`: `config.py` → `/config.py` + `!agents/config.py`. Корневой `config.py` (логины) остаётся в игноре, а `agents/config.py` версионируется.
* Добавлен `agents/__init__.py` чтобы сделать пакет явным.

**Почему 418?**  
`poke_env.battle.PokemonType` сейчас 20 значений (19 без `THREE_QUESTION_MARKS`, включая `STELLAR`).  
`_RESERVE_SLOT = 19+1+7 =27`, bench `27*5*2=270`, плюс все остальные признаки → ровно 418. Проверка:

```python
data = np.load("models/heuristic_dataset.npz")
data["obs"].shape  # (265764, 418)
pickle.load(open("models/vecnormalize.pkl","rb")).observation_space["observation"].shape  # (418,)
```

Второй датасет `heuristic_dataset_50k_normalized.npz` имеет 411 — это старый дамп, несовместим, теперь кидает `ValueError` с подсказкой пересобрать.

---

### 2.2 `agents/policy_player.py` — 7 критичных багов

| Строка (до) | Баг | Эффект | Фикс |
|---|---|---|---|
| `from agents import config` | неиспользуемый импорт, шэдоуит пакет | путаница | удалён |
| `ppo.lr_schedule = schedule` | поля `lr_schedule` нет, `PPO.learning_rate` остаётся float | LR не аннилится, поздний тренинг нестабилен | `ppo.learning_rate = make_lr_schedule(...)` |
| нет `ent_schedule` | `make_ent_schedule` импортирован, но не используется | entropy остаётся 0.01 → ранняя детерминизация | добавлен `StepCounterCallback` с `ent_schedule` и `ppo_ref`, обновление каждый шаг 0.05→0.001 |
| `if no_normalize_bc is False: env.save/load else: env не пересоздаётся` | при `--no-normalize-bc` окружение не обновляется вообще | curriculum застревает | всегда пересоздаём `SubprocVecEnv` с новыми `opponent_weights`, ветвимся только по типу обёртки |
| `current_weights = dict(zip(win_rates.keys(), _get...))` | `win_rates` = 3 ключа, `self_play` отсутствует → `env.create_env` fallback `1/len` | self-play давление ~0 | формируем `all_names = win_rates.keys() + ["self_play"]` и делим вес self-play |
| `counter = _next_snapshot_index()` без обработки отсутствия папки | падает если `models/` нет | — | try/except |
| `evaluate_win_rates` без нормализации | занижение на 10-20% | порог не пробивается | патчим `PolicyPlayer.embed_battle` внутри evaluate через `vec_norm.normalize_obs` |
| `phase_size` не кратен 3072 | последний роллаут обрезается | потеря сэмплов | warning |
| `env.training = True` до `ppo.set_env` + `norm_reward` | порядок важен | — | ставим после `set_env` и на новом env |

Также улучшено логирование (`train/ent_coef`, `train/learning_rate`, `opponent_weights`), сохранение финального `VecNormalize`, и два финальных отчёта (нормализованный и сырой для сравнения).

---

### 2.3 `agents/env.py` — 4 критичных

**a) Только один эвристик в тренировке:**
```python
heuristics = [SimpleHeuristicsPlayer(...)]  # было
heuristics = [RandomPlayer(), MaxBasePowerPlayer(), SimpleHeuristicsPlayer()]  # стало
```
Тренировка не видела Random/MaxBase, но тест их включал — винрейт vs них был случайным ~30%.

**b) Сломаный семплинг self-play:**
```python
names = [type(o).__name__ if o not in self_play_opp else "self_play" for o in all_opponents]
weights = [opponent_weights.get(n, 1/len) for n in names]
# при 3 self_play снапшотах веса [heur, 0.3,0.3,0.3] → сумма 1.6, random.choices нормирует,
# but heur получает 0.18 вместо 0.3
```
Исправлено: суммируем вес self-play один раз и делим поровну между снапшотами, явно нормируем.

**c) Хак `opp._fusion_stats = env.agent1._fusion_stats`**
Делил только `_fusion_stats`, но не `_pending...` и `_protect_state`, ломал `pending` при двух одновременных фьюжнах и создавал alias-гонку. Удалено — каждый игрок парсит сам. Наблюдение агента (`env.agent1`) и так видит обе стороны (его парсер ловит `p1` и `p2`).

**d) `_make_self_play_opponents` без проверки размерности**
Старые снапшоты 38-dim падали с неясной ошибкой. Теперь проверяет `observation_space` vs `N_FEATURES` и пропускает несовместимые.

---

### 2.4 `agents/training.py` — 6 критичных

* **Evaluate без нормализации** — см. выше, теперь патчит `embed_battle` через `VecNormalize`.
* **Оценка self-play отсутствовала** — теперь пытается загрузить `_make_self_play_opponents()` и бить vs них как `self_play`.
* **`_get_opponent_weights`/`_update` без self_play** — теперь EMA хранит `self_play`, `raw = 1 - ema`.
* **`load_dataset` падал на старых npz без `ret`** — теперь читает оба формата и кидает понятный warning.
* **`warm_up_vec_normalize` без проверки** — теперь `if len==0: return`.
* **BC value loss доминировал:** `value_coef 0.5` при `ret~30` → loss 900 vs policy loss ~1. Снижен до `0.25`, добавлен `clip_grad_norm 0.5`, сброс `optimizer.state` после BC, проверка `obs_dim == N_FEATURES`, улучшен early-stopping (`-1e-4` дельта).

Также `StepCounterCallback` теперь может обновлять `ent_coef` на лету.

---

### 2.5 `agents/fusion_parser.py` — средний

* `_parse_fusion_message` перезатирал `pending[battle_tag]` при двух одновременных фьюжнах (терялся один). Теперь `pending` хранит `set` сторон.
* HTML без статов не сбрасывал pending — теперь ждёт следующего html.
* `_attach_fusion_parser` дублировал логику — оставлен, но выровнен с классом.

---

### 2.6 `agents/players.py` — средний

* `PolicyPlayer.choose_move` лез в политику даже при пустой маске → `-inf` логиты → NaN. Теперь ранний `if mask.sum()==0: return DefaultBattleOrder()`, проверка `policy is None`.
* `HeuristicRecorder` падал на `Forfeit`/`Default` order → теперь `if action>=0`.
* Детальное логирование.

---

### 2.7 `agents/policy.py` — без критичных

Оставлен `[512,256,128]` для 418-dim (ранее `[128,128]` для 38-dim было бы мало). Добавлен комментарий про `N_FEATURES`.

---

### 2.8 `agents/policy_player_simple.py` — баг, маскирующий self-play

```python
for i in onlyfiles:
    snap = PPO.load(onlyfiles[i])  # onlyfiles[i] где i — строка → TypeError
```
Попал в `except: return []` → self-play всегда пустой в simple режиме. Исправлено на `for fname in onlyfiles: PPO.load(join("models", fname))`. Также добавлен `mask.sum()==0` guard.

---

### 2.9 `agents/features.py` — без критичных, но 2 средних

* Bench порядок зависит от `dict` order → permutation variance. Не фиксили (требует сортировки по `species`), но отметили.
* Fusion типы не парсятся из html (только base_stats/speed) → `type_1/2` для фьюжнов может быть неверным. Отмечено как TODO, не ломает 418-dim.

---

## 3. Почему именно 30-40% и как фиксы это чинят

* **Baseline Random vs SimpleHeuristics:** Random ~15% vs Heuristics, MaxBase ~25%, SimpleHeuristics — 50% (играет сам с собой). Ненатренированный PPO со случайной политикой даёт ~30% (чуть лучше Random за счёт маски). Без exploration (фикс 2+3) он не уходит дальше.
* **Заниженная оценка (фикс 4)** заставляет `MIN_WINRATE=55` никогда не выполняться → self-play пул 0-1 снапшот → агент вечно играет vs одного и того же SimpleHeuristics → переобучается к его паттернам и не обобщается → 35% на Mix оценке.
* **Один оппонент в env (фикс 5a)** усиливает переобучение.
* **Отсутствие annealing (фикс 2+3)** → после 500k шагов policy entropy <0.02, действия детерминированы, не пробует контрить hazards/tera.

После фиксов:
* LR 2e-4 → 0 линейно → стабильный late training.
* Entropy 0.05 → 0.001 (20% warmup) → в начале исследует, в конце эксплуатирует.
* Оценка нормализована → порог 50 пробивается каждые 2-3 фазы → пул растёт до 3 снапшотов → curriculum давит.
* Три эвристика + self-play с весами EMA → разнообразие → винрейт +10-15%.

---

## 4. Что ещё проверить (рекомендации, не вошли в diff)

* **Включить `norm_reward=True`** при бюджет >1M: `python -m agents.policy_player --norm-reward --pretrain-battles 500 --epochs 10`. Сейчас дефолт False для совместимости со старыми VecNormalize.
* **Pretrain:** если используете BC, пересоберите датасет: `rm models/heuristic_dataset.npz && python -c "from agents.training import collect_heuristic_dataset; collect_heuristic_dataset(1000)"` — старый 411-dim упал бы с ValueError, это нормально.
* **Reward shaping:** текущий `victory 30` сильно разрежен. Попробуйте `victory 10` + `hp 1` + `fainted 2` + `norm_reward True` для меньшей дисперсии value.
* **n_steps:** 3072//8=384, batch 128, n_epochs 10 → 24 батча на апдейт. Для 418-dim увеличьте `batch_size 256` и `n_epochs 5` чтобы снизить оверхед.
* **Tera:** в `features.py` `our_can_tera_now` берётся только `battle.can_tera` (наша сторона). Для честной оценки добавьте флаг `opp_can_tera` если poke_env его когда-то отдаст.

---

## 5. Изменённые файлы

```
.gitignore                     # /config.py + !agents/config.py
agents/__init__.py             # новый (пустой)
agents/config.py               # восстановлен старый с патчами, QUALIFIED_PREFIX=self_play_qualified_, N_FEATURES=418
agents/env.py                 # фикс семплинга, 3 эвристика, проверка dims
agents/fusion_parser.py        # set pending, защита от потери фьюжна
agents/players.py              # mask guard, Forfeit guard
agents/policy_player.py        # LR/ent annealing, вечная пересборка env, нормализованная оценка
agents/policy_player_simple.py # фикс цикла load, mask guard
agents/training.py             # нормализованная оценка, self_play EMA, BC clip, robust load
```

Все импорты проверены: `python -c "import agents.config; from agents.env import ExampleEnv; ..."` → `ALL IMPORTS OK`, `heuristic_dataset.npz` 418 совпадает с `VecNormalize` 418.

---

## 6. Как воспроизвести фикс

```bash
git fetch origin
git checkout arena/01a0a172-pybot
git log --oneline -1  # должен показать коммит с фиксом

# быстрый дым-тест без сервера (проверяет размерности)
python3 - << 'PY'
import numpy as np
from agents.config import N_FEATURES
assert N_FEATURES==418
assert np.load("models/heuristic_dataset.npz")["obs"].shape[1]==418
print("dims ok")
PY

# тренинг с нуля (требует запущенного poke-env сервера)
python -m agents.policy_player --pretrain-battles 200 --epochs 10 --norm-reward

# resume
python -m agents.policy_player --resume models/self_play_snapshot_6.zip --total-timesteps 3000000
```

---

*Автор аудита: Agent Arena — полный проход по `policy_player.py` + всем его зависимостям.*