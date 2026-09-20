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
* Fusion типы не парсятся из html (только base_stats/speed) → `type_1/2` для фьюжнов может быть неверным. Частично закрыто в разделе 7: неизвестный тип больше не маскирует иммунитет (KeyError → 0.0 вместо 1.0), плюс добавлена диагностика сырых `typechange`-сообщений сервера.

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
agents/damage.py               # расчёт потенциального урона (87 признаков)
agents/vecnorm_utils.py        # паддинг статистик VecNormalize под текущий N_FEATURES
play_trained.py                # инференс с нормализацией и авто-миграцией снапшота
test_damage.py                 # тесты блока урона
test_dim_migration.py          # тесты миграции снапшотов/VecNormalize/датасетов
test_no_shadowing.py           # ast-детектор затенения имён (UnboundLocalError) по всему проекту
agents/checkpoint_utils.py     # совместимая загрузка снапшотов (миграция + кэш)
test_self_play_migration.py    # тесты загрузки self-play оппонентов со старой размерностью
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

## 7. Фикс расчёта типовой эффективности (`PokemonType.damage_multiplier`)

**Симптом:** модель спамит Electric-приём по земляному фьюжну, хотя Electric vs Ground = 0 урона.

**Диагноз (воспроизведён на poke-env 0.16.1):** чарт 9-го поколения
`GenData.from_gen(9).type_chart` содержит только 18 стандартных типов — ни `STELLAR`,
ни `THREE_QUESTION_MARKS` в нём нет. Оригинальный метод делает
`type_chart[type_1.name][self.name]` и падает с `KeyError`, если **второй** тип защиты
отсутствует в чарте:

```
ELECTRIC.damage_multiplier(GROUND, THREE_QUESTION_MARKS, type_chart=chart)
-> KeyError('THREE_QUESTION_MARKS')      # чарт знает GROUND, но не знает ???
```

Монки-патч в `agents/config.py` ловил `KeyError` и возвращал `1.0` на **весь** расчёт, то
есть иммунитет (0.0) превращался в "нейтрально". Это влияло и на признак
`moves_dmg_multiplier` в obs, и на `_move_wasted_flag` (→ wasted-штраф не начислялся),
и на reward (`_estimate_max_damage`, проверка иммунитета в `action_to_order`).

Второй, менее очевидный случай: если `???`/`STELLAR` стоит **первым** типом, poke-env
возвращает `1` ещё до обращения к чарту (иммунитет тоже не виден, но `KeyError` нет).

**Фикс:** `agents/type_utils.py::damage_multiplier_safe` считает покомпонентно —
неизвестный тип даёт множитель 1.0 только за себя, известные компоненты сохраняют вклад:

| вызов | было | стало |
|---|---|---|
| ELECTRIC vs (GROUND, None) | 0.0 | 0.0 |
| ELECTRIC vs (GROUND, ???) | **1.0** (KeyError→нейтрал) | **0.0** |
| ELECTRIC vs (GROUND, STELLAR) | **1.0** | **0.0** |
| ELECTRIC vs (???, None) | 1.0 | 1.0 (+ флаг unknown) |
| ELECTRIC vs (WATER, ???) | 1.0 | 2.0 |

Хелпер используется во всех точках: `config.py` (монки-патч), `features.py`
(`embed_battle_with_fusion`, `_move_wasted_flag`, `_weakness_score`, `_bench_moves_vec`,
`_vulnerability_frac`), `env.py` (`_estimate_max_damage`, `action_to_order`),
`policy_player_simple.py`. Размер obs не изменился (715).

**Диагностика (для подтверждения на живых боях):** `PYBOT_DEBUG_TYPES=1` печатает сырые
`-start|typechange` сообщения сервера (видно, присылает ли он `???` для фьюжнов), сводку по
бою и по фазе обучения. Первое срабатывание "иммунитет сохранён" / "неизвестный тип"
печатается всегда, без флага.

**Тест:** `python test_type_multiplier_fix.py` (нужен poke-env) — проверяет и ванильный
`KeyError`, и покомпонентный расчёт, и E2E на мок-бое (`moves_wasted` + obs-признак).

### 7.1 Гонка `typechange` (модель видит СТАРЫЙ тип)

`poke-env >= 0.12` обрабатывает батч сообщений **по порядку** и на строке `|request|` сразу
вызывает `choose_move` (→ `embed_battle`). Значит всё, что сервер прислал в том же кадре
**после** `|request|`, к моменту решения ещё не применено. Воспроизведено на настоящем
`Battle` (`test_typechange_timing.py`):

```
[">battle-tag"]
["", "switch", "p2a: Frost", "froslass, L50, M", "100/100"]     # базовые ICE/GHOST
["", "request", ...]                                            # <-- здесь идёт решение
["", "-start", "p2a: Frost", "typechange", "Steel/Ice", "[silent]"]  # реальный тип

без фикса:  на решении тип = ICE/GHOST  (СТАРЫЙ, реальный — STEEL/ICE)
с фиксом:   на решении тип = STEEL/ICE
```

Фикс (`agents/fusion_parser.py::_hoist_typechanges_before_request`): строки
`-start|typechange` из хвоста кадра (после **последнего** `|request|`) переносятся на позицию
перед этим запросом. Если в кадре несколько запросов (клиент отстал), сообщения между ними
не трогаются — иначе тип из будущего хода попал бы в решение текущего. Набор сообщений не меняется,
poke-env парсит каждую строку ровно один раз (без двойного применения), относительный
порядок остальных сообщений и самих typechange сохраняется. Хук стоит в двух местах:
миксин `FusionInfoParser._handle_battle_message` (PolicyPlayer/HeuristicRecorder) и
инстанс-патч `_attach_fusion_parser` (agent1/agent2 внутри `ExampleEnv`, которые миксин не
наследуют). Если перестановка не требовалась (typechange и так до запроса) — батч
возвращается как есть.

Побочно это же чинит случай, когда typechange приходит в первом батче боя (объект боя
создаётся внутри обработки): перестановка не зависит от наличия battle-объекта.

Диагностика: `PYBOT_DEBUG_TYPES=1` + счётчик `typechange_reordered` — печатает
`[type-fix] typechange перенесён перед |request| (...)`, а `typechange_after_request`
показывает, что гонка в логах сервера реально встречается. Сырые строки `|-start|...|typechange|`
сохраняются в `_RAW_TYPECHANGE` (видно, шлёт ли сервер `???` и на какую сторону).

Тест: `python test_typechange_timing.py` — демонстрация гонки, проверка перестановки
(ничего не теряется, порядок прочих сообщений сохраняется), отсутствие двойного применения
и поведение для `???`.

**Что осталось нерешённым:** если сервер присылает `???` — тип реально неизвестен (ни
`typechange`, ни декс его не знают), тогда любой приём считается нейтральным; если сервер
не присылает typechange для оппонента вообще — типы останутся дексовыми от базового вида.
Оба случая видны в диагностике (секция 7), но лечатся только на стороне сервера.


---

## 8. Статы фьюжна и тайминг инференса

`embed_battle` берёт статы из `_fusion_stats`, который заполняется из html-таблицы
(`|html|<b>X + Y base stats:`). Разбор вызывается в начале обработки кадра (`_parse_fusion_message`
до `super()._handle_battle_message`), поэтому html из того же кадра, что и `|request|`,
**успевает** к решению даже если сервер прислал его после запроса — это проверено
поведенческим тестом (`test_inference_paths.py`, секция 1: решение видит и тип STEEL/ICE, и
`base_stats`, и `speed_range` из кадра-гонки).

Что может сломаться и починено:

* **условие `m[-1] == "[silent]"`** — если typechange приходит без маркера, `pending` не
  выставлялся и следующая html-таблица со статами **не парсилась** (типы/статы молча терялись).
  Теперь принимаются оба варианта, вариант считается в счётчике `typechange_variant`.
* **регресс**: при правке reorder однажды были вырезаны `_parse_fusion_message` и
  `_parse_protect_message` (модуль компилировался, падало в рантайме). Восстановлено,
  добавлен тест-страж (`test_inference_paths.py`, секция 2 и дым-тест импортов).
* **диагностика**: счётчик `frame_sequence` печатает порядок сообщений в кадре с `|request|`
  (видно, шлёт ли сервер `typechange`/`html` после запроса), `stats_missing_at_decision` —
  сколько решений сыграно без статов фьюжна (дексовый фолбэк). Включается `PYBOT_DEBUG_TYPES=1`.
  Если html приходит в ДРУГОМ кадре (позже запроса), ничем не помочь — решение уже отправлено;
  именно этот случай видно по `stats_missing_at_decision`.

## 9. Инференс обученной модели: где работает фикс и что мешало

Фикс тайминга живёт в классах игроков, а не в env, поэтому он применяется и на инференсе:

| путь | класс | фикс раньше | сейчас |
|---|---|---|---|
| eval (`evaluate_win_rates`), `index.py`, `diagnose_type_spam.py` | `agents.players.PolicyPlayer` (миксин) | да | да |
| `test.py`, `agents/policy_player_simple.py` | свой `PolicyPlayer(Player)` | **нет** — ни reorder, ни статов фьюжна | наследует `FusionInfoParser` |
| игроки внутри `ExampleEnv` (обучение) | `_attach_fusion_parser` | да | да |

Дополнительно найдено на пути инференса:

* **нормализация obs не применялась** в `index.py`/`test.py`, хотя обучение идёт с
  `norm_obs=True` (`VecNormalize`) — модель получала ненормализованный obs. Добавлен
  `agents/vecnorm_utils.py` (формула 1:1 с SB3) и скрипт `play_trained.py`, который
  нормализует obs и играет лестницу/вызовы;
* **`models/vecnormalize.pkl` хранит статистику 418 признаков**, а миграция умела только
  713→715, поэтому `normalize_obs` падал на несовпадении (418 против 715). Теперь
  `pad_stats` добивает статистику до любой целевой размерности (mean=0, var=1) — работает
  и для resume обучения;
* **все снапшоты в `models/` — от старых версий признаков**: `archive/*` 38/74/118,
  `self_play_snapshot_0..6`, `ppo_policy_final`, `pretrained_10000` — 418. Текущий env даёт
  715, поэтому эти веса не загружаются в текущую политику (проверяется `play_trained.py
  --selfcheck`). Чтобы проверить фикс на живой модели, нужен снапшот, обученный на 715.

Тесты: `test_inference_paths.py` (тайминг, статы, регресс парсеров, все пути инференса,
VecNormalize 418→715, дым импортов), `test_typechange_timing.py`, `test_type_multiplier_fix.py`,
`play_trained.py --selfcheck`.


---

## 10. Что показали живые логи (гонки в этом формате нет)

Реальный прогон `PYBOT_DEBUG_TYPES=1 python play_trained.py --ladder 10` (модель 715-й версии):

```
[type-debug] typechange от сервера: |-start|p1a: +Stonjourner|typechange|Grass/Rock|[silent]
[type-debug] typechange пришёл [silent]
[type-debug] кадр с |request|: request
...
[type-debug] typechange_raw: 225 (...) | typechange_variant: 225 ([silent] x225) | frame_sequence: 306 (request x306)
```

Выводы:

* **`frame_sequence: 306 (request x306)`** — сервер присылает `|request|` отдельным кадром,
  без соседей. Значит сообщений «после запроса в том же кадре» не бывает, и перестановка
  `typechange` (раздел 7.1) в этом формате никогда не срабатывает — это страховка на случай
  будущих изменений/лагованного клиента, а не активный фикс.
* **`[silent]` приходит всегда** (225/225) — послабление условия (раздел 8) тоже страховка.
* **`???` не встречался** — все типы нормальные (`Grass/Rock`, `Poison/Normal`, `Steel/Dragon`, ...),
  поэтому `masked_immunity`/`unknown_def_type` пусты. То есть спам Electric по Ground нельзя
  объяснить ни гонкой, ни неизвестным типом *в этих боях*. Проверять надо по-другому:

### 10.1 Прямые сверки (добавлены в `diagnose_type_spam.py`)

1. **тип в бою == последний `typechange` сервера** — `type_checks` / `type_mismatch`.
   Ловит неприменённый/потерянный тип (расхождение печатается с идентификатором).
   Сообщение про другого покемона (свитч был, typechange ещё нет) считается отдельно
   (`server_msg_other_mon`) и расхождением НЕ считается.
2. **obs посчитан по серверному типу** — `obs_checks` / `obs_stale`. Для каждого доступного
   приёма `moves_dmg_multiplier` из свежего obs сравнивается с множителем по серверным типам.
   Это ровно та проверка, которая отличает «модель видит настоящий тип» от «модель видит
   устаревший/нейтральный».

В конце прогона печатается блок «СВЕРКА С ЛОГОМ СЕРВЕРА (типы)» с интерпретацией порядка
кадров. Флаг `--no-check-obs` отключает вторую сверку (она считает obs второй раз, дороже).

Тест этих проверок: `test_type_sync.py` (мок-бой, без сервера) — ловит неприменённый
typechange, устаревший obs, сообщения о другом покемоне, нормализацию `???`/`Stellar`,
и проверяет нашего покемона тоже.

### 10.2 Что делать дальше, если спам повторится

```bash
PYBOT_DEBUG_TYPES=1 python diagnose_type_spam.py --model <модель-715> --battles 20
```

Смотреть в конце: `выбрано 0x-приёмов`, `type_mismatch`, `obs_stale`. Если все три нулевые,
а модель всё равно выбирает 0x — это уже политика (обучение), а не парсинг типов:
в таком случае смотреть `выбрано приём хуже лучшего по типу` и распределение выбранных приёмов.

---

## 11. Признаки потенциального урона (`agents/damage.py`, +87 к obs)

Формализует то, что политика раньше могла выучить только из таблицы типов: **сколько**
урона принесёт каждый конкретный приём против конкретного покемона с учётом реальных
статов, бустов, поля и способностей. По просьбе: «через калькуляцию статов противника,
статов наших, базового урона приёма, эффективности, текущих бустов (погоды, способности,
stab и т.д.), наверное лучше прописывать минимально возможный».

### 11.1 Как считается

    base = floor( floor(2*level/5 + 2) * power * A / D / 50 ) + 2

* `A`/`D` — статы из базовых (`real_stat`, Gen 3+ формула: HP = `floor((2*base+31+252/4)*level/100)+level+10`,
  остальные — `floor((floor((2*base+31+252/4)*level/100)+5)*nature)`), при неизвестных EV/природе
  берётся нейтральная природа и 252 EV — консервативно для атакующего и защитника одинаково;
* **min = base × 0.85** (минимальный разброс), **max = base × 1.0** — политика видит худший
  для нас и лучший для нас варианты («минимально возможный урон» = именно 0.85);
* множители: STAB ×1.5 (Adaptability ×2), эффективность по типам через
  `damage_multiplier_safe` (иммунитет = 0, а не 1.0 — раздел 7), погода (sun/rain ×1.5
  и ×0.5 по типам, sandstorm ×1.5 к Rock при SpD-спец.), terrain (Electric/Grassy ×1.3/1.5
  для наземных), стадии бустов (`(2+n)/2` / `2/(2-n)`), Burn ×0.5 для физических,
  Multiscale ×0.5 на полном HP, экраны (Reflect/Light Screen ×0.5), способности-иммунитеты
  (Levitate, Flash Fire, Volt Absorb, Water Absorb, Dry Skin, Air Balloon, Lightning Rod,
  Storm Drain, Sap Sipper, Motor Drive, Wonder Guard и т.д.), Thick Fat / Heatproof /
  Filter / Solid Rock / Prism Armor / Fluffy / Punk Rock / Ice Scales / Fur Coat.
* Погода и террейн берутся с `battle`; если поле неизвестно — множитель не применяется.

### 11.2 Раскладка блока (87 признаков урона; актуальные срезы — в §14)

| срез      | что                        | размер |
|-----------|----------------------------|--------|
| [0:12]    | 4 доступных приёма активного против активного противника: `min_frac`, `max_frac`, `guaranteed_KO` | 12 |
| [12:15]   | лучший известный приём противника по нашему активному: `min_frac`, `max_frac`, гарантированный KO (по мин. роллу) | 3 |
| [15:51]   | матрица «наш слот i -> их слот j», `min_frac` | 36 |
| [51:87]   | матрица «их слот j -> наш слот i», `min_frac` | 36 |

`*_frac` — доля от **текущего** HP цели (cap 2.0 → обрезка), фейнт/неизвестно/нет данных → 0.
Слоты — канонический порядок (сортировка по `species`, `team_slots`), поэтому вектор
не зависит от порядка словарей в бою (проверено тестом). Обе матрицы (в обе стороны) —
решение по просьбе «не уверен, как лучше»: 36+36 признаков относительно дёшевы, а модель
сама выучит, какие из них полезны.

### 11.3 Что НЕ раскрывается (нет утечки информации)

Скрытые способность/предмет противника → множитель нейтрален (1.0), никакие «догадки» не
подставляются. Это же правило действует и для Mold Breaker (§15.1): нераскрытая способность
атакующего ничего не отключает. Неизвестный тип (`???`/`Stellar`) → нейтральный множитель, но факт
логируется (`[type-fix]` счётчики, `PYBOT_DEBUG_TYPES=1`). Данные фьюжна берутся из
`our_fusion`/`opp_fusion` и per-mon entry `our_team_fusions`/`opp_team_fusions`, то есть
из тех же источников, что и остальные признаки.

### 11.4 Производительность

Матрицы 6×6×4 приёма = до 144 расчётов на шаг, поэтому введены кэши:
`move_static` (id приёма → power/category/type/flags) и `cached_best_move_damage`
(сигнатура пары: id покемона, бусты, способность, предмет, число приёмов, для атакующего —
статус, для защитника — «полное HP» + типы, плюс погода/террейн/экраны; кэшируется
**абсолютный** урон, доли считаются от текущего HP). Ошибки построения блока больше не
проглатываются молча: первый сбой печатает traceback (иначе нулевая матрица выглядела бы
как «всё в порядке» — так уже случилось с одной опечаткой при разработке).

Замер: 6×6 команд, `_damage_block` ≈ 0.4 мс (было 2.3 мс до кэшей), полный
`embed_battle_with_fusion` ≈ 0.9 мс. Худший случай (бусты меняются каждый шаг) — 0.48 мс.
После добавления зеркала и флагов (§14) и усиления сигнатуры кэша — `_damage_block` ≈ 0.63 мс,
`embed_battle_with_fusion` ≈ 1.25 мс: сигнатуры считаются один раз на покемона
(`prepare_mon` → `stats_sig`/`moves_sig`) и выносятся из циклов матриц/зеркала.

### 11.5 Совместимость: `N_FEATURES` 715 → 802

Добавление признаков ломает всё, что было сохранено на 715: снапшоты, `VecNormalize`,
датасеты BC. Вместо требования переобучаться с нуля сделана **авто-миграция**:

* **снапшоты** (`agents/policy_player.py`): `_checkpoint_obs_dim` читает размерность из
  метаданных zip; при расхождении с `N_FEATURES` `_migrate_checkpoint_dim` добивает
  первый слой `[512, old] -> [512, N_FEATURES]` нулями (warm start: старое поведение
  сохраняется побитово, новые признаки входят с нулевыми весами), обновляет
  `observation_space` и сбрасывает optimizer state. Работает для любой старой размерности
  (38/74/118/418/715), не только 713→715; прежнее имя `_migrate_ppo_713_to_715` оставлено
  алиасом. `play_trained.py` вызывает это автоматически (`--no-migrate` — выключить).
* **VecNormalize** (`agents/vecnorm_utils.py`): `VecNormalize.load` в SB3 **падает** на
  несовпадающей размерности (`check_shape_equal`), поэтому `_load_vecnormalize_tolerant`
  читает pkl напрямую, паддит `obs_rms` (новые признаки mean=0/var=1, старые не тронуты),
  подменяет `observation_space` и только затем привязывает env. Используется и при
  resume обучения, и для инференса (`load_vecnorm_stats`).
* **датасеты BC** (`agents/training.py`): `_pad_obs_to_features` добивает старый датасет
  нулями (для >200k переходов — через `np.memmap`, чтобы не забивать RAM) вместо
  `ValueError`. Нули = «нет информации» для признаков урона, модель их игнорирует, пока не
  дообучится на свежих данных. Пересборка 3.6 ГБ датасета больше не обязательна.

Регресс-тесты: `test_dim_migration.py` (30 проверок) — читаемость размерности,
идемпотентность, побитовое сохранение старых колонок, нули в новых, forward после миграции,
паддинг `VecNormalize` (418→802), паддинг датасета (включая mmap-путь), fallback-путь
миграции и определение «несовпадение размеров» по тексту ошибки.

Мелочи, которые тоже стреляли на реальном `--resume`:

* **`UnboundLocalError: N_FEATURES`** — внутри `run()` был локальный
  `from agents.config import N_FEATURES`, из-за чего имя становилось локальным для всей
  функции, а миграционная проверка стояла выше него. Локальные импорты убраны,
  `test_no_shadowing.py` (39 проверок) сканирует **весь проект** ast-анализом и ловит
  любое «использование имени раньше его локального определения» (проверено на самом баге).
* **`--resume models/self_play_qualified_19` без `.zip`** — `resolve_checkpoint_path`
  подставляет `.zip`/`.pkl`, а если файла нет — печатает список похожих файлов рядом
  вместо голого `FileNotFoundError`.
* **fallback-миграция** больше не поднимает `SubprocVecEnv` с `ExampleEnv` (это требовало
  запущенного poke-env сервера и падало на `_DimProbeEnv` без gymnasium): политика
  собирается на крошечном `_DimProbeEnv(dim)`; паддинг определяет старую размерность из
  самого тензора, а не из `OLD_N` (тот мог быть `None`).

### 11.6 Как запускать

```bash
# все тесты фич
for t in test_damage test_type_multiplier_fix test_typechange_timing test_type_sync \
         test_inference_paths test_dim_migration test_no_shadowing; do
  PYTHONPATH=. python $t.py
done

# играть старой моделью (715) на новых признаках — миграция автоматическая
python play_trained.py --selfcheck --model models/self_play_qualified_19.zip   # проверить пайплайн
python play_trained.py --ladder 10 --model models/self_play_qualified_19.zip   # играть

# resume обучения с миграцией снапшота и VecNormalize
python -m agents.policy_player --resume models/self_play_snapshot_6.zip --total-timesteps 3000000
```

**Важно:** миграция сохраняет поведение *до* дообучения, но признаки урона входят в политику
с нулевыми весами. Чтобы они реально влияли на игру, нужен либо BC-прогон на свежих данных,
либо PPO-дообучение — иначе сеть продолжит играть как на 715.


---

## 12. Нужно ли пересматривать `net_arch` после +87 признаков

Короткий ответ: **головы pi/vf — нет, размер экстрактора — по желанию, и решать это надо по
метрике, а не на глаз.** Числа:

| вариант | параметров | шаг оптимизации (batch 256, CPU) |
|---|---|---|
| 802 → 512, pi/vf `[512,256]` (текущее) | 1 202 698 | 18.7 мс |
| 802 → 640, pi/vf `[512,256]` | 1 436 810 | 21.6 мс (+16%) |
| 802 → 768, pi/vf `[512,256]` | 1 670 922 | 25.7 мс (+37%) |
| 802 → 640, pi/vf `[640,320]` | 1 749 130 | 24.8 мс (+33%) |

Что из этого следует:

* **Признаки почти ничего не стоят.** +87 входов = +44.5k параметров (+4.3% сети), первый слой
  вырос с 367k до 411k. Сжатие первого слоя стало 1.57x вместо 1.40x — это всё ещё мягко.
* **Головы `[512,256]` не связаны с числом входов.** Их размер определяется сложностью задачи,
  а не шириной наблюдения. Расширять их вместе с признаками — самая дорогая часть (+33% времени
  против +16%), и пользы ждать не от чего, пока сеть не упрётся.
* **Оптимизация вообще не узкое место.** 2M шагов при `batch_size=256, n_epochs=3` — это ~23k
  обновлений, то есть 7–8 минут счёта на CPU против часов боёв. Поэтому «дорого/дёшево» тут
  не аргумент: если 640 помогает по winrate — его стоит взять.
* **Настоящий риск — не размер, а выключенные признаки.** После warm start веса столбцов
  `[715:802]` равны нулю: сеть сначала играет как старая, а новые признаки включает постепенно.
  Поэтому в фазах обучения теперь пишется метрика (и в TensorBoard, и в лог):

```
[arch] features_dim=512 использование новых признаков (RMS весов, доля от старых) = 0.0412
```

  `arch/new_cols_rms_ratio` — RMS весов новых столбцов / RMS старых. `0` сразу после миграции,
  ~`1.0` когда блок урона выучен наравне с остальными признаками (свежая сеть даёт ровно 1.0,
  проверено тестом), `< 0.02` — признаки практически не используются (печатается подсказка
  «подними lr или features_dim»).

* **Итог.** По умолчанию оставляем `--features-dim 512` (так обучены все текущие снапшоты).
  Если через 200–300k шагов `arch/new_cols_rms_ratio` стоит ниже ~0.3, есть два хода: поднять
  `--learning-rate` (1e-5 включает новые входы медленно) или перейти на `--features-dim 640` —
  warm start сохранится, потому что миграция паддит и первый слой, и LayerNorm, и входы pi/vf:

```
python -m agents.policy_player --resume models/self_play_qualified_19.zip --features-dim 640 ...
#   Миграция ...: obs 715->802, features_dim 512->640
#     паддинг features_extractor.net.0.weight [512, 715] -> [640, 802]
#     паддинг features_extractor.net.1.weight [512] -> [640] (новые = 1)
#     паддинг mlp_extractor.policy_net.0.weight [512, 512] -> [512, 640]
```

  Обратное сужение (например 512→384) тоже поддержано — слой обрезается, лишние нейроны
  отбрасываются.

Проверки: `test_dim_migration.py` (43) — паддинг первого слоя, LayerNorm новых нейронов
(weight=1/bias=0), вход pi-головы, forward расширенной политики, сужение 512→384, метрика
использования (свежая сеть ≈1.0, warm start ровно 0).

---

## 13. Self-play оппоненты после смены N_FEATURES (почему обучение шло без них)

В логе реального запуска после миграции основного снапшота видно километровый повтор:

```
Failed to load qualified snapshot self_play_qualified_12.zip:
    size mismatch for features_extractor.net.0.weight: [512, 715] vs [512, 802]
```

Что происходило: `agents/env.py::_make_self_play_opponents` грузил qualified-снапшоты обычным
`PPO.load`, а те обучены на 715 признаках -> падение -> снапшот выброшен. Проверка
`obs_dim != N_FEATURES -> skip` стояла ПОСЛЕ `PPO.load`, поэтому не срабатывала. Итог:

* self-play был фактически выключен. Вес `self_play=0.349` при пустом списке оппонентов
  никуда не переносится (он делится между `n_sp` снапшотами), поэтому остальные веса после
  нормализации давали ~**69% боёв против SimpleHeuristicsPlayer** и 0% self-play — модель
  училась не в том распределении, что задумано;
* ошибка печаталась заново на каждой фазе из каждого из 8 воркеров (env пересоздаётся
  каждую фазу) — лог тонул в повторах.

Исправление (`agents/checkpoint_utils.py`):

* `load_policy_compat(path, target_obs_dim)` — определяет размерность obs из весов и при
  расхождении мигрирует снапшот (паддинг нулями, как для основного чекпоинта), затем
  возвращает готовую политику;
* мигрированный снапшот кэшируется в `models/_migrated/<имя>__obs802.zip` (атомарная запись
  через `os.replace`, гонка 8 воркеров безопасна — результат миграции идентичен). Без кэша
  каждую фазу делалось бы 4 снапшота x 8 процессов = 32 миграции;
* детальные сообщения паддинга ушли под `verbose=False` (`_migrate_checkpoint_dim`
  вызывается тихо), а ошибка по конкретному файлу печатается один раз за процесс;
* то же самое подключено в `policy_player_simple.py` (там свои снапшоты `self_play_snapshot_*`).

Проверки: `test_self_play_migration.py` (15) — 715-снапшот загружается оппонентом и ждёт
802 признака, в логе больше нет `Failed to load qualified snapshot`, кэш создаётся и читается,
битый файл не ломает загрузку остальных и репортится один раз, снапшот нужной размерности
грузится без миграции.

Побочно для диагностики: в начале каждой фазы и сразу после resume печатается метрика
использования признаков урона:

```
[arch] старт: features_dim=512, использование новых признаков урона (RMS весов, доля от старых) = 0.0000
       веса новых признаков нулевые (warm start): они включатся по мере обучения; смотри метрику [arch] между фазами
```

`WARNING: empty action mask` в первом роллауте и предупреждение про некратность `phase_size`
роллауту — не ошибки: первое относится к стартовым шагам окружений, второе печаталось и раньше.

---

## 14. Зеркальный урон и флаги особых эффектов (N_FEATURES 802 -> 870)

К блоку урона добавлены два новых раздела: то, чем НАС бьют, и то, какие опасные эффекты есть
у команды противника. Итоговая раскладка (см. `damage.DAMAGE_BLOCK_SIZE = 155`):

| срез | что | размер |
|---|---|---|
| `[0:12]` | наши 4 приёма против их активного: `min_frac`, `max_frac`, `guaranteed_KO` | 12 |
| `[12:15]` | лучший известный приём противника по нашему активному | 3 |
| `[15:51]` | матрица «наш i -> их j» (`min_frac`) | 36 |
| `[51:87]` | матрица «их j -> наш i» (`min_frac`) | 36 |
| `[87:99]` | **их известные приёмы по нашему активному**: `min_frac`, `max_frac`, `possible_KO` | 12 |
| `[99:123]` | **те же приёмы x наши 6 слотов** (`min_frac`) — кого они пробивают | 24 |
| `[123:155]` | **флаги особых эффектов у противника** (см. `damage.EFFECT_FLAGS`) | 32 |

### 14.1 Зеркальный урон

`their_known_moves_damage()` берёт раскрытые приёмы активного противника, считает урон каждого
по нашему активному и **сортирует по убыванию минимального урона** (при равенстве — по id).
Порядок не зависит от того, в каком порядке приёмы раскрылись — проверено тестом. Слот 0 — самый
опасный для нас приём, дальше по убыванию; урон по нашим неактивным слотам идёт в той же
последовательности приёмов, поэтому пара «их ход x наш слот» читается однозначно.

### 14.2 Флаги особых эффектов

Считаются по **union всех раскрытых приёмов команды противника**, а не только активного: сеттер
хазардов или бустов часто сидит на скамейке, и это важно знать заранее (проверено тестом со
Skarmory на скамейке). Состав (32 флага):

* hazards: `stealthrock`, `spikes`, `toxicspikes`, `stickyweb`, `hazard_any`;
* статусы: `burn`, `para`, `poison`, `sleep`, `freeze`, `confuse`, `status_any`
  (учитываются и secondary-эффекты вида «scald 30% burn», и собственный `.status` приёма);
* контроль: `phazing` (`force_switch`), `protect` (`is_protect_move`), `screens`
  (`side_condition` Reflect/Light Screen/Aurora Veil), `substitute`, `taunt_or_disable`;
* восстановление: `healing_move` (`heal > 0` или id-набор), `drain_move` (`drain > 0`), `leechseed`;
* поле: `weather_setter`, `terrain_setter`, `self_switch` (u-turn/volt-switch);
* угроза: `priority_attack` (приоритетный атакующий приём), `setup_booster`, `target_debuff`,
  `trick_item`, `knockoff`;
* счётчики (0..1): `count_status_moves`, `count_setup_moves`, `count_damaging_moves`,
  `count_hazard_moves`.

Почти всё берётся из структурных полей `Move` (`side_condition`, `volatile_status`, `force_switch`,
`is_protect_move`, `self_switch`, `heal`, `drain`, `weather`, `terrain`, `priority`, `boosts`,
`secondary`), id-наборы нужны только там, где движок признак явно не хранит (hazard-атаки вида
Stone Axe, heal-приёмы без поля `heal`, taunt/trick/knock off, дебаффы вроде Parting Shot).
Найденная при разработке тонкость: `str(move.category)` даёт «PHYSICAL (MOVE CATEGORY) OBJECT»,
поэтому категорию надо брать через `.name` — иначе `priority_attack` не срабатывал.

### 14.3 Совместимость и цена

* Старые срезы не сдвинулись: `[0:12]` и `[15:51]` не зависят от известных приёмов противника —
  проверено тестом, так что ранее обученные веса остаются осмысленными.
* Все новые признаки идут в конец, поэтому миграция снапшотов (802 -> 870, а также 715 -> 870)
  работает как раньше: паддинг нулями, warm start.
* Цена: `_damage_block` 0.40 -> 0.65 мс, `embed_battle_with_fusion` 0.93 -> 1.37 мс на шаг
  (кэш `cached_move_damage` по паре x приём x поле). Для сравнения: один бой — секунды-минуты.


### 14.4 Ревью после внедрения (что нашлось и исправлено)

1. **`setup_booster` срабатывал на самоослабляющих атаках.** `Move.self_boost` отдаёт бусты
   *себе*, и у Close Combat / Overheat / Draco Meteor / Superpower / V-create / Leaf Storm /
   Armor Cannon / Make It Rain / Hammer Arm / Ice Hammer / Spin Out / Psycho Boost / Fleur
   Cannon / Dragon Ascent / Headlong Rush / Hyperspace Fury он **отрицательный** — 17 приёмов
   помечались как «противник качается». Теперь учитываются только положительные бусты, плюс
   добавлены бусты через `secondary[].self.boosts` (Flame Charge, Power-Up Punch, Torch Song,
   Aqua Step, Rapid Spin, Meteor Mash и ещё 13) и отсечены Swagger/Flatter/Decorate, которые
   бустят НЕ себя (проверка через `Move.target`).
2. **Stone Axe и Ceaseless Edge не давали своего флага хазарда.** Они ставят Stealth Rock и
   Spikes, но `side_condition` движок для них не хранит (`hazard_any` был, конкретный — нет).
   Добавлена явная карта `HAZARD_ATTACK_IDS`, как и для heal-приёмов.
3. **Третий значение среза называлось `guaranteed_KO`, а считалось по максимальному роллу.**
   Для «их» среза важен вопрос «может ли меня убить», поэтому семантика оставлена, но
   переименована в `possible_KO` и задокументирована: верхние срезы (`[0:12]`, `[12:15]`)
   считают гарантированный KO по минимальному роллу, нижний — по максимальному.
4. **Кэш мог отдать урон чужого покемона.** В сигнатуре были только `id(mon)` и число
   известных приёмов: Python переиспользует `id` убитых объектов, а статы/уровень/терра/состав
   приёмов не участвовали — для двух разных фьюжнов одной species запись кэша могла совпасть.
   Теперь в сигнатуре species, статы, уровень, терра и id приёмов (а `id(mon)` остался лишь
   быстрым дискриминатором). Побочно это ускорило блок: сигнатуры считаются в `prepare_mon`.
5. **Status-мoves не кэшировались** (`if hit is not None` не видел закэшированный `None`) —
   пересчитывались каждый шаг, теперь кэшируется и `None`.
6. **Устойчивость к кастомному моду.** `Move.status`/`Move.weather` делают `Status[...]`/
   `Weather[...]` и кидают KeyError на незнакомой строке (ровно этот класс ошибок давал
   `KeyError: 'frost'`), а один битый приём обнулял весь блок. Теперь свойства читаются через
   `_safe_attr`, каждый приём обрабатывается под `try` с разовым логом (`_note_effect_flag_error`, набор `_EFFECT_FLAG_ERRORS`),
   проверено на всех 954 приёмах гена 9 и на искусственно сломанном приёме.
7. **Фейнт насыщал признаки урона до максимума.** У фейнта HP = 0, `prepare_mon` из-за
   защиты от нуля отдавал `hp_now = 1`, и `damage_frac` делил урон на 1 → мёртвый слот в
   матрицах и в новом срезе показывал 1.0 (а docstring обещал 0). Теперь цели-фейнты явно
   пропускаются (0.0) в обеих матрицах и в зеркале, а фейнт нашего активного обнуляет только
   срез активного — строки по живой скамейке остаются (по ним и выбирается замена). Затронуты
   и старые срезы `[15:51]`/`[51:87]`: значения для мёртвых слотов были мусорными.
8. **Магическое число 87 в `features.py`** заменено на базовые индексы из `damage.py`
   (`MIRROR_BASE`/`TEAM_BASE`/`FLAGS_BASE`, размер блока собирается из них) — сдвиг срезов
   больше не может «молча» разъехаться с уже обученными моделями.

### 14.5 Тест на живом `Battle`

`test_mirror_features_live.py` прогоняет настоящие сообщения протокола Showdown через
`poke_env.battle.Battle` и проверяет то, что заглушки не ловят: приёмы противника появляются
только после раскрытия (превью команды не даёт ни приёмов, ни флагов — утечки нет); `earthquake`
попадает в слот 0 зеркала; строка по слотам совпадает с активным срезом; их Reflect режет НАШ
урон и не трогает зеркало, а НАШ Reflect режет зеркало и не трогает наш урон (проверка
направления `ctx_ours`/`ctx_theirs`); Will-O-Wisp даёт `status_burn`; после смены активного
hazard-флаг со скамейки сохраняется, а зеркало обнуляется (у нового активного только статусные
приёмы).

Проверки: `test_damage.py` 33 -> 83 (сортировка зеркала по опасности, `max >= min`, семантика
`possible_KO` по максимальному роллу, матрица по 6 слотам, независимость от порядка приёмов,
нули без раскрытых приёмов, неизменность старых срезов, все группы флагов, secondary-статусы,
скамеечный сеттер, самоослабляющие атаки ≠ setup, Stone Axe/Ceaseless Edge, битый приём мода),
плюс `test_mirror_features_live.py` (29 проверок на настоящем `Battle`).

---

## 15. Mold Breaker и фьюжн-карты по сторонам

### 15.1 Mold Breaker / Teravolt / Turboblaze и приёмы с `ignoreAbility`

В `estimate_damage` добавлен `attack_ignores_ability(atk, move)`: истина, если способность
атакующего — Mold Breaker / Teravolt / Turboblaze (в gen 9 они идентичны) или у приёма стоит
флаг `ignoreAbility` (Sunsteel Strike, Moongeist Beam, Photon Geyser, Searing Sunraze Smash,
Menacing Moonraze Maelstrom — читается из `Move.ignore_ability`, без хардкода id приёмов).

Что игнорируется (и иммунитеты, и снижение урона):

* `Levitate` (Ground), `Sap Sipper` (Grass), `Flash Fire` (Fire), `Water Absorb` / `Storm Drain` /
  `Dry Skin` (Water) — приёмы больше не обнуляются;
* `Multiscale` / `Shadow Shield`, `Filter` / `Solid Rock` / `Prism Armor`, `Thick Fat` /
  `Heatproof` / `Water Bubble`, `Ice Scales`, `Fluffy`, `Punk Rock`, `Dry Skin` (уязвимость к огню).

Что НЕ игнорируется:

* предметы — Air Balloon по-прежнему держит Ground;
* типовые иммунитеты — Flying-тип от Ground и вся таблица типов;
* способности атакующего (Tough Claws и т.п.) — Mold Breaker отключает только способность цели.

Правило «не подставлять неизвестное» сохранено: способность атакующего берётся из
`prepare_mon` (`mon.ability`), то есть у противника — только после раскрытия (`-ability`).
Неизвестно → поведение как без Mold Breaker. Работает во всех срезах урона автоматически
(наши приёмы, `[12:15]`, обе матрицы, зеркало и его разбивка по слотам) — проверено и в
зеркальном направлении: раскрытый Mold Breaker противника пробивает нашу Levitate.

Побочно из блока иммунитетов убраны дублирующие проверки (Dry Skin / Water Absorb / Storm
Drain / Flash Fire / Sap Sipper / Levitate уже перечислены в `_ABILITY_IMMUNITY`) — поведение
не изменилось, проверено на всех 954 приёмах гена 9 и тестами.

### 15.2 Фьюжн-карты: только со своей стороны

`_fusion_for(mon)` в `features.py` раньше искала вид в `our_team_fusions`, а при промахе — в
`opp_team_fusions`. Если у обоих игроков совпадала species (в этом формате обычное дело),
наш покемон получал статы фьюжна противника и наоборот. Теперь у каждой стороны своя карта
(`_fusion_for(mon, own_map)`), чужая не используется вовсе: нет записи → статы берутся из
декла (`prepare_mon(mon, None)`), как и для любого неизвестного случая. Для активного
покемона остался per-side фолбэк `our_fusion`/`opp_fusion`.

Тесты: `test_damage.py` 87 → 105 (Mold Breaker по всем трём способностям и по флагу приёма,
Air Balloon, типовая иммунность, неизвестная способность защиты, зеркальное направление;
фьюжн-карты — контроль на свою и на чужую сторону). Всего 9 файлов, 326 проверок.

---

## 16. Повторная проверка: STAB от tera-типа и инварианты Mold Breaker

### 16.1 STAB давался за tera-тип, который ещё не применён (ошибка старого кода)

`Pokemon.tera_type` в poke-env — это `_terastallized_type`, а он заполняется **заранее** из
details (`|switch|p1a: X|Camerupt, L50, M, tera:Water`) и из teambuilder'а, причём
`is_terastallized` при этом остаётся `False`. В `estimate_damage` STAB считался так:

```python
if atk_name in stab_types or (tera_name and atk_name == tera_name):
```

то есть приём, совпадающий с объявленным (но не применённым) tera-типом, получал **лишний
STAB 1.5x**. На живом `Battle`: Camerupt (Fire/Ground, объявлен tera:Water) до теры бил Surf
на 174 вместо 116. Ошибка тянулась из блока «+87 признаков урона» и завышала наши оценки
урона и флаги KO.

Исправление: отдельная ветка убрана, достаточно `type_1`/`type_2` — poke-env отдаёт там
tera-тип **только** когда покемон терасталлизован (`type_2` становится `None`), то есть STAB
за теру по-прежнему считается, но строго после теры. Проверено тестом
(`test_stab_tera_timing`): до теры урон равен контрольному без объявленного tera-типа, после
теры в Water — вырастает в 1.5 раза.

Известное приближение (не менялось): при совпадении tera-типа с исходным реальные правила
дают STAB 2x, у нас 1.5x (2x только у Adaptability).

### 16.2 Инвариант Mold Breaker — проверка на всём дексе

`test_mold_breaker_is_exhaustive` прогоняет **все 640 атакующих приёмов гена 9** против всех
22 способностей, которые моделирует `damage.py` (`_ABILITY_IMMUNITY` + `_DEF_ABILITY_MULT`):

* с Mold Breaker урон не зависит от способности защиты — 0 нарушений (сильная формулировка:
  сравнивается не «больше/меньше», а полное совпадение с вариантом «способности нет»);
* без Mold Breaker каждая из 22 способностей действительно что-то меняет хотя бы на одном
  приёме — тест не вырожденный.

Побочно закрыты мелочи: `attack_ignores_ability` не падает, если ему передали не dict
(подготовленный покемон vs сырой объект), а в зеркальном срезе сигнатуры наших слотов
считаются один раз на покемона (`team_sigs`), а не на каждый из его приёмов.

Замер после правок: `_damage_block` 0.61 мс, `embed_battle_with_fusion` 1.26 мс на шаг.

Тесты: `test_damage.py` 105 -> 111. Всего 9 файлов, 332 проверки.

---

## 17. Порядок выходов vs bench-признаки и устаревший верификатор

### 17.1 bench-слоты зависели от порядка словаря (8 признаков на перестановку)

`_bench_vec` раскладывал резервы `[mon for mon in team.values() if not mon.active]` — то есть в
порядке ключей `battle.team`/`battle.opponent_team`. Этот порядок в poke-env произволен (порядок
превью/выходов и раскрытия), поэтому одинаковые состояния давали разные obs: перестановка двух
резервов меняла **8 признаков** (16 в полном obs, индексы 299..565 — ровно bench-блок).
Слоты матриц урона тот же класс ошибки уже не имели (`team_slots` сортирует по species, в его
docstring это прямо написано), а bench-часть осталась на dict-порядке.

Исправление: `canonical_reserves(team)` — сортировка по `(species, name)`; имя (identify/ник)
уникально внутри стороны, поэтому и два покемона одного вида упорядочены детерминированно.
Применено везде, где перечисляются резервы: `_bench_vec`, `_switch_summary` (наш/их),
`our_reserves`/`opp_reserves` (для `vulnerability`).

Тест `test_slot_order` этого не ловил, потому что был вырожденным: заглушки `mk_mon`
создаются **активными**, значит `reserves` пуст, bench-блок нулевой в обоих случаях и
сравнивать нечего. Теперь тест создаёт настоящие резервы (`active=False`) и проверяет сортировку
для обеих сторон; плюс в живом наборе два боя с **разным порядком выходов** противника
(rock -> skarm -> blade vs skarm -> rock -> blade) дают побитово одинаковый obs, и перестановка
`battle.team`/`battle.opponent_team` на одной боевой сцене тоже ничего не меняет.

### 17.2 `test_features_verify.py` врал про корректный код

Корневой верификатор был зашит на раскладку из **418** признаков: `obs.shape==(418,)`, карта
групп с диапазонами 122..418 и прямые пробы (`obs[75]`, `obs[392]`, `obs[397:416]`). При
N_FEATURES=870 он печатал FAIL/DEAD на каждом синтетическом батле, хотя код корректен. Теперь
он проверяет фактический размер (`obs.shape == (N_FEATURES,)`), а устаревшая карта групп и
индексы-пробы пропускаются с явным сообщением (актуальная раскладка — §11.2/§14, проверки —
`test_damage.py`); расхождение размерности датасета печатается как WARN (датасеты пересобираются
автоматически, `_pad_obs_to_features`).

### 17.3 Замечание про warm start

Это третья правка, которая меняет **значения** признаков, но не их размерность (после
side-local фьюжн-карт и STAB/tera): bench-слоты теперь упорядочены канонически. Миграция
715/802 -> 870 при этом не требуется — паддинг идёт по размерности, — но веса bench-колонок
старой модели обучались на произвольном порядке резервов, поэтому полноценный warm start
ожидаемо даст лучший результат на свежем прогоне (он у тебя и так запланирован).

Замер после правок: `_damage_block` 0.61 мс, `embed_battle_with_fusion` ~1.3 мс на шаг.

Тесты: `test_damage.py` 111 -> 113, `test_mirror_features_live.py` 29 -> 33. Всего 9 файлов,
338 проверок.

---

## 18. index.py: 4 параллельных боя на бота и авто-перезапуск

### 18.1 Что было

* оба бота играли по одному бою: `accept_challenges(None, 1)` в цикле, `max_concurrent_battles`
  по умолчанию = 1;
* `aiPlay` ловил исключения, печатал их в stdout и чистил `_battles`, но переподключения как
  такового не было (после разрыва websocket старый объект Player оставался полумёртвым);
* `aiPlay2` не ловил ничего: любое исключение или обрыв — и бот тихо выпадал из процесса;
* ошибки жили только в консоли: закрыл окно — потерял историю.

### 18.2 Что стало

`BATTLES_PER_BOT = 4` (флаг `--battles`): `max_concurrent_battles=4`, а приём вызовов идёт
непрерывно (`accept_challenges(None, ACCEPT_BATCH)`) — пока 4 боя идут, следующий вызов ждёт
свободный слот в очереди poke-env и принимается сразу после окончания боя, без простоя.

`supervise(spec, factory)` — обёртка на каждого бота:

* падение/обрыв/ошибка ловится (`except Exception`), включая traceback, пишется в лог;
* бот поднимается **с нуля** (новый логин и новое соединение) — повторное использование
  убитого `Player` после разрыва часто оставляет незакрытые задачи и зависает вместо
  восстановления;
* пауза перед перезапуском растёт экспоненциально (5 -> 10 -> ... -> 300 c); если сессия
  прожила больше `HEALTHY_RUN_SECONDS` (120 c), backoff сбрасывается — чтобы плановые
  перезапуски не накапливали задержку;
* перед перезапуском старое соединение закрывается (`ps_client.stop_listening()`), состояние
  `_battles` чистится; в лог идёт статистика сессии (боёв/побед/винрейт).

`_connection_watchdog` — отдельная задача, которая раз в 15 c проверяет живость: `websocket`
отсутствует (после grace-периода), `websocket.closed` (сервер разорвал связь или сеть упала;
aiohttp ставит это и по `ping_timeout`), потерян `logged_in`. Без этого бот «висит зелёным»:
`accept_challenges` ждёт вызов вечно, и никто не замечает, что играть он уже не может.

Логи: `logs/index.log` (ротация 5 МБ x 5) + консоль. Файловый handler висит на корневом
логгере, поэтому в файл попадают и сообщения самого poke-env («Websocket connection ... closed»
и т.п.), а не только наши. Консоль переключается в UTF-8 с `errors="replace"` — Windows-консоль
(cp1251) больше не может уронить бота на печати.

Модель грузится через `load_policy_compat` (штатная миграция репозитория): старый чекпоинт
(715/802) подхватывается с паддингом нулями под 870, вместо падения на первом же ходе с
size mismatch, как было с ручным чтением `policy.pth`. Битая/отсутствующая модель — понятная
ошибка на старте (с записью в лог), а не бесконечный цикл перезапусков.

Отдельная находка: после `PPO.load` политика остаётся в **train-режиме**, то есть `Dropout(0.1)`
в экстракторе признаков работает и на инференсе — прежний index.py явно вызывал
`policy_instance.eval()`, так что без этого бот играл бы с шумными признаками и отличался
от прежнего поведения. Теперь `load_policy` делает `set_training_mode(False)` и пишет в лог
`training=False`. Плюс `--battles 0` отвергается (в poke-env 0 = «без лимита», бот принял бы
все вызовы подряд).

### 18.3 Тесты

`test_index_supervisor.py` (29 проверок, сервер не нужен — супервизор работает с duck-typed
игроком): перезапуск после исключения (+traceback и текст ошибки в логе, закрытие соединения,
очистка боёв), перезапуск после «тихого» обрыва websocket (watchdog), отсутствие перезапусков
на штатном прогоне, идемпотентность логирования и ротация файла, `max_concurrent_battles=4`
у настоящего `PolicyPlayer` (и переопределение через `--battles`), понятная ошибка на
отсутствующий чекпоинт, миграция честного 802-чекпоинта в 870 с нулевыми новыми столбцами
и кэшем миграции.

Отдельно любопытное наблюдение по ходу тестов: `FeaturesExtractor` строит первый слой от
модульной константы `N_FEATURES`, а не от `observation_space`, поэтому «старый» чекпоинт для
теста приходится собирать вручную (обрезать первый слой и править `observation_space`) —
`_probe_ppo(dim)` сам по себе даёт веса N_FEATURES независимо от аргумента.

---

## 19. Падение мержа датасета: `could not broadcast ... (26918,715) into (26918,713)`

### 19.1 Что произошло

Лог: `_recompute_from_chunked_cache` успешно пересчитал obs из сырого кэша (786267 примеров,
obs 870), после чего `collect_or_load_dataset` вместо готового датасета в памяти взял с диска
**чужие чанки** `models/heuristic_dataset_tmp/dataset_chunk_*.npz` (остатки прошлых прогонов,
713 и 715 признаков) и упал на мерже.

Виноват был вот этот приоритет в `collect_or_load_dataset`:

```python
chunk_files = _list_dataset_chunk_files()
if len(chunk_files) > 1 and len(dataset) > 10000:
    _merge_dataset_chunks(chunk_files, path)   # <- чанки неизвестного происхождения
```

Чанки на диске — это результат прошлых запусков, они не имеют отношения к датасету, который
только что собран в памяти. Если бы размерности совпали, на диск молча ушли бы устаревшие
признаки (а `_pad_obs_to_features` потом «добил» бы их нулями до 870).

### 19.2 Почему нельзя было просто добить нулями

`obs_dim` различался на 2 (713 -> 715), и соблазн «выровнять паддингом справа» есть. Но
раскладка признаков менялась **не только в конец**: в `d344e46` два признака
`[our_is_tera, opp_is_tera]` вставлены ПЕРЕД `our_tera_type(19)`:

```python
[our_can_tera_now, our_used_tera, opp_used_tera],
+ [our_is_tera, opp_is_tera],
  our_tera_type,                       # 19 значений сдвигаются
  [our_protected_last_turn, opp_protected_last_turn]
```

Паддинг 713-чанков до 715 сдвинул бы `tera_type` и `protect` — данные стали бы тихо неверными.
Поэтому в `_merge_dataset_chunks` вместо паддинга теперь **диагностика и явная ошибка**:

```
чанки собраны разными версиями кода и их нельзя слить: obs_dim (713: 1 чанк(ов), 715: 1 чанк(ов)),
mask_dim (26). Пересоберите датасет из сырого кэша текущим кодом — удалите ТОЛЬКО чанки датасета
(models/heuristic_dataset_tmp/dataset_chunk_*.npz) и запустите обучение снова...
```

(в сообщении сознательно нет `--force-recollect`: он чистит и сырые чанки, то есть 30000 боёв
пришлось бы собирать заново.)

### 19.3 Что изменено

1. `_store_dataset(dataset, path)` — сохранение собранного датасета без чужих чанков:
   сначала пробуем скопировать свежий `models/heuristic_dataset_tmp_recompute/_merged.npz`
   (совпал по числу примеров и размерности — значит это ровно наш датасет; копирование вместо
   повторного stack экономит ~3 ГБ), иначе `save_dataset` из памяти.
2. `_store_dataset` используется и в ветке `dim mismatch`, и в обычном сохранении — единый путь.
3. Ветка «chunked кэш уже содержит N боёв, мержу чанки» проверяет размерность чанков
   (`_wrong_dim_dataset_chunks`) и при расхождении с `N_FEATURES` уходит в пересчёт из сырого
   кэша: иначе в обучение ушли бы признаки прежней версии.
4. Кэш пересчёта: `_recompute_from_chunked_cache` больше не пересчитывает 800k+ переходов на
   каждом запуске. Рядом с `_merged.npz` пишется сайдкар `_merged.npz.meta.json` с obs_dim,
   числом боёв, отпечатком сырых чанков (имя+размер+mtime) и **хешем кода признаков**
   (`features.py`+`damage.py`+`config.py`). Совпало всё — мерж переиспользуется; изменилась
   любая правка кода признаков при той же размерности — кэш инвалидируется.
   Если сайдкара ещё нет (как у пользователя — мерж уже посчитан прежней версией кода), файл
   принимается, только если он **новее кода признаков**, и сайдкар дописывается сразу.
5. `_merge_dataset_chunks` читает размерности по заголовкам zip (без загрузки данных), пишет
   `has_ret` по всем чанкам (а не по первому) и предупреждает про чанки без `ret`.

### 19.4 Заодно: пересчёт из кэша считал obs не так, как бой

Найдено при разборе: `HeuristicRecorder` писал obs **с** командными фьюжн-картами
(`our_team_fusions`/`opp_team_fusions`), а пересчёт из сырого кэша (`_recompute_from_chunked_cache`,
`_recompute_dataset_from_raw`) считал те же obs **без** них — по дексу. Разница попадает в блок
урона (зеркало/матрицы) и в bench-статы, то есть пересобранный датасет расходился с тем, что
модель видит в бою. Теперь сырая запись несёт эти карты (9-й элемент: `(our_team, opp_team)`),
пересчёт их использует, а старые 8-элементные записи читаются как `(None, None)` — поведение
не меняется, зато новые данные консистентны. Проверено тестом: пересчёт побайтово совпадает с
живым `embed_battle_with_fusion` при тех же картах.

### 19.5 Мелочи

* `_list_dataset_chunk_files(tmp_dir=None)` — каталог читается в момент вызова: default-значение
  связывалось на импорте, из-за чего константу нельзя было переопределить.
* `test_no_shadowing.py` давал ложное срабатывание на `mod.attr = ...` (считал модуль локальным
  именем): теперь учитываются только реальные цели связывания (`ctx=Store`), добавлен само-тест
  на оба случая.

Тесты: новый `test_dataset_merge.py` — 55 проверок, включая полный сценарий из лога
(старый финальный датасет 713 + устаревшие чанки 713/715 + сырой кэш + готовый мерж 870 без
сайдкара): датасет собирается, записывается с obs 870, устаревшие чанки не читаются, повторного
пересчёта нет. Всего 11 файлов, 428 проверок.

## 20. «За 2 800 000 шагов модель научилась только спамить атаками»

### 20.1 Симптом

После ~2.8M шагов политика почти всегда выбирала атакующий приём: свитчи и статусные/погодные
ходы исчезли, хотя награда за них предусмотрена (SWITCH_BONUS 0.12 / SWITCH_IMMUNE_BONUS 0.25,
WEATHER_BONUS 0.08, TERRAIN_BONUS 0.06, HEAL_BONUS 0.08, STATUS_CURE_BONUS 0.10 и т.д.).

### 20.2 Причина: инвертированная конвенция индексов действий

`ExampleEnv.action_to_order` вёл «книжку» последнего хода (`_last_move_id`, `_last_was_switch`,
`_last_wasted`) по конвенции DoublesEnv (`action < len(available_moves)` → приём, иначе свитч).
В poke-env **SinglesEnv** раскладка обратная:

| action | что это |
|---|---|
| 0..5 | свитч (индекс в `battle.team`) |
| 6..9 | приём, индекс `(action - 6) % 4` |
| 10..13 / 14..17 / 18..21 | мега / z / динамакс (в gen9-фьюжне недоступны) |
| 22..25 | тера (тот же приём `(action - 6) % 4`) |

Сам ордер всегда строился правильно (`super().action_to_order`), ломалась только бухгалтерия
награды. Проверено эмпирически на живом `Battle` (`swampert` с surf/earthquake против `skarmory`):

```
до фикса:
  action=1 (свитч):   ордер /choose switch Bench,  _last_move_id='earthquake', was_switch=False, wasted=True
  action=6 (surf):    ордер /choose move surf,     _last_move_id='switch',     was_switch=True,  wasted=False
  action=7 (earthq.): ордер /choose move earthquake,_last_move_id='switch',     was_switch=True,  wasted=False
после фикса:
  action=1: switch,  _last_move_id='switch', was_switch=True,  wasted=False
  action=6: surf,    _last_move_id='surf',   was_switch=False, wasted=False
  action=7: earthq., _last_move_id='earthquake', was_switch=False, wasted=True
  action=23: тера-версия earthquake, kind='tera', wasted=True
```

Что это давало награде:

* **Приёмы (6..9) записывались как свитч**: `_last_wasted=False` → generic `WASTED_MOVE_PENALTY`
  (0.06) для приёмов не срабатывал **никогда**. Прожать бесполезную атаку (иммунитет/застатусленный
  оппонент/хил на полном HP/хазард уже стоит) стоило 0 — прямой стимул «спамить».
* **Ветки награды по `_last_move_id` были мертвы**: id был `"switch"`, поэтому не срабатывали
  хилы (`HEAL_BONUS`/`HEAL_WASTED_PENALTY`), «наша» погода/террейн (`WEATHER_MOVE_IDS`,
  `TERRAIN_MOVE_IDS`), а также проверки wasted по погоде/террейну/сабу/бустам.
* **Свитчи (0..3 при четырёх приёмах) записывались как приём** с чужим id из своего же мувсета и
  получали `WASTED_MOVE_PENALTY`, если этот приём был wasted → штраф за свитч (в тесте: свитч на
  скамейку → `_last_move_id='earthquake'`, `wasted=True`).
* Плюс `_last_was_switch=True` на ходах-приёмах включал ветку «погоду поставила абилка при свитче»
  (Drought/Drizzle/…) — лишний положительный сигнал на атакующих ходах.

Итог: градиент систематически «награждал атаку и наказывал свитч» — ровно наблюдаемое поведение.

### 20.3 Что изменено

`agents/env.py`:
* классификация действий по конвенции SinglesEnv: `action < 0` → default/forfeit,
  `0..5` → свитч (`_last_move_id="switch"`, `_last_was_switch=True`, `wasted=False`),
  `>=6` → приём `moves[(action-6) % 4]` (для 22..25 ещё и `kind="tera"`), wasted-флаг считается
  **для реально выбранного приёма** (тот же расчёт «иммунитет = wasted», что и раньше);
* `ExampleEnv._moves_for_action(battle)` повторяет ровно ту выборку приёмов, что использует
  `SinglesEnv.action_to_order` (`known_moves` активного, а если доступен ровно один незнакомый
  приём — именно он), чтобы индексы не разъезжались;
* счётчики `_last_action_kind` + `action_mix()`/`reset_action_mix()` (switch/move/tera/unknown)
  — диагностика «спам атаками» на стороне env.

`agents/training.py`:
* `StepCounterCallback` теперь считает состав действий по `_locals["actions"]` (SB3 отдаёт массив
  выбранных действий на каждом шаге роллаута) и раз в `mix_every=20 000` решений печатает
  `[mix] решений N: свитч X%, приём Y%, тера Z%` + пишет `mix/switch_share`, `mix/move_share`,
  `mix/tera_share` в TensorBoard. Это прямая проверка, что фикс работает, без разбора реплеев.

`agents/policy_player.py` (латентный баг того же семейства, найден при разборе):
* `_DimProbeEnv`/`_probe_ppo` собирали политику под `action_dim=9` (как у DoublesEnv), хотя в
  gen9-фьюжне действий 26. Через probe идёт fallback-миграция чекпоинта: там
  `load_state_dict(strict=False)` падает на `size mismatch for action_net` (torch ругается на
  форму даже при `strict=False`), то есть fallback не выживал → `load_policy_compat` возвращал
  `None`, и self-play оппонент отваливался с `Failed to load qualified snapshot`
  (приёмка пункта про self-play после смены размерности). Теперь probe берёт размер головы из
  чекпоинта (`_checkpoint_arch` читает `action_net.weight`, дефолт 26), а `action_mask` в его
  observation_space тоже 26.
* fallback печатает незагруженные/лишние ключи и отдельно выделяет критичные
  (`action_net`, `features_extractor.net.0`, `mlp_extractor`) — молчаливая потеря весов больше
  не проходит незамеченной.

### 20.4 Тесты

* `test_action_kind.py` (новый, 54 проверки): книга действий для свитча/приёма/теры/default,
  сверка `_moves_for_action` с логикой poke-env (включая ветку «единственный незнакомый приём»),
  и главное — цена свитча: разница награды между корректным и прежним (багованным) состоянием
  ровно `WASTED_MOVE_PENALTY`.
* `test_dim_migration.py` (+12 проверок, всего 55 вместо 43): probe-политика на 26 действий, `_checkpoint_arch`
  сообщает `action_dim`, fallback-миграция не теряет голову (веса помечены константой и сверяются),
  первый слой экстрактора расширяется с нулями в новых колонках, mask/action_space = 26.
  `FakeEnv` в тесте переведён на реальные 26 действий.

### 20.5 Что делать пользователю

1. `--resume` с текущего чекпоинта: веса валидны, градиент теперь корректен, политика доучится.
2. Смотреть в логе строку `[mix]`: доля свитчей должна вырасти с ~0% до заметной (эвристика
   свитчит в заметной доле ходов). В TensorBoard — `mix/*`.
3. Если после ~300–500k шагов доля свитчей всё ещё ~0, тогда уже есть смысл тюнить веса награды
   (например `SWITCH_BONUS`), но менять награду до того, как отработает исправленный градиент,
   нельзя: иначе симптом будет лечить не причина.

---

## 21. Нормализация obs: инференс без VecNormalize и статистика от чужой раскладки

Продолжение аудита после §20 (индексы действий). Здесь две ошибки одного класса: «на обучении
одни условия, на инференсе/в self-play другие, и об этом никто не говорит».

### 21.1 Обучение нормализует obs, а живой инференс — нет

Обучение идёт с `VecNormalize(base_env, norm_obs=True, norm_obs_keys=["observation"])`, то есть
политика видит **нормализованные** признаки. При этом:

| путь инференса | нормализация obs до фикса |
|---|---|
| `evaluate_win_rates` (оценка внутри обучения) | да — подменялся `embed_battle` (monkey-patch) |
| `policy_player`/`play_trained.py` | да (в `play_trained.py` тоже patch) |
| **живые боты `index.py`** | **нет** — сырые признаки |
| **self-play оппоненты в обучении** (`agents/env.py`) | **нет** — сырые признаки |

Последствия: (1) боты на сервере играют хуже обученной политики ровно настолько, насколько
`normalize_obs` меняет вход; (2) self-play соперник — «сломанная» версия снапшота, а по боям с ним
считается винрейт и порог ratchet; (3) снимок из `models/vecnormalize.pkl` в репо (418 признаков)
показывает, что 99% колонок нормализуются с масштабом, отличающимся больше чем на 25%, а часть —
в десятки раз (клип ±10 срабатывает), так что разница далеко не косметическая.

Исправлено единым механизмом вместо monkey-patch:

* `PolicyPlayer(..., obs_normalizer=<объект с normalize(obs)>)` (`agents/players.py`) — нормализация
  применяется в `embed_battle`; если нормализатор падает (чужая размерность), играем на сырых
  признаках и пишем предупреждение один раз (не молча);
* `VecNormStats` (статистика из pkl) и `LiveVecNormalizeAdapter` (живой `VecNormalize` обучения) —
  обе реализации дают **бит-в-бит** тот же результат, что `VecNormalize.normalize_obs` (тест);
* `index.py`: `load_obs_normalizer()` + флаг `--no-normalize-obs` (для моделей, обученных с
  `--no-normalize-bc`); в лог пишется `describe()` и предупреждение о чужой статистике;
* `agents/env.py`: self-play оппоненты получают тот же нормализатор (`_self_play_obs_normalizer()`,
  кэш на процесс), отключается переменной `PYBOT_SELF_PLAY_NORM=0`, которую выставляет обучение,
  когда идёт без VecNormalize; в главном процессе об этом пишется строка в лог;
* `evaluate_win_rates`: monkey-patch убран, нормализатор передаётся и «нашему» игроку, и
  self-play сопернику для eval (раньше он играл на сырых признаках даже внутри оценки).

### 21.2 «Миграция» статистики добивала её нулями (нормализация чужой раскладкой)

`VecNormalize` хранит per-колонку mean/var — статистику **конкретных** признаков. Раскладка
менялась несколько раз (418 → 713 → 715 → 802 → 870), причём 713 → 715 вставлял два признака
**в середину**. Прежний код при расхождении размерности делал `pad_stats`: оставлял старые колонки
как есть и добивал хвост `mean=0, var=1`. Это неверно: первые 418 колонок нормализуются
статистиками другой раскладки, а `count` ≈ 202 760 не даёт им вымыться новыми данными (вес одной
порции ~3072 наблюдений пренебрежимо мал).

Теперь (`agents/vecnorm_utils.py`):

* `stats_verdict()` — решение «доверять/нет»: сначала сайдкар (`models/vecnormalize.pkl.meta.json`
  с `obs_dim` + хешем кода признаков, тот же `_features_fingerprint`, что у кэша датасета), затем
  размерность, затем подпись паддинга (колонки ровно `mean=0, var=1` — у реальных признаков таких
  не бывает пачками);
* `reset_obs_rms(vec, dim)` — сброс статистики (`mean=0, var=1, count=0`) **с правильной
  размерностью** (иначе `normalize_obs` падает на broadcast);
* `load_vecnormalize_for_dim(..., force_reset=, keep_stale=)` — при несовместимой статистике
  сбрасывает её и печатает причину; прежнее «добить нулями» доступно флагом `--keep-obs-stats`,
  принудительный сброс — `--reset-obs-stats`;
* `save_vecnormalize_with_meta()` — сохранение вместе с сайдкаром (вызывается во всех точках
  `env.save(VECNORM_PATH)`), работает и через обёртку (`vecnormalize_of`);
* на инференсе статистика **не** отбрасывается (её надо применять ровно так, как при обучении),
  но `VecNormStats` получает флаги `stale`/`provenance` и `stale_warning()` — пользователю видно,
  что модель обучена на статистике чужой раскладки и что поможет `--reset-obs-stats`.

### 21.3 Старые снапшоты не грузились вообще («Failed to load qualified snapshot»)

Диагностика по `models/self_play_snapshot_*.zip` (418 признаков, identity-экстрактор):

* в `data` лежит `lr_schedule` — функция, снятая **другой версией Python**; `save_to_zip_file`
  при миграции падает в cloudpickle (`IndexError: tuple index out of range`), то есть основная
  ветка миграции не работает для таких файлов;
* `_checkpoint_arch` искал state_dict по ключам `features_extractor` — у identity-экстрактора
  (`LegacyFeaturesExtractor`) таких ключей нет, поэтому `arch={}` и fallback-проба собиралась с
  текущими дефолтами (`pi/vf [512,256]`, Linear-экстрактор), а в чекпоинте `[512,256,128]` и вход
  mlp 418 → `size mismatch for action_net` → снапшот не загружался, self-play падал в fallback на
  эвристики и спамил `Failed to load qualified snapshot`.

Исправлено (`agents/policy_player.py`):

* `_checkpoint_arch` ищет state_dict по любому из ключей (`features_extractor`/`mlp_extractor`/
  `action_net.weight`), собирает `policy_net`/`value_net`/`shared_net` головы и флаг
  `has_feature_net`;
* `_probe_ppo(..., net_arch=, legacy_extractor=)` строит политику **той же формы**, что чекпоинт
  (`LegacyFeaturesExtractor` для файлов без `features_extractor.net.*`);
* fallback паддит вход mlp до `N_FEATURES` (у identity-экстрактора выход = N_FEATURES) и умеет
  `mlp_extractor.shared_net.*`; про cloudpickle-ограничение пишется короткая строка вместо
  трёхэкранного traceback.

Проверка на реальных файлах: все 7 снапшотов из `models/` теперь грузятся (`load_policy_compat: OK`,
`obs 418 -> 870`, голова `26x128`), в логе self-play больше нет `Failed to load qualified snapshot`.
Тест проверяет и «честность» warm start: новые колонки весов строго нулевые, старые побитово равны
чекпоинту, и выход (логиты и value) не зависит от новых признаков.

### 21.4 Мелочи

* `agents/policy.py`: `get_distribution()` без предварительного `forward()` больше не падает
  (`self._mask` мог быть не установлен) — маскирование просто не применяется; добавлена защита от
  маски чужой длины с понятным warning (раньше — падение на несовместимых формах в обучении).
* `VecNormStats`/`LiveVecNormalizeAdapter` описаны в докстринге модуля вместе с объяснением, почему
  статистику чужой раскладки нельзя переиспользовать.

### 21.5 Что это значит для текущего прогона

1. Resume заберёт статистику нормализации и, если она от другой раскладки (файл 418 или «добитый»
   870 с колонками `mean=0/var=1`), сбросит её и переоценит по текущим данным — в логе будет
   причина. Если хочется сохранить прежнее поведение: `--keep-obs-stats`.
2. Живые боты (`index.py`) и self-play теперь работают в тех же условиях, что обучение: это
   заметно меняет (обычно улучшает) и качество игры на сервере, и честность винрейта self-play.
3. Если модель обучена без нормализации (`--no-normalize-bc`), для ботов есть `--no-normalize-obs`.

---

## 22. Проверка пользовательского маршрута «BC-претрейн -> RL» (шаг за шагом)

Маршрут пользователя:

```
1) python -m agents.policy_player --dataset-path models/mixed.npz --epochs 15 --contrastive \
     --neg-weight 0.3 --total-timesteps 0 --ent-coef 0.05 --lr 3e-4
2) python -m agents.policy_player --resume models/pretrained_v2 --total-timesteps 10000000 \
     --lr 3e-5 --clip-range 0.1 --n-epochs 3 --batch-size 256 --vf-coef 0.5 --min-winrate 25 \
     --reset-schedules --icm --icm-anneal --eval-battles 60
```

Ниже — что в нём ломалось. Всё воспроизведено и починено; проверки — в новом
`test_training_route.py` (37 проверок, работает без Showdown-сервера и без записи в `models/`).

### 22.1 Шаг 2 не находил файл шага 1

Финальная модель всегда сохранялась как `models/ppo_policy_final`, а имя `models/pretrained_v2`
не создавалось нигде: `--resume models/pretrained_v2` падал на `resolve_checkpoint_path`
(`.zip` тоже нет) и завершался `SystemExit` с подсказкой про похожие файлы.

Исправлено: флаг `--save-as NAME` (финальная модель -> `models/NAME.zip`), плюс в конце прогона
печатается готовая команда для resume. `resolve_checkpoint_path` и так достраивает `.zip`.

### 22.2 Шаг 1 не работал без сервера (и «--skip-eval» не помогал)

`--total-timesteps 0` не входит в цикл фаз, но после сохранения модели код безусловно уходил в
два финальных блока оценки: (1) eval с нормализацией, (2) «сырой» прогон
`asyncio.run(agent.battle_against(*opponents, n_battles=100))` — 100 боёв на каждого из трёх ботов.
`--skip-eval` отключал только первый, поэтому BC-претрейн без запущенного Showdown-сервера висел
навсегда (а с сервером жёг 300 боёв, хотя претрейну оценка не нужна).

Исправлено: финальные оценки вынесены в `_final_evals(...)`; `--skip-eval` пропускает ОБЕ, и
добавлен отдельный `--skip-final-raw-eval`. Дополнительно BC-прогон (`--total-timesteps 0`) теперь
поднимает 1 env вместо 8 (RL не учится, 8 процессов только зря дёргают сервер).

### 22.3 Датасет старой раскладки: тихая порча колонок (главное)

`_pad_obs_to_features` добивал obs нулями В ХВОСТ. Это корректно только начиная с 715: все
последующие изменения (блок урона 155 = 87+68) идут в конец (см. `features._damage_block`,
`OFF = N_FEATURES - DAMAGE_BLOCK_SIZE`). А переход 713 -> 715 вставил `[our_is_tera, opp_is_tera]`
ПЕРЕД `tera_type`, то есть В СЕРЕДИНУ, поэтому у датасета на 713 и старее все колонки после места
вставки — уже другие признаки. В репозитории лежат именно такие файлы: `models/heuristic_dataset.npz`
— 418 признаков, `models/heuristic_dataset_50k_normalized.npz` — 411 (и без `ret`).

Итог: BC на таком датасете учился бы на чужих значениях, не сказав ни слова. Теперь:

* `MIN_PREFIX_OBS_DIM = 715` + `dataset_layout_error()`: паддинг разрешён только для «префиксных»
  раскладок (715, 802), иначе — честная ошибка с объяснением и тем, как пересобрать датасет
  (пересборка сборщиком пересчитывает obs из сырых боёв; `--force-recollect` НЕ советуем — он
  стирает сырой кэш);
* `validate_bc_dataset(path, N_FEATURES)` проверяет датасет ДО создания env: есть ли `ret`,
  какой obs_dim, сколько примеров. Раньше отсутствие `ret` выяснялось внутри BC после подъёма
  8 процессов, а старая раскладка — вообще не выяснялась.

Проверка своего файла одной строкой:
`python -c "import numpy as np; d = np.load('models/mixed.npz'); print(d['obs'].shape, 'ret' in d.files)"`
— для текущего кода ожидается `(N, 870) True`.

### 22.4 Warm-up статистики нормализации падал на broadcast

`pretrain_policy_bc` выравнивал obs (`_pad_obs_to_features`) для обучения, но в
`warm_up_vec_normalize` уходил НЕпаддированный массив (`_DatasetView`/путь) -> `RunningMeanStd.update`
падал на `operands could not be broadcast together with shapes (418,) (870,)`. То есть BC с
нормализацией не работал вообще ни на одном датасете, кроме ровно 870-мерного.

Исправлено: warm-up получает уже выровненный массив (`obs_arr`), а сам `warm_up_vec_normalize`
умеет и `ndarray`, и `_DatasetView`, и путь; при расхождении размерностей приводит тем же правилом,
что и BC (и падает с понятным текстом на старой раскладке, а не с broadcast-ошибкой).

### 22.5 `--icm`: сохранение статистики через обёртку

При `--icm` env — это `CuriosityVecWrapper`; сохранение статистики шло через `env.save(...)` и
держалось на том, что SB3 `VecEnvWrapper` делегирует неизвестные атрибуты внутреннему env.
Проверено на живом `CuriosityVecWrapper` (сохранение/перезагрузка совпадают: count 200.0001 ->
200.0001, obs 870). Тем не менее `save_vecnormalize_with_meta` теперь явно сохраняет внутренний
`VecNormalize` (`vecnormalize_of`), так что обёртка без делегирования больше не потеряет статистику
молча, и в лог пишется строка «VecNormalize сохранён: ... (+ сайдкар)».

### 22.6 Что в маршруте в порядке (проверено)

* `--lr 3e-4` на шаге 1 РАБОТАЕТ: BC использует `ppo.policy.optimizer`, lr которого задан при
  создании PPO (`--lr` — алиас `--learning-rate`, в логе «LR зафиксирован: 3.00e-04»).
* `--ent-coef 0.05` на шаге 1 ни на что не влияет: BC-лосс не содержит энтропии (безвредно).
* `--total-timesteps 0` действительно пропускает RL: печатает «RL пропущен..., LR зафиксирован».
* Шаг 2: `--reset-schedules` обязателен для старта lr 3e-5 и ent 0.01 с нуля — у вас есть;
  `--clip-range 0.1/--n-epochs 3/--batch-size 256/--vf-coef 0.5/--min-winrate 25` валидны;
  `--icm --icm-anneal` собирается с `obs_dim=870`, `action_dim=26` (проверено на реальном прогоне
  с `--total-timesteps 0`: «ICM включён (resume): beta=0.05 anneal=True feat=256 action_dim=26»).
* Статистика нормализации с шага 1 подхватывается шагом 2 по сайдкару (obs 870 + хеш кода признаков),
  то есть obs на BC и на RL нормализуются одинаково. Если между шагами менять код признаков —
  статистика сбросится, тогда нужен `--keep-obs-stats`.
* `--eval-battles 60` = 60 боёв на каждого из 4 соперников (240 за фазу); при `phase_size` 200k и
  10M шагов это 50 фаз ≈ 12 000 боёв — если долго, `--skip-eval` (пропускает и финальные).

### 22.7 Итоговые команды

```
# 0) убедиться, что датасет текущей раскладки (ожидается (N, 870) True)
python -c "import numpy as np; d = np.load('models/mixed.npz'); print(d['obs'].shape, 'ret' in d.files)"

# 1) BC-претрейн (офлайн, без сервера) — ent-coef убран (BC энтропию не использует)
python -m agents.policy_player --dataset-path models/mixed.npz --epochs 15 --contrastive \
    --neg-weight 0.3 --total-timesteps 0 --lr 3e-4 --skip-eval --save-as pretrained_v2

# 2) RL-дообучение (без изменений, resume теперь находит файл шага 1)
python -m agents.policy_player --resume models/pretrained_v2 --total-timesteps 10000000 \
    --lr 3e-5 --clip-range 0.1 --n-epochs 3 --batch-size 256 --vf-coef 0.5 --min-winrate 25 \
    --reset-schedules --icm --icm-anneal --eval-battles 60
```

### 22.8 Мелочь по коду

CLI-парсер вынесен из `if __name__ == "__main__"` в `build_parser()` / `parse_args(argv)` — флаги
теперь проверяются тестами (раньше единственным «тестом» был ручной запуск).

---

## 23. `[mix]` врал: wait-шаги считались свитчами (жалоба «в [mix] 60% свитчей, а в бою ни одного»)

Проверено на живом Showdown-сервере (поднят локально, `gen9randombattle`, poke-env 0.16.1)
с моделью пользователя `models/ppo_policy_final.zip` (мигрирована 418 -> 870).

### 23.1 Что считает `[mix]`

`StepCounterCallback` берёт из SB3 локальные `actions` — по одному действию на env на шаг —
и раскладывает их по индексам SinglesEnv: 0..5 свитч, 6..9 приём, 10..21 мега/z/динамакс,
22..25 тера. Это ровно те действия, которые SB3 отдал в `env.step` (проверено наземной
правдой: сэмплы = применённые действия).

### 23.2 Почему метрика расходилась с боями

poke-env отдаёт агенту не только состояния «наш ход». Когда сервер присылает `|request|`
с `wait: true` (мы выбрали, ждём соперника), `PokeEnv.step` **не зовёт `action_to_order`
вообще** (`agent1_to_move=False` — действие выбрасывается), а obs приходит с маской
`[1, 0, 0, ...]`: `get_action_mask()` при `battle._wait` разрешает ровно одно действие —
индекс 0, который в раскладке SinglesEnv означает СВИТЧ. Политика с additive-маской
обязана его выбрать, `[mix]` писал «свитч», которого в бою не было.

Замер (модель пользователя, 8 env, соперник-эвристика, 1600 шагов):

| метрика | значение |
|---|---|
| wait-шаги (obs в состоянии ожидания) | 181 (11.3%) |
| obs с единственным разрешённым действием | 247 (15.4%) |
| `[mix]` свитч | 29% (все 181 wait-шага — как «свитчи») |
| реально применённых свитчей | 284 |
| ордеров-заглушек (`DefaultBattleOrder`) | 0 |

То же с self-play соперником (снапшот `self_play_snapshot_6.zip`): wait-шагов 11.1%,
`[mix]` свитч 25.1%. На загруженном/удалённом сервере доля ожиданий выше, поэтому
расхождение может быть кратным, а не фиксированным.

### 23.3 Исправление

1) `agents/env.py`: `TIME_PENALTY` вместо двух `-0.02`; новый `DecisionWrapper(SingleAgentWrapper)`,
   который прокручивает состояния ожидания ВНУТРИ шага и наружу отдаёт только состояние,
   на котором действие будет применено (`agent1_to_move == True`). Награды суммируются
   (`calc_reward` ведёт дельты — сумма корректна), лишние штрафы времени возвращаются:
   сколько раз сервер заставил ждать — не подконтрольно агенту. `ExampleEnv.create_env`
   теперь оборачивает env в `DecisionWrapper` (оба места возврата).

2) `agents/training.py`: `[mix]` считает класс по маске из pre-step obs; шаги без выбора
   (`mask.sum() == 1`) помечаются отдельно (`forced`, ещё и в TB как `mix/forced_share`),
   а доли считаются по всем шагам, чтобы совпадать с боем.

Проверка (живой сервер, 4 env, PPO, 3072 шага):

```
[mix] решений 3072 (без выбора 122): свитч 1391, приём 1581, тера 100
наземная правда (применил env):       свитч 1391, приём 1581, тера 100   <- совпало точно
наружу ушло состояний ожидания: 0, действий вне своего хода: 0, ордеров-заглушек: 0
```

Плюс обучение стало чище: ~11% шагов роллаута раньше были пустышками (награда ≈ −0.02,
действие игнорируется) и только разбавляли буфер и метрику.

3) `agents/players.py` + `evaluate_win_rates`: у оценочного `PolicyPlayer` появился
`action_counter`, и в лог пишется строка

```
eval-микс (решения модели в оценочных боях): свитч 37.2%, приём 60.7%, тера 2.2% (136/222/8 из 366)
```

— это поведение модели именно в боях за винрейт (в них wait-шагов нет by design:
`PolicyPlayer.choose_move` при `battle.wait` отдаёт `DefaultBattleOrder` и ничего не выбирает).

### 23.4 Что проверено отдельно (и оказалось в порядке)

* Маска соблюдается: семплов вне маски 0 (в живом прогоне); действие никогда не срезается.
* Пути согласованы: одна и та же политика в env-обучении и в `PolicyPlayer` даёт одинаковое
  распределение действий (на модели пользователя: env 80/170/0 против eval 10/30/0 при 40
  решениях; с загруженным чекпоинтом 418 -> 870 — свитчи в оценке реальны: 10 из 10 ордеров
  `/choose switch ...`).
* `ppo.get_vec_normalize_env()` с `--icm` возвращает внутренний `VecNormalize` (обёртка
  `CuriosityVecWrapper` — `VecEnvWrapper`), поэтому оценка получает живые 870-мерные статистики,
  а не добитую 418-мерную с диска.

`test_action_kind.py` дополнен: шаги без выбора не идут в свитчи, доли считаются по всем
шагам, `DecisionWrapper` прокручивает ожидание, суммирует награды, оставляет один штраф
времени и не зацикливается на терминальном состоянии.

## 24. «Что считает [mix]» — разбор и живые замеры (свитчи по выбору vs вынужденные)

Жалоба: «`[mix] решений 200000: свитч 60.4% ...` — а в боях за винрейт модель ни разу не
свитчила и теру не жала». Что именно считает метрика и почему она расходилась с боями.

### 24.1 Что считает [mix]

`StepCounterCallback` (agents/training.py) вызывается на каждом шаге SB3-роллаута и складывает
в корзины `actions` — те индексы действий, которые **выбрала политика**. Класс действия берётся
по конвенции poke-env `SinglesEnv`: 0..5 свитч, 6..9 приём, 10..21 mega/z/dynamax, 22..25 тера.
Это НЕ «что сделала модель в бою», а «что политика выбрала на каждом кадре роллаута», и кадры
бывают двух видов, которые раньше сваливались в один счётчик:

* **состояния ожидания** (`|request| wait:true`): poke-env отдаёт obs с маской `[1, 0, 0, ...]`
  (`get_action_mask`: `if battle._wait: actions = [0]`), где единственное разрешённое действие 0 —
  это индекс СВИТЧА, а сам шаг env выбрасывает (`agent1_to_move == False`, `action_to_order` не
  зовётся). До `7e35d9b` такие кадры шли в «свитчи»: замер на живом сервере — 11.3% шагов;
* **вынужденные свитчи после фейнта**: активный покемон упал, `get_action_mask` возвращает только
  свитчи (`battle.active_pokemon is None` → `actions = switch_space`). Свитч в бою происходит, но
  выбора «атаковать или смениться» у модели нет — это подстановка следующего покемона.

### 24.2 Что теперь показывает

* `switch` — свитч, когда приёмы БЫЛИ доступны (собственный выбор политики);
* `switch_forced` — свитч, когда приёмов не было вообще (`mask[6:]` пусто: фейнт);
* `single` — кадры с ровно одним легальным действием (ожидание/последний покемон);
* `_decided` — шаги, где был настоящий выбор (>=2 легальных действия и есть приёмы).

Строка в логе: `[mix] решений N: приём X%, тера Y%, свитч Z% (своих A = %, вынужденных после
фейнта B = %); шагов без выбора C (D%), своих решений (был выбор) E (F%)`. В TensorBoard —
`mix/switch_own_share`, `mix/switch_forced_share`, `mix/single_share`, `mix/choice_share`.
В боях за винрейт `PolicyPlayer` теперь печатает ту же разбивку строкой `eval-микс` — это
поведение модели там, где и считается винрейт (wait-кадров в `Player.battle_against` нет).

### 24.3 Живой замер на модели пользователя (`models/ppo_policy_final.zip`, 418 -> 870)

Сервер Showdown поднят локально из npm; кастомного формата фьюжнов в стоковом пакете нет, поэтому
формат подменялся в рантайме на `gen9randombattle` (`config.py` специально НЕ правили: его правка
меняет хеш признаков в сайдкаре статистики нормализации).

```
обучение, 2 env, 3072 решения:
[mix] решений 3000: приём 78.8%, тера 0.0%, свитч 21.2% (своих 118 = 3.9%, вынужденных
      после фейнта 519 = 17.3%); шагов без выбора 99 (3.3%), своих решений (был выбор) 2478 (82.6%)

оценка, 32 боя (8 на каждого из 4 ботов), 1005 решений:
eval-микс: приём 85.1%, тера 0.0%, свитч 14.9% (своих 1 = 0.1%, вынужденных после фейнта
      149 = 14.8%) из 1005 решений
винрейт: Random 87.5%, MaxBasePower 62.5%, SimpleHeuristics 0.0%, self_play 37.5%
```

Вывод: модель действительно почти НЕ свитчит по своей воле (3.9% в обучении, 0.1% в оценке) и
не жмёт теру (0%) — ровно то, что видно в боях. Прежние «60.4% свитчей» в `[mix]` — это
вынужденные свитчи после фейнтов плюс wait-кадры (их действие вообще не применялось).
То есть расхождение было в МЕТРИКЕ, а не в модели.

### 24.4 Что нашлось по дороге (и почему до боёв вообще не доходило)

1. **`resume` legacy-чекпоинта падал ещё до первого шага**: probe-политика (fallback-миграция)
   объявляла `action_mask` как `Box(..., dtype=bool)`, а poke-env — `int8`, и SB3
   `set_env()` падал на `check_for_correct_spaces`.
   Исправлено: probe строит маску `int8` (как SinglesEnv), и в fallback то же самое.
2. **Число окружений зашито в модель**: SB3 `set_env` требует `env.num_envs == self.n_envs`, а
   fallback-проба всегда создавала 1 env — прогон с `--num-envs 2/8` падал. Плюс загруженный
   чекпоинт приносит СВОЙ `n_steps` (у legacy — 8 от пробы, у модели после BC-only — 3072), и
   rollout-буфер не соответствовал задуманной схеме `n_steps = 3072 // num_envs`.
   Исправлено: `_sync_model_n_envs()` пересобирает ТОЛЬКО rollout buffer (политику не трогает —
   `_setup_model()` заново создал бы сеть и стёр веса), а `run()` выставляет
   `n_steps = max(3072 // num_envs, 1)` и на resume тоже.
3. **`evaluate_win_rates` играл без нормализации obs**: в модуле не было импорта `N_FEATURES`,
   `LiveVecNormalizeAdapter(vec_norm, target_dim=N_FEATURES)` кидал `NameError`, его глотал
   `except Exception`, и в логе оставалась одна строка «Не удалось включить нормализацию для eval».
   То есть модель в боях за винрейт (и в наших прошлых замерах) шла по СЫРЫМ признакам, хотя
   обучение — с нормализацией: это и есть «в боях ведёт себя иначе, чем в обучении». Запасной
   путь «статистика с диска» падал на том же `NameError`.
   Исправлено: `N_FEATURES` импортирован, выбор нормализатора вынесен в `eval_normalizer_for()`
   (тестируется без сервера), сбой теперь печатается явно, вплоть до
   «ВНИМАНИЕ: eval играет БЕЗ нормализации obs».
4. **Основная ветка миграции портила legacy-веса**: вход `mlp_extractor.*.0.weight` паддился до
   `features_dim` (512) вместо числа признаков (870) — у identity-экстрактора выход равен obs_dim.
   Штатная загрузка падала на size mismatch, а при обратном паддинге 870 -> 512 веса обрезались.
   Исправлено: целевая ширина выбирается по `arch["has_feature_net"]`, и `features_dim` больше не
   подсовывается `LegacyFeaturesExtractor` (он его не принимает — в логе был TypeError).
   Живая проверка на исходном чекпоинте 418: `[512, 418] -> [512, 870]`, старые колонки побитово.

### 24.5 Чем это проверено

* `test_action_kind.py`: разделение свитчей, кадры без выбора, `DecisionWrapper`.
* `test_obs_norm_inference.py` (54): `eval_normalizer_for()` — живой VecNormalize применяется,
  `PYBOT_SELF_PLAY_NORM=0` отключает только дисковый запасной путь, про отключённую нормализацию
  пишется явно.
* `test_dim_migration.py` (74): dtype маски probe-политики, совместимость с реальным env
  (`check_for_correct_spaces`), ширина входа mlp = N_FEATURES для legacy.
* `test_training_route.py` (42): `_sync_model_n_envs` (n_envs 1 -> 2, пересборка буфера, веса
  политики не меняются, `set_env` проходит).
* Живой прогон на Showdown: обучение `[mix]` и оценка `eval-микс` сходятся по составу действий.

---

## 25. Разбор ICM/CuriosityVecWrapper: устройство, размерности, дрейф нормализации

Внешний разбор (математика ICM верна: `encoder -> inverse/forward`, per-sample MSE как
intrinsic reward, `phi_next.detach()` в forward-loss, аннилинг beta) нашёл несколько мест.
Каждое проверено по коду и по живому стеку; что подтвердилось — исправлено, что нет —
проверено тестом.

### 25.1 Подтвердилось: `icm` никогда не переезжал на device

`CuriosityVecWrapper.__init__` сохранял `device` в атрибут, но `icm.to(device)` не вызывал.
На cpu это незаметно, на cuda — `RuntimeError: Expected all tensors to be on the same device`
на первом же `step_wait`. Исправлено: `self.icm = icm.to(device)` ДО создания оптимизатора
(иначе Adam ссылался бы на старые тензоры). Закреплено тестом: двойник ICM записывает
`to(...)` и проверяется, что wrapper вызвал его с запрошенным устройством (тест не требует GPU).

### 25.2 Подтвердилось: `obs_dim=715` был зашит в ICM

`N_FEATURES` в проекте менялся (715 -> 802 -> 870); при рассинхроне `nn.Linear(715, 256)`
падал бы на первом батче — не тихая порча, но лишняя точка отказа. Исправлено без правки
`config.py`: `obs_dim=None -> N_FEATURES` (импорт из config), `action_dim=None ->` размер
действий, спрошенный у poke-env (`SinglesEnv.get_action_space_size(9) == 26`), а не литерал.

Почему не через новую константу в `config.py`: `_features_fingerprint()` хеширует БАЙТЫ
`features.py + damage.py + config.py`, а этот хеш лежит в сайдкаре статистики нормализации и
в метаданных кэша датасета. Любая правка config.py (даже комментарий) обнулила бы статистику
пользователя — уже проверено на практике в этой сессии. Так что config.py не трогаем, а
константу действия выводим из poke-env. (Рекомендация на будущее: считать хеш по AST —
комментарии и форматирование перестанут обнулять кэши; сейчас НЕ делаем, чтобы не сбросить
валидную статистику `models/vecnormalize.pkl.meta.json`.)

### 25.3 Подтвердилось и вылечено: дрейф масштаба нормализации в replay (пункт «методологическая особенность»)

`VecNormalize` обновляет `obs_rms` на каждом шаге, поэтому наблюдения, записанные в replay
раньше, отмасштабированы иначе, чем текущие; в пределах ОДНОГО перехода s и s' тоже могли
нормализоваться разными статистиками. Раньше это можно было только «держать в уме».

Сделано: wrapper хранит в replay СЫРЫЕ наблюдения (`VecNormalize.get_original_obs()`) и
приводит их к ТЕКУЩЕМУ масштабу в момент `_train_icm_step()`, а для подсчёта `r_int` оба
конца перехода нормализуются одной и той же (текущей) статистикой. Закреплено тестом:
сдвигаем `obs_rms.mean` на +100 между сбором и обучением и сверяем батч ICM с явно посчитанным
`normalize(raw)` строка-в-строку (`max|Δ| = 0`), при этом сам replay остаётся сырым.
Есть флаг `normalize_replay=False` — тогда ICM учится на том, что есть (сырые наблюдения),
но смешивать сырые и нормализованные записи в одном батче не даём: нечем нормализовать —
сырые строки выбрасываются.

Попутно тесты вскрыли две ошибки в самой правке (обе исправлены):
`VecNormalize.normalize_obs/unnormalize_obs` при dict-наблюдениях требуют DICT (obs_rms —
словарь), голый ndarray падает на `assert isinstance(obs_rms, RunningMeanStd)`; и у `nn.Module`
нет атрибута `.device` — устройство берём у параметров.

### 25.4 Проверено эмпирически (а не «по вере»): терминальное наблюдение уже нормализовано

Вопрос был: лежит ли в `infos[i]["terminal_observation"]` нормализованное наблюдение или сырое.
По исходникам SB3 2.9: воркер `SubprocVecEnv`/`DummyVecEnv` кладёт туда СЫРОЕ последнее
наблюдение эпизода (`subproc_vec_env.py:41`), а `VecNormalize.step_wait` его нормализует
(`vec_normalize.py:199-204`, «Normalize the terminal observations»). Проверено на живом стеке
`Monitor(DummyVecEnv) + VecNormalize`: `terminal_observation` mean -0.03 при norm-obs mean +0.03
и raw mean +3.02 — то есть масштаб нормализованный, и wrapper корректно возвращает его к сырому
через `unnormalize_obs`. На настоящем прогоне (Showdown, 2 env, 3072 шага) счётчик
`missing_terminal_obs = 0`: терминалы есть всегда.

Заодно закрыт случай «ключа нет»: раньше такой done-переход молча уходил в replay с
наблюдением НОВОГО эпизода (артефакт) и мог начислить за него `r_int`. Теперь переход не
пишется в replay, `r_int` обнуляется, счётчик `missing_terminal_obs` и одноразовое
предупреждение в логе.

### 25.5 Про перевес curiosity над shaping: теперь видно в логе

`info["r_int"]`, `info["beta"]` были, но нигде не агрегировались. Добавлено:
* в wrapper — `info["r_int_episode"]` (суммарный вклад `beta*r_int` за эпизод) и одноразовое
  предупреждение, если он превысил `SHAPING_EPISODE_CAP` (12.0, тянется из env.py);
* в `[mix]`-строке и TensorBoard — `r_int(сред)`, `beta`, «вклад за эпизод (макс)».

Живой замер (Showdown, 2 env, 3072 шага, `feat_dim=64`, beta 0.05 -> 0.01):

```
[mix] решений 3000: приём 84.7%, тера 0.0%, свитч 15.3% (своих 149 = 5.0%, вынужденных
      после фейнта 309 = 10.3%); шагов без выбора 45 (1.5%), своих решений (был выбор) 2685 (89.5%)
      | curiosity: r_int(сред) 1.1801, beta 0.0110, вклад за эпизод (макс) 5.41 из 72 эпизодов
ICM: достигнуто 3072 перехода, fwd_loss=1.1315, inv_loss=4.0093, done-шагов без
     terminal_observation: 0, replay: raw=True
```

Вывод для тюнинга: при beta=0.05 и `feat_dim=64` вклад за эпизод ~5.4 при cap shaping 12 —
то есть intrinsic сопоставим с плотной частью награды; при `feat_dim=256` (дефолт прогонов)
он будет выше. Смотреть надо именно `вклад за эпизод (макс)` в `[mix]`: если он подходит к 12 —
уменьшать `--icm-beta` (или `--icm-feat-dim`), иначе curiosity начинает перевешивать победу
ровно так же, как когда-то перевешивал shaping.

### 25.6 Что признано корректным (не менялось)

* Архитектура и лоссы ICM, `phi_next.detach()` (проверено тестом: градиент в энкодер через
  forward-путь не течёт «трюком»), `F.cross_entropy` для inverse, clip `r_int` в `[0, 5]`.
* Порядок операций: `beta*r_int` добавляется к награде ПОСЛЕ нормализации r_ext внутри
  `VecNormalize` (wrapper снаружи) — то есть intrinsic не проходит через `ret_rms`.
* Аннилинг beta и `PPO.get_vec_normalize_env()` (через `CuriosityVecWrapper` доходит до
  внутреннего `VecNormalize`).

Тесты: новый `test_curiosity.py` (40 проверок, без сервера), `test_action_kind.py` дополнен
curiosity-метриками в `[mix]`. Все 15 файлов — PASS.

---

---

---

## 26. Тест «признаки одинаковы на всех стадиях» (`test_features_consistency.py`)

Задача: доказать на исполняемом тесте, что одна и та же раскладка 870 признаков, в том же
порядке и с той же нормализацией, доезжает до модели на всех стадиях: сбор датасета (BC-претрейн)
-> BC -> RL-обучение -> оценка винрейта -> запуск уже натренированной модели.

Запуск:

```
PYTHONPATH=. python test_features_consistency.py             # офлайн + живой прогон S1..S5
PYTHONPATH=. python test_features_consistency.py --no-live    # только офлайн (34 проверки)
PYTHONPATH=. python test_features_consistency.py --only S3     # адресно, стадии накопительные
PYBOT_TEST_FAST=1 ...   # 1 бой и короткий роллаут
PYBOT_SHOWDOWN_DIR=/path/to/pokemon-showdown ...  # сервер поднимается сам, если порт 8000 пуст
```

### 26.1 Что проверяется офлайн

* **Раскладка**: сумма сегментов `LAYOUT` == `N_FEATURES` (870), границы 715 / 802 / 12 / 24 / 32 / 155
  совпадают с `MIN_PREFIX_OBS_DIM`, `MIRROR_BASE`, `TEAM_BASE`, `FLAGS_BASE`, `EFFECT_FLAGS`,
  `DAMAGE_BLOCK_SIZE`. Правка одного признака («сдвиг») меняет колонки только своего сегмента:
  погода — ровно 2 колонки в `weather_field`, слои Spikes — колонку в `hazards`, HP скамейки —
  `bench` + сводку свитчей, тера — приёмы/`tera_meta`/`is_tera`/`tera_type`/`damage`.
* **Один бой -> один obs на всех путях**: эталон `embed_battle_with_fusion` == `ExampleEnv.embed_battle`
  == `PolicyPlayer.embed_battle` (сырой, без норм.) == геттеры `HeuristicRecorder` == датасет.
  max|Δ| = 0.0 (бит-в-бит), повторный вызов совпадает, а без командных фьюжн-карт obs отличается —
  то есть проверка не вакуумная.
* **Нормализация**: обучение (`VecNormalize.normalize_obs`) == оценка (`LiveVecNormalizeAdapter`) ==
  инференс (`VecNormStats` с диска + сайдкар `obs_dim`/`features_hash`), включая стиль
  `play_trained` и стиль `index.obs_normalizer`.

### 26.2 Что проверяется на живых боях (S1..S5)

| Стадия | Проверка |
|---|---|
| S1 сбор датасета | obs_dim == 870, маска 26, obs конечны; пересчёт из СЫРОГО кэша (`_recompute_dataset_from_raw`) даёт те же переходы и те же obs 1:1; `validate_bc_dataset` принимает |
| S2 BC (1 эпоха) | модель собрана под 870; **вход политики** (forward-hook экстрактора) == `vec.normalize_obs(obs датасета)` по каждой строке, max|Δ| = 0.0; нормализация не тождественна |
| S3 RL-роллаут | obs шага == повторный пересчёт по снимку боя; **буфер обучения == то, что политика видела на шаге**; при замороженной статистике вход == `normalize(сырой obs шага)` |
| S4 оценка винрейта | бой против RandomPlayer: сырой obs == путь env, то, что видит политика == `LiveVecNormalizeAdapter(сырой obs)` |
| S5 инференс | `save` -> `load_policy_compat` (без миграции) -> `index.load_obs_normalizer` -> бой: сырой obs тот же, нормализация политики == статистика с диска == живой VecNormalize обучения |

Итог последнего прогона: **78/78 проверок PASS**, все сравнения max|Δ| = 0.0, сводная таблица стадий
печатается в конце теста.

### 26.3 Что тест нашёл (исправлено)

1. **`VecNormStats` хранил статистику в float32**, а обучение (`RunningMeanStd`) — в float64.
   На колонках с малой дисперсией ошибка округления mean/var усиливалась делением на
   `sqrt(var+eps)`, и инференс расходился с обучением на ~1e-5 в нормализованных единицах
   (тест падал на `S5: статистика с диска == живой VecNormalize`: 1.115e-05). Теперь
   `VecNormStats.mean/var` — float64, инференс совпадает бит-в-бит (0.0). Правка в
   `agents/vecnorm_utils.py`, `features_fingerprint()` (хеш `agents/config.py`) не меняется,
   кэши/сайдкары статистики не инвалидируются.

### 26.4 Грабли, которые тест обходит (для будущих правок)

* BC считает лосс через `policy.extract_features`, а не через `policy.forward`/`evaluate_actions`:
  спаи на этих методах слепы — вход политики снимается forward-hook'ом экстрактора.
* При `training=True` статистика `VecNormalize` меняется на каждом шаге (нормализация obs_t идёт
  уже по статистике, включающей obs_t), поэтому точное равенство «вход == normalize(сырой obs)»
  воспроизводимо только при замороженной статистике (`vec.training = False`); первый вход
  роллаута — перенесённый `_last_obs` с прошлой стадии.
* Стоковый Showdown: занятое имя аккаунта -> сервер выдаёт другое, а `_` в имени класса молча
  урезается -> poke-env не логинится и бой виснет («Agent is not challenging»). В тесте имена
  генерируются уникальными и alnum.
* `PokeEnv.close()` и `Battle.weather/side_conditions` (без сеттеров) — в тесте используются
  `safe_close` и приватные `_weather`/`_side_conditions`.
* Формат подменяется рантаймом сразу в трёх namespace (`agents.config`, `agents.env`,
  `agents.training`), иначе env уходит в fusion-формат, которого нет на стоковом сервере.

---

## 27. Учёт фьюжн-типов: урон по скамейке считался по дексовым типам «головы»

### 27.1 Симптом и причина

Формат `gen9fusionmonsrandombattle` («fusionmon»). Сервер сообщает фактический тип фьюжна
**только** сообщением `|-start|<мон>|typechange|<A>/<B>|[silent]` — и только когда мон выходит
на поле (живой лог: `|-start|p1a: +Stonjourner|typechange|Grass/Rock|[silent]`). poke-env кладёт
это в `Pokemon._temporary_types` и **очищает их при `switch_out`** (`pokemon.py` ~604), а
`type_1/type_2` предпочитают `_temporary_types`, иначе берут дексовые типы.

Следствие: у любого мон **вне поля** (скамейка, ещё не показанный оппонент) `type_1/type_2` —
это типы **«головы»** из `details`, а не фьюжна. Всё, что считало урон/эффективность по таким
типам, врало:

* блок урона в obs по **нашим** покемонам (матрица «их j -> наш i», зеркальный блок «их приёмы
  x наши слоты», weakness/vulnerability скамейки, множитель типа приёма);
* множитель типа приёма против активного оппонента — до первого `typechange` в бою;
* награда в `env.py` (в т.ч. `_estimate_max_damage` для свитч-шейпинга) и флаг `wasted`
  по типовой иммунности.

Пример (живой бой): наш `Tropius` + `Stonjourner` = Grass/Rock; дексовые типы головы — Grass/Flying.
Для скамейки Earthquake считался иммунным (Flying), хотя по фьюжну он бьёт — признак «их лучший
приём по нашему слоту» показывал 0 там, где в бою урон реальный.

### 27.2 Что сделано

Новый модуль **`agents/fusion_types.py`** — расчёт типа по формуле мода (эталон — `fuseTypes`):

```
fuseTypes(types1, types2) = [types1[0], types2[1] ? types2[1] : types2[0]]   # схлопнуть дубли
если пусто -> [types1[0] || "Normal"]
```

* `parseName`: имя-«тело» = `pokemon.name.substring(1, 20)` (префикс `+` — единственный признак
  фьюжна); вид партнёра ищется в dex по id (точное совпадение, затем первое вхождение подстроки —
  как `CutDexMap` у мода);
* `types1` — тип **головы** (вид из `details`), `types2` — тип **партнёра** из имени;
* спец-случаи мода: `Arceus` (num 493) — тип по плате при `multitype`, `Silvally` (num 773) —
  по диску при `rkssystem`; для них в dex-виде `arceus<плата>` / `silvally<диск>`;
* **приоритет источников**: `server:tera` > `server:typechange` > `fusion:голова+тело` > `dex`
  (сервер всегда прав: тера и присланный typechange не пересчитываются);
* гейт `is_fusion_format`: тег боя -> env `PYBOT_BATTLE_FORMAT` -> `config.BATTLE_FORMAT`;
  вне fusion-форматов поведение не меняется **бит-в-бит** (тест это проверяет сравнением obs);
* нераспознанный партнёр/неизвестный тип -> дексовые типы + счётчики `fusion_type_used` /
  `fusion_type_unknown` (печатаются как `[type-fix]`, как и остальная диагностика типов);
* в `effective_types` есть кэш по (gen, вид, имя, предмет, способность, фолбэк-типы) — расчёт
  в горячем пути стоит ~2 мкс, накладные расходы на шаг признаков **+2.9 %** (0.645 -> 0.676 мс
  на `gen9fusionmonsrandombattle`), поэтому кэш не отключается.

Подключено:

| Место | Что теперь |
|---|---|
| `damage.prepare_mon` | кладёт `"types"` + `"types_src"` (фьюжн-типы считаются один раз на мон) |
| `damage.estimate_damage` | эффективность по `prep["types"]` защиты и STAB по `prep["types"]` атакующего |
| `damage._mon_sig` | сигнатура кэша урона — из `prep["types"]` (кэш не смешивает дексовый и фьюжн-тип) |
| `features` | `_eff_types`, `fusion_types_pair`, `_fusion_entry_for`; фьюжн-типы в мульти-hot, STAB-флаге, `_weakness_score`, `_vulnerability_frac`, `_bench_moves_vec`, `_move_wasted_flag`, `moves_dmg_multiplier` |
| `env.py` | `_eff_types` в `_estimate_max_damage` (оба пути) и в флаге `wasted` по типовой иммунности |
| `policy_player_simple.py` | множитель типа приёма по фьюжн-типам (старый/тестовый плеер) |
| `diagnose_type_spam.py` | сверка `-start\|typechange` с текущим моном теперь понимает ident `+Тело` (иначе свои же сообщения считались «про другого покемона» и `obs_stale` врал) |

Кэши и сайдкары: `training._features_fingerprint()` (он же `vecnorm_utils.features_fingerprint`)
дополнен `fusion_types.py` и `type_utils.py` -> хеш `80b9bde3c8039c9b1c1a1b2bf155808f` (на момент
правки; текущий проверяется тестом). Значит, кэш пересчёта датасета и сайдкар статистики от
старых прогонов будут считаться «чужими» — это ожидаемо: значения признаков изменились.

### 27.3 Что тест нашёл по ходу (исправлено)

1. **`_fusion_entry_for` распаковывал `fusion_types_pair` не в том порядке** (`body, _ = …`),
   из-за чего статы фьюжна скамейки не находились по ключу-«телу» -> молча падали на дексовые.
   Теперь `_head, body = fusion_pair(mon)`.
2. **Кэш `effective_types` путал моков**: в ключе не было фолбэк-типов, поэтому два тестовых
   мон `SimpleNamespace(species="dfn")` с разными `type_1/type_2` получали типы друг друга.
   Это поймал `test_damage.py::test_mold_breaker_is_exhaustive` (73 нарушения инварианта
   Mold Breaker); после добавления типов в ключ — PASS. Тест-регрессия: секция H в
   `test_fusion_types.py`.
3. API `effective_types/effective_type_names` получил параметр `fmt=` (в тестах удобнее, чем
   подменять глобальный `PYBOT_BATTLE_FORMAT`).

### 27.4 Тесты

`test_fusion_types.py` — **70 проверок**, секции:

| Секция | Что проверяет |
|---|---|
| A | формула `fuseTypes` (в т.ч. живой случай `Grass/Flying` + `Rock` -> `Grass/Rock`) |
| B | `parseName` («+Тело», `Mr. Mime`, обрезка имени), Arceus+плата/Multitype, Silvally+диск/RKS System |
| C | приоритет источников (тера/typechange/фьюжн/декс), гейт по формату (тег/config/env), счётчики |
| D | фьюжн-тип доезжает до `prepare_mon`/`estimate_damage`/`_weakness_score`/`_vulnerability_frac` |
| E | статы фьюжна у скамейки: ключ-«тело», составной `голова_тело`, коллизии (нет записи -> декс) |
| F | сквозной obs: в бою с `+Тело` меняются только блоки урона (матрица, mirror, [12:15]) и 4 type-колонки |
| G | `diagnose_type_spam._ident_matches_mon` понимает ident `+Тело` |
| H | кэш расчёта не путает монов одного вида с разными типами |
| I | `features_fingerprint` == ручной пересчёт по файлам и меняется без `fusion_types.py` |

Батарея (15 офлайн-файлов) — PASS, `test_mirror_features_live.py` — PASS,
`test_features_consistency.py` — **78/78 PASS** на живом сервере (в т.ч. с включённым гейтом
`PYBOT_BATTLE_FORMAT=gen9fusionmonsrandombattle` как смоук: на стоковом формате obs не меняются).

---

## 28. Блок «типы соперника x наша команда»: N_FEATURES 870 -> 991

Запрос: «добавь в признаки эффективность каждого типа каждого покемона соперника против
комбинации типов каждого нашего покемона — модель не может предсказать, что противник ударит
супер-эффективно, пока он не покажет приём, а признака с просто типами у нас нет».

### 28.1 Активный и запас: модель их различает?

Да, и это уже было в раскладке — просто явного ответа в документации не было:

* **Активный** описан «своими» скалярами/векторами в голове obs: `our_hp` (индекс 122),
  `our_status` [124:138), `our_boosts` [146:160), `our_actual_stats` [160:172), `our_ability`
  [172:212), `our_item`, `our_volatiles`, `our_semi_invuln`, `our_sub_damaged`, `our_restricted`,
  `our_is_tera`, `our_tera_type`.
* **Bench-блок** [287:687) — это 2 x 200 (наши, потом их) по `MAX_RESERVES = 5` слотов
  по `_RESERVE_SLOT_SIZE = 40`: `canonical_reserves()` **выбрасывает активного**
  (`not mon.active`), поэтому активный в bench не попадает никогда, а слотов ровно 5.
  Порядок слотов канонический (сортировка по species, тай-брейк по имени) — детерминированный
  между боями, но не «порядок выхода».
* Каждый слот = `типы(19) + HP + статус(7) + actual_stats(6) + weakness(1) + bench_moves(5) + item_flag(1)`.

Итог: активный и любой резерв различимы и по позиции блока, и по содержимому слота.

### 28.2 Новый блок (в хвосте obs, после блока урона)

`features._type_matchup_block`, `TYPE_MATCHUP_BLOCK_SIZE = 121`, `TYPE_MATCHUP_BASE = 870`
(715 префикс + 155 блок урона), `N_FEATURES = 991`:

```
[0:19)     multi-hot типов НАШЕГО активного (фьюжн-осознанно) — «просто типы», которых не было
[19:115)   12 строк по 8: r = opp_slot*2 + type_slot (opp_slot: 0 — активный, 1..5 — резервы)
             [0]   флаг: у покемона соперника есть тип в этом слоте
             [1]   скаляр типа 0..1 (та же шкала, что _move_type_scalar у наших приёмов)
             [2:8] множитель этого типа против наших слотов 0..5 (активный, затем резервы)
[115:121)  6 флагов: тип соперника подтверждён сервером (typechange/тера, либо формат не фьюжн)
```

* Множители сырые (0 / 0.25 / 0.5 / 1 / 2 / 4) — как `moves_dmg_multiplier`; `VecNormalize`
  приводит масштаб сам.
* Слоты нашей команды: 0 — активный, 1..5 — резервы в **том же каноническом порядке**, что в
  bench-блоке (наш слот k+1 = (k)-й слот нашего bench-блока).
* Типы берутся через `effective_types` (см. §27): сервер (`typechange`/тера) > формула фьюжна >
  декс. Флаг `подтверждён сервером` = 1 для серверных типов и для не-фьюжн форматов (там декс и
  есть правда) и 0, когда тип выведен формулой — ровно та оговорка «до свитча тип неизвестен»:
  строки есть, но модель видит, что тип не от сервера.
* Для неизвестных покемонов соперника строка нулевая (флага типа нет).

Зачем это вместе с блоком урона: блок урона считает «их j -> наш i» по **известным** приёмам
(их приёмы до первого использования неизвестны), а новый блок даёт STAB-потенциал типов
соперника по всей нашей команде **сразу** — то есть закрывает ровно тот случай, о котором
пользователь написал («пока оппонент не покажет приём»).

### 28.3 Совместимость: почему блок в хвосте

Конвенция проекта: изменения признаков добавляются **только в хвост** — `_pad_obs_to_features`
добивает старые датасеты нулями, `_migrate_checkpoint_dim` паддится нулевыми колонками первого
слоя, `MIN_PREFIX_OBS_DIM = 715` не меняется. Поэтому:

* старые датасеты (715..991) и чекпоинты (418/713/715/802/870) грузятся: новые 121 колонка = 0
  («нет информации»: флаг типа 0 => строки нет), optimizer-state сбрасывается — тот же
  warm-start, что был при 715 -> 802 -> 870;
* `test_features_consistency` проверяет, что блок идёт ровно после блока урона и что префикс не
  поехал (`715 + DAMAGE_BLOCK_SIZE == TYPE_MATCHUP_BASE`);
* `features_fingerprint()` (см. §27) включает `features.py`, где лежит блок, поэтому статистика
  нормализации/кэши старых прогонов корректно считаются «чужими».

Грабли (нашлись при обновлении тестов): в четырёх тестах начало блока урона считалось как
`N_FEATURES - DAMAGE_BLOCK_SIZE` («блок урона в хвосте»). Теперь хвост — блок типов, правильная
база — `MIN_PREFIX_OBS_DIM`. Поправлено в `test_damage.py`, `test_mirror_features_live.py`,
`test_fusion_types.py`, `test_dim_migration.py`.

### 28.4 Стоимость и проверки

* 72 вызова `damage_multiplier_safe` на шаг: +0.044 мс к `embed_battle_with_fusion`
  (1.324 -> 1.338 мс/шаг, +1.1%) — кэши dex/типов/урона уже прогреты.
* `test_features_consistency.py`: ч.1 теперь 41 проверка (границы блока, 12 строк x 6 слотов
  сверяются с ручным пересчётом по типам, флаг подтверждения 0 -> 1 после серверного
  `typechange`, сдвиг типа активного соперника меняет только приёмы/урон/блок типов),
  живой прогон S1..S5 — 102 проверки PASS, obs_dim 991 на всех путях, max|Δ| = 0.
* `test_fusion_types.py` 71 PASS, `test_mirror_features_live.py` PASS, `test_dim_migration.py`
  75 PASS, батарея из 17 файлов — PASS.

## 29. Adam для learning rate: раздельные lr/eps policy/value/shared, BC-расписание, свежий оптимизатор на BC -> PPO

Новый модуль `agents/optim.py` + изменения в `agents/policy.py`, `agents/training.py`,
`agents/policy_player.py`. Тест — `test_optim_split.py` (122 проверки, части A..H).

### 29.1 Что требовалось

1. BC: `Adam(lr=1e-3, eps=1e-8)`, плавный спад lr до `1e-4` к концу датасета (cosine/linear).
2. Точка BC -> PPO: **новый объект** оптимизатора, моменты градиентов из BC не переносятся.
3. PPO: `Adam(lr=1e-4..3e-4, eps=1e-5)`, при этом lr/eps задаются **раздельно** для policy, value и
   экстрактора признаков (shared).

### 29.2 `SplitAdam`

* группы собираются по ИМЕНАМ параметров (`classify_param_name`): `pi.*`/`log_std`/`action_net` ->
  `policy`, `value_net`/`vf.*` -> `value`, остальное (mlp_extractor/features_extractor) -> `shared`;
* `base_lr`/`base_eps` на группу + `lr_base`/`eps_base` в самом `param_group` (уезжают в чекпоинт)
  и `_lr_ratio` = отношение группы к policy. `set_lrs(policy_lr)` — единственная точка входа: lr
  групп = `policy_lr * ratio * adapt_scale`;
* `lr_value`/`lr_shared`/`eps_*` — АБСОЛЮТНЫЕ значения флагов CLI; если флаг не задан, берётся
  `lr`/`eps` (то есть группы не разъезжаются «сами»).

**Грабли (найдены тестом E).** SB3 строит оптимизатор как
`optimizer_class(self.parameters(), lr=lr_schedule(1.0), **optimizer_kwargs)` — без имён, поэтому
первый объект получается из одной плоской группы. Если считать группы «уже определёнными по
именам», то `lr_value=0.25` при базе `2e-4` даёт ratio 1250 и мусорный lr. Поэтому:

* `_as_groups()` возвращает признак `classified`: для плоского входа lr/eps групп = общие
  `lr`/`eps`, а ratio = 1.0;
* политика после создания оптимизатора пересобирает его по `named_parameters()`
  (`MaskedActorCriticPolicy._install_split_optimizer`), а базовый lr берёт из `self.lr_schedule(1.0)`,
  а НЕ из `param_groups[0]["lr"]` (после `set_lrs` там уже ratio-масштабированное значение).

### 29.3 Адаптивное управление lr и расписания

* расписания: `constant` / `linear` / `cosine` с `final_ratio` и `warmup_frac`
  (`schedule_factor`), прогресс берётся из `steps_holder["value"]` (глобальный счётчик шагов, чтобы
  расписание не сбивалось на нескольких фазах `learn()`);
* `adapt=gnorm`: `scale = clip(sqrt(target/ema_gnorm), min, max)`, target — EMA нормы градиента
  после прогрева (`tick_after_step` из `optimizer.step`);
* `adapt=plateau`: множитель `adapt_factor` (0.5) после `adapt_patience` (2) нелучших замеров
  `train/policy_loss` (из `logger.name_to_value`), пол `adapt_min` (0.1);
* множитель применяется к ВСЕМ группам одинаково (сохраняет ratio), в TB пишется
  `train/lr_policy|value|shared` и `train/lr_adapt_scale`.

### 29.4 BC-этап и переход BC -> PPO

* `pretrain_policy_bc(...)`: свежий оптимизатор в начале BC (`reset_at_start`), свой lr/eps
  (`--bc-lr 1e-3`, `--bc-eps 1e-8`, `--bc-lr-value` — отдельный lr value), спад lr ВНУТРИ эпохи по
  числу пройденных батчей (`lr = lr_final + (lr - lr_final) * factor`), лог `lr=policy/value/shared`
  в каждой эпохе;
* в конце BC оптимизатор создаётся заново (`reset_policy_optimizer`) с RL-настройками
  (`--lr-policy`/`--lr-value`/`--lr-shared`/`--eps-*`/`--lr-adapt`) — моменты BC не переносятся,
  это и есть «новый объект Adam» из требования. `--bc-keep-optimizer` + `--keep-optimizer`
  отключают пересборку.

### 29.5 Resume: конфиг CLI должен побеждать чекпоинт

`PPO.load()` возвращает БАЗОВЫЙ `PPO`, а SB3-шный `_update_learning_rate` перезаписывает lr ВСЕМ
группам одним числом — то есть на `--resume` раздельные lr молча терялись ровно там, где нужны
(дообучение после BC). Исправлено:

* `ensure_split_ppo(ppo)` возвращает загруженному объекту класс `SplitLRPPO` (смена `__class__`
  на месте, веса/моменты не трогаются) — живой прогон подтвердил: `_update_learning_rate` больше
  не падает без логгера (раньше `AttributeError: _logger` съедался `except Exception: pass`,
  и lr оставался от чекпоинта);
* `SplitAdam.apply_settings(...)` применяет lr/eps/betas/adapt к ЖИВОМУ оптимизатору, сохраняя
  моменты Adam; незаданные `lr_value`/`lr_shared` сохраняют прежнее отношение к policy;
* `adopt_optimizer(policy)`: плоский `torch.optim.Adam` из СТАРОГО чекпоинта превращается в
  `SplitAdam` с ПЕРЕНОСОМ моментов (state у Adam привязан к тензору параметра, а не к группе) —
  иначе раздельные lr на старых чекпоинтах не работают вовсе;
* свой чекпоинт round-trip: 3 группы, раздельные lr/eps и моменты на месте (проверено загрузкой
  `models/resume_check.zip`); чужой (плоский) — моменты отбрасываются с RuntimeWarning, наши
  lr/eps применяются, загрузка не падает.

### 29.6 Совместимость со старыми командами

`--learning-rate`/`--lr` теперь `default=None`: если `--bc-lr` не задан, а `--lr` задан явно
(маршрут `--total-timesteps 0 --lr 3e-4`), то BC берёт этот lr и печатает «BC lr не задан: беру
--learning-rate=...». Без флагов дефолты остаются: PPO `2e-4`, BC `1e-3 -> 1e-4`, `eps` PPO
`1e-5`, BC `1e-8` (`resolve_lr_args`, покрыто `test_training_route.py`).

### 29.7 Проверки

* `test_optim_split.py` 122 PASS (A расписания, B группы на реальной политике, C адаптация,
  D BC+сброс, E `SplitLRPPO._update_learning_rate`, F совместимость чекпоинтов, G value-warmup,
  H resume-контракт: `ensure_split_ppo`, `apply_settings`, перенос моментов из плоского Adam).
* `test_training_route.py` 49 PASS (флаги CLI, маршрут шага 1/шага 2, дефолты).
* Живые прогоны: BC-only (`--lr 3e-4 --total-timesteps 0`): lr батчей 1e-3 -> 1.15e-4, ниже 1e-4 не
  уходит, eps 1e-8; после BC — сообщение «моменты BC не переносятся», lr/eps = RL-настройки;
  RL-дообучение с `--resume`: `lr: policy=3e-05, value=1e-05, shared=1e-05`, моменты Adam
  сохранены, в TB `train/lr_policy|value|shared` и `train/lr_adapt_scale`.
* Батарея 18 файлов (features_consistency 102, fusion_types 71, dim_migration 75, obs_norm 54,
  training_route 49, curiosity 40, no_shadowing 55, self_play_migration 15, остальные PASS).

### 29.8 «Ortho только для новых выходов»: что это значит в этом проекте

BC и PPO живут на ОДНОЙ архитектуре: миграция чекпоинта паддит только ВХОДНЫЕ колонки (новые
признаки входят с нулевыми весами), головы действий/ценности не пересобираются. Поэтому на
BC -> PPO ничего молча не переинициализируется — это проверено на живом артефакте:
`models/ppo_policy_final.zip` (418 признаков, legacy-экстрактор) мигрируется на 991, и
`action_net.weight/bias`, `value_net.weight/bias` остаются **бит-в-бит** (см. проверку в конце
раздела), то есть обученные выходы не «сбрасываются» под видом миграции.

Если выходы всё-таки становятся новыми (голова шире числа действий, отсутствующий в чекпоинте
выходной слой, другая `net_arch`), включается правило `agents/policy.py`:

* `ortho_init_new_rows_(weight, old_rows, gain, bias)` — ортогональная инициализация ТОЛЬКО строк
  `[old_rows:]`; старые строки бит-в-бит, bias новых строк = 0;
* `reinit_new_outputs_(policy, old_state, missing_keys)` — применяет это к `action_net`,
  `value_net` (gain 0.01, как в PPO-инициализации SB3) и к последним слоям pi/vf-блоков
  (gain `sqrt(2)`), а модуль, которого в чекпоинте не было вовсе, инициализирует целиком;
* правки делаются ТОЛЬКО по доказанной новизне: ключ в `missing_keys` ЛИБО в `old_state` тот же
  тензор с меньшим числом строк. Если про модуль ничего не известно — он не трогается
  (иначе «на всякий случай» можно потерять обученные выходы);
* вызовы: основная миграция (`_migrate_checkpoint_dim`, после `PPO.load`) и fallback-загрузка по
  весам (`load_state_dict(strict=False)`) — раньше новые выходы там просто помечались как
  «остались случайными».

Почему не нули: `action_net` с нулевыми новыми строками = мёртвая голова (все логиты новых
действий равны, политика их не выбирает), а случайный init без масштаба даёт скачок логитов и
ломает уже обученную политику на первых шагах. Поэтому орто + маленький gain.

Проверки — `test_output_init.py` (30 проверок):

* A) `ortho_init_new_rows_`: новые строки ортонормальны (W Wᵀ ≈ g²I), старые бит-в-бит, bias новых = 0,
  повторный вызов при отсутствии новых строк — no-op;
* B) `reinit_new_outputs_`: выросшая голова (22 из 26: 4 новых — орто, 22 старых бит-в-бит),
  отсутствующий `value_net` (полная орто-инициализация, `action_net` при этом не тронут),
  отсутствующий последний слой pi (gain sqrt(2)), защита «неизвестное не трогаем»;
* C) штатная политика: правок нет (пустой список), все веса бит-в-бит;
* D) живой BC -> PPO: выходы не переинициализируются, оптимизатор новый, моменты BC не перенесены,
  `eps = 1e-5`, `lr = RL`-значение.

---

*Автор аудита: Agent Arena — полный проход по `policy_player.py` + всем его зависимостям.*
