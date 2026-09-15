"""
Сбор датасета из реплеев Pokemon Showdown высокого рейтинга.

Использование:
  python -m agents.collect_replay_dataset --format gen9randombattle --min-rating 1800 --count 1000 --output models/replay_dataset.npz
  python -m agents.collect_replay_dataset --format gen9fusionmonsrandombattle --min-rating 1500 --count 500 --output models/replay_fusion.npz

Логика:
  1) search.json?format=<format> — пагинация по uploadtime/before, фильтр по rating.
  2) GET /<id>.json (+ .inputlog для точного action) — кэшируем в models/replay_cache/
  3) Парсим лог через poke_env Battle (аналогично live боям) + fusion_parser для фьюжн-статов.
     Для каждого хода снимаем obs = embed_battle_with_fusion(battle) ДО того как ход исполнился,
     mask = SinglesEnv.get_action_mask(battle) (fallback к inferred если пусто),
     action = order_to_action(move/switch) для игрока который ходил.
     Сохраняем per-player траектории, в конце считаем ret = ±30 * 0.99**(steps_remaining) по |win|.

Зависимости: requests (опционально urllib), poke_env, numpy.
"""
import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# не импортируем torch здесь — датасет собирается без него
from poke_env.battle import Battle
from poke_env.environment import SinglesEnv

from .config import N_FEATURES, BATTLE_FORMAT
from .features import embed_battle_with_fusion

import ssl
import urllib.request
import urllib.error
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

REPLAY_BASE = "https://replay.pokemonshowdown.com"
SEARCH_URL = REPLAY_BASE + "/search.json"
CACHE_DIR = Path("models/replay_cache")

# для маппинга move id -> action
def _move_id_to_action(battle: Battle, move_id: str, terastallize: bool = False) -> int | None:
    """Находит индекс хода в battle.active_pokemon.moves / available_moves -> action 6-9 (+ tera)."""
    if battle.active_pokemon is None:
        return None
    # mvs логика как в SinglesEnv.order_to_action
    known_moves = list(battle.active_pokemon.moves.values())[:4]
    known_ids = [m.id for m in known_moves]
    avail_ids = [m.id for m in battle.available_moves] if battle.available_moves else []
    if len(avail_ids) == 1 and avail_ids and avail_ids[0] not in known_ids:
        mvs = battle.available_moves  # type: ignore
    else:
        mvs = known_moves  # type: ignore
    mvs_ids = [m.id for m in mvs]
    # move_id из лога — нормализуем через to_id_str (lower, убираем пробелы/дефисы)
    from poke_env.data.normalize import to_id_str
    norm = to_id_str(move_id)
    if norm in mvs_ids:
        idx = mvs_ids.index(norm)
    elif move_id.lower() in [m.id.lower() for m in mvs]:
        idx = [m.id.lower() for m in mvs].index(move_id.lower())
    else:
        # move ещё не в known_moves (редкий случай первого хода) — добавляем временно
        # пробуем найти по известным moves всего покемона (включая невидимые)
        # fallback: если не нашли — считаем что это первый слот
        return None
    gimmick = 4 if terastallize else 0
    return 6 + idx + 4 * gimmick

def _switch_to_action(battle: Battle, poke_identifier: str) -> int | None:
    """poke_identifier: 'p1a: Tauros' или 'Tauros, L82, M' — маппим на team index 0-5."""
    # identifier может быть 'p1a: Tauros' или просто вид
    # battle.team keys like 'p1: Tauros'
    # нам нужно найти base_species в team
    # Извлечём вид из identifier
    # для '|switch|p1a: Tauros|Tauros-Paldea-Combat, L82, M|...' -> identifier = 'p1a: Tauros'
    # для '|switch|p2a: Koraidon|...' similarly
    # Для inputLog '>p1 switch 2' — там индекс 1-based в оригинальной команде (1-6)
    # Но для log-based парсинга — ищем по species
    species = poke_identifier.split(":")[-1].strip().split(",")[0].strip()
    # нормализуем
    from poke_env.data.normalize import to_id_str
    norm = to_id_str(species)
    team_list = list(battle.team.values())
    for idx, mon in enumerate(team_list):
        if to_id_str(mon.base_species) == norm or to_id_str(mon.species) == norm:
            return idx
    # fallback — поиск по полному имени
    for idx, mon in enumerate(team_list):
        if mon.base_species.lower() in species.lower() or species.lower() in mon.base_species.lower():
            return idx
    return None

def _infer_mask(battle: Battle) -> List[int]:
    """Fallback если battle.available_moves пусто (нет request). Инферим из battle state."""
    mask = SinglesEnv.get_action_mask(battle)
    if sum(mask) > 0:
        return mask
    # fallback: все живые бенчи + все известные мувы активного
    size = SinglesEnv.get_action_space_size(battle.gen)
    inferred = [0]*size
    # switches: все не fainted, не активные
    for i, mon in enumerate(battle.team.values()):
        if not mon.fainted and not mon.active and not battle.trapped:
            inferred[i] = 1
    if battle.active_pokemon is not None and not battle.trapped:
        known = list(battle.active_pokemon.moves.values())[:4]
        for idx, mv in enumerate(known):
            # считаем что PP >0 (в реплее не знаем точно, но если ход был использован — точно >0)
            inferred[6+idx] = 1
            if battle.can_tera:
                inferred[22+idx] = 1
        if not known:
            inferred[6] = 1
    # если всё ещё пусто — разрешим default
    if sum(inferred) == 0:
        inferred[6] = 1
    return inferred

def _download_with_retry(url: str, retries=3, sleep=1.0) -> str | None:
    for attempt in range(retries):
        # 1) пробуем requests
        if HAS_REQUESTS:
            try:
                r = requests.get(url, timeout=15, headers={"User-Agent": "PyBot/1.0"})
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", "5"))
                    logger.warning(f"429 {url}, жду {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.text
            except Exception as e_req:
                logger.debug(f"requests failed {e_req}, пробую urllib")
        # 2) urllib с unverified SSL (работает в sandbox где requests падает по SSL)
        try:
            ctx = ssl._create_unverified_context()
            req = urllib.request.Request(url, headers={"User-Agent": "PyBot/1.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"download {url} attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(sleep*(attempt+1))
    return None

def search_replay_ids(fmt: str, min_rating: int | None, count: int) -> List[Dict[str, Any]]:
    """Пагинация search.json?format=fmt&before=uploadtime"""
    ids = []
    before = None
    seen = set()
    # Showdown формат в URL — без скобок, lower, без пробелов: "[Gen 9] Random Battle" -> gen9randombattle
    # Пользователь может передать уже нормализованное (gen9randombattle) или человеческое
    # Нормализуем для запроса: lower, убрать не-буквоцифры
    def normalize_format(f: str) -> str:
        f = f.strip().lower()
        # если уже gen9... оставляем
        if f.startswith("gen"):
            return re.sub(r"[^a-z0-9]", "", f)
        # иначе "[Gen 9] Random Battle" -> gen9randombattle
        return re.sub(r"[^a-z0-9]", "", f)
    fmt_q = normalize_format(fmt)
    logger.info(f"Поиск реплеев format={fmt} -> query={fmt_q}, min_rating={min_rating}, need={count}")
    # Для очень нишевых форматов (fusion) search может вернуть [] — пробуем также passthrough без format
    tried_without_format = False
    while len(ids) < count:
        url = f"{SEARCH_URL}?format={fmt_q}"
        if before is not None:
            url += f"&before={before}"
        # альтернативный способ пагинации для старых версий API: &page=
        text = _download_with_retry(url)
        if text is None:
            logger.error(f"search failed {url}")
            break
        try:
            data = json.loads(text)
        except Exception as e:
            logger.error(f"json parse search {e}")
            break
        if not data:
            if not tried_without_format and fmt_q != "gen9randombattle":
                logger.warning(f"search {fmt_q} вернул 0, пробую без фильтра и отфильтрую локально")
                tried_without_format = True
                # сбросим и попробуем более широкий поиск (только если fusion пустой)
                # но чтобы не выкачать весь Showdown, ограничимся 1 попыткой с очень широким
                # для fusion попробуем также gen9fusionmonsrandombattle -> ген9randombattle как fallback?
                # просто выйдем и вернём что есть
                break
            logger.info("search вернул пусто — конец")
            break
        # data — список 50+1
        has_more = len(data) > 50
        batch = data[:50]
        for entry in batch:
            rid = entry.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            rating = entry.get("rating")
            # фильтр по рейтингу: если min_rating задан и рейтинг null — пропускаем (unrated)
            if min_rating is not None:
                if rating is None:
                    continue
                if rating < min_rating:
                    continue
            # дополнительный фильтр по формату если искали широко
            if tried_without_format:
                if fmt_q not in entry.get("format", "").lower().replace(" ", "").replace("[", "").replace("]", ""):
                    continue
            ids.append(entry)
            if len(ids) >= count:
                break
        if len(ids) >= count:
            break
        if not has_more:
            break
        # пагинация: берем uploadtime последнего в batch
        try:
            before = batch[-1].get("uploadtime")
            if before is None:
                break
        except Exception:
            break
        # защита от вечного цикла
        if len(ids) % 200 == 0:
            logger.info(f"  собрано {len(ids)}/{count} ...")
        time.sleep(0.3)  # не спамим
    logger.info(f"Найдено {len(ids)} реплеев под фильтр")
    return ids[:count]

def download_replay(replay_id: str, cache_dir: Path = CACHE_DIR) -> Dict[str, Any] | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{replay_id}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    url = f"{REPLAY_BASE}/{replay_id}.json"
    text = _download_with_retry(url)
    if text is None:
        return None
    try:
        data = json.loads(text)
        # сохраняем кэш
        try:
            cache_path.write_text(text, encoding="utf-8")
        except Exception:
            pass
        return data
    except Exception as e:
        logger.warning(f"json parse {replay_id}: {e}")
        return None

def download_inputlog(replay_id: str, cache_dir: Path = CACHE_DIR) -> str | None:
    cache_path = cache_dir / f"{replay_id}.inputlog"
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception:
            pass
    url = f"{REPLAY_BASE}/{replay_id}.inputlog"
    text = _download_with_retry(url)
    if text is not None:
        try:
            cache_path.write_text(text, encoding="utf-8")
        except Exception:
            pass
        return text
    return None

def parse_replay_to_samples(replay_json: Dict[str, Any], inputlog_text: str | None = None, only_winners: bool = False) -> List[Tuple[np.ndarray, np.ndarray, int, str, int]]:
    """
    Возвращает список (obs, mask, action, battle_tag, won_flag) для обоих игроков.
    won_flag — 1 если этот игрок выиграл, 0 если проиграл (для ret).
    Для BC используем позже ret = 30/-30 * gamma**remaining.
    """
    replay_id = replay_json.get("id", "unknown")
    fmt = replay_json.get("format", "")
    # Определим gen из формата: "[Gen 9] ..." -> 9, default 9
    m = re.search(r"gen\s*(\d+)", fmt.lower())
    gen = int(m.group(1)) if m else 9
    # battle_format для конфига — используем BATTLE_FORMAT если совпадает, иначе переопределяем gen локально
    log = replay_json.get("log", "")
    if not log:
        logger.warning(f"{replay_id} пустой log")
        return []
    # Определим игроков и победителя из log
    winner = None
    # log содержит "|win|username"
    for line in log.split("\n"):
        if line.startswith("|win|"):
            winner = line.split("|")[2].strip()
            break
    players = replay_json.get("players", [])
    # inputlog парсинг для точного action (move/switch + tera)
    # inputlog формат: ">p1 move closecombat" / ">p2 switch 3" / ">p1 move hydropump terastallize"
    input_moves: Dict[int, Dict[str, str]] = defaultdict(dict)  # turn -> {p1: "move ...", p2: ...}
    # но inputlog не нумерует turn, просто последовательность. Более надёжно — маппим по порядку ходов в log.
    # Упростим: соберём очередь инпутов по игрокам
    p1_inputs = []
    p2_inputs = []
    if inputlog_text:
        for line in inputlog_text.split("\n"):
            line=line.strip()
            if line.startswith(">p1 "):
                p1_inputs.append(line[4:].strip())  # "move closecombat" etc
            elif line.startswith(">p2 "):
                p2_inputs.append(line[4:].strip())

    # Создаём два Battle для двух перспектив
    logger_battle = logging.getLogger("replay_battle")
    logger_battle.setLevel(logging.CRITICAL)
    battle_p1 = Battle(battle_tag=replay_id, username=players[0] if players else "p1", logger=logger_battle, gen=gen)
    battle_p2 = Battle(battle_tag=replay_id, username=players[1] if len(players)>1 else "p2", logger=logger_battle, gen=gen)
    # Установим player_role вручную — Battle обычно ставит его из |player| сообщения, но мы форсируем
    # чтобы оба battle корректно различали team/opponent
    battle_p1._player_role = "p1"
    battle_p2._player_role = "p2"
    # fusion / protect state эмулируем как в live
    from agents.fusion_parser import _parse_fusion_message, _parse_protect_message

    fusion_store: Dict[str, Dict[str, Dict]] = {}
    pending_store: Dict[str, Any] = {}
    protect_store: Dict[str, Dict] = {}

    # Разбиваем log на линии и обрабатываем последовательно
    # Нам нужно на каждом |turn| снимать obs ДО хода, а action брать из inputlog / log
    log_lines = [l for l in log.split("\n") if l.strip() != ""]
    # Будем итерировать и на каждом turn фиксировать состояние
    # Для упрощения: перед обработкой событий очередного turn'a — снимаем obs для обоих игроков
    # Затем обрабатываем события этого turn'a и маппим действия

    # Очереди инпутов — индекс по turn (начиная с 1)
    # В inputlog первый ход — turn 1, второй — turn 2, etc. Но switch на старте не считается turn
    # Проще: используем pointer per player
    p1_ptr = 0
    p2_ptr = 0

    # Храним траектории per player tag
    # tag для группировки как в HeuristicRecorder: battle_tag + player
    p1_tag = f"{replay_id}_p1"
    p2_tag = f"{replay_id}_p2"
    p1_traj: List[Tuple[np.ndarray, np.ndarray, int]] = []
    p2_traj: List[Tuple[np.ndarray, np.ndarray, int]] = []

    # Вспомогательная функция снятия obs
    def snapshot(battle: Battle, fusion_entry, opp_fusion_entry, protect_state, player_role: str):
        # fusion_entry — для нашей стороны
        # protect_state — берём из protect_store
        our_side = player_role
        opp_side = "p2" if our_side == "p1" else "p1"
        ps = protect_store.get(replay_id, {})
        our_prot = 1.0 if ps.get(f"last_{our_side}", False) else 0.0
        opp_prot = 1.0 if ps.get(f"last_{opp_side}", False) else 0.0
        try:
            obs = embed_battle_with_fusion(
                battle,
                our_fusion=fusion_entry,
                opp_fusion=opp_fusion_entry,
                our_protected_last_turn=our_prot,
                opp_protected_last_turn=opp_prot,
            )
        except Exception as e:
            logger.debug(f"embed failed {replay_id} {player_role}: {e}")
            return None
        mask = _infer_mask(battle)
        mask_arr = np.array(mask, dtype=np.int8)
        # sanity: Box
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, mask_arr

    # Обработка лога с turn-границами
    # Мы будем идти по log_lines, и при встрече "|turn|N" — это начало нового хода
    # Снимаем obs перед обработкой событий этого хода (т.е. состояние в начале turn N)
    current_turn = 0
    # Нужно обработать начальные |switch| до первого turn как team preview (не снимаем obs)
    # Для этого просто прокрутим до первого |turn|
    idx = 0
    # Сначала обработаем все события до первого turn (team preview, start)
    # Но obs для turn 1 нужен после стартовых switch'ей
    # Поэтому найдём первый turn
    first_turn_idx = None
    for i, line in enumerate(log_lines):
        if line.startswith("|turn|"):
            first_turn_idx = i
            break
    if first_turn_idx is None:
        # нет turn — возможно быстрая победа? просто обработаем весь лог
        first_turn_idx = len(log_lines)

    # Обработаем preview часть (0 .. first_turn_idx-1) для обоих battle
    for i in range(first_turn_idx):
        line = log_lines[i]
        split = line.split("|")
        # fusion/protect парсинг — делаем как split_messages с battle_tag префиксом
        # Для fusion: нужен split_messages[0][0] = ">replay_id"
        # Упростим: передаём напрямую в парсеры
        fake_split = [[f">{replay_id}"]] + [split]
        try:
            _parse_fusion_message(fusion_store, pending_store, fake_split)
            _parse_protect_message(protect_store, fake_split)
        except Exception:
            pass
        # Battle parse
        # Battle.parse_message ожидает split где split[1] это сообщение
        # line = "|switch|p1a: Tauros|..." -> split = ['', 'switch', 'p1a: Tauros', ...]
        try:
            battle_p1.parse_message(split)
            battle_p2.parse_message(split)
        except Exception as e:
            logger.debug(f"parse preview {line}: {e}")

    # Теперь обрабатываем по ходам
    # current_turn уже после preview, следующий индекс = first_turn_idx
    turn_idx = first_turn_idx
    turn_number = 1
    while turn_idx < len(log_lines):
        line = log_lines[turn_idx]
        if not line.startswith("|turn|"):
            turn_idx+=1
            continue
        # Начало хода turn_number
        # Найдём границы текущего turn (нужны ДО snapshot, чтобы инферить available_moves)
        next_turn_idx = None
        for j in range(turn_idx+1, len(log_lines)):
            if log_lines[j].startswith("|turn|"):
                next_turn_idx = j
                break
        if next_turn_idx is None:
            next_turn_idx = len(log_lines)
        turn_slice = log_lines[turn_idx+1: next_turn_idx]

        # --- FIX для replay: нет |request| -> available_moves пусто -> moves_base/wasted/acc/pp DEAD
        # Инферим доступные ходы перед snapshot, иначе embed даёт -1/1/0
        # Также восстанавливаем moves активного покемона из предстоящего |move| (иначе moves пусто на 1-м ходу)
        for _batt in (battle_p1, battle_p2):
            try:
                # если у активного нет moves — добавим из turn_slice (первый увиденный мув этого игрока)
                if _batt.active_pokemon is not None and len(_batt.active_pokemon.moves) == 0:
                    # найдём предстоящий |move| этого игрока в turn_slice
                    role = _batt.player_role  # p1/p2
                    for _l in turn_slice:
                        if _l.startswith("|move|") and f"|{role}a:" in _l:
                            try:
                                mv_name = _l.split("|")[3]
                                from poke_env.data.normalize import to_id_str as _toid
                                from poke_env.battle import Move
                                mid = _toid(mv_name)
                                # создаём Move, добавляем в актив
                                if mid not in _batt.active_pokemon.moves:
                                    mv = Move(mid, gen=_batt.gen)
                                    _batt.active_pokemon.moves[mid] = mv
                                    # также в team dict
                                    _batt.active_pokemon.moves[mid] = mv
                            except Exception:
                                pass
                            break
                # теперь инферим available_moves/switches если пусто
                if not _batt.available_moves and _batt.active_pokemon is not None and not _batt.trapped:
                    known = list(_batt.active_pokemon.moves.values())[:4]
                    if known:
                        _batt._available_moves = known  # type: ignore
                if not _batt.available_switches:
                    # все живые неактивные
                    switches = [m for m in _batt.team.values() if not m.fainted and not m.active]
                    if switches:
                        _batt._available_switches = switches  # type: ignore
            except Exception as e:
                logger.debug(f"prep battle {e}")

        # Снимаем obs для обоих игроков ПЕРЕД ходом
        # Fusion entries для каждого игрока
        # fusion_store[replay_id] = {"p1": {...}, "p2": {...}}
        f_entry = fusion_store.get(replay_id, {})
        # Для battle_p1: наша сторона p1, оппонент p2
        # Для battle_p2: наоборот
        p1_fus = f_entry.get("p1")
        p2_fus = f_entry.get("p2")
        # Для p1 battle: our = p1, opp = p2
        obs_p1 = snapshot(battle_p1, p1_fus, p2_fus, protect_store, "p1")
        obs_p2 = snapshot(battle_p2, p2_fus, p1_fus, protect_store, "p2")
        # Определим действия этого хода из inputlog / log
        # В inputlog: каждая запись соответствует выбору игрока на этот turn (если игрок ходил)
        # Но не все turn'ы оба игрока ходят (может быть force switch)
        # Попробуем взять следующие инпуты из очереди
        # Для log-based fallback: смотрим |move|/|switch| внутри этого turn до следующего |turn|

        # Собираем действия из turn_slice
        # |move|p1a: ...|MoveName|p2a: ... и |switch|p1a: ...
        # Для каждого такого события — это действие игрока p1 или p2
        # Но нам нужно маппить на action ДО хода, а не после. Текущий obs — перед ходом, действие — то что в этом turn_slice совершил игрок.
        # Если в turn_slice оба игрока сделали ход (обычно 2 move), то для каждого игрока свой action
        # Если один switch — только один игрок

        # Парсим actions из slice
        actions_this_turn: Dict[str, Tuple[str, str, bool]] = {}  # role -> (type, id, tera)
        # Сначала попробуем inputlog (более точный для tera, т.к. в log tera в отдельной строке |-terastallize|)
        # inputlog очередь: p1_inputs[p1_ptr] соответствует этому turn если p1 ходил
        # Но как узнать ходил ли p1 в этот turn? Смотрим turn_slice — если там есть |move|p1a или |switch|p1a -> ходил
        p1_did_move = any("|p1a:" in l for l in turn_slice)
        p2_did_move = any("|p2a:" in l for l in turn_slice)
        # Также бывает |cant| — тогда игрок не мог ходить (par/slp)
        # Для inputlog: если игрок не ходил, в inputlog для этого turn не будет записи (или будет pass?)
        # Упростим: если p1_did_move — берём следующий p1 input
        p1_input = None
        p2_input = None
        if p1_did_move and p1_ptr < len(p1_inputs):
            p1_input = p1_inputs[p1_ptr]
            p1_ptr+=1
        if p2_did_move and p2_ptr < len(p2_inputs):
            p2_input = p2_inputs[p2_ptr]
            p2_ptr+=1

        # Парсим input -> (type, id, tera)
        def parse_input(inp: str):
            # inp examples: "move closecombat", "switch 3", "move hydropump terastallize", "move uturn", "pass"
            if inp is None:
                return None
            if inp.startswith("move "):
                parts = inp.split()
                move_id = parts[1]
                tera = "terastallize" in inp
                return ("move", move_id, tera)
            elif inp.startswith("switch "):
                # "switch 2" — 1-based index в команде (1..6) как в inputLog? Для random team это индекс в оригинальном порядке
                # Но для action нам нужен team index 0..5 — это как раз (int(parts[1])-1)
                # Однако battle.team order может отличаться (отсортирован по появлению). Для упрощения маппим через switch identifier из log
                # Поэтому для switch лучше брать из log, а не из input
                return ("switch", inp.split()[1], False)
            elif inp == "pass" or inp.startswith("pass"):
                return None
            return None

        p1_parsed = parse_input(p1_input) if p1_input else None
        p2_parsed = parse_input(p2_input) if p2_input else None

        # Если input не дал action (или нет inputlog) — парсим из log
        # Для log: ищем |move| и |switch| и сопоставляем
        # Собираем log actions per role
        log_actions: Dict[str, Tuple[str, str, bool]] = {}
        tera_this_turn = set()  # роли которые терасталлили в этом ходу
        for l in turn_slice:
            if l.startswith("|-terastallize|"):
                # "|-terastallize|p2a: Koraidon|Fire"
                try:
                    who = l.split("|")[2][:2]  # p1 or p2
                    tera_this_turn.add(who)
                except Exception:
                    pass
            if l.startswith("|move|"):
                # "|move|p1a: Tauros|Close Combat|p2a: Hypno"
                try:
                    parts = l.split("|")
                    who_full = parts[2]  # p1a: Tauros
                    role = who_full[:2]
                    move_name = parts[3]
                    # tera?
                    tera = role in tera_this_turn
                    log_actions[role] = ("move", move_name, tera)
                except Exception:
                    pass
            elif l.startswith("|switch|") or l.startswith("|drag|"):
                try:
                    parts = l.split("|")
                    who_full = parts[2]  # p1a: Tauros
                    role = who_full[:2]
                    # details = parts[3] like "Tauros-Paldea-Combat, L82, M"
                    species = parts[3].split(",")[0].strip()
                    log_actions[role] = ("switch", species, False)
                except Exception:
                    pass

        # Теперь для каждого игрока выбираем action приоритет: input parsed если есть, иначе log
        # p1
        chosen_p1 = None
        if p1_did_move:
            if p1_parsed and p1_parsed[0] == "move":
                # p1_parsed move id — маппим
                chosen_p1 = p1_parsed
            elif "p1" in log_actions:
                chosen_p1 = log_actions["p1"]
            else:
                # fallback — если в turn_slice есть p1 move но мы не распарсили, пробуем log
                pass
        # p2
        chosen_p2 = None
        if p2_did_move:
            if p2_parsed and p2_parsed[0] == "move":
                chosen_p2 = p2_parsed
            elif "p2" in log_actions:
                chosen_p2 = log_actions["p2"]

        # Для switch — input даёт индекс, log даёт species — предпочитаем log для species,
        # но для action индекса нужен team mapping, который лучше из battle
        # Если chosen из input был switch — заменяем на log switch если есть
        if chosen_p1 and chosen_p1[0] == "switch" and "p1" in log_actions and log_actions["p1"][0]=="switch":
            chosen_p1 = log_actions["p1"]
        if chosen_p2 and chosen_p2[0] == "switch" and "p2" in log_actions and log_actions["p2"][0]=="switch":
            chosen_p2 = log_actions["p2"]

        # Теперь конвертим chosen -> action index и сохраняем если obs есть
        if obs_p1 is not None and chosen_p1 is not None:
            obs, mask = obs_p1
            act = None
            if chosen_p1[0] == "move":
                act = _move_id_to_action(battle_p1, chosen_p1[1], terastallize=chosen_p1[2])
                # если не нашли из-за unknown moves — инферим как 6
                if act is None:
                    # fallback: считаем что ход — первый слот
                    act = 6
                    if chosen_p1[2]:
                        act = 22
            else:  # switch
                # chosen_p1[1] может быть species или индекс
                if chosen_p1[1].isdigit():
                    # input switch index 1-6 -> 0-5
                    try:
                        act = int(chosen_p1[1]) - 1
                        # защита: если в battle team другой порядок — всё равно 0-5 валиден, но может быть не тот покемон
                        # оставим как есть
                    except Exception:
                        act = None
                else:
                    act = _switch_to_action(battle_p1, chosen_p1[1])
            if act is not None and 0 <= act < len(mask):
                # проверим что action в маске, если нет — расширим маску (fallback)
                if mask[act] == 0:
                    # если маска не содержит этот action, добавим (иногда из-за inferred mismatch)
                    mask = mask.copy()
                    mask[act] = 1
                p1_traj.append((obs, mask, act))

        if obs_p2 is not None and chosen_p2 is not None:
            obs, mask = obs_p2
            act = None
            if chosen_p2[0] == "move":
                act = _move_id_to_action(battle_p2, chosen_p2[1], terastallize=chosen_p2[2])
                if act is None:
                    act = 6
                    if chosen_p2[2]:
                        act = 22
            else:
                if chosen_p2[1].isdigit():
                    try:
                        act = int(chosen_p2[1]) - 1
                    except Exception:
                        act = None
                else:
                    act = _switch_to_action(battle_p2, chosen_p2[1])
            if act is not None and 0 <= act < len(mask):
                if mask[act] == 0:
                    mask = mask.copy()
                    mask[act] = 1
                p2_traj.append((obs, mask, act))

        # Теперь обрабатываем события этого turn для обновления battle state (для следующего turn)
        for line in turn_slice:
            split = line.split("|")
            fake_split = [[f">{replay_id}"]] + [split]
            try:
                _parse_fusion_message(fusion_store, pending_store, fake_split)
                _parse_protect_message(protect_store, fake_split)
            except Exception:
                pass
            try:
                battle_p1.parse_message(split)
                battle_p2.parse_message(split)
            except Exception as e:
                logger.debug(f"parse turn {turn_number} {line}: {e}")
        # также обработать саму строку |turn|N (инкремент)
        try:
            battle_p1.parse_message(["", "turn", str(turn_number)])
            battle_p2.parse_message(["", "turn", str(turn_number)])
        except Exception:
            pass
        turn_number+=1
        turn_idx = next_turn_idx

    # После цикла — у нас траектории per player
    # Нужно посчитать ret per trajectory based on winner
    samples = []
    # winner null -> draw, skip? (редко)
    if winner is None:
        # попробуем из replay_json rating? но win неизвестен — пропускаем реплей
        logger.debug(f"{replay_id} no winner {players}")
        return []
    p1_won = (winner == (players[0] if players else "p1"))
    p2_won = (winner == (players[1] if len(players)>1 else "p2"))
    # gamma как в training._compute_bc_returns
    gamma = 0.99
    victory_value = 30.0
    if only_winners:
        trajectories = [(traj, won, tag) for traj, won, tag in [(p1_traj, p1_won, p1_tag), (p2_traj, p2_won, p2_tag)] if won]
    else:
        trajectories = [(p1_traj, p1_won, p1_tag), (p2_traj, p2_won, p2_tag)]
    for traj, won, tag in trajectories:
        if not traj:
            continue
        outcome = victory_value if won else -victory_value
        n = len(traj)
        for i, (obs, mask, act) in enumerate(traj):
            steps_remaining = n - i -1
            ret = outcome * (gamma ** steps_remaining)
            samples.append((obs, mask, act, ret, tag))  # tag для дебага, но сохраним без tag?
    # samples — каждый элемент (obs, mask, act, ret)
    # Для совместимости с heuristic_dataset: (obs, mask, action, ret)
    # Вернём в формате list of (obs, mask, action, ret)
    return [(obs, mask, act, ret) for (obs, mask, act, ret, tag) in samples]

def collect_replay_dataset(
    fmt: str,
    min_rating: int | None,
    count: int,
    output: str,
    cache_dir: str = "models/replay_cache",
    max_workers: int = 4,
    only_winners: bool = False,
    refresh: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, int, float]]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    # --- кэш-first: если в кэше уже есть достаточно реплеев под фильтр — берём их без поиска ---
    def _format_match(entry_fmt: str, fmt_q: str) -> bool:
        # entry_fmt: "[Gen 9] Random Battle" or "gen9randombattle"
        norm = re.sub(r"[^a-z0-9]", "", entry_fmt.lower())
        return fmt_q in norm or norm in fmt_q

    fmt_q = re.sub(r"[^a-z0-9]", "", fmt.strip().lower()) if fmt.strip().lower().startswith("gen") else re.sub(r"[^a-z0-9]", "", fmt.lower())
    cached_entries: List[Dict[str, Any]] = []
    if not refresh and cache_path.exists():
        for jp in cache_path.glob("*.json"):
            # пропускаем inputlog json? только *.json реплеев (они содержат "id" и "log")
            if jp.suffix != ".json" or jp.name.endswith(".inputlog"):
                continue
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
                # фильтр по формату
                entry_fmt = data.get("format", "") or data.get("formatid", "")
                if fmt and not _format_match(entry_fmt, fmt_q):
                    continue
                rating = data.get("rating")
                if min_rating is not None:
                    if rating is None or rating < min_rating:
                        continue
                # проверим что есть log
                if not data.get("log"):
                    continue
                cached_entries.append({"id": data.get("id", jp.stem), "_data": data})
                if len(cached_entries) >= count:
                    break
            except Exception:
                continue
        if len(cached_entries) >= count:
            logger.info(f"Кэш хит: {len(cached_entries)} реплеев уже в {cache_path} под {fmt} {min_rating}+ — беру из кэша без поиска")
            # преобразуем к виду search entries
            ids = [{"id": e["id"], "rating": e["_data"].get("rating"), "format": e["_data"].get("format", fmt)} for e in cached_entries[:count]]
            # дальше пойдёт ветка кэшированного download (download_replay вернёт из файла)
        else:
            if cached_entries:
                logger.info(f"Кэш: {len(cached_entries)}/{count} подходят, докачаю ещё {count - len(cached_entries)}")
            # берём с запасом 30% + 100 (раньше было ×3 → 50к → 150к и час поиска)
            need = int((count - len(cached_entries)) * 1.3) + 100 if cached_entries else int(count * 1.3) + 100
            ids = search_replay_ids(fmt, min_rating, need)
            # если нашли кэшированные — добавим их в начало, чтобы не качать дубликаты
            if cached_entries:
                cached_ids_set = {e["id"] for e in cached_entries}
                # prepend cached
                cached_search = [{"id": e["id"], "rating": e["_data"].get("rating"), "format": e["_data"].get("format", fmt)} for e in cached_entries]
                # фильтруем дубликаты из search
                ids = cached_search + [e for e in ids if e["id"] not in cached_ids_set]
                ids = ids[:need + len(cached_entries)]
    else:
        # берём с запасом 30% + 100
        need = int(count * 1.3) + 100
        ids = search_replay_ids(fmt, min_rating, need)
    logger.info(f"Скачиваю {len(ids)} реплеев (цель {count})...")
    all_samples: List[Tuple[np.ndarray, np.ndarray, int, float]] = []
    ok_replays = 0
    failed = 0
    for idx, entry in enumerate(ids):
        # основной стоп — по количеству реплеев, а не по семплам (раньше было count*10 -> обрезало 1000 до 210)
        if ok_replays >= count:
            break
        rid = entry["id"]
        # скачка
        replay_json = download_replay(rid, cache_path)
        if replay_json is None:
            failed+=1
            continue
        inputlog = download_inputlog(rid, cache_path)
        try:
            samples = parse_replay_to_samples(replay_json, inputlog, only_winners=only_winners)
        except Exception as e:
            logger.warning(f"parse {rid} failed: {e}")
            import traceback; traceback.print_exc()
            failed+=1
            continue
        if not samples:
            failed+=1
            continue
        all_samples.extend([(obs, mask, act, ret) for (obs, mask, act, ret) in samples])
        ok_replays+=1
        if ok_replays % 20 == 0:
            logger.info(f"  {ok_replays}/{count} реплеев ok, {len(all_samples)} семплов, failed {failed}")
        time.sleep(0.05)
    logger.info(f"Итого: {ok_replays} реплеев, {len(all_samples)} семплов, {failed} failed")
    # семплы уже с ret, сохраняем как heuristic_dataset
    if not all_samples:
        logger.error("Нет семплов — проверь формат/рейтинг, может fusion пусто. Попробуй --format gen9randombattle")
        return []
    # обрезка по count реплеев уже сделана, но если нужно ровно count реплеев — уже
    # сохраняем
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obs_arr = np.stack([s[0] for s in all_samples]).astype(np.float32)
    mask_arr = np.stack([s[1] for s in all_samples]).astype(np.int8)
    action_arr = np.array([s[2] for s in all_samples], dtype=np.int64)
    ret_arr = np.array([s[3] for s in all_samples], dtype=np.float32)
    # проверка размерности
    assert obs_arr.shape[1] == N_FEATURES, f"obs dim {obs_arr.shape[1]} != {N_FEATURES}"
    assert mask_arr.shape[1] == SinglesEnv.get_action_space_size(9), f"mask {mask_arr.shape[1]}"
    np.savez_compressed(out_path, obs=obs_arr, mask=mask_arr, action=action_arr, ret=ret_arr)
    logger.info(f"Датасет сохранён: {out_path} ({len(all_samples)} семплов, {ok_replays} реплеев)")
    # также выводим статистику по win/loss ret
    try:
        print(f"ret mean {ret_arr.mean():.2f} min {ret_arr.min():.1f} max {ret_arr.max():.1f}")
    except Exception:
        pass
    return all_samples

def main():
    ap = argparse.ArgumentParser(description="Сбор датасета из реплеев Showdown")
    ap.add_argument("--format", type=str, default="gen9randombattle", help="Формат Showdown, напр. gen9randombattle / gen9ou / gen9fusionmonsrandombattle")
    ap.add_argument("--min-rating", type=int, default=1700, help="Минимальный рейтинг реплея (null реплеи скипаются). 0 чтобы брать все")
    ap.add_argument("--count", type=int, default=1000, help="Сколько реплеев скачать (успешных)")
    ap.add_argument("--output", type=str, default="models/replay_dataset.npz", help="Куда сохранить npz")
    ap.add_argument("--cache-dir", type=str, default="models/replay_cache")
    ap.add_argument("--max-workers", type=int, default=4, help="Не используется сейчас (синхронно), оставлен для совместимости")
    ap.add_argument("--only-winners", action="store_true", help="Брать только ходы победителей (иначе учим и проигравших). Рекомендуется для трансферa randombattle->fusion")
    ap.add_argument("--refresh", action="store_true", help="Игнорировать кэш и перекачать (по умолчанию берёт из кэша если хватает)")
    args = ap.parse_args()
    min_rating = None if args.min_rating == 0 else args.min_rating
    collect_replay_dataset(
        fmt=args.format,
        min_rating=min_rating,
        count=args.count,
        output=args.output,
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
        only_winners=args.only_winners,
        refresh=args.refresh,
    )

if __name__ == "__main__":
    main()
