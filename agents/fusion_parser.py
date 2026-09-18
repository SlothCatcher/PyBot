import re
from .features import _STATS_TABLE_RE, _SPEED_RANGE_RE

# protectmoves которые действительно используют -singleturn как защиту
_PROTECT_SINGLETURN_KEYWORDS = {
    "protect", "detect", "kingsshield", "king's shield", "spikyshield", "banefulbunker",
    "obstruct", "silktrap", "burningbulwark", "maxguard", "craftyshield", "matblock",
    "quickguard", "wideguard",
}
# endure/focus punch / magic coat / snatch / followme - тоже -singleturn но НЕ защита
# их исключаем: если в строке нет protect-ключевого слова — игнор

_TITLE_RE = re.compile(r"<b>\s*([^<]+?)\s*base stats", re.IGNORECASE)

# диагностика: что сервер реально присылает в -start|typechange (tag -> {side: raw}).
# Это ключевое доказательство при разборе "модель не видит иммунитет": если тут "???",
# то poke-env получает THREE_QUESTION_MARKS и типовая эффективность становится нейтральной.
_RAW_TYPECHANGE: dict[str, dict[str, str]] = {}


def get_raw_typechange(battle_tag: str, side: str):
    return _RAW_TYPECHANGE.get(battle_tag, {}).get(side)

def _parse_protect_message(protect_state: dict, split_messages):
    battle_tag = split_messages[0][0].strip(">").strip() if split_messages and split_messages[0] else None
    if battle_tag is None:
        return
    # утечка: чистим на win/tie так же как fusion — иначе словарь растет весь прогон
    for m in split_messages:
        if len(m) > 1 and m[1] in ("win", "tie"):
            protect_state.pop(battle_tag, None)
            return
    state = protect_state.setdefault(battle_tag, {"p1": False, "p2": False, "last_p1": False, "last_p2": False})
    for m in split_messages:
        if len(m) <= 1:
            continue
        if m[1] == "-singleturn" and len(m) >= 3:
            side = m[2][:2] if len(m) > 2 else None
            if side not in ("p1", "p2"):
                continue
            # точный фильтр: только Protect-семейство, а не любой singleturn
            joined = "|".join(m).lower()
            # m[3] иногда содержит имя эффекта: "move: Protect"
            if any(kw in joined for kw in _PROTECT_SINGLETURN_KEYWORDS):
                state[side] = True
            else:
                # fallback: если формат showdown без имени (старые сервера) — оставляем старый широкий триггер?
                # но по ревью — лучше пропустить чем дать ложную защиту
                # поэтому игнор
                pass
        elif m[1] == "turn":
            state["last_p1"], state["last_p2"] = state["p1"], state["p2"]
            state["p1"], state["p2"] = False, False

def _parse_fusion_message(store: dict, pending: dict, split_messages):
    battle_tag = split_messages[0][0].strip(">").strip() if split_messages and split_messages[0] else None
    if battle_tag is None:
        return
    for m in split_messages:
        if len(m) <= 1:
            continue
        msg_type = m[1]

        if msg_type in ("win", "tie"):
            store.pop(battle_tag, None)
            pending.pop(battle_tag, None)
            _RAW_TYPECHANGE.pop(battle_tag, None)
            continue

        # диагностика: логируем ЛЮБОЙ typechange (в т.ч. без [silent]), чтобы видеть,
        # какие типы сервер отдаёт для фьюжнов (PYBOT_DEBUG_TYPES=1 включает печать)
        if msg_type == "-start" and len(m) >= 4 and m[3] == "typechange":
            _side = m[2][:2] if len(m) > 2 else ""
            if _side in ("p1", "p2"):
                _raw = "|".join(m)
                _RAW_TYPECHANGE.setdefault(battle_tag, {})[_side] = _raw
                try:
                    from .type_utils import note_typechange_raw
                except ImportError:
                    from type_utils import note_typechange_raw
                note_typechange_raw(_raw)

        if msg_type == "-start" and len(m) >= 4 and m[3] == "typechange" and m[-1] == "[silent]":
            side = m[2][:2]
            if side not in ("p1", "p2"):
                continue
            species_raw = m[2].split(":", 1)[-1].strip().lstrip("+").strip()
            # храним per-side species чтобы различать одновременные свитчи
            cur = pending.get(battle_tag)
            if cur is None or isinstance(cur, str) or isinstance(cur, set):
                # миграция старого формата -> dict
                if isinstance(cur, str):
                    # был один сайд строкой — превращаем в dict с unknown species
                    cur = {cur: ""} if cur in ("p1", "p2") else {}
                elif isinstance(cur, set):
                    cur = {s: "" for s in cur if s in ("p1", "p2")}
                elif cur is None:
                    cur = {}
                pending[battle_tag] = cur
            # если уже есть тот же side — перезаписываем (новый свитч той же стороны)
            # если теперь оба сайда pending — оставляем оба, но html будем матчить по title
            cur = pending[battle_tag]
            cur[side] = species_raw

        elif msg_type == "html":
            pend = pending.get(battle_tag)
            if not pend or not isinstance(pend, dict) or len(pend) == 0:
                continue
            html = m[2] if len(m) > 2 else ""
            stats_m = _STATS_TABLE_RE.search(html)
            speed_m = _SPEED_RANGE_RE.search(html)
            if not (stats_m or speed_m):
                continue
            # пытаемся вытащить название из title: "<b>Corviknight + Froslass base stats:</b>"
            title_species = ""
            tm = _TITLE_RE.search(html)
            if tm:
                title_species = tm.group(1).strip()
            # нормализуем title для матчинга: берём lower без пробелов
            title_norm = title_species.lower().replace(" ", "").replace("-", "")
            matched_side = None
            matched_species_raw = None
            if len(pend) == 1:
                # один pending — однозначно его
                matched_side, matched_species_raw = next(iter(pend.items()))
            elif len(pend) >= 2:
                # коллизия: пытаемся сматчить по имени из html
                # если title содержит species_raw одной из сторон — берём её
                for side, species_raw in list(pend.items()):
                    if not species_raw:
                        continue
                    # species_raw может быть "Corviknight" а title "Corviknight+Froslass"
                    sr_norm = species_raw.lower().replace(" ", "").replace("-", "")
                    if sr_norm and sr_norm in title_norm:
                        matched_side = side
                        matched_species_raw = species_raw
                        break
                if matched_side is None:
                    # безопасный компромисс из ревью: лучше отбросить оба чем дать неверные данные обоим
                    pending.pop(battle_tag, None)
                    continue
                # нашли — раздаём только этой стороне, вторая остаётся pending на свой html
            # запись: и в side-слот (для активного) и в by_species (для скамейки)
            entry = store.setdefault(battle_tag, {}).setdefault(matched_side, {})
            if stats_m:
                hp, atk, d, spa, spd, spe = map(int, stats_m.groups())
                entry["base_stats"] = {"hp": hp, "atk": atk, "def": d, "spa": spa, "spd": spd, "spe": spe}
            if speed_m:
                entry["speed_range"] = tuple(map(int, speed_m.groups()))
            # дублируем в per-species хранилище
            if matched_species_raw:
                try:
                    # ключ — нормализованный id как в PokeEnv (to_id_str)
                    from poke_env.data.normalize import to_id_str
                    sid = to_id_str(matched_species_raw) or matched_species_raw.lower().replace(" ", "").replace("-", "")
                except Exception:
                    sid = matched_species_raw.lower().replace(" ", "").replace("-", "")
                # также пробуем распарсить вторую часть фьюжна из title для полноты
                # титул "Corviknight + Froslass" — положим обе половинки если они различаются
                by_species_store = store.setdefault(battle_tag, {}).setdefault(f"{matched_side}_by_species", {})
                species_entry = by_species_store.setdefault(sid, {})
                if stats_m:
                    hp, atk, d, spa, spd, spe = map(int, stats_m.groups())
                    species_entry["base_stats"] = {"hp": hp, "atk": atk, "def": d, "spa": spa, "spd": spd, "spe": spe}
                if speed_m:
                    species_entry["speed_range"] = tuple(map(int, speed_m.groups()))
                # также положим под ключ титула если он отличается (например фьюжн id "corviknightfroslass")
                if title_species and "+" in title_species:
                    try:
                        # фьюжн id часто просто конкат без плюса, но оставим
                        fusion_id = title_species.lower().replace(" ", "").replace("+", "").replace("-", "")
                        if fusion_id != sid and fusion_id not in by_species_store:
                            fusion_entry = by_species_store.setdefault(fusion_id, {})
                            if stats_m:
                                fusion_entry["base_stats"] = dict(species_entry["base_stats"]) if "base_stats" in species_entry else {"hp": hp, "atk": atk, "def": d, "spa": spa, "spd": spd, "spe": spe}
                            if speed_m and "speed_range" in species_entry:
                                fusion_entry["speed_range"] = species_entry["speed_range"]
                    except Exception:
                        pass
            # убираем отработанный сайд из pending, остальных оставляем
            try:
                del pend[matched_side]
                if len(pend) == 0:
                    pending.pop(battle_tag, None)
                else:
                    pending[battle_tag] = pend
            except Exception:
                pending.pop(battle_tag, None)


class FusionInfoParser:
    """Миксин: ловит -start|typechange + html-сообщения с базовыми статами фьюжна."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fusion_stats: dict[str, dict[str, dict]] = {}
        self._pending_stats_side: dict[str, dict[str, str] | None] = {}
        self._protect_state = {} 

    async def _handle_battle_message(self, split_messages):
        _parse_fusion_message(self._fusion_stats, self._pending_stats_side, split_messages)
        _parse_protect_message(self._protect_state, split_messages)
        await super()._handle_battle_message(split_messages)

    def get_protected_last_turn(self, battle, is_ours: bool) -> float:
        state = self._protect_state.get(battle.battle_tag, {})
        side = battle.player_role if is_ours else ("p2" if battle.player_role == "p1" else "p1")
        return 1.0 if state.get(f"last_{side}", False) else 0.0
    
    def get_fusion_entry(self, battle, is_ours: bool) -> dict | None:
        side = battle.player_role if is_ours else ("p2" if battle.player_role == "p1" else "p1")
        return self._fusion_stats.get(battle.battle_tag, {}).get(side)

    def get_team_fusion_map(self, battle, is_ours: bool) -> dict:
        side = battle.player_role if is_ours else ("p2" if battle.player_role == "p1" else "p1")
        return self._fusion_stats.get(battle.battle_tag, {}).get(f"{side}_by_species", {}) or {}


def _attach_fusion_parser(player):
    if hasattr(player, "_fusion_stats"):
        return
    player._fusion_stats = {}
    player._pending_stats_side = {}
    player._protect_state = {}
    original_handle = player._handle_battle_message

    async def patched_handle(split_messages, _orig=original_handle):
        _parse_fusion_message(player._fusion_stats, player._pending_stats_side, split_messages)
        _parse_protect_message(player._protect_state, split_messages)
        await _orig(split_messages)

    player._handle_battle_message = patched_handle
