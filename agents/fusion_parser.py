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


# --- тайминг typechange ---------------------------------------------------------------
# poke-env обрабатывает строки батча ПО ПОРЯДКУ и на строке |request| сразу вызывает
# choose_move (→ embed_battle). Значит typechange, пришедший в том же батче ПОСЛЕ |request|,
# к решению ещё не применён, и модель видит СТАРЫЙ тип покемона. Лечим перестановкой:
# переносим строки typechange перед |request| внутри батча. Набор сообщений не меняется,
# каждая строка парсится poke-env'ом ровно один раз (без двойного применения), взаимный
# порядок самих typechange и всех прочих сообщений сохраняется.


def _is_typechange_message(m) -> bool:
    return len(m) > 4 and m[1] == "-start" and m[3] == "typechange"


def _collect_typechanges(split_messages) -> list:
    """[(ident, types_raw, after_request)] — для диагностики."""
    out = []
    try:
        req_idx = None
        for i, m in enumerate(split_messages):
            if len(m) > 1 and m[1] == "request":
                req_idx = i
                break
        for i, m in enumerate(split_messages):
            if _is_typechange_message(m):
                after = req_idx is not None and i > req_idx
                out.append((m[2], m[4], after))
                if after:
                    try:
                        from .type_utils import note_typechange_raw
                    except ImportError:
                        from type_utils import note_typechange_raw
                    note_typechange_raw("|".join(m), after_request=True)
    except Exception:
        pass
    return out


def _hoist_typechanges_before_request(split_messages):
    """Переносит все -start|typechange строки батча перед |request|.

    Возвращает (новый список, число перенесённых строк).
    """
    try:
        if not split_messages or len(split_messages) < 3:
            return split_messages, 0
        header, rest = split_messages[0], split_messages[1:]
        req_pos = None
        for i, m in enumerate(rest):
            if len(m) > 1 and m[1] == "request":
                req_pos = i
                break
        if req_pos is None:
            return split_messages, 0
        tc_all = [m for m in rest if _is_typechange_message(m)]
        if len(tc_all) <= len([m for m in rest[:req_pos] if _is_typechange_message(m)]):
            return split_messages, 0  # все typechange и так до запроса
        before = [m for m in rest[:req_pos] if not _is_typechange_message(m)]
        after = [m for m in rest[req_pos:] if not _is_typechange_message(m)]
        return [header, *before, *tc_all, *after], len(tc_all)
    except Exception:
        return split_messages, 0


def _reorder_batch(player, split_messages):
    """Диагностика + перестановка перед передачей батча в poke-env."""
    try:
        _collect_typechanges(split_messages)
    except Exception:
        pass
    reordered, n = _hoist_typechanges_before_request(split_messages)
    if n and reordered is not split_messages:
        try:
            from .type_utils import note_typechange_reordered
        except ImportError:
            from type_utils import note_typechange_reordered
        note_typechange_reordered(f"{n} стр. перенесено перед |request|")
    return reordered


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
        # FIX тайминга: typechange, пришедший ПОСЛЕ |request| в этом же батче, переносим перед
        # запросом — иначе poke-env применит его уже после выбора хода и модель увидит старый тип
        split_messages = _reorder_batch(self, split_messages)
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
        # тот же фикс тайминга, что и в миксине (agent1/agent2 в env — не миксин)
        split_messages = _reorder_batch(player, split_messages)
        await _orig(split_messages)

    player._handle_battle_message = patched_handle
