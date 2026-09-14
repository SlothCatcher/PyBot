from .features import _STATS_TABLE_RE, _SPEED_RANGE_RE

def _parse_protect_message(protect_state: dict, split_messages):
    battle_tag = split_messages[0][0].strip(">").strip() if split_messages and split_messages[0] else None
    if battle_tag is None:
        return
    state = protect_state.setdefault(battle_tag, {"p1": False, "p2": False, "last_p1": False, "last_p2": False})
    for m in split_messages:
        if len(m) <= 1:
            continue
        if m[1] == "-singleturn" and len(m) >= 2:
            side = m[2][:2] if len(m) > 2 else None
            if side in ("p1", "p2"):
                state[side] = True
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
            continue

        if msg_type == "-start" and len(m) >= 4 and m[3] == "typechange" and m[-1] == "[silent]":
            pending[battle_tag] = m[2][:2]
        elif msg_type == "html":
            side = pending.get(battle_tag)
            if side:
                html = m[2] if len(m) > 2 else ""
                stats_m = _STATS_TABLE_RE.search(html)
                speed_m = _SPEED_RANGE_RE.search(html)
                if stats_m or speed_m:
                    entry = store.setdefault(battle_tag, {}).setdefault(side, {})
                    if stats_m:
                        hp, atk, d, spa, spd, spe = map(int, stats_m.groups())
                        entry["base_stats"] = {"hp": hp, "atk": atk, "def": d, "spa": spa, "spd": spd, "spe": spe}
                    if speed_m:
                        entry["speed_range"] = tuple(map(int, speed_m.groups()))
                    pending[battle_tag] = None


class FusionInfoParser:
    """Миксин: ловит -start|typechange + html-сообщения с базовыми статами фьюжна."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fusion_stats: dict[str, dict[str, dict]] = {}
        self._pending_stats_side: dict[str, str | None] = {}
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