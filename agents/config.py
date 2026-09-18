from poke_env.battle.pokemon_type import PokemonType

try:
    from .type_utils import damage_multiplier_safe, note_keyerror
except ImportError:  # запуск модуля вне пакета
    from type_utils import damage_multiplier_safe, note_keyerror
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import DefaultBattleOrder, Player
import numpy as np

BATTLE_FORMAT = "gen9fusionmonsrandombattle"
N_FEATURES = 715  # 713 +2 is_tera (our/opp active terastallized now). Было 713: 120 moves(4*30) +4 faint/hp +14 status +8 hazards +4 switches +14 boosts(7+7) +12 actual +40 ability(20+20) +10 weather/field +9 trick/tail/screens +1 speed +2 revealed +2 semi +2 sub +1 restr +22 volatiles +22 items +400 bench +2 vuln +3 tera +19 tera_type +2 protect . 715 = +2 is_tera
VECNORM_PATH = "models/vecnormalize.pkl"
SELF_PLAY_PATH = "models/self_play_snapshot"
QUALIFIED_PREFIX = "self_play_qualified_"
MIN_WINRATE_TO_QUALIFY = 25  # было 50 -> 30, но у тебя после BC 31.7% упал до 25% и deadlock, поэтому 25 + fallback на обычные снапшоты

# --- монки-патчи библиотеки poke-env, применяются один раз при импорте ---

_original_order_to_action = SinglesEnv.order_to_action

def _safe_order_to_action(order, battle, fake=False, strict=True):
    try:
        return _original_order_to_action(order, battle, fake=fake, strict=strict)
    except ValueError:
        pass

    try:
        fallback_order = Player.choose_random_move(battle)
        return _original_order_to_action(fallback_order, battle, fake=fake, strict=strict)
    except ValueError:
        pass

    try:
        return _original_order_to_action(DefaultBattleOrder(), battle, fake=fake, strict=strict)
    except ValueError:
        # Последний рубеж: берём индекс первого действия, разрешённого маской
        mask = SinglesEnv.get_action_mask(battle)
        for idx, allowed in enumerate(mask):
            if allowed:
                return idx
        return 0  # маска тоже пуста — возвращаем что угодно, PPO это переживёт как один плохой шаг
SinglesEnv.order_to_action = staticmethod(_safe_order_to_action)

_original_action_to_order = SinglesEnv.action_to_order

def _safe_action_to_order(action, battle, fake=False, strict=True):
    mask = SinglesEnv.get_action_mask(battle)
    if sum(mask) == 0:
        return DefaultBattleOrder()
    # poke_env ожидает np.int64 с методом .item(), а PolicyPlayer отдаёт Python int
    try:
        if isinstance(action, int) and not hasattr(action, "item"):
            action = np.int64(action)
        elif isinstance(action, np.ndarray) and action.ndim == 0:
            action = np.int64(action.item())
        return _original_action_to_order(action, battle, fake=fake, strict=strict)
    except (ValueError, AttributeError, TypeError):
        return DefaultBattleOrder()

SinglesEnv.action_to_order = staticmethod(_safe_action_to_order)

_original_damage_multiplier = PokemonType.damage_multiplier

def _safe_damage_multiplier(self, *args, **kwargs):
    # ВАЖНО: раньше тут был `except KeyError: return 1.0` — это маскировало иммунитет.
    # Чарт gen9 не знает типов STELLAR / THREE_QUESTION_MARKS, поэтому для фьюжна с
    # нераспознанным вторым типом ("???") вызов ELECTRIC.damage_multiplier(GROUND, ???)
    # кидал KeyError, и на весь расчёт возвращался нейтрал 1.0 вместо иммунитета 0.0.
    # Теперь считаем покомпонентно: неизвестный тип нейтрален только за себя,
    # известный Ground сохраняет свой 0.0.
    try:
        type_1 = args[0] if len(args) > 0 else kwargs.get("type_1")
        type_2 = args[1] if len(args) > 1 else kwargs.get("type_2")
        chart = kwargs.get("type_chart")
        return damage_multiplier_safe(self, type_1, type_2, chart)
    except Exception:
        # последний рубеж: оригинал, и только потом нейтрал (с логом, чтобы не молчать)
        try:
            return _original_damage_multiplier(self, *args, **kwargs)
        except KeyError as err:
            note_keyerror(self, args[0] if args else kwargs.get("type_1"),
                          args[1] if len(args) > 1 else kwargs.get("type_2"), err)
            return 1.0

PokemonType.damage_multiplier = _safe_damage_multiplier
