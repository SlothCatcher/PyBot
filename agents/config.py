from poke_env.battle.pokemon_type import PokemonType
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import DefaultBattleOrder, Player

BATTLE_FORMAT = "gen9fusionmonsrandombattle"
N_FEATURES = 418
VECNORM_PATH = "models/vecnormalize.pkl"
SELF_PLAY_PATH = "models/self_play_snapshot"
QUALIFIED_PREFIX = "self_play_qualified_"
MIN_WINRATE_TO_QUALIFY = 50

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
    try:
        return _original_action_to_order(action, battle, fake=fake, strict=strict)
    except ValueError:
        return DefaultBattleOrder()

SinglesEnv.action_to_order = staticmethod(_safe_action_to_order)

_original_damage_multiplier = PokemonType.damage_multiplier

def _safe_damage_multiplier(self, *args, **kwargs):
    try:
        return _original_damage_multiplier(self, *args, **kwargs)
    except KeyError:
        return 1.0

PokemonType.damage_multiplier = _safe_damage_multiplier
