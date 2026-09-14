"""Конфиг обучения. Вынесен чтобы избежать циркулярных импортов и magic numbers.

Не игнорируется git'ом: корневой config.py (логин/пароль) по-прежнему в игноре через /config.py,
а этот файл трекается.
"""
BATTLE_FORMAT = "gen9fusionmonsrandombattle"
# Размер наблюдения считается в agents/features.py -> embed_battle_with_fusion.
# При изменении признаков не забыть пересчитать и обновить vecnormalize.pkl:
# TYPE_LIST 19 + statuses 7 + bench 27*5*2 + ... = 418
N_FEATURES = 418

QUALIFIED_PREFIX = "qualified_"
SELF_PLAY_PATH = "models/self_play_snapshot"
VECNORM_PATH = "models/vecnormalize.pkl"
# Порог винрейта vs SimpleHeuristics чтобы снапшот попал в self-play пул.
# 55 было слишком высоко (пул пустой -> нет давления self-play -> плато 30-40%).
# Снижаем до 48-50 чтобы качественные снапшоты попадали регулярно, но мусор отсеивался.
MIN_WINRATE_TO_QUALIFY = 50
