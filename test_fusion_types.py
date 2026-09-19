#!/usr/bin/env python3
"""Фьюжн-типы: считаем тип покемона так же, как мод, когда сервер молчит.

Зачем: в формате фьюжнов (`*fusionmon*`) Showdown не отдаёт нестандартный тип заранее —
`|-start|<ident>|typechange|<t1>/<t2>|[silent]` приходит только когда покемон вышел на поле,
а при свитче poke-env очищает `_temporary_types` (снова дексовые типы «головы»). Поэтому урон
по покемонам вне поля (матрица 6x6 «их j -> наш i», блок «их приёмы x наши слоты», weakness/
vulnerability, множитель типа приёма) считался по НЕВЕРНОМУ типу.

Тест проверяет:
  A) формула мода `fuseTypes` (включая реальный случай из живого лога: голова Grass/Flying +
     тело Stonjourner -> Grass/Rock, ровно как прислал сервер);
  B) разбор имени («+Тело»), поиск вида, спец-случаи Arceus/Silvally (платы/диски);
  C) приоритет источников: тера > typechange сервера > наш расчёт фьюжна > декс;
  D) гейт по формату: без "fusionmon" в id формата ничего не меняется;
  E) что фьюжн-тип реально доезжает до урона и признаков (obs-блоки урона, weakness, vulnerability,
     статы фьюжна у скамейки по ключу-«телу»).

Запуск: PYTHONPATH=. python test_fusion_types.py
"""
import os
import sys
from contextlib import contextmanager

import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poke_env.battle import Battle, Move
from poke_env.battle.move import MoveSet
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.data import GenData

from agents import config as _cfg  # noqa: F401  (монки-патчи типовой эффективности)
from agents import fusion_types as ft
from agents import type_utils
from agents.damage import estimate_damage, prepare_mon
from agents.damage import DAMAGE_BLOCK_SIZE, FLAGS_BASE, MIRROR_BASE, TEAM_BASE
from agents.training import MIN_PREFIX_OBS_DIM
from agents.features import (
    TYPE_MATCHUP_BLOCK_SIZE, _fusion_entry_for, _vulnerability_frac, _weakness_score,
    embed_battle_with_fusion,
)

OK = 0
FAIL = []


def check(name, cond, extra=""):
    global OK
    if cond:
        OK += 1
        print(f"OK   {name}" + (f": {extra}" if extra else ""))
    else:
        FAIL.append(name)
        print(f"FAIL {name}" + (f": {extra}" if extra else ""))


def check_eq(name, got, want):
    check(name, got == want, f"got={got!r} want={want!r}")


@contextmanager
def fmt_env(value):
    """Подмена формата прогона (env-переменная важнее конфига)."""
    old = os.environ.get("PYBOT_BATTLE_FORMAT")
    os.environ["PYBOT_BATTLE_FORMAT"] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("PYBOT_BATTLE_FORMAT", None)
        else:
            os.environ["PYBOT_BATTLE_FORMAT"] = old


@contextmanager
def config_fmt(value):
    """Подмена config.BATTLE_FORMAT (когда env-переменной нет)."""
    old = getattr(_cfg, "BATTLE_FORMAT", None)
    old_env = os.environ.pop("PYBOT_BATTLE_FORMAT", None)
    _cfg.BATTLE_FORMAT = value
    try:
        yield
    finally:
        _cfg.BATTLE_FORMAT = old
        if old_env is not None:
            os.environ["PYBOT_BATTLE_FORMAT"] = old_env


class FakeBattle:
    def __init__(self, tag):
        self.battle_tag = tag


def mk_mon(details, name=None, gen=9, item=None, ability=None):
    """Реальный poke-env Pokemon: species из details, отображаемое имя — из name."""
    mon = Pokemon(details=details, name=name or details.split(",")[0], gen=gen)
    if item is not None:
        mon._item = item
    if ability is not None:
        mon._ability = ability
    return mon


def main() -> int:
    real_fmt = ft.current_format()
    print(f"формат по умолчанию: {real_fmt!r}; fusion-гейт: {ft.is_fusion_format()}")
    print("=" * 78)
    print("A. Формула мода fuseTypes")
    print("-" * 78)

    # Реальный случай из живого лога: |-start|p1a: +Stonjourner|typechange|Grass/Rock|[silent]
    # Голова — травиный мон (Tropius: Grass/Flying), тело — Stonjourner (Rock, один тип).
    check_eq("голова Grass/Flying + тело Rock -> Grass/Rock (как сервер в бою)",
             ft.fuse_type_names(("Grass", "Flying"), ("Rock",)), ("Grass", "Rock"))
    check_eq("берём ВТОРОЙ тип тела (Ice/Ghost + Steel/Flying)",
             ft.fuse_type_names(("Ice", "Ghost"), ("Steel", "Flying")), ("Ice", "Flying"))
    check_eq("тело с одним типом даёт его первый (Fire/Flying + Rock)",
             ft.fuse_type_names(("Fire", "Flying"), ("Rock",)), ("Fire", "Rock"))
    check_eq("одинаковые первые типы схлопываются (Normal + Normal)",
             ft.fuse_type_names(("Normal",), ("Normal",)), ("Normal",))
    check_eq("дубликат по первому типу схлопывается (Grass/Flying + Grass/Fighting)",
             ft.fuse_type_names(("Grass", "Flying"), ("Grass", "Fighting")), ("Grass", "Fighting"))
    check_eq("нет тела -> только голова", ft.fuse_type_names(("Water",), ()), ("Water",))
    check_eq("нет головы -> только тело", ft.fuse_type_names((), ("Rock",)), ("Rock",))
    check_eq("совсем пусто -> Normal (фолбэк мода)",
             ft.fuse_type_names((), ()), ("Normal",))
    check_eq("пустая строка типа не попадает в результат",
             ft.fuse_type_names(("Grass", ""), (None, "Rock")), ("Grass", "Rock"))

    print("-" * 78)
    print("B. Имя («+Тело») и разбор вида")
    print("-" * 78)

    check_eq("'+Stonjourner' -> 'Stonjourner'", ft.partner_name("+Stonjourner"), "Stonjourner")
    check_eq("без '+' это не партнёр", ft.partner_name("Stonjourner"), None)
    check_eq("'+' без имени -> None", ft.partner_name("+"), None)
    check_eq("обрезка как substring(1, 20)",
             ft.partner_name("+" + "a" * 30), "a" * 19)
    check_eq("'Mr. Mime' -> mrmime", ft.resolve_species("Mr. Mime")[0], "mrmime")
    check_eq("как нашли (точное совпадение)", ft.resolve_species("Stonjourner")[1], "точно")
    check_eq("неизвестное имя -> None", ft.resolve_species("Zzzzzzz")[0], None)

    # спец-случаи мода относятся к ПАРТНЁРУ (то, что распарсено из имени), голова — своя
    FUS = "gen9fusionmonsrandombattle"
    arceus = mk_mon("Tropius, L50, M", name="+Arceus", item="dracoplate", ability="multitype")
    check_eq("тело Arceus + Draco Plate + Multitype -> второй тип Dragon (Grass/Dragon)",
             ft.effective_type_names(arceus, fmt=FUS), ("GRASS", "DRAGON"))
    arceus_plain = mk_mon("Tropius, L50, M", name="+Arceus", item="leftovers", ability="multitype")
    check_eq("тело Arceus без платы -> Normal (Grass/Normal)",
             ft.effective_type_names(arceus_plain, fmt=FUS), ("GRASS", "NORMAL"))
    silvally = mk_mon("Tropius, L50, M", name="+Silvally", item="firememory", ability="rkssystem")
    check_eq("тело Silvally + Fire Memory + RKS System -> Fire (Grass/Fire)",
             ft.effective_type_names(silvally, fmt=FUS), ("GRASS", "FIRE"))
    check_eq("плата без Multitype не даёт тип платы", 
             ft.partner_species_of(mk_mon("Tropius, L50, M", name="+Arceus", item="dracoplate",
                                          ability="levitate"))[0], "arceus")

    print("-" * 78)
    print("C. Приоритет источников типа")
    print("-" * 78)

    fused = mk_mon("Tropius, L50, M", name="+Stonjourner")
    check_eq("голова берётся из details, тело — из имени",
             ft.fusion_pair(fused), ("tropius", "stonjourner"))
    check_eq("декс-типы головы (Tropius)", ft.dex_types("tropius"), ("Grass", "Flying"))

    tera = mk_mon("Tropius, L50, M", name="+Stonjourner")
    tera._terastallized = True
    tera._terastallized_type = PokemonType.ICE
    t1, t2, src = ft.effective_types(tera)
    check_eq("тера побеждает расчёт фьюжна (тип ICE)", (str(t1.name), t2), ("ICE", None))
    check_eq("источник — сервер (тера)", src, "server:tera")

    temp = mk_mon("Tropius, L50, M", name="+Stonjourner")
    temp._temporary_types = [PokemonType.POISON, PokemonType.NORMAL]
    t1, t2, src = ft.effective_types(temp)
    check_eq("typechange сервера побеждает расчёт (Poison/Normal)",
             (str(t1.name), str(t2.name)), ("POISON", "NORMAL"))
    check_eq("источник — сервер (typechange)", src, "server:typechange")

    with fmt_env("gen9fusionmonsrandombattle"):
        t1, t2, src = ft.effective_types(fused)
        check_eq("в фьюжн-формате тип считается по формуле мода",
                 (str(t1.name), str(t2.name)), ("GRASS", "ROCK"))
        check("источник помечен как расчёт фьюжна", src.startswith("fusion:tropius+stonjourner"), src)
        check_eq("и это ровно серверный 'Grass/Rock' из живого лога",
                 "/".join(ft.effective_type_names(fused)).lower(), "grass/rock")
        check("счётчик fusion_type_used растёт",
              sum(type_utils.counter_snapshot("fusion_type_used").values()) > 0)
        # мон с такой же головой, но без партнёра — не фьюжн
        plain = mk_mon("Tropius, L50, M", name="+Tropius")
        plain._name = "Tropius"
        t1, t2, src = ft.effective_types(plain)
        check_eq("имя без '+' -> дексовые типы", (str(t1.name), str(t2.name)), ("GRASS", "FLYING"))
        check("источник помечен как декс", src.startswith("dex"), src)
        check("счётчик fusion_type_unknown растёт",
              sum(type_utils.counter_snapshot("fusion_type_unknown").values()) > 0)

    with fmt_env("gen9randombattle"):
        t1, t2, src = ft.effective_types(fused)
        check_eq("не фьюжн-формат: тип не подменяется", (str(t1.name), str(t2.name)), ("GRASS", "FLYING"))
        check_eq("источник — dex", src, "dex")

    with config_fmt("gen9fusionmonsrandombattle"):
        check("формат из config подхватывается", ft.is_fusion_format())
    with config_fmt("gen9randombattle"):
        check("не фьюжн-формат из config не считается фьюжном", not ft.is_fusion_format())
    check("тег боя важнее конфига (fusion в теге)",
          ft.is_fusion_format(FakeBattle("battle-gen9fusionmonsrandombattle-123")))
    with fmt_env("gen9fusionmonsrandombattle"):
        check("тег боя важнее конфига (стоковый тег -> False)",
              not ft.is_fusion_format(FakeBattle("battle-gen9randombattle-123")))

    print("-" * 78)
    print("D. Фьюжн-тип доезжает до урона и признаков")
    print("-" * 78)

    with fmt_env("gen9fusionmonsrandombattle"):
        our = mk_mon("Tropius, L50, M", name="+Stonjourner")
        opp = mk_mon("Garchomp, L50, M", name="+Garchomp")
        our_prep, opp_prep = prepare_mon(our), prepare_mon(opp)
        check_eq("prepare_mon: типы фьюжна в prep",
                 tuple(str(t.name) for t in our_prep["types"]), ("GRASS", "ROCK"))
        check("prepare_mon: источник записан", str(our_prep["types_src"]).startswith("fusion:"),
              our_prep["types_src"])

        earthquake = Move("earthquake", gen=9)
        r_fused = estimate_damage(opp_prep, our_prep, earthquake)
        check("Ground по фьюжну Grass/Rock бьёт (нет ложного иммунитета)",
              r_fused is not None and r_fused[0] > 0, f"dmg={r_fused}")

        with fmt_env("gen9randombattle"):
            # готовим монов заново: prepare_mon кэширует типы, поэтому сравнение — на свежих
            dex_prep = prepare_mon(mk_mon("Tropius, L50, M", name="+Stonjourner"))
            atk_prep = prepare_mon(mk_mon("Garchomp, L50, M", name="+Garchomp"))
            stat = estimate_damage(atk_prep, dex_prep, earthquake)
            check_eq("без фьюжн-расчёта тот же удар считался бы иммунным (0.0)", stat, (0.0, 0.0))
            check_eq("в декс-формате prep остаётся дексовым",
                     tuple(str(t.name) for t in dex_prep["types"]), ("GRASS", "FLYING"))

        # STAB считается по фьюжн-типам атакующего
        toxic = Move("sludgebomb", gen=9)
        s_fused = estimate_damage(our_prep, opp_prep, toxic)
        check("STAB по фьюжн-типам атакующего считается", s_fused is not None and s_fused[0] > 0,
              f"dmg={s_fused}")

        # weakness/vulnerability по нашим покемонам вне поля (то, на что жаловались)
        bench = mk_mon("Tropius, L50, M", name="+Stonjourner")
        attacker = mk_mon("Zapdos, L50, M", name="+Zapdos")
        chart = GenData.from_gen(9).type_chart
        weak_fused = _weakness_score(bench, attacker, chart)
        vuln_fused = _vulnerability_frac([bench], attacker, chart)
        with fmt_env("gen9randombattle"):
            weak_dex = _weakness_score(bench, attacker, chart)
            vuln_dex = _vulnerability_frac([bench], attacker, chart)
        check("weakness скамейки считается по фьюжн-типу",
              abs(weak_fused - weak_dex) > 1e-6, f"фьюжн={weak_fused:.3f} декс={weak_dex:.3f}")
        check("vulnerability скамейки считается по фьюжн-типу",
              abs(vuln_fused - vuln_dex) > 1e-6, f"фьюжн={vuln_fused:.3f} декс={vuln_dex:.3f}")

    print("-" * 78)
    print("E. Статы фьюжна у скамейки: ключ-«тело» и коллизии")
    print("-" * 78)

    entry_body = {"base_stats": {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}}
    entry_other = {"base_stats": {"hp": 1, "atk": 1, "def": 1, "spa": 1, "spd": 1, "spe": 1}}
    bench = mk_mon("Tropius, L50, M", name="+Stonjourner")
    check("запись находится по виду-«телу» (ключ парсера из |switch|)",
          _fusion_entry_for(bench, {"stonjourner": entry_body}) is entry_body)
    check("составной ключ 'голова_тело' приоритетнее простого",
          _fusion_entry_for(bench, {"stonjourner": entry_other, "tropius_stonjourner": entry_body})
          is entry_body)
    check("точный ключ по голове по-прежнему работает",
          _fusion_entry_for(mk_mon("Stonjourner, L50, M"), {"stonjourner": entry_body}) is entry_body)
    check("нет записи -> None (статы из декса, без догадок)",
          _fusion_entry_for(bench, {"garchomp": entry_body}) is None)
    check("пустая карта -> None", _fusion_entry_for(bench, None) is None)

    print("-" * 78)
    print("F. Сквозная проверка obs: урон по НАШИМ покемонам считается по фьюжн-типу")
    print("-" * 78)

    def mk_battle():
        """Бой как его шлёт сервер: ident = "+Тело", details = голова."""
        b = Battle(battle_tag="battle-gen9fusionmonsrandombattle-9", username="Me",
                   logger=logging.getLogger("quiet"), gen=9)
        for m in (["", "player", "p1", "Me", "", ""], ["", "player", "p2", "Opp", "", ""],
                  ["", "start"],
                  ["", "switch", "p1a: +Stonjourner", "Tropius, L50, M", "100/100"],
                  ["", "switch", "p2a: +Garchomp", "Garchomp, L50, M", "100/100"]):
            b.parse_message(m)
        # наш резерв (фьюжн): poke-env добавляет монов в team по ident — делаем так же
        from poke_env.battle.pokemon import Pokemon
        b._team["p1: +Stonjourner2"] = Pokemon(details="Swampert, L50, M", name="+Stonjourner", gen=9)
        # приём противника модель знает (revealed): только Earthquake, чтобы проверить иммунитет
        b.opponent_active_pokemon._moves = MoveSet({"earthquake": Move("earthquake", gen=9)})
        return b

    b1 = mk_battle()
    with fmt_env("gen9fusionmonsrandombattle"):
        obs_fusion = embed_battle_with_fusion(b1, None, None)
        check_eq("в бою виден фьюжн-тип нашего активного",
                 ft.effective_type_names(b1.active_pokemon), ("GRASS", "ROCK"))
        check_eq("и фьюжн-тип нашего резерва (Swampert + Stonjourner)",
                 ft.effective_type_names(b1.team["p1: +Stonjourner2"]), ("WATER", "ROCK"))
    with fmt_env("gen9randombattle"):
        obs_dex = embed_battle_with_fusion(b1, None, None)
        check_eq("в декс-формате тот же бой даёт дексовый тип",
                 ft.effective_type_names(b1.active_pokemon), ("GRASS", "FLYING"))

    check_eq("obs одинаковой размерности", obs_fusion.shape, obs_dex.shape)
    # блок урона начинается сразу после префикса (за ним идёт блок типов соперника)
    base = int(MIN_PREFIX_OBS_DIM)
    check_eq("блок урона начинается ровно после префикса", int(base), 715)
    check_eq("за блоком урона идёт блок типов соперника",
             int(obs_fusion.shape[0]) - base - int(DAMAGE_BLOCK_SIZE), int(TYPE_MATCHUP_BLOCK_SIZE))

    d_best = float(np.abs(obs_fusion[base + 12:base + 15] - obs_dex[base + 12:base + 15]).max())
    check("лучший приём противника по нашему активному считается по фьюжн-типу",
          d_best > 1e-6, f"max|Δ| в блоке [12:15] = {d_best:.3f}")
    check_eq("Earthquake по фьюжну Grass/Rock НЕ иммунен (доля > 0)",
             bool(obs_fusion[base + 12] > 0), True)
    check_eq("а по дексовому Grass/Flying он был иммунным (0.0)",
             float(obs_dex[base + 12]), 0.0)
    d_matrix = float(np.abs(obs_fusion[base + 51:base + 87] - obs_dex[base + 51:base + 87]).max())
    check("матрица «их j -> наш i» считается по фьюжн-типам нашей команды",
          d_matrix > 1e-6, f"max|Δ| в блоке [51:87] = {d_matrix:.3f}")
    d_mirror = float(np.abs(obs_fusion[base + MIRROR_BASE:base + TEAM_BASE]
                            - obs_dex[base + MIRROR_BASE:base + TEAM_BASE]).max())
    check("зеркальный блок «их приёмы по нам» тоже (mirror)",
          d_mirror > 1e-6, f"max|Δ| в [{MIRROR_BASE}:{TEAM_BASE}] = {d_mirror:.3f}")
    # вне блока урона меняются только признаки, зависящие от типа мон (multi-hot типов, STAB)
    changed = np.where(np.abs(obs_fusion[:base] - obs_dex[:base]) > 1e-6)[0]
    check("вне блока урона меняются только type-зависимые колонки (немного)",
          0 < changed.size < 120,
          f"изменено {changed.size} колонок из {base}: {list(changed[:12])}"
          f"{'...' if changed.size > 12 else ''}")

    print("-" * 78)
    print("G. Диагностика: ident «+Тело» относится к своему мону")
    print("-" * 78)

    import diagnose_type_spam as diag
    fused_mon = mk_mon("Tropius, L50, M", name="+Stonjourner")
    check("typechange '+Stonjourner' — про этого мон (по телу)",
          diag._ident_matches_mon(fused_mon, "p1a: +Stonjourner"))
    check("typechange про чужого мон — не про этого",
          not diag._ident_matches_mon(fused_mon, "p1a: +Garchomp"))
    check("обычный случай (имя == вид) продолжает работать",
          diag._ident_matches_mon(mk_mon("Snorlax, L50, M"), "p2a: +Snorlax"))

    print("-" * 78)
    print("H. Кэш расчёта не путает монов одного вида с разными типами")
    print("-" * 78)

    import types as _types
    from poke_env.battle import PokemonType as PT

    def mk_ns(types, ability="sandveil"):
        return _types.SimpleNamespace(species="dfn", type_1=types[0],
                                      type_2=(types[1] if len(types) > 1 else None),
                                      types=[x for x in types if x], ability=ability, item=None,
                                      name=None)

    norm = mk_ns((PT.NORMAL, PT.POISON))
    water = mk_ns((PT.WATER,))
    check_eq("первый мон получил свои типы", ft.effective_type_names(norm), ("NORMAL", "POISON"))
    check_eq("второй мон с тем же видом — свои (кэш не перепутал)",
             ft.effective_type_names(water), ("WATER",))

    print("-" * 78)
    print("I. Кэши датасета/статистики инвалидируются при правке расчёта типов")
    print("-" * 78)

    import hashlib
    from agents.training import _features_fingerprint

    agents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents")

    def _fp(files):
        h = hashlib.md5()
        for n in files:
            with open(os.path.join(agents_dir, n), "rb") as f:
                h.update(f.read())
        return h.hexdigest()

    full = ("features.py", "damage.py", "fusion_types.py", "type_utils.py", "config.py")
    check_eq("fingerprint признаков == ручной пересчёт по тем же файлам",
             _features_fingerprint(), _fp(full))
    check("без fusion_types.py хеш другой (иначе кэши остались бы «валидными»)",
          _fp(("features.py", "damage.py", "type_utils.py", "config.py")) != _fp(full))

    print("=" * 78)
    if FAIL:
        print(f"ПРОВАЛЕНО ({len(FAIL)}): {FAIL}")
        return 1
    print(f"Все проверки пройдены ({OK}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
