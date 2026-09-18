"""Тест-детектор затенения имён: локальный импорт/присвоение внутри функции, которые
делают имя локальным, а использование стоит раньше определения -> UnboundLocalError.

Именно этот баг поймал пользователь на resume:
    run() ... if _resume_dim != N_FEATURES:   # raньше локального импорта
             ...
             from agents.config import N_FEATURES   # делает N_FEATURES локальной для run()

Тест статический (ast), сервера/моделей не требует. Запуск:
    PYTHONPATH=/home/user/PyBot /tmp/venv_pe/bin/python test_no_shadowing.py
"""
import ast
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {".git", "__pycache__", "models", "tb_logs", "venv", ".venv"}

OK, FAIL = [], []


def check(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f": {extra}" if extra else ""))


def module_globals(tree: ast.Module) -> set:
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name != "*":
                    names.add(a.asname or a.name)
    return names


def _own_nodes(node):
    """Узлы тела функции БЕЗ вложенных областей (вложенные def/class/lambda имеют свои)."""
    out = []
    stack = list(getattr(node, "body", []))
    while stack:
        cur = stack.pop()
        out.append(cur)
        for child in ast.iter_child_nodes(cur):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            stack.append(child)
    return out


def _assigned_names(node):
    """Имена, которые становятся локальными в этой функции (без вложенных областей)."""
    defined = {}
    for arg in list(getattr(node.args, "posonlyargs", [])) + list(node.args.args) + list(node.args.kwonlyargs):
        defined[arg.arg] = min(defined.get(arg.arg, 10**9), node.lineno)
    if node.args.vararg:
        defined[node.args.vararg.arg] = min(defined.get(node.args.vararg.arg, 10**9), node.lineno)
    if node.args.kwarg:
        defined[node.args.kwarg.arg] = min(defined.get(node.args.kwarg.arg, 10**9), node.lineno)

    for child in _own_nodes(node):
        targets = []
        if isinstance(child, ast.Assign):
            targets = child.targets
        elif isinstance(child, ast.AnnAssign):
            targets = [child.target]
        elif isinstance(child, ast.AugAssign):
            targets = [child.target]
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            targets = [child.target]
        elif isinstance(child, ast.withitem):
            targets = [child.optional_vars]
        elif isinstance(child, ast.ExceptHandler) and child.name:
            defined[child.name] = min(defined.get(child.name, 10**9), child.lineno)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for a in child.names:
                if a.name == "*":
                    continue
                nm = a.asname or a.name.split(".")[0]
                defined[nm] = min(defined.get(nm, 10**9), child.lineno)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            for nm in child.names:
                defined.pop(nm, None)
        line = getattr(child, "lineno", None)
        if line is None and targets:
            line = getattr(targets[0], "lineno", None)
        for t in targets:
            if t is None or line is None:
                continue
            for sub in ast.walk(t):
                if isinstance(sub, ast.Name):
                    defined[sub.id] = min(defined.get(sub.id, 10**9), line)
    return defined


def find_shadow_problems(path: str):
    """Возвращает список (имя, строка использования, строка определения, функция)."""
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    globs = module_globals(tree)
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        defined = _assigned_names(node)
        shadowing = {k: v for k, v in defined.items() if k in globs}
        if not shadowing:
            continue
        for child in _own_nodes(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                first_def = shadowing.get(child.id)
                if first_def is not None and child.lineno < first_def:
                    problems.append((child.id, child.lineno, first_def, node.name))
    return problems


def check_file(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    try:
        ast.parse(src)
    except SyntaxError as e:
        check(f"{path}: парсится", False, str(e))
        return
    problems = find_shadow_problems(path)
    if problems:
        detail = "; ".join(f"{nm} (исп. стр.{u} < опр. стр.{d}, функция {fn})"
                           for nm, u, d, fn in problems[:4])
        check(f"{os.path.relpath(path, ROOT)}: нет использования локального имени до определения",
              False, detail)
    else:
        check(f"{os.path.relpath(path, ROOT)}: нет использования локального имени до определения", True)


def test_detector_catches_bug():
    """Детектор обязан ловить ровно тот баг, который поймал пользователь на resume."""
    buggy = """import os
from agents.config import N_FEATURES
from stable_baselines3 import PPO


def run(resume_from):
    dim = 715
    if dim != N_FEATURES:
        print("migrate", N_FEATURES)
    from agents.config import N_FEATURES
    return PPO
"""
    clean = """import os
from agents.config import N_FEATURES


def run(dim):
    if dim != N_FEATURES:
        print("ok", N_FEATURES)
"""
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "bad.py")
        good = os.path.join(td, "good.py")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write(buggy)
        with open(good, "w", encoding="utf-8") as fh:
            fh.write(clean)
        caught = find_shadow_problems(bad)
        check("детектор ловит затенение N_FEATURES (тот самый UnboundLocalError)",
              any(nm == "N_FEATURES" and fn == "run" for nm, _, _, fn in caught), str(caught))
        check("детектор не ругается на корректный код", find_shadow_problems(good) == [])


def test_resolve_checkpoint_path():
    from agents.policy_player import explain_missing_checkpoint, resolve_checkpoint_path

    with tempfile.TemporaryDirectory() as td:
        plain = os.path.join(td, "model")
        zipped = plain + ".zip"
        with open(zipped, "wb") as fh:
            fh.write(b"x")
        check("resolve_checkpoint_path дополняет .zip", resolve_checkpoint_path(plain) == zipped)
        check("resolve_checkpoint_path не трогает существующий путь",
              resolve_checkpoint_path(zipped) == zipped)
        check("resolve_checkpoint_path не выдумывает путь для несуществующего файла",
              resolve_checkpoint_path(os.path.join(td, "nope")) == os.path.join(td, "nope"))
        hint = explain_missing_checkpoint(plain)
        check("подсказка про отсутствующий снапшот упоминает найденные файлы",
              "model.zip" in hint and "не найден" in hint, hint.replace("\n", " | ")[:90])


def main():
    files = []
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for nm in sorted(names):
            if nm.endswith(".py"):
                files.append(os.path.join(base, nm))
    print(f"Проверяю {len(files)} файлов на затенение имён...")
    for f in files:
        check_file(f)
    test_detector_catches_bug()
    test_resolve_checkpoint_path()

    print("-" * 74)
    if FAIL:
        print(f"ПРОВАЛЕНО: {len(FAIL)} -> {FAIL}")
        return 1
    print(f"Все проверки пройдены ({len(OK)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
