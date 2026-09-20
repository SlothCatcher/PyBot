#!/usr/bin/env python3
"""Ортогональная инициализация ТОЛЬКО новых выходов (требование к BC -> PPO / resume).

Правило: если архитектура выходов чуть изменилась (голова шире числа действий, отсутствующий
в чекпоинте выходной слой, другой net_arch), новые выходные строки нельзя оставлять случайными
(скачок логитов) или нулевыми (мёртвая голова с одинаковыми логитами). Их инициализируют
ортогонально с малым gain, а СТАРЫЕ строки обязаны остаться бит-в-бит.

Что проверяется:
  A) `ortho_init_new_rows_`: новые строки ортогональны (W Wᵀ ≈ g²·I), старые — бит-в-бит,
     bias новых строк = 0, повторный вызов при отсутствии новых строк ничего не делает;
  B) `reinit_new_outputs_`: выросшая голова (4 новых действия из 26), отсутствующий value_net,
     отсутствующий последний слой pi/vf, выросший log_std (std=1 у новых действий);
  C) политика в штатном состоянии (все выходы на месте) НЕ переинициализируется — пустой список
     правок, значит на BC -> PPO ничего не «пересобирается молча»;
  D) живой BC -> PPO на настоящей политике: веса после перехода бит-в-бит равны весам на выходе BC
     (никаких орто-правок), а оптимизатор при этом новый (моменты не переносятся).

Запуск: PYTHONPATH=. python test_output_init.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gymnasium import spaces

from agents.config import N_FEATURES
from agents.policy import MaskedActorCriticPolicy, ortho_init_new_rows_, reinit_new_outputs_
from agents.optim import SplitAdam

OK = 0
FAIL: list = []


def check(name, cond, extra=""):
    global OK
    if cond:
        OK += 1
        print(f"OK   {name}" + (f": {extra}" if extra else ""))
    else:
        FAIL.append(name)
        print(f"FAIL {name}" + (f": {extra}" if extra else ""))


def check_close(name, got, want, tol=1e-9):
    d = abs(float(got) - float(want))
    check(name, d <= tol, f"got={float(got):.6g} want={float(want):.6g} (Δ={d:.2e})")


def make_policy(features_dim=64, action_dim=26, **optimizer_kwargs):
    obs_space = spaces.Dict({
        "observation": spaces.Box(-1.0, 4.0, shape=(N_FEATURES,), dtype=np.float32),
        "action_mask": spaces.Box(0, 1, shape=(action_dim,), dtype=np.int8),
    })
    opt_kwargs = dict(optimizer_kwargs)
    opt_kwargs.pop("optimizer_class", None)
    return MaskedActorCriticPolicy(
        obs_space, spaces.Discrete(action_dim), lr_schedule=lambda _p: 2e-4,
        features_extractor_kwargs=dict(features_dim=features_dim),
        optimizer_class=SplitAdam, optimizer_kwargs=opt_kwargs,
    )


def obs_batch(n=4, action_dim=26, seed=0):
    g = torch.Generator().manual_seed(int(seed))
    return {
        "observation": torch.randn((n, N_FEATURES), generator=g, dtype=torch.float32),
        "action_mask": torch.ones((n, action_dim), dtype=torch.int8),
    }


def part_a_helper():
    print("=" * 78)
    print("A. ortho_init_new_rows_: орто только для новых строк, старые не трогаются")
    print("=" * 78)
    torch.manual_seed(0)
    old_rows, new_rows = 6, 4
    w = torch.randn(old_rows + new_rows, 5)
    bias = torch.randn(old_rows + new_rows)
    w_before, bias_before = w.clone(), bias.clone()

    n = ortho_init_new_rows_(w, old_rows, gain=0.01, bias=bias)
    check("A: вернулось число новых строк", n == new_rows, f"{n}")
    check("A: старые строки веса бит-в-бит",
          torch.equal(w[:old_rows], w_before[:old_rows]))
    check("A: старые bias бит-в-бит", torch.equal(bias[:old_rows], bias_before[:old_rows]))
    check("A: bias новых строк занулён", bool((bias[old_rows:] == 0).all()))
    block = w[old_rows:]
    gram = block @ block.t()
    expected = torch.eye(new_rows) * (0.01 ** 2)
    check_close("A: новые строки ортонормальны с gain=0.01 (max|Δ|)",
                float((gram - expected).abs().max()), 0.0, tol=1e-10)
    check("A: новые строки НЕ равны старым (была реальная инициализация)",
          not torch.allclose(block, w_before[old_rows:]))

    w2 = w.clone()
    check("A: при отсутствии новых строк функция ничего не делает",
          ortho_init_new_rows_(w2, old_rows=w2.shape[0], gain=0.01) == 0
          and torch.equal(w2, w))
    check("A: вырожденный вход (не 2D) не падает",
          ortho_init_new_rows_(torch.randn(3), 1, gain=0.01) == 0)


def part_b_reinit_outputs():
    print("=" * 78)
    print("B. reinit_new_outputs_: выросшая голова / отсутствующие выходы / log_std")
    print("=" * 78)
    torch.manual_seed(1)
    policy = make_policy(features_dim=64)

    # (1) голова «выросла» с 22 до 26 действий: 4 новых строки — ortho, 22 старые — бит-в-бит
    old_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    old_state["action_net.weight"] = old_state["action_net.weight"][:22].clone()
    old_state["action_net.bias"] = old_state["action_net.bias"][:22].clone()
    with torch.no_grad():
        policy.action_net.weight.data[:22] = torch.randn(22, old_state["action_net.weight"].shape[1]) * 0.1
        policy.action_net.bias.data[:22] = torch.randn(22) * 0.1
    others = {k: v.detach().clone() for k, v in policy.state_dict().items()
              if not k.startswith("action_net")}
    before = policy.action_net.weight.detach().clone()
    notes = reinit_new_outputs_(policy, old_state=old_state, verbose=False)
    check("B: правка только по action_net", notes and all("action_net" in n for n in notes), str(notes))
    check("B: старые 22 строки сохранены бит-в-бит",
          torch.equal(policy.action_net.weight.detach()[:22], before[:22]))
    gram = policy.action_net.weight.detach()[22:] @ policy.action_net.weight.detach()[22:].t()
    check_close("B: 4 новых выхода ортонормальны gain=0.01",
                float((gram - torch.eye(4) * 1e-4).abs().max()), 0.0, tol=1e-10)
    check("B: bias новых выходов = 0", bool((policy.action_net.bias.detach()[22:] == 0).all()))
    check("B: остальные выходы (value_net, mlp pi/vf) не тронуты",
          all(torch.equal(others[k], v) for k, v in policy.state_dict().items() if k in others))

    # (1b) неполный old_state БЕЗ missing_keys -> не переинициализируем ничего (защита весов)
    policy_ns = make_policy(features_dim=64)
    snap_ns = {k: v.detach().clone() for k, v in policy_ns.state_dict().items()}
    notes_ns = reinit_new_outputs_(policy_ns, old_state={"action_net.weight": torch.zeros(22, 64)},
                                   verbose=False)
    check("B: без missing_keys неизвестные выходы не трогаются",
          all("value_net" not in n and "policy_net" not in n and "value_net.last" not in n
              for n in notes_ns), str(notes_ns))
    check("B: value_net и mlp остались бит-в-бит",
          all(torch.equal(snap_ns[k], v) for k, v in policy_ns.state_dict().items()
              if not k.startswith("action_net")))

    # (2) value_net отсутствовал в чекпоинте целиком -> полная ortho-инициализация
    policy2 = make_policy(features_dim=64)
    snap2 = {k: v.detach().clone() for k, v in policy2.state_dict().items()}
    policy2.value_net.weight.data.fill_(0.0)
    notes2 = reinit_new_outputs_(policy2, old_state={k: v for k, v in snap2.items()
                                                     if not k.startswith("value_net")},
                                 missing_keys=["value_net.weight", "value_net.bias"], verbose=False)
    check("B: отсутствующий value_net помечен как новый выход",
          any("value_net" in n for n in notes2), str(notes2))
    wv = policy2.value_net.weight.detach()
    check_close("B: value_net проинициализирован ортогонально (норма строки = 0.01)",
                float(wv.norm(dim=1).mean()), 0.01, tol=1e-6)
    check("B: value_net не остался нулевым", float(wv.abs().sum()) > 0)
    check("B: action_net при этом бит-в-бит",
          torch.equal(snap2["action_net.weight"], policy2.action_net.weight.detach()))

    # (3) отсутствующий последний слой pi-блока (другая net_arch) -> ortho с gain=sqrt(2)
    policy3 = make_policy(features_dim=64)
    snap3 = {k: v.detach().clone() for k, v in policy3.state_dict().items()}
    pi_last = [k for k in snap3 if k.startswith("mlp_extractor.policy_net.") and k.endswith(".weight")][-1]
    notes3 = reinit_new_outputs_(policy3,
                                 old_state={k: v for k, v in snap3.items() if k != pi_last},
                                 missing_keys=[pi_last], verbose=False)
    check("B: новый слой pi помечен", any("policy_net" in n for n in notes3), f"{pi_last} -> {notes3}")
    new_w = policy3.state_dict()[pi_last]
    check_close("B: новый слой pi ортогонален (gain=sqrt(2), норма строки)",
                float(new_w.norm(dim=1).mean()), float(np.sqrt(2)), tol=1e-4)
    check("B: value_net не тронут",
          torch.equal(snap3["value_net.weight"], policy3.value_net.weight.detach()))


def part_c_no_silent_reinit():
    print("=" * 78)
    print("C. Штатная политика: выходы не переинициализируются (никаких молчаливых правок)")
    print("=" * 78)
    torch.manual_seed(2)
    policy = make_policy(features_dim=64)
    state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    notes = reinit_new_outputs_(policy, old_state=state, verbose=False)
    check("C: правок нет (пустой список)", notes == [], str(notes))
    same = all(torch.equal(before[k], v) for k, v in policy.state_dict().items())
    check("C: все веса бит-в-бит", same)


def part_d_bc_transition():
    print("=" * 78)
    print("D. Живой BC -> PPO: веса не переинициализируются, а оптимизатор — новый")
    print("=" * 78)
    from agents.training import pretrain_policy_bc
    from agents.optim import SplitAdam as _SA

    torch.manual_seed(3)
    policy = make_policy(features_dim=64)
    ppo_like = type("P", (), {})()
    ppo_like.policy = policy
    ppo_like.learning_rate = 2e-4

    n = 32
    dataset = []
    for i in range(n):
        obs = obs_batch(1)[ "observation"][0]
        dataset.append((obs, np.ones(26, dtype=np.int8), int(i % 26), float(np.random.randn())))

    pretrain_policy_bc(ppo_like, dataset, epochs=1, batch_size=8, normalize=False,
                       lr=1e-3, lr_final=1e-4, reset_optimizer=True, rl_lr=3e-4,
                       rl_eps=1e-5, verbose=False)
    snap = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    notes = reinit_new_outputs_(policy, old_state=snap, verbose=False)
    check("D: после BC -> PPO выходы не переинициализируются", notes == [], str(notes))
    check("D: оптимизатор — новый SplitAdam", isinstance(policy.optimizer, _SA),
          type(policy.optimizer).__name__)
    check("D: моменты BC не перенесены", len(policy.optimizer.state) == 0,
          str(len(policy.optimizer.state)))
    check_close("D: eps нового Adam = 1e-5 (правило PPO)",
                policy.optimizer.eps_by_group()["policy"], 1e-5, tol=1e-12)
    check_close("D: lr нового Adam = RL-значение 3e-4",
                policy.optimizer.lr_by_group()["policy"], 3e-4, tol=1e-12)
    head_before = snap["action_net.weight"]
    check("D: голова действий осталась бит-в-бит",
          torch.equal(head_before, policy.state_dict()["action_net.weight"]))


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    part_a_helper()
    print("-" * 78)
    part_b_reinit_outputs()
    print("-" * 78)
    part_c_no_silent_reinit()
    print("-" * 78)
    part_d_bc_transition()
    print("=" * 78)
    if FAIL:
        print(f"ПРОВАЛЕНО ({len(FAIL)}): {FAIL}")
        return 1
    print(f"Все проверки пройдены ({OK}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
