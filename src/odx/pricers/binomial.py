"""CRR binomial tree pricer."""

from __future__ import annotations

import numpy as np


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int = 100,
    option_type: str = "call",
    exercise_style: str = "american",
    q: float = 0.0,
) -> float:
    """CRR binomial tree for European or American options."""
    is_call = option_type.strip().lower() in ("call", "c")
    is_american = exercise_style.strip().lower() in ("american", "am")

    if T <= 0:
        payoff = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return float(payoff)

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)

    growth = np.exp((r - q) * dt)
    p = (growth - d) / (u - d)

    j = np.arange(n_steps + 1)
    S_T = S * (u ** j) * (d ** (n_steps - j))
    V = np.maximum(S_T - K, 0.0) if is_call else np.maximum(K - S_T, 0.0)

    for i in range(n_steps - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        if is_american:
            j = np.arange(i + 1)
            S_i = S * (u ** j) * (d ** (i - j))
            payoff = np.maximum(S_i - K, 0.0) if is_call else np.maximum(K - S_i, 0.0)
            V = np.maximum(V, payoff)

    return float(V[0])
