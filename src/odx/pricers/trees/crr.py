"""American option pricer using a Cox Ross Rubinstein binomial tree.

This module prices vanilla American and European options (calls and puts) with an
optional continuous dividend yield.

American options can be exercised at any time up to expiry, so the tree uses
backward induction with early exercise at each node.

References:
    Cox, Ross, Rubinstein (1979)
    Hull, Options, Futures and Other Derivatives
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple, Union

import numpy as np


class OptionType(str, Enum):
    call = "call"
    put = "put"


class ExerciseStyle(str, Enum):
    american = "american"
    european = "european"


@dataclass(frozen=True)
class CRRResult:
    price: float
    delta: float
    gamma: float
    theta: float
    exercise_nodes: Optional[Sequence[Tuple[int, int]]] = None


def _parse_option_type(cp: Union[str, OptionType]) -> OptionType:
    if isinstance(cp, OptionType):
        return cp
    s = str(cp).strip().lower()
    if s in ("c", "call"):
        return OptionType.call
    if s in ("p", "put"):
        return OptionType.put
    raise ValueError("option_type must be 'call' or 'put'")


def _parse_exercise_style(style: Union[str, ExerciseStyle]) -> ExerciseStyle:
    if isinstance(style, ExerciseStyle):
        return style
    s = str(style).strip().lower()
    if s in ("am", "american"):
        return ExerciseStyle.american
    if s in ("eu", "european"):
        return ExerciseStyle.european
    raise ValueError("exercise_style must be 'american' or 'european'")


def _validate_inputs(S0: float, K: float, T: float, sigma: float, n_steps: int) -> None:
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")


class CRRBinomialTreePricer:
    """Cox Ross Rubinstein binomial tree pricer for vanilla options.

    Parameters
    ----------
    S0 : float
        Spot price
    K : float
        Strike
    T : float
        Time to expiry in years
    r : float
        Risk free rate
    sigma : float
        Volatility
    n_steps : int
        Number of binomial steps
    option_type : str
        'call' or 'put'
    exercise_style : str
        'american' or 'european'
    q : float
        Continuous dividend yield
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int = 200,
        option_type: Union[str, OptionType] = "call",
        exercise_style: Union[str, ExerciseStyle] = "american",
        q: float = 0.0,
    ) -> None:
        _validate_inputs(S0, K, T, sigma, n_steps)

        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.n_steps = int(n_steps)
        self.q = float(q)

        self.option_type = _parse_option_type(option_type)
        self.exercise_style = _parse_exercise_style(exercise_style)

        self.dt = self.T / self.n_steps
        self.u = float(np.exp(self.sigma * np.sqrt(self.dt)))
        self.d = 1.0 / self.u
        self.disc = float(np.exp(-self.r * self.dt))

        growth = float(np.exp((self.r - self.q) * self.dt))
        self.p = (growth - self.d) / (self.u - self.d)

        if not (0.0 < self.p < 1.0):
            raise ValueError(
                "Risk neutral probability is outside (0, 1). "
                "Try increasing n_steps or check inputs."
            )

    def _payoff(self, S: np.ndarray) -> np.ndarray:
        if self.option_type == OptionType.call:
            return np.maximum(S - self.K, 0.0)
        return np.maximum(self.K - S, 0.0)

    def _stock_nodes(self, step: int) -> np.ndarray:
        j = np.arange(step + 1, dtype=float)
        return self.S0 * (self.u ** j) * (self.d ** (step - j))

    def price(self, track_exercise: bool = False) -> CRRResult:
        n = self.n_steps

        S_T = self._stock_nodes(n)
        V = self._payoff(S_T)

        exercise_nodes: Optional[list[Tuple[int, int]]] = [] if track_exercise else None

        V1 = None
        S1 = None
        V2 = None
        S2 = None

        for i in range(n - 1, -1, -1):
            hold = self.disc * (self.p * V[1:] + (1.0 - self.p) * V[:-1])

            if self.exercise_style == ExerciseStyle.american:
                S_i = self._stock_nodes(i)
                ex = self._payoff(S_i)
                V_i = np.maximum(ex, hold)

                if track_exercise:
                    idx = np.where(ex > hold)[0]
                    for j in idx:
                        exercise_nodes.append((i, int(j)))
            else:
                V_i = hold

            if i == 2:
                V2 = V_i.copy()
                S2 = self._stock_nodes(2)
            elif i == 1:
                V1 = V_i.copy()
                S1 = self._stock_nodes(1)

            V = V_i

        price0 = float(V[0])

        delta, gamma, theta = self._greeks_from_levels(price0, V1, S1, V2, S2)

        return CRRResult(
            price=price0,
            delta=delta,
            gamma=gamma,
            theta=theta,
            exercise_nodes=exercise_nodes,
        )

    def _greeks_from_levels(
        self,
        V0: float,
        V1: Optional[np.ndarray],
        S1: Optional[np.ndarray],
        V2: Optional[np.ndarray],
        S2: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        delta = float("nan")
        gamma = float("nan")
        theta = float("nan")

        if V1 is not None and S1 is not None and len(V1) == 2:
            denom = S1[1] - S1[0]
            if denom != 0:
                delta = float((V1[1] - V1[0]) / denom)

        if V2 is not None and S2 is not None and len(V2) == 3:
            Su, Sm, Sd = float(S2[2]), float(S2[1]), float(S2[0])
            Vu, Vm, Vd = float(V2[2]), float(V2[1]), float(V2[0])

            denom_up = Su - Sm
            denom_dn = Sm - Sd
            if denom_up != 0 and denom_dn != 0:
                delta_up = (Vu - Vm) / denom_up
                delta_dn = (Vm - Vd) / denom_dn
                mid_width = (Su - Sd) / 2.0
                if mid_width != 0:
                    gamma = float((delta_up - delta_dn) / mid_width)

            theta = float((Vm - V0) / (2.0 * self.dt))

        return delta, gamma, theta


def crr_american_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_steps: int = 200,
    q: float = 0.0,
) -> float:
    pr = CRRBinomialTreePricer(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        n_steps=n_steps, option_type=option_type,
        exercise_style="american", q=q,
    )
    return pr.price().price


def crr_european_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_steps: int = 200,
    q: float = 0.0,
) -> float:
    pr = CRRBinomialTreePricer(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        n_steps=n_steps, option_type=option_type,
        exercise_style="european", q=q,
    )
    return pr.price().price
