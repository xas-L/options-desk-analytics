"""Numerical utilities: safe math functions and finite-difference Greeks."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

_LOG_FLOOR = 1e-300


def safe_log(x: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.log(np.maximum(arr, _LOG_FLOOR))


def safe_sqrt(x: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.sqrt(np.maximum(arr, 0.0))


def finite_difference_delta(
    pricer_fn: Callable[..., float],
    S: float,
    h: float = 0.01,
    **pricer_kwargs: object,
) -> float:
    v_up = pricer_fn(S=S + h, **pricer_kwargs)
    v_dn = pricer_fn(S=S - h, **pricer_kwargs)
    return (v_up - v_dn) / (2.0 * h)


def finite_difference_gamma(
    pricer_fn: Callable[..., float],
    S: float,
    h: float = 0.01,
    **pricer_kwargs: object,
) -> float:
    v_up = pricer_fn(S=S + h, **pricer_kwargs)
    v_0 = pricer_fn(S=S, **pricer_kwargs)
    v_dn = pricer_fn(S=S - h, **pricer_kwargs)
    return (v_up - 2.0 * v_0 + v_dn) / (h * h)


def finite_difference_vega(
    pricer_fn: Callable[..., float],
    sigma: float,
    h: float = 0.001,
    **pricer_kwargs: object,
) -> float:
    v_up = pricer_fn(sigma=sigma + h, **pricer_kwargs)
    v_dn = pricer_fn(sigma=sigma - h, **pricer_kwargs)
    return (v_up - v_dn) / (2.0 * h)
