"""Monte Carlo pricer for barrier options under risk-neutral GBM.

Supports all eight barrier types (down/up, in/out, call/put).

Key implementation notes
------------------------
Path generation
    Log-returns are accumulated with np.cumsum so the entire path matrix is
    built in two vectorised operations with no Python loop over time steps.

Antithetic variates
    Half the standard-normal draws Z are generated; the antithetic half is
    simply -Z.  Both halves are stacked before the path matrix is built, so
    there is exactly one call to np.random.normal regardless of whether
    antithetic variates are enabled.

Payoff calculation
    Barrier breach is detected with a single np.min / np.max reduction over
    the path axis — one call, all paths simultaneously.  No Python loop over
    paths; no Python loop over time steps inside the payoff routine.

Continuity correction
    The Broadie-Glasserman-Kou correction adjusts the barrier level ONCE,
    before paths are generated, using the actual sigma passed to the pricer.
    The original code hardcoded sigma=0.2 inside the per-path payoff method;
    that bug is fixed here.

Barrier hit statistic
    Derived from the boolean array already produced by the payoff routine.
    No second loop over paths.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Option type helpers
# ---------------------------------------------------------------------------

_VALID_OPTION_TYPES = frozenset([
    "down_and_out_call", "down_and_out_put",
    "up_and_out_call",   "up_and_out_put",
    "down_and_in_call",  "down_and_in_put",
    "up_and_in_call",    "up_and_in_put",
])


def _parse_option_type(option_type: str) -> Tuple[bool, bool, bool]:
    """Return (is_down, is_out, is_call) for a valid option type string."""
    ot = option_type.strip().lower()
    if ot not in _VALID_OPTION_TYPES:
        raise ValueError(
            f"Invalid option_type '{option_type}'. "
            f"Must be one of: {sorted(_VALID_OPTION_TYPES)}"
        )
    return "down" in ot, "out" in ot, "call" in ot


# ---------------------------------------------------------------------------
# Vectorised path generation
# ---------------------------------------------------------------------------

def _simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    N_sim: int,
    N_steps: int,
    Z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate GBM paths from a pre-supplied or freshly drawn normal matrix.

    Parameters
    ----------
    S0, r, sigma, T : float
        Standard GBM parameters.
    N_sim : int
        Number of paths to generate.
    N_steps : int
        Number of time steps.
    Z : ndarray of shape (N_sim, N_steps), optional
        Pre-drawn standard normals.  If None, drawn internally.

    Returns
    -------
    paths : ndarray of shape (N_sim, N_steps + 1)
    """
    dt = T / N_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    if Z is None:
        Z = np.random.standard_normal((N_sim, N_steps))

    # Accumulate log-returns; prepend column of zeros for S0
    log_increments = drift + diffusion * Z                        # (N_sim, N_steps)
    log_paths = np.empty((N_sim, N_steps + 1), dtype=np.float64)
    log_paths[:, 0] = 0.0
    np.cumsum(log_increments, axis=1, out=log_paths[:, 1:])
    return S0 * np.exp(log_paths)


def _draw_normals_with_antithetic(N_sim: int, N_steps: int) -> np.ndarray:
    """Return (N_sim, N_steps) standard normals; bottom half is antithetic.

    If N_sim is odd the last path has no antithetic pair and a fresh draw is
    appended so the returned matrix always has exactly N_sim rows.
    """
    half = N_sim // 2
    Z_half = np.random.standard_normal((half, N_steps))
    Z = np.vstack([Z_half, -Z_half])
    if N_sim % 2 == 1:                                           # odd sim count
        Z = np.vstack([Z, np.random.standard_normal((1, N_steps))])
    return Z


# ---------------------------------------------------------------------------
# Continuity correction (Broadie-Glasserman-Kou 1997)
# ---------------------------------------------------------------------------

_BGK_BETA = 0.5826  # beta = -zeta(1/2) / sqrt(2*pi)


def apply_continuity_correction(
    B: float,
    sigma: float,
    T: float,
    N_steps: int,
    option_type: str,
) -> float:
    """Adjust the barrier level to approximate continuous monitoring.

    The BGK correction shifts B by exp(±beta * sigma * sqrt(dt)) so that the
    discretely monitored price matches the continuously monitored price to
    first order.

    Direction of shift
    ------------------
    Down barriers are shifted inward (lowered for out, raised for in) so that
    the discrete simulation is less likely to miss a crossing near the barrier.
    Up barriers are shifted in the opposite sense.

    Parameters
    ----------
    B : float
        Unadjusted barrier level.
    sigma : float
        Annualised volatility — must be the same sigma used to simulate paths.
    T, N_steps : float, int
        Used to compute dt = T / N_steps.
    option_type : str
        One of the eight valid barrier type strings.

    Returns
    -------
    float
        Adjusted barrier level.
    """
    dt = T / N_steps
    eps = _BGK_BETA * sigma * np.sqrt(dt)
    is_down, is_out, _ = _parse_option_type(option_type)

    if is_down:
        # down-out: lower barrier (paths less likely to breach by rounding)
        # down-in:  raise barrier (symmetric argument)
        sign = -1.0 if is_out else +1.0
    else:
        # up-out: raise barrier
        # up-in:  lower barrier
        sign = +1.0 if is_out else -1.0

    return B * np.exp(sign * eps)


# ---------------------------------------------------------------------------
# Vectorised payoff calculation
# ---------------------------------------------------------------------------

def _compute_payoffs(
    paths: np.ndarray,
    K: float,
    B_eff: float,
    is_down: bool,
    is_out: bool,
    is_call: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute discounted-payoff numerator and barrier indicator for all paths.

    Parameters
    ----------
    paths : ndarray of shape (N_sim, N_steps + 1)
    K : float
        Strike.
    B_eff : float
        Effective barrier (already continuity-corrected if requested).
    is_down, is_out, is_call : bool
        Option type flags from ``_parse_option_type``.

    Returns
    -------
    payoffs : ndarray of shape (N_sim,)
        Undiscounted payoff for each path.
    barrier_crossed : ndarray of bool, shape (N_sim,)
        True where the barrier was breached on that path.
    """
    # ------------------------------------------------------------------
    # 1. Barrier breach: one reduction over the time axis — no path loop
    # ------------------------------------------------------------------
    # Note: we check the *monitoring nodes* only (columns 1..N_steps for
    # discrete monitoring).  Column 0 is S0 which by construction satisfies
    # the barrier condition for any sensible parameter set; including it would
    # silently knock out every path if S0 == B.  We exclude it to match the
    # standard discrete-monitoring convention.
    monitoring_prices = paths[:, 1:]   # shape (N_sim, N_steps)

    if is_down:
        barrier_crossed = monitoring_prices.min(axis=1) <= B_eff
    else:
        barrier_crossed = monitoring_prices.max(axis=1) >= B_eff

    # ------------------------------------------------------------------
    # 2. Terminal payoff
    # ------------------------------------------------------------------
    S_T = paths[:, -1]
    if is_call:
        intrinsic = np.maximum(S_T - K, 0.0)
    else:
        intrinsic = np.maximum(K - S_T, 0.0)

    # ------------------------------------------------------------------
    # 3. Apply barrier condition
    # ------------------------------------------------------------------
    if is_out:
        payoffs = intrinsic * (~barrier_crossed)
    else:
        payoffs = intrinsic * barrier_crossed

    return payoffs, barrier_crossed


# ---------------------------------------------------------------------------
# Main pricer class
# ---------------------------------------------------------------------------

class BarrierOptionsPricer:
    """Monte Carlo pricer for barrier options under risk-neutral GBM.

    Supports all eight barrier types (down/up × in/out × call/put).
    All heavy computation is vectorised over paths; there are no Python-level
    loops over paths or time steps in the hot path.
    """

    valid_option_types: List[str] = sorted(_VALID_OPTION_TYPES)

    # ------------------------------------------------------------------
    # Public path-generation wrapper (kept for notebook use)
    # ------------------------------------------------------------------

    def simulate_gbm_paths(
        self,
        S0: float,
        r: float,
        sigma: float,
        T: float,
        N_sim: int,
        N_steps: int,
    ) -> np.ndarray:
        """Simulate GBM price paths.

        Parameters
        ----------
        S0 : float
            Initial asset price.
        r : float
            Annualised risk-free rate.
        sigma : float
            Annualised volatility.
        T : float
            Time to maturity in years.
        N_sim : int
            Number of Monte Carlo paths.
        N_steps : int
            Number of discrete time steps.

        Returns
        -------
        ndarray of shape (N_sim, N_steps + 1)
        """
        return _simulate_gbm_paths(S0, r, sigma, T, N_sim, N_steps)

    # ------------------------------------------------------------------
    # Continuity correction (public wrapper)
    # ------------------------------------------------------------------

    def apply_continuity_correction(
        self,
        B: float,
        sigma: float,
        T: float,
        N_steps: int,
        option_type: str,
    ) -> float:
        """Return the BGK-adjusted barrier level. See module-level docstring."""
        return apply_continuity_correction(B, sigma, T, N_steps, option_type)

    # ------------------------------------------------------------------
    # Core Monte Carlo pricer
    # ------------------------------------------------------------------

    def monte_carlo_pricer(
        self,
        S0: float,
        K: float,
        B: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        N_sim: int,
        N_steps: int,
        monitoring_type: str = "discrete",
        confidence_level: float = 0.95,
        antithetic: bool = False,
    ) -> Tuple[float, float, float, Dict]:
        """Price a barrier option by Monte Carlo simulation.

        Parameters
        ----------
        S0 : float
            Initial asset price.
        K : float
            Strike price.
        B : float
            Barrier level.
        T : float
            Time to maturity in years.
        r : float
            Annualised risk-free rate.
        sigma : float
            Annualised volatility.
        option_type : str
            One of: down_and_out_call, down_and_out_put, up_and_out_call,
            up_and_out_put, down_and_in_call, down_and_in_put,
            up_and_in_call, up_and_in_put.
        N_sim : int
            Total number of simulation paths (including antithetic pairs).
        N_steps : int
            Number of discrete monitoring steps.
        monitoring_type : str
            ``'discrete'`` — check barrier only at the N_steps monitoring
            dates.  ``'continuous_approx'`` — apply the BGK continuity
            correction to B so that the discrete simulation approximates
            continuous monitoring.
        confidence_level : float
            Coverage for the reported confidence interval (default 0.95).
        antithetic : bool
            If True, half the paths use negated normal draws (antithetic
            variates) for variance reduction at no extra cost.

        Returns
        -------
        price : float
            Discounted Monte Carlo estimate.
        ci_lower, ci_upper : float
            Confidence interval bounds at ``confidence_level``.
        stats : dict
            Diagnostic statistics — see keys below.

        Stats keys
        ----------
        mean_payoff, std_payoff, standard_error, mc_error,
        barrier_hit_percentage, computation_time, effective_simulations,
        convergence_ratio
        """
        t0 = time.perf_counter()

        is_down, is_out, is_call = _parse_option_type(option_type)

        # ------------------------------------------------------------------
        # 1. Continuity correction — applied ONCE here with the real sigma,
        #    not inside a per-path loop with a hardcoded dummy value.
        # ------------------------------------------------------------------
        if monitoring_type == "continuous_approx":
            B_eff = apply_continuity_correction(B, sigma, T, N_steps, option_type)
        elif monitoring_type == "discrete":
            B_eff = float(B)
        else:
            raise ValueError(
                f"monitoring_type must be 'discrete' or 'continuous_approx', "
                f"got '{monitoring_type}'"
            )

        # ------------------------------------------------------------------
        # 2. Draw random normals — antithetic pairs built here, not by
        #    extracting and negating log-returns from already-generated paths.
        # ------------------------------------------------------------------
        if antithetic:
            Z = _draw_normals_with_antithetic(N_sim, N_steps)
        else:
            Z = np.random.standard_normal((N_sim, N_steps))

        # ------------------------------------------------------------------
        # 3. Generate all paths in one vectorised call.
        # ------------------------------------------------------------------
        paths = _simulate_gbm_paths(S0, r, sigma, T, N_sim, N_steps, Z=Z)

        # ------------------------------------------------------------------
        # 4. Compute payoffs and barrier indicator — fully vectorised,
        #    no Python loop over paths or time steps.
        # ------------------------------------------------------------------
        payoffs, barrier_crossed = _compute_payoffs(
            paths, K, B_eff, is_down, is_out, is_call
        )

        # ------------------------------------------------------------------
        # 5. Price and confidence interval.
        # ------------------------------------------------------------------
        disc = np.exp(-r * T)
        mean_payoff = float(payoffs.mean())
        price = disc * mean_payoff

        std_payoff = float(payoffs.std(ddof=1))
        n = len(payoffs)
        se = std_payoff / np.sqrt(n)
        mc_error = disc * se

        alpha = 1.0 - confidence_level
        z = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))
        ci_lower = price - z * mc_error
        ci_upper = price + z * mc_error

        # ------------------------------------------------------------------
        # 6. Diagnostics — barrier hit percentage from the boolean array
        #    already computed in step 4; no second loop needed.
        # ------------------------------------------------------------------
        barrier_hit_pct = float(barrier_crossed.mean()) * 100.0
        computation_time = time.perf_counter() - t0

        diagnostics: Dict = {
            "mean_payoff": mean_payoff,
            "std_payoff": std_payoff,
            "standard_error": se,
            "mc_error": mc_error,
            "barrier_hit_percentage": barrier_hit_pct,
            "computation_time": computation_time,
            "effective_simulations": n,
            "convergence_ratio": mc_error / price if price > 0.0 else float("inf"),
        }

        return price, ci_lower, ci_upper, diagnostics

    # ------------------------------------------------------------------
    # Convergence analysis
    # ------------------------------------------------------------------

    def analyze_convergence(
        self,
        S0: float,
        K: float,
        B: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        N_steps: int,
        sim_counts: Optional[List[int]] = None,
        monitoring_type: str = "discrete",
    ) -> Dict:
        """Price the option at increasing simulation counts to study convergence.

        Parameters
        ----------
        sim_counts : list of int, optional
            Simulation counts to sweep.  Defaults to
            [1_000, 5_000, 10_000, 50_000, 100_000, 500_000].

        Returns
        -------
        dict with keys: sim_counts, prices, errors, computation_times
        """
        if sim_counts is None:
            sim_counts = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

        prices, errors, times = [], [], []
        for n in sim_counts:
            p, _, _, d = self.monte_carlo_pricer(
                S0, K, B, T, r, sigma, option_type, n, N_steps, monitoring_type
            )
            prices.append(p)
            errors.append(d["mc_error"])
            times.append(d["computation_time"])

        return {
            "sim_counts": sim_counts,
            "prices": prices,
            "errors": errors,
            "computation_times": times,
        }

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        S0: float,
        K: float,
        B: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        N_sim: int,
        N_steps: int,
        parameter: str,
        range_pct: float = 0.10,
        num_points: int = 11,
    ) -> Dict:
        """Sweep one input parameter and record how the price changes.

        Parameters
        ----------
        parameter : str
            One of 'S0', 'K', 'B', 'T', 'r', 'sigma'.
        range_pct : float
            Half-width of sweep as a fraction of the base value.
        num_points : int
            Number of grid points.

        Returns
        -------
        dict with keys: parameter, values, prices, base_value, base_price
        """
        valid_params = {"S0", "K", "B", "T", "r", "sigma"}
        if parameter not in valid_params:
            raise ValueError(f"parameter must be one of {valid_params}")

        base = {"S0": S0, "K": K, "B": B, "T": T, "r": r, "sigma": sigma}
        base_value = base[parameter]
        grid = np.linspace(base_value * (1 - range_pct), base_value * (1 + range_pct), num_points)

        prices = []
        for val in grid:
            p_args = {**base, parameter: float(val)}
            p, _, _, _ = self.monte_carlo_pricer(
                p_args["S0"], p_args["K"], p_args["B"],
                p_args["T"], p_args["r"], p_args["sigma"],
                option_type, N_sim, N_steps,
            )
            prices.append(p)

        return {
            "parameter": parameter,
            "values": grid,
            "prices": prices,
            "base_value": base_value,
            "base_price": prices[num_points // 2],
        }


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def print_detailed_results(
    price: float,
    conf_lower: float,
    conf_upper: float,
    stats: Dict,
    option_params: Dict,
) -> None:
    """Print a formatted summary of pricing results."""
    width = 60
    print("\n" + "=" * width)
    print("BARRIER OPTION PRICING RESULTS")
    print("=" * width)

    print(f"\nOption Type : {option_params['option_type'].replace('_', ' ').title()}")
    print(f"S0          : {option_params['S0']:.4f}")
    print(f"K           : {option_params['K']:.4f}")
    print(f"B           : {option_params['B']:.4f}")
    print(f"T           : {option_params['T']:.4f} years")
    print(f"r           : {option_params['r']:.4%}")
    print(f"σ           : {option_params['sigma']:.4%}")

    print(f"\nSimulation Parameters:")
    print(f"  Paths     : {stats['effective_simulations']:,}")
    print(f"  Steps     : {option_params['N_steps']:,}")
    print(f"  Monitoring: {option_params.get('monitoring_type', 'discrete')}")

    print(f"\n{'-' * 30}")
    print("PRICING RESULTS")
    print(f"{'-' * 30}")
    cv = option_params.get("confidence_level", 0.95)
    print(f"Price                : {price:.6f}")
    print(f"{cv:.0%} Confidence Interval: [{conf_lower:.6f}, {conf_upper:.6f}]")
    print(f"MC Standard Error    : ±{stats['mc_error']:.6f}")
    print(f"Convergence Ratio    : {stats['convergence_ratio']:.4f}")

    print(f"\n{'-' * 30}")
    print("DIAGNOSTICS")
    print(f"{'-' * 30}")
    print(f"Mean Payoff (undiscounted): {stats['mean_payoff']:.6f}")
    print(f"Payoff Std Dev            : {stats['std_payoff']:.6f}")
    print(f"Barrier Hit Rate          : {stats['barrier_hit_percentage']:.2f}%")
    print(f"Computation Time          : {stats['computation_time']:.3f}s")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Interactive CLI for the barrier option pricer."""
    pricer = BarrierOptionsPricer()

    defaults = {
        "S0": 100.0,
        "K": 100.0,
        "B": 90.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
        "option_type": "down_and_out_call",
        "N_sim": 100_000,
        "N_steps": 252,
        "monitoring_type": "discrete",
        "confidence_level": 0.95,
    }

    print("Monte Carlo Barrier Options Pricer")
    print("===================================")

    if input("Use default parameters? (y/n): ").strip().lower() != "y":
        try:
            for key in ("S0", "K", "B", "T", "r", "sigma"):
                raw = input(f"  {key} (default {defaults[key]}): ").strip()
                if raw:
                    defaults[key] = float(raw)
            print(f"  Available types: {pricer.valid_option_types}")
            raw = input(f"  option_type (default {defaults['option_type']}): ").strip()
            if raw:
                defaults["option_type"] = raw
            raw = input(f"  N_sim (default {defaults['N_sim']}): ").strip()
            if raw:
                defaults["N_sim"] = int(raw)
            raw = input(f"  N_steps (default {defaults['N_steps']}): ").strip()
            if raw:
                defaults["N_steps"] = int(raw)
            raw = input(
                f"  monitoring_type discrete/continuous_approx "
                f"(default {defaults['monitoring_type']}): "
            ).strip()
            if raw:
                defaults["monitoring_type"] = raw
        except ValueError as exc:
            print(f"  Invalid input ({exc}), reverting to defaults.")

    print("\nPricing…")
    price, ci_lo, ci_hi, stats = pricer.monte_carlo_pricer(
        defaults["S0"], defaults["K"], defaults["B"],
        defaults["T"], defaults["r"], defaults["sigma"],
        defaults["option_type"], defaults["N_sim"], defaults["N_steps"],
        defaults["monitoring_type"], defaults["confidence_level"],
    )
    print_detailed_results(price, ci_lo, ci_hi, stats, defaults)

    if input("\nRun additional analysis? (y/n): ").strip().lower() != "y":
        return

    print("  1. Convergence analysis")
    print("  2. Sensitivity analysis")
    choice = input("  Choice (1/2): ").strip()

    if choice == "1":
        print("\nConvergence analysis…")
        conv = pricer.analyze_convergence(
            defaults["S0"], defaults["K"], defaults["B"],
            defaults["T"], defaults["r"], defaults["sigma"],
            defaults["option_type"], defaults["N_steps"],
            monitoring_type=defaults["monitoring_type"],
        )
        hdr = f"{'Simulations':<14}{'Price':<14}{'MC Error':<14}{'Time (s)':<10}"
        print(hdr)
        print("-" * len(hdr))
        for n, p, e, t in zip(conv["sim_counts"], conv["prices"],
                               conv["errors"], conv["computation_times"]):
            print(f"{n:<14,}{p:<14.6f}{e:<14.6f}{t:<10.3f}")

    elif choice == "2":
        print("  Parameters: S0 K B T r sigma")
        param = input("  Parameter to vary: ").strip()
        if param not in {"S0", "K", "B", "T", "r", "sigma"}:
            print("  Invalid parameter.")
            return
        sens = pricer.sensitivity_analysis(
            defaults["S0"], defaults["K"], defaults["B"],
            defaults["T"], defaults["r"], defaults["sigma"],
            defaults["option_type"], defaults["N_sim"] // 5,
            defaults["N_steps"], param,
        )
        print(f"\nSensitivity to {param}:")
        print(f"  {'Value':<14}{'Price':<14}")
        print("  " + "-" * 28)
        for v, p in zip(sens["values"], sens["prices"]):
            print(f"  {v:<14.6f}{p:<14.6f}")


if __name__ == "__main__":
    main()