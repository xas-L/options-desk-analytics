# Known Limitations

This is a research and prototyping library, not a production trading system. This
file is meant to be accurate rather than flattering, and it is kept in sync with
the code. Where a limitation showed up in real-data work, the observed numbers are
given so the size of the problem is clear.

Last reviewed: 2026-08-03.

## Rates

The pricers take a flat risk-free rate. There is no OIS or SOFR curve bootstrap in
this repo, so every discount factor is exp(-rT) at a single input rate.

This matters more than it sounds. When the forward is recovered from put-call
parity by regressing C - P against K, the fitted rate absorbs anything that is not
the pure risk-free rate, borrow cost above all. On the FEZ chain used in the
research note, the parity-implied rate came out at 0.16% and 2.88% for the two
surviving expiries against a 3.8% external T-bill proxy. That is a gap worth 364
and 91 vol-bp respectively at the ATM strike. Until an external curve is supplied
for the risk-free leg, the rate and the borrow are not separately identified.

## Dividends

Two constructions are supported: a single continuous yield, and a discrete schedule
built from actual historical distribution dates and amounts rolled forward. The
discrete path is the one used in the equity index forward research note.

What is not modelled: dividend growth or forecasting of any kind (the schedule is
rolled forward at the same calendar timing with flat amounts), announcement risk,
withholding tax, and any term structure of implied dividends recovered from the
option chain itself. `implied_dividend_yield` inverts a forward to a flat yield and
is a diagnostic, not a dividend model.

For ETF proxies specifically, the fund's own smoothed distribution schedule masks
the concentrated dividend timing of the underlying index constituents. Any result
obtained on an ETF understates the dividend timing effect you would see on native
index options.

## Borrow cost

`implied_borrow_cost` backs out a residual from spot, forward, rate and yield. It
is a residual and nothing more. Because it is computed against an assumed r and q,
it inherits every error in both. There is no borrow term structure, no hard to
borrow modelling, and no reconciliation against observable lending rates. The
honest reading of the current output is "rate error plus borrow error combined",
not "the borrow fee".

## American exercise

Longstaff-Schwartz (`pricers/lsm.py`, Laguerre basis) and a CRR binomial tree
(`pricers/binomial.py`) both price American options. The Barone-Adesi-Whaley
approximation is used in the real-data chain pipeline to convert American quotes to
European equivalents before put-call parity or any surface fit, because otherwise
early exercise premium contaminates the recovered forward and is not separable from
the dividend timing effect.

Limitations. The analytic pricers, the Heston FFT and the characteristic function
methods are European only, by construction. Neither LSM nor the binomial tree is
dividend-aware in the discrete sense: both take a continuous yield, so early
exercise decisions around a known ex-dividend date are not handled correctly. On
the FEZ data the measured early exercise premium was under 3 vol-bp even at 1.49y,
which is small, but that is a statement about one moderately liquid ETF at one
snapshot and should not be generalised.

## Volatility surface

`check_ssvi_arbitrage` enforces the two Gatheral-Jacquier butterfly conditions,
theta * phi * (1 + |rho|) < 4 and theta * phi^2 * (1 + |rho|) <= 4. It does not
check the calendar spread condition. The power-law phi used here has a known
sufficient condition for calendar arbitrage freedom involving gamma in (0, 1/2] and
eta * (1 + |rho|) <= 2, and that is not currently imposed. So the surface is
butterfly-checked, not statically arbitrage-free, and any claim to the contrary
elsewhere in the docs is wrong.

The fit is a penalty method inside a differential evolution search over box bounds.
That combination pushes parameters onto boundaries. On real FEZ data the 1.49y
slice converged with rho at -0.999, sitting exactly on its bound, and the 0.32y
slice failed its own butterfly check with phi pinned at the upper bound. A
well-identified fit should not sit on a boundary. Reported rho and phi should be
treated as directionally right, not precise. A constrained solver over an
unconstrained reparameterisation would be the fix.

Fit quality on real data is not close to market-making standards. Spread-weighted
RMSE was 367 vol-bp at 0.32y and 191 vol-bp at 1.49y. Spread weighting (inverse
squared bid-ask) improved the longer expiry from 234 bp and did essentially nothing
at the short one, which says the short-dated problem is something other than wing
quote noise.

## Market data

Quotes come from free retail sources. There is no NBBO, no exchange timestamps, and
no guarantee that the spot print and the option chain were captured at the same
instant. Mid prices are used throughout and the mid is not a tradeable price.

Chains thin out fast under real filters. On the FEZ snapshot only 2 of 5 expiries
survived the requirement for enough two-sided quotes clearing intrinsic value after
the American correction. That is a property of the instrument, not a filtering bug,
but it means single-name and ETF results here rest on very few slices.

The LOBSTER loader handles the free sample format only, which is Level 10 L2 data
for a small number of symbols on a single 2012 session.

## Market making and queue simulation

`QueueTracker` assumes our order joins at the back of the queue and never improves
position other than through trades and cancellations. Cancellations are handled by
scaling cancelled size by a fixed probability of being ahead of us, which is a
mean-field approximation rather than a sampled one, so it captures the average and
not the dispersion. True queue position requires order-by-order (MBO) data, which
the free LOBSTER sample does not provide.

The Avellaneda-Stoikov and Whalley-Wilmott implementations follow the papers
directly. Fill intensity parameters are not calibrated to observed fills, so
absolute fill rates and PnL from these modules are illustrative.

## Dispersion

`implied_correlation` assumes a single uniform pairwise correlation across all
constituents. This is the standard "dirty" implied correlation proxy and it is
known to be biased when the correlation structure is heterogeneous or when index
and constituent option maturities do not line up. Weights are treated as static.
There is no dividend or borrow adjustment on the constituent legs.

## Testing

Coverage is broad but the tests are mostly internal consistency checks: round
trips through the library's own functions, and calibration against data generated
by the same model being calibrated. The strongest tests are the analytic limit
checks, for example Heston FFT converging to Black-Scholes as vol-of-vol goes to
zero, and put-call parity holding across pricers. There is no validation against an
independent reference implementation such as QuantLib, and no published option
price benchmarks. Some tests assert only that a price is positive, which is close to
worthless. Treat the suite as a regression guard, not as evidence of correctness.

## C++ extension and performance

The native extension covers Black-Scholes with Greeks and a radix-2 FFT Heston
pricer. Everything else is Python or NumPy. The extension is optional and the
library falls back to Python when it is not built, so results are reproducible
without a toolchain but timings are not.

No benchmarks are published in this repo. Any speedup figure quoted elsewhere
should be treated as unverified until there is a committed benchmark script with
hardware stated.

## Not covered at all

Multi-curve discounting and collateral. Term structure of repo. Transaction costs
and market impact beyond a simple slippage model. Exchange-specific microstructure,
auctions and halts. Any form of live connectivity, order management or risk limit
enforcement. Corporate actions other than cash dividends.