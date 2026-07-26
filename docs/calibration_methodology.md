# Calibration Methodology

Volatility calibration is a critical component of options pricing. ODX utilises several advanced techniques to fit theoretical models to observed market data while strictly preventing arbitrage opportunities.

## SVI and SSVI Surface Calibration

The Stochastic Volatility Inspired (SVI) model and its surface extension (SSVI) parametrise the implied volatility smile using a set of continuous equations. 

We fit these models to market data using a least-squares objective function that minimises the difference between the model's total variance and the observed market variance. Because standard gradient descent algorithms can easily get trapped in local minima, we utilise the differential evolution algorithm to explore the parameter space globally.

### Arbitrage Constraints

A major risk in volatility surface fitting is generating a surface that allows for theoretical risk-free profits. We explicitly enforce Gatheral's conditions to prevent this:
* **Calendar Arbitrage**: The total variance must not decrease alongside increasing time to maturity. We satisfy this structurally by ensuring our time-scaling parameters are strictly positive.
* **Butterfly Arbitrage**: The probability density function implied by the surface must remain positive. We map this constraint into a penalty function within our optimiser. Where the optimiser explores a parameter set that violates the butterfly conditions, it is hit with a massive numerical penalty, forcing it back into the arbitrage-free region.

## Heston Model Calibration

The Heston model assumes that the underlying asset's variance follows a stochastic process. We calibrate its five parameters (initial variance, long-term variance, mean reversion speed, volatility of volatility, and correlation) to the market smile.

For speed, we use a Fast Fourier Transform (FFT) pricer during the calibration loop. The FFT allows us to price options across multiple strikes simultaneously, which is significantly faster than using Monte Carlo simulation. 

We apply the L-BFGS-B optimisation algorithm to navigate the parameter space. To ensure numerical stability, we impose the Feller condition as a soft penalty. The Feller condition guarantees that the variance process never hits zero, which prevents our numerical integrators from failing in extreme scenarios.

## Numerical Caveats

When calibrating stochastic models, researchers should be aware of a few numerical realities:

1. **Monte Carlo Noise**: When attempting to calibrate models via Monte Carlo simulation (such as our rough Bergomi implementation), the inherent randomness introduces noise into the objective function. Gradient-based optimisers will often fail here. We rely on gradient-free methods like Nelder-Mead to handle noisy evaluations.
2. **Scaling**: Optimisers perform best when parameters are on a similar scale. Heston parameters can vary wildly in magnitude (correlation is bounded between -1 and 1, while mean reversion can be much larger).
3. **Local Minima**: Implied volatility surfaces often have multiple local minima. While we use robust starting guesses and bounds, complex models may still require manual tuning or alternative starting points to find the true global optimum.
