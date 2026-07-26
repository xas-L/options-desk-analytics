"""Two-regime Markov-switching variance model."""

from __future__ import annotations

import numpy as np
from typing import Tuple


class MarkovRegimeSwitchingVariance:
    """Two-regime Markov-switching model for variance (Normal vs Crisis).
    
    Models variance dynamics where the base level of variance switches between two regimes
    governed by a transition probability matrix.
    """

    def __init__(
        self,
        var_normal: float,
        var_crisis: float,
        p_normal_to_crisis: float,
        p_crisis_to_normal: float
    ) -> None:
        """Initialize the regime-switching model.
        
        Args:
            var_normal: Variance level in normal regime (regime 0).
            var_crisis: Variance level in crisis regime (regime 1).
            p_normal_to_crisis: Probability of switching from 0 to 1 in one time step (dt).
            p_crisis_to_normal: Probability of switching from 1 to 0 in one time step (dt).
        """
        self.var_levels = np.array([var_normal, var_crisis])
        self.p_01 = p_normal_to_crisis
        self.p_10 = p_crisis_to_normal
        self.p_00 = 1.0 - self.p_01
        self.p_11 = 1.0 - self.p_10

        # Transition matrix P[i, j] = P(State_t+1 = j | State_t = i)
        self.transition_matrix = np.array([
            [self.p_00, self.p_01],
            [self.p_10, self.p_11]
        ])

    def stationary_distribution(self) -> np.ndarray:
        """Calculate the long-term stationary distribution of the regimes.
        
        Returns:
            Array [pi_0, pi_1] of unconditional probabilities for each regime.
        """
        # pi * P = pi, pi * (P - I) = 0, sum(pi) = 1
        pi_0 = self.p_10 / (self.p_01 + self.p_10)
        pi_1 = self.p_01 / (self.p_01 + self.p_10)
        return np.array([pi_0, pi_1])

    def unconditional_variance(self) -> float:
        """Calculate the unconditional expected variance."""
        pi = self.stationary_distribution()
        return float(np.dot(pi, self.var_levels))

    def simulate_regime_path(self, initial_state: int, num_steps: int) -> np.ndarray:
        """Simulate a path of regimes.
        
        Args:
            initial_state: 0 (Normal) or 1 (Crisis).
            num_steps: Number of steps to simulate.
            
        Returns:
            Array of shape (num_steps,) containing the regime sequence (0s and 1s).
        """
        states = np.zeros(num_steps, dtype=int)
        states[0] = initial_state
        
        for t in range(1, num_steps):
            curr_state = states[t-1]
            prob_switch_to_1 = self.transition_matrix[curr_state, 1]
            if np.random.rand() < prob_switch_to_1:
                states[t] = 1
            else:
                states[t] = 0
                
        return states

    def simulate_variance_path(self, initial_state: int, num_steps: int) -> np.ndarray:
        """Simulate a path of realized variances based on the regime sequence."""
        states = self.simulate_regime_path(initial_state, num_steps)
        return self.var_levels[states]
