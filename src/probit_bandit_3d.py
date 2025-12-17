#!/usr/bin/env python3
"""
3D Probit Bandit Analysis

Decision based on probit of (mu_+ - mu_-) where:
  mu_+ = (sum y_t[a=+1]) / (sum [a=+1])
  mu_- = (sum y_t[a=-1]) / (sum [a=-1])

Sufficient statistics:
  Y = sum_t y_t       (total reward)
  A = sum_t a_t       (action imbalance)
  G = sum_t a_t*y_t   (gamma, reward-action correlation)

Decision variable:
  mu_+ - mu_- = 2(GT - AY) / (T^2 - A^2)

Policy:
  P(a=+1) = Phi(beta * (GT - AY) / (T^2 - A^2))
"""

import numpy as np
from scipy.stats import norm
from scipy.special import comb
import matplotlib.pyplot as plt
from collections import defaultdict


class ProbitBandit3D:
    """3D probit bandit with (Y, A, G) sufficient statistics."""

    def __init__(self, m_plus, m_minus, beta, eta_plus, eta_minus):
        """
        Initialize bandit.

        Parameters:
        -----------
        m_plus, m_minus : int
            Prior pseudo-counts for each arm
        beta : float
            Inverse temperature
        eta_plus, eta_minus : float
            True reward probabilities for arm +1 and -1
        """
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.beta = beta
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus

        # Initial statistics from prior
        # Prior: m_plus pulls of +1, m_minus pulls of -1
        # Expected rewards: m_plus * eta_prior, m_minus * eta_prior
        # For symmetric prior, assume eta_prior = 0.5
        self.T0 = m_plus + m_minus
        self.A0 = m_plus - m_minus  # Initial action imbalance
        # Initial Y and G depend on prior reward assumption
        # For simplicity, start from (Y0, A0, G0) = (0, 0, 0) at t=0

    def decision_variable(self, Y, A, G, T):
        """
        Compute mu_+ - mu_- = 2(GT - AY) / (T^2 - A^2)

        Handle edge cases where denominator is small.
        """
        denom = T**2 - A**2
        if abs(denom) < 1e-10:
            # Edge case: all actions same sign
            return 0.0
        return 2 * (G * T - A * Y) / denom

    def action_prob(self, Y, A, G, T):
        """P(a=+1 | Y, A, G, T)"""
        delta_mu = self.decision_variable(Y, A, G, T)
        return norm.cdf(self.beta * delta_mu)

    def simulate_trajectory(self, T_horizon, seed=None):
        """
        Simulate a single trajectory.

        Returns:
        --------
        history : dict with keys 'Y', 'A', 'G', 't'
            Each is array of length T_horizon+1
        """
        if seed is not None:
            np.random.seed(seed)

        Y = np.zeros(T_horizon + 1)
        A = np.zeros(T_horizon + 1)
        G = np.zeros(T_horizon + 1)

        for t in range(T_horizon):
            # Current state
            T_curr = self.T0 + t
            p_plus = self.action_prob(Y[t], A[t], G[t], T_curr)

            # Sample action
            a = 1 if np.random.rand() < p_plus else -1

            # Sample reward
            eta = self.eta_plus if a == 1 else self.eta_minus
            y = 1 if np.random.rand() < eta else 0

            # Update sufficient statistics
            Y[t+1] = Y[t] + y
            A[t+1] = A[t] + a
            G[t+1] = G[t] + a * y

        return {'Y': Y, 'A': A, 'G': G, 't': np.arange(T_horizon + 1)}

    def monte_carlo_moments(self, T_horizon, n_samples=10000, seed=None):
        """
        Estimate moments via Monte Carlo.

        Returns:
        --------
        means : array of shape (T_horizon+1, 3) for [Y, A, G]
        covs : array of shape (T_horizon+1, 3, 3)
        third_moments : dict of third-order cumulants
        fourth_moments : dict of fourth-order cumulants
        third_se : dict of standard errors for third-order cumulants
        fourth_se : dict of standard errors for fourth-order cumulants
        """
        if seed is not None:
            np.random.seed(seed)

        # Storage
        samples = np.zeros((n_samples, T_horizon + 1, 3))

        for i in range(n_samples):
            traj = self.simulate_trajectory(T_horizon)
            samples[i, :, 0] = traj['Y']
            samples[i, :, 1] = traj['A']
            samples[i, :, 2] = traj['G']

        # Compute means
        means = np.mean(samples, axis=0)

        # Compute covariances
        covs = np.zeros((T_horizon + 1, 3, 3))
        for t in range(T_horizon + 1):
            covs[t] = np.cov(samples[:, t, :].T)

        # Compute standardized third and fourth cumulants with standard errors
        third_cumulants = {}
        fourth_cumulants = {}
        third_se = {}
        fourth_se = {}

        for t in range(T_horizon + 1):
            centered = samples[:, t, :] - means[t]
            stds = np.sqrt(np.diag(covs[t])) + 1e-10
            standardized = centered / stds

            # Third-order (skewness-like)
            for i in range(3):
                for j in range(i, 3):
                    for k in range(j, 3):
                        key = (i, j, k)
                        if key not in third_cumulants:
                            third_cumulants[key] = np.zeros(T_horizon + 1)
                            third_se[key] = np.zeros(T_horizon + 1)
                        vals = standardized[:, i] * standardized[:, j] * standardized[:, k]
                        third_cumulants[key][t] = np.mean(vals)
                        third_se[key][t] = np.std(vals) / np.sqrt(n_samples)

            # Fourth-order (kurtosis-like) - just diagonal for now
            for i in range(3):
                key = (i, i, i, i)
                if key not in fourth_cumulants:
                    fourth_cumulants[key] = np.zeros(T_horizon + 1)
                    fourth_se[key] = np.zeros(T_horizon + 1)
                vals = standardized[:, i]**4 - 3  # Excess
                fourth_cumulants[key][t] = np.mean(vals)
                fourth_se[key][t] = np.std(vals) / np.sqrt(n_samples)

        return means, covs, third_cumulants, fourth_cumulants, third_se, fourth_se


class MasterEquation3D:
    """
    Master equation for 3D probit bandit.

    State space: (Y, A, G) where Y, A, G are integers.
    Transitions:
      (Y, A, G) -> (Y+1, A+1, G+1) with prob P(a=+1) * eta_+
      (Y, A, G) -> (Y,   A+1, G)   with prob P(a=+1) * (1-eta_+)
      (Y, A, G) -> (Y+1, A-1, G-1) with prob P(a=-1) * eta_-
      (Y, A, G) -> (Y,   A-1, G)   with prob P(a=-1) * (1-eta_-)
    """

    def __init__(self, m_plus, m_minus, beta, eta_plus, eta_minus):
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.beta = beta
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.T0 = m_plus + m_minus

    def decision_variable(self, Y, A, G, T):
        """mu_+ - mu_-"""
        denom = T**2 - A**2
        if abs(denom) < 1e-10:
            return 0.0
        return 2 * (G * T - A * Y) / denom

    def action_prob(self, Y, A, G, T):
        """P(a=+1 | Y, A, G, T)"""
        delta_mu = self.decision_variable(Y, A, G, T)
        return norm.cdf(self.beta * delta_mu)

    def evolve(self, T_horizon, max_states=50000):
        """
        Evolve the master equation.

        Returns:
        --------
        distributions : list of dicts, each mapping (Y, A, G) -> probability
        """
        # Initial distribution: delta at (0, 0, 0)
        dist = {(0, 0, 0): 1.0}
        distributions = [dist.copy()]

        for t in range(T_horizon):
            T_curr = self.T0 + t
            new_dist = defaultdict(float)

            for (Y, A, G), prob in dist.items():
                if prob < 1e-15:
                    continue

                p_plus = self.action_prob(Y, A, G, T_curr)
                p_minus = 1 - p_plus

                # Four possible transitions
                # a=+1, y=1
                new_dist[(Y+1, A+1, G+1)] += prob * p_plus * self.eta_plus
                # a=+1, y=0
                new_dist[(Y, A+1, G)] += prob * p_plus * (1 - self.eta_plus)
                # a=-1, y=1
                new_dist[(Y+1, A-1, G-1)] += prob * p_minus * self.eta_minus
                # a=-1, y=0
                new_dist[(Y, A-1, G)] += prob * p_minus * (1 - self.eta_minus)

            # Prune small probabilities if state space too large
            if len(new_dist) > max_states:
                items = sorted(new_dist.items(), key=lambda x: -x[1])
                new_dist = dict(items[:max_states])
                total = sum(new_dist.values())
                new_dist = {k: v/total for k, v in new_dist.items()}

            dist = dict(new_dist)
            distributions.append(dist.copy())

            if (t + 1) % 10 == 0:
                print(f"  t = {t+1}: {len(dist)} states")

        return distributions

    def compute_moments(self, distributions):
        """
        Compute moments from distribution sequence.

        Returns:
        --------
        means, covs, third_cumulants, fourth_cumulants
        """
        T_horizon = len(distributions) - 1
        means = np.zeros((T_horizon + 1, 3))
        covs = np.zeros((T_horizon + 1, 3, 3))

        for t, dist in enumerate(distributions):
            states = np.array(list(dist.keys()))
            probs = np.array(list(dist.values()))

            # Means
            means[t] = np.sum(states * probs[:, None], axis=0)

            # Covariance
            centered = states - means[t]
            for i in range(3):
                for j in range(3):
                    covs[t, i, j] = np.sum(probs * centered[:, i] * centered[:, j])

        # Third and fourth cumulants (standardized)
        third_cumulants = {}
        fourth_cumulants = {}

        for t, dist in enumerate(distributions):
            states = np.array(list(dist.keys()))
            probs = np.array(list(dist.values()))

            centered = states - means[t]
            stds = np.sqrt(np.diag(covs[t])) + 1e-10
            standardized = centered / stds

            # Third order
            for i in range(3):
                for j in range(i, 3):
                    for k in range(j, 3):
                        key = (i, j, k)
                        if key not in third_cumulants:
                            third_cumulants[key] = np.zeros(T_horizon + 1)
                        third_cumulants[key][t] = np.sum(
                            probs * standardized[:, i] * standardized[:, j] * standardized[:, k]
                        )

            # Fourth order (diagonal)
            for i in range(3):
                key = (i, i, i, i)
                if key not in fourth_cumulants:
                    fourth_cumulants[key] = np.zeros(T_horizon + 1)
                fourth_cumulants[key][t] = np.sum(probs * standardized[:, i]**4) - 3

        return means, covs, third_cumulants, fourth_cumulants


def plot_gaussianity_convergence(means_mc, covs_mc, third_mc, fourth_mc,
                                  means_me=None, covs_me=None, third_me=None, fourth_me=None,
                                  third_se=None, fourth_se=None,
                                  figname='../fig/probit_3d_gaussianity.png'):
    """Plot convergence to Gaussianity with uncertainty bands."""

    T = len(means_mc) - 1
    t_arr = np.arange(T + 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Marginal skewnesses
    labels_3 = ['Y', 'A', 'G']
    for i, (key, label) in enumerate(zip([(0,0,0), (1,1,1), (2,2,2)], labels_3)):
        ax = axes[0, i]
        y = third_mc[key]
        ax.plot(t_arr, y, 'b-', label='MC', linewidth=1.5)
        if third_se is not None and key in third_se:
            se = third_se[key]
            ax.fill_between(t_arr, y - 2*se, y + 2*se, color='blue', alpha=0.2)
        if third_me is not None and key in third_me:
            ax.plot(t_arr[:len(third_me[key])], third_me[key], 'r--',
                   label='Master Eq', linewidth=1.5)
        ax.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time t')
        ax.set_ylabel(f'Skewness({label})')
        ax.set_title(f'Marginal skewness of {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 2: Marginal excess kurtoses
    for i, (idx, label) in enumerate(zip([0, 1, 2], labels_3)):
        ax = axes[1, i]
        key = (idx, idx, idx, idx)
        y = fourth_mc[key]
        ax.plot(t_arr, y, 'b-', label='MC', linewidth=1.5)
        if fourth_se is not None and key in fourth_se:
            se = fourth_se[key]
            ax.fill_between(t_arr, y - 2*se, y + 2*se, color='blue', alpha=0.2)
        if fourth_me is not None and key in fourth_me:
            ax.plot(t_arr[:len(fourth_me[key])], fourth_me[key], 'r--',
                   label='Master Eq', linewidth=1.5)
        ax.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time t')
        ax.set_ylabel(f'Excess kurtosis({label})')
        ax.set_title(f'Marginal excess kurtosis of {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f'Saved: {figname}')


def plot_coskewness(third_mc, third_me=None, third_se=None, figname='../fig/probit_3d_coskewness.png'):
    """Plot co-skewness evolution with uncertainty bands."""
    T = len(third_mc[(0,0,0)]) - 1
    t_arr = np.arange(T + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Co-skewness kappa_YAG
    ax = axes[0]
    y = third_mc[(0,1,2)]
    ax.plot(t_arr, y, 'b-', label='MC', linewidth=2)
    if third_se is not None and (0,1,2) in third_se:
        se = third_se[(0,1,2)]
        ax.fill_between(t_arr, y - 2*se, y + 2*se, color='blue', alpha=0.2)
    if third_me is not None and (0,1,2) in third_me:
        ax.plot(t_arr[:len(third_me[(0,1,2)])], third_me[(0,1,2)], 'r--',
               label='Master Eq', linewidth=1.5)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'Co-skewness $\kappa_{YAG}/(\sigma_Y\sigma_A\sigma_G)$')
    ax.set_title('Three-way correlation (co-skewness)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: All third-order cumulants
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    labels = ['YYY', 'AAA', 'GGG', 'YYA', 'YYG', 'YAA', 'YAG', 'YGG', 'AAG', 'AGG']
    keys = [(0,0,0), (1,1,1), (2,2,2), (0,0,1), (0,0,2), (0,1,1), (0,1,2), (0,2,2), (1,1,2), (1,2,2)]
    for i, (key, label) in enumerate(zip(keys, labels)):
        if key in third_mc:
            ax.plot(t_arr, third_mc[key], color=colors[i], label=label, linewidth=1)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Standardized 3rd cumulant')
    ax.set_title('All third-order cumulants')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f'Saved: {figname}')


def plot_means_and_covariance(means_mc, covs_mc, means_me=None, covs_me=None,
                               figname='../fig/probit_3d_moments.png'):
    """Plot means and covariance evolution."""
    T = len(means_mc) - 1
    t_arr = np.arange(T + 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    labels = ['Y', 'A', 'G']

    # Row 1: Means
    for i, label in enumerate(labels):
        ax = axes[0, i]
        ax.plot(t_arr, means_mc[:, i], 'b-', label='MC', linewidth=1.5)
        if means_me is not None:
            ax.plot(t_arr[:len(means_me)], means_me[:, i], 'r--',
                   label='Master Eq', linewidth=1.5)
        ax.set_xlabel('Time t')
        ax.set_ylabel(f'E[{label}]')
        ax.set_title(f'Mean of {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 2: Variances
    for i, label in enumerate(labels):
        ax = axes[1, i]
        ax.plot(t_arr, covs_mc[:, i, i], 'b-', label='MC', linewidth=1.5)
        if covs_me is not None:
            ax.plot(t_arr[:len(covs_me)], covs_me[:, i, i], 'r--',
                   label='Master Eq', linewidth=1.5)
        ax.set_xlabel('Time t')
        ax.set_ylabel(f'Var({label})')
        ax.set_title(f'Variance of {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f'Saved: {figname}')


def plot_correlation_matrix(covs_mc, figname='../fig/probit_3d_correlations.png'):
    """Plot correlation evolution."""
    T = len(covs_mc) - 1
    t_arr = np.arange(T + 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ['Y', 'A', 'G']

    # Compute correlations
    corr_YA = covs_mc[:, 0, 1] / (np.sqrt(covs_mc[:, 0, 0] * covs_mc[:, 1, 1]) + 1e-10)
    corr_YG = covs_mc[:, 0, 2] / (np.sqrt(covs_mc[:, 0, 0] * covs_mc[:, 2, 2]) + 1e-10)
    corr_AG = covs_mc[:, 1, 2] / (np.sqrt(covs_mc[:, 1, 1] * covs_mc[:, 2, 2]) + 1e-10)

    ax.plot(t_arr, corr_YA, 'b-', label='Corr(Y, A)', linewidth=1.5)
    ax.plot(t_arr, corr_YG, 'r-', label='Corr(Y, G)', linewidth=1.5)
    ax.plot(t_arr, corr_AG, 'g-', label='Corr(A, G)', linewidth=1.5)

    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Correlation coefficient')
    ax.set_title('Pairwise correlations between Y, A, G')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f'Saved: {figname}')


def plot_summary_figure(means_mc, covs_mc, third_mc, fourth_mc,
                        means_me=None, covs_me=None, third_me=None, fourth_me=None,
                        third_se=None, fourth_se=None,
                        figname='../fig/probit_3d_summary.png'):
    """Generate a comprehensive summary figure with uncertainty bands."""
    T = len(means_mc) - 1
    t_arr = np.arange(T + 1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    labels = ['Y', 'A', 'G']

    # Row 1, Col 1-3: Marginal skewnesses
    for i, (key, label) in enumerate(zip([(0,0,0), (1,1,1), (2,2,2)], labels)):
        ax = axes[0, i]
        y = third_mc[key]
        ax.plot(t_arr, y, 'b-', linewidth=2)
        if third_se is not None and key in third_se:
            se = third_se[key]
            ax.fill_between(t_arr, y - 2*se, y + 2*se, color='blue', alpha=0.2)
        if third_me is not None and key in third_me:
            ax.plot(t_arr[:len(third_me[key])], third_me[key], 'r--', linewidth=1.5)
        ax.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time t')
        ax.set_ylabel(f'Skewness({label})')
        ax.set_title(f'Skewness({label})')
        ax.grid(True, alpha=0.3)

    # Row 1, Col 4: Co-skewness
    ax = axes[0, 3]
    y = third_mc[(0,1,2)]
    ax.plot(t_arr, y, 'purple', linewidth=2, label='Co-skewness')
    if third_se is not None and (0,1,2) in third_se:
        se = third_se[(0,1,2)]
        ax.fill_between(t_arr, y - 2*se, y + 2*se, color='purple', alpha=0.2)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\kappa_{YAG}$')
    ax.set_title('Co-skewness (3-way)')
    ax.grid(True, alpha=0.3)

    # Row 2, Col 1-3: Marginal excess kurtoses
    for i, (idx, label) in enumerate(zip([0, 1, 2], labels)):
        ax = axes[1, i]
        key = (idx, idx, idx, idx)
        y = fourth_mc[key]
        ax.plot(t_arr, y, 'b-', linewidth=2)
        if fourth_se is not None and key in fourth_se:
            se = fourth_se[key]
            ax.fill_between(t_arr, y - 2*se, y + 2*se, color='blue', alpha=0.2)
        if fourth_me is not None and key in fourth_me:
            ax.plot(t_arr[:len(fourth_me[key])], fourth_me[key], 'r--', linewidth=1.5)
        ax.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time t')
        ax.set_ylabel(f'Ex. Kurt({label})')
        ax.set_title(f'Excess Kurtosis({label})')
        ax.grid(True, alpha=0.3)

    # Row 2, Col 4: Correlations
    ax = axes[1, 3]
    corr_YA = covs_mc[:, 0, 1] / (np.sqrt(covs_mc[:, 0, 0] * covs_mc[:, 1, 1]) + 1e-10)
    corr_YG = covs_mc[:, 0, 2] / (np.sqrt(covs_mc[:, 0, 0] * covs_mc[:, 2, 2]) + 1e-10)
    corr_AG = covs_mc[:, 1, 2] / (np.sqrt(covs_mc[:, 1, 1] * covs_mc[:, 2, 2]) + 1e-10)
    ax.plot(t_arr, corr_YA, 'b-', label='Y-A', linewidth=1.5)
    ax.plot(t_arr, corr_YG, 'r-', label='Y-G', linewidth=1.5)
    ax.plot(t_arr, corr_AG, 'g-', label='A-G', linewidth=1.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Correlation')
    ax.set_title('Pairwise correlations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f'Saved: {figname}')


def main():
    """Run analysis."""
    print('='*70)
    print('3D PROBIT BANDIT: GAUSSIANITY ANALYSIS')
    print('='*70)
    print()

    # Parameters
    m_plus = 10
    m_minus = 10
    beta = 0.5
    eta_plus = 0.6
    eta_minus = 0.4
    T_horizon = 300  # Extended time horizon (MC only beyond ~80)
    T_master = 80    # Master equation limit due to O(T^3) state space
    n_samples = 20000  # More samples for smoother curves

    print(f'Parameters:')
    print(f'  m_plus = {m_plus}, m_minus = {m_minus}')
    print(f'  beta = {beta}')
    print(f'  eta_plus = {eta_plus}, eta_minus = {eta_minus}')
    print(f'  T_horizon (MC) = {T_horizon}')
    print(f'  T_horizon (Master Eq) = {T_master}')
    print(f'  n_samples = {n_samples}')
    print()

    # Monte Carlo
    print('Running Monte Carlo simulation...')
    bandit = ProbitBandit3D(m_plus, m_minus, beta, eta_plus, eta_minus)
    means_mc, covs_mc, third_mc, fourth_mc, third_se, fourth_se = bandit.monte_carlo_moments(
        T_horizon, n_samples=n_samples, seed=42
    )
    print(f'  Completed {n_samples} trajectories')
    print()

    # Master equation (shorter time due to O(T^3) state space)
    print('Running Master Equation...')
    master = MasterEquation3D(m_plus, m_minus, beta, eta_plus, eta_minus)
    distributions = master.evolve(T_master)
    means_me, covs_me, third_me, fourth_me = master.compute_moments(distributions)
    print()

    # Generate all figures
    print('Generating figures...')
    print()

    # Figure 1: Main gaussianity convergence (2x3 grid)
    plot_gaussianity_convergence(means_mc, covs_mc, third_mc, fourth_mc,
                                  means_me, covs_me, third_me, fourth_me,
                                  third_se, fourth_se)

    # Figure 2: Co-skewness and all third-order cumulants
    plot_coskewness(third_mc, third_me, third_se)

    # Figure 3: Means and variances
    plot_means_and_covariance(means_mc, covs_mc, means_me, covs_me)

    # Figure 4: Correlations
    plot_correlation_matrix(covs_mc)

    # Figure 5: Comprehensive summary (2x4 grid)
    plot_summary_figure(means_mc, covs_mc, third_mc, fourth_mc,
                        means_me, covs_me, third_me, fourth_me,
                        third_se, fourth_se)

    # Summary table
    print()
    print('NON-GAUSSIANITY MEASURES AT SELECTED TIMES (Monte Carlo):')
    print()
    print('t      | skew(Y)    | skew(A)    | skew(G)    | co-skew    | kurt(Y)    | kurt(A)    | kurt(G)')
    print('-' * 105)
    for t in [0, 25, 50, 100, 150, 200, 300]:
        if t <= T_horizon:
            sk_Y = third_mc[(0,0,0)][t]
            sk_A = third_mc[(1,1,1)][t]
            sk_G = third_mc[(2,2,2)][t]
            co_sk = third_mc[(0,1,2)][t]  # co-skewness kappa_YAG
            ku_Y = fourth_mc[(0,0,0,0)][t]
            ku_A = fourth_mc[(1,1,1,1)][t]
            ku_G = fourth_mc[(2,2,2,2)][t]
            print(f'{t:>6} | {sk_Y:>10.4f} | {sk_A:>10.4f} | {sk_G:>10.4f} | {co_sk:>10.4f} | '
                  f'{ku_Y:>10.4f} | {ku_A:>10.4f} | {ku_G:>10.4f}')

    print()
    print('For Gaussian: skewness = 0, excess kurtosis = 0')
    print()

    # Check convergence
    final_skews = [third_mc[(i,i,i)][T_horizon] for i in range(3)]
    final_kurts = [fourth_mc[(i,i,i,i)][T_horizon] for i in range(3)]

    print(f'At t={T_horizon}:')
    print(f'  Max |skewness|: {max(abs(s) for s in final_skews):.4f}')
    print(f'  Max |excess kurtosis|: {max(abs(k) for k in final_kurts):.4f}')
    print(f'  Co-skewness: {third_mc[(0,1,2)][T_horizon]:.4f}')

    if max(abs(s) for s in final_skews) < 0.1 and max(abs(k) for k in final_kurts) < 0.2:
        print('  -> Approaching Gaussian!')
    else:
        print('  -> Still non-Gaussian')

    print()
    print('='*70)
    print('FIGURES GENERATED:')
    print('='*70)
    print('  ../fig/probit_3d_gaussianity.png   - Marginal skewness & kurtosis')
    print('  ../fig/probit_3d_coskewness.png    - Co-skewness and all 3rd cumulants')
    print('  ../fig/probit_3d_moments.png       - Means and variances')
    print('  ../fig/probit_3d_correlations.png  - Pairwise correlations')
    print('  ../fig/probit_3d_summary.png       - Comprehensive 2x4 summary')


if __name__ == '__main__':
    main()
