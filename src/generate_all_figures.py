#!/usr/bin/env python3
"""
Generate All Figures for Bandit FPE Analysis

This script generates all figures used in the LaTeX documentation.
Run from the src/ directory: python generate_all_figures.py

Figures generated:
  1. late_time_gaussian_convergence.png - Main results on Gaussian convergence
  2. truncation_rate_error.png - Detailed truncation error analysis
  3. rates_comparison.png - Exact vs ansatz rate comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

from truncation_rate_error import (
    compute_truncation_rate_error,
    plot_truncation_rate_error,
    plot_rates_comparison
)


# Default parameters
DEFAULT_PARAMS = {
    'm_plus': 10,
    'm_minus': 10,
    'beta': 0.3,
    'eta_plus': 0.6,
    'eta_minus': 0.4,
}


def get_derived_params(params):
    """Compute derived parameters from base parameters."""
    eta_bar = (params['eta_plus'] + params['eta_minus']) / 2
    delta_eta = (params['eta_plus'] - params['eta_minus']) / 2
    return eta_bar, delta_eta


def generate_late_time_gaussian_convergence(params=None, T=300,
                                            figname='../fig/late_time_gaussian_convergence.png'):
    """
    Generate the main late-time Gaussian convergence figure (4 panels).

    Panel 1: K=2 truncation rate error decay
    Panel 2: Standardized cumulants (skewness, kurtosis) decay
    Panel 3: Exact vs ansatz variance rates comparison
    Panel 4: Power-law decay verification (log-log)
    """
    if params is None:
        params = DEFAULT_PARAMS

    # Only use K=2 (higher K has known issues)
    K_values = [2]

    eta_bar, delta_eta = get_derived_params(params)

    print(f'Computing T={T} steps for late-time Gaussian convergence...')
    times, exact_rates, ansatz_rates, errors, exact_cumulants = compute_truncation_rate_error(
        params['m_plus'], params['m_minus'], params['beta'],
        eta_bar, delta_eta, T, K_values
    )

    # Compute standardized cumulants
    sigma = np.sqrt(exact_cumulants[:, 1])
    skewness = exact_cumulants[:, 2] / sigma**3
    ex_kurtosis = exact_cumulants[:, 3] / sigma**4

    # Integer time array for plotting
    t_int = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Truncation rate error decay (log scale)
    ax = axes[0, 0]
    ax.semilogy(t_int, errors[2][:, 0] + 1e-16, 'b-',
                label=r'$\epsilon_1^{K=2}$ (mean)', linewidth=1.5)
    ax.semilogy(t_int, errors[2][:, 1] + 1e-16, 'r-',
                label=r'$\epsilon_2^{K=2}$ (variance)', linewidth=1.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Truncation rate error')
    ax.set_title(r'$K=2$ closure error decays as $p \to$ Gaussian')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Standardized cumulants decay
    ax = axes[0, 1]
    t_cumulants = np.arange(T + 1)
    ax.plot(t_cumulants, skewness, 'b-', label='Skewness', linewidth=1.5)
    ax.plot(t_cumulants, ex_kurtosis, 'r-', label='Excess kurtosis', linewidth=1.5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Standardized cumulant')
    ax.set_title(r'Non-Gaussianity decays: $p \to$ Gaussian')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Exact vs ansatz variance rate comparison
    ax = axes[1, 0]
    ax.plot(t_int, exact_rates[:, 1], 'k-', linewidth=2, label='Exact')
    ax.plot(t_int, ansatz_rates[2][:, 1], 'r--', linewidth=1.5, alpha=0.8, label='K=2 ansatz')
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\Delta\kappa_2(t)$')
    ax.set_title('Variance rate: exact vs K=2 ansatz')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Power law check - log-log plot
    ax = axes[1, 1]
    t_start = 50  # skip transient
    t_arr = np.arange(t_start, T)
    ax.loglog(t_arr, errors[2][t_start:T, 0] + 1e-16, 'b-',
              label=r'$\epsilon_1^{K=2}$', linewidth=1.5)
    ax.loglog(t_arr, errors[2][t_start:T, 1] + 1e-16, 'r-',
              label=r'$\epsilon_2^{K=2}$', linewidth=1.5)

    # Fit power law to late time
    def power_law(t, a, b):
        return a * t**b

    t_fit_start = 100
    t_fit = np.arange(t_fit_start, T)
    try:
        popt1, _ = curve_fit(power_law, t_fit, errors[2][t_fit_start:T, 0] + 1e-16, p0=[1, -2])
        popt2, _ = curve_fit(power_law, t_fit, errors[2][t_fit_start:T, 1] + 1e-16, p0=[1, -2])
        ax.loglog(t_fit, power_law(t_fit, *popt1), 'b--', alpha=0.7,
                  label=f'fit: $t^{{{popt1[1]:.2f}}}$')
        ax.loglog(t_fit, power_law(t_fit, *popt2), 'r--', alpha=0.7,
                  label=f'fit: $t^{{{popt2[1]:.2f}}}$')
    except Exception:
        pass

    ax.set_xlabel('Time t')
    ax.set_ylabel('Error')
    ax.set_title('Power law decay of truncation error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f'Saved: {figname}')

    # Print summary
    print()
    print('=' * 70)
    print('LATE-TIME GAUSSIAN CONVERGENCE SUMMARY')
    print('=' * 70)
    print()
    print('As t -> infinity:')
    print(f'  - Skewness -> 0 (t={T}: {skewness[-1]:.4f})')
    print(f'  - Excess kurtosis -> 0 (t={T}: {ex_kurtosis[-1]:.4f})')
    print('  - K=2 truncation error -> 0')

    return times, errors, exact_cumulants


def generate_truncation_error_figures(params=None, T=1000):
    """
    Generate truncation_rate_error.png and rates_comparison.png.
    Uses the existing plotting functions from truncation_rate_error.py.

    Note: T=1000 captures extended late-time behavior where:
    - Error rises in transient (t=0-50), peaks, then decays
    - By t~500, truncation error reaches machine precision (~1e-14)
    - Standardized cumulants decay toward zero (skewness -0.06, ex.kurt 0.01 at t=1000)
    """
    if params is None:
        params = DEFAULT_PARAMS

    # Only use K=2 (higher K has known issues)
    K_values = [2]

    eta_bar, delta_eta = get_derived_params(params)

    print(f'Computing T={T} steps for truncation error analysis...')
    times, exact_rates, ansatz_rates, errors, exact_cumulants = compute_truncation_rate_error(
        params['m_plus'], params['m_minus'], params['beta'],
        eta_bar, delta_eta, T, K_values
    )

    # Generate figures using existing functions
    plot_truncation_rate_error(times, exact_rates, ansatz_rates, errors,
                               '../fig/truncation_rate_error.png',
                               exact_cumulants=exact_cumulants)
    plot_rates_comparison(times, exact_rates, ansatz_rates,
                          '../fig/rates_comparison.png')

    return times, exact_rates, ansatz_rates, errors


def print_standardized_cumulants_table(params=None, T=300, times_to_show=None):
    """Print table of standardized cumulants at selected times."""
    if params is None:
        params = DEFAULT_PARAMS
    if times_to_show is None:
        times_to_show = [0, 50, 100, 200, 300]

    eta_bar, delta_eta = get_derived_params(params)

    print(f'\nComputing standardized cumulants for T={T}...')
    _, _, _, _, exact_cumulants = compute_truncation_rate_error(
        params['m_plus'], params['m_minus'], params['beta'],
        eta_bar, delta_eta, T, [2]
    )

    print('\nStandardized cumulants (measures of non-Gaussianity):')
    print('t      | sigma        | skewness     | ex. kurtosis')
    print('-' * 60)
    for t in times_to_show:
        if t <= T:
            sigma = np.sqrt(exact_cumulants[t, 1])
            k3 = exact_cumulants[t, 2]
            k4 = exact_cumulants[t, 3]
            skew = k3 / sigma**3 if sigma > 0 else 0
            kurt = k4 / sigma**4 if sigma > 0 else 0
            print(f'{t:>6} | {sigma:>12.4f} | {skew:>12.4f} | {kurt:>12.4f}')


def main():
    """Generate all figures."""
    print('=' * 70)
    print('GENERATING ALL FIGURES FOR BANDIT FPE ANALYSIS')
    print('=' * 70)
    print()

    # Figure 1: Late-time Gaussian convergence (main result)
    print('\n[1/3] Generating late_time_gaussian_convergence.png...')
    generate_late_time_gaussian_convergence(T=300)

    # Figures 2 & 3: Truncation error and rates comparison
    print('\n[2/3] Generating truncation_rate_error.png...')
    print('[3/3] Generating rates_comparison.png...')
    generate_truncation_error_figures(T=1000)

    # Print summary table
    print_standardized_cumulants_table()

    print('\n' + '=' * 70)
    print('ALL FIGURES GENERATED SUCCESSFULLY')
    print('=' * 70)
    print('\nFigures saved to ../fig/:')
    print('  - late_time_gaussian_convergence.png')
    print('  - truncation_rate_error.png')
    print('  - rates_comparison.png')


if __name__ == '__main__':
    main()
