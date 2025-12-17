"""
Compare exact master equation evolution with Gaussian cumulant dynamics.

The master equation is:
  p_{n+1}(gamma) = R^+(gamma-1) p_n(gamma-1) + R^-(gamma+1) p_n(gamma+1)

where R^+(gamma) = (1 + eta_bar + Delta_eta * b(gamma)) / 2
      R^-(gamma) = (1 - eta_bar - Delta_eta * b(gamma)) / 2
      b(gamma) = 2*Phi(beta*gamma) - 1

The Gaussian cumulant dynamics are:
  mu' = mu + eta_bar + Delta_eta * <b>
  sigma'^2 = sigma^2 + 1 - <v>^2 + 2*Delta_eta*Cov(gamma, b)

where <b> = 2*Phi(beta_tilde * mu) - 1
      beta_tilde = beta / sqrt(1 + beta^2 * sigma^2)
      Cov(gamma, b) = 2*beta*sigma^2 / sqrt(1 + beta^2*sigma^2) * phi(beta_tilde * mu)
"""

import numpy as np
from scipy.stats import norm
from scipy.special import erfc
import matplotlib.pyplot as plt


def build_initial_pdf(m_plus, m_minus, gamma_grid):
    """
    Build initial PDF from m_+ visits to arm + and m_- visits to arm -.
    Initial gamma = m_+ - m_- with binomial-like spread.

    For large m, this is approximately Gaussian with:
      mean = m_+ - m_-
      variance = m_+ + m_-  (each step adds variance 1)
    """
    mu0 = m_plus - m_minus
    sigma0_sq = m_plus + m_minus
    sigma0 = np.sqrt(sigma0_sq)

    # Discretized Gaussian on integer grid
    pdf = norm.pdf(gamma_grid, loc=mu0, scale=sigma0)
    pdf /= pdf.sum()  # Normalize for discrete grid
    return pdf, mu0, sigma0_sq


def Phi(x):
    """Standard normal CDF (probit)."""
    return norm.cdf(x)


def phi(x):
    """Standard normal PDF."""
    return norm.pdf(x)


def compute_rates(gamma_grid, beta, eta_bar, delta_eta):
    """Compute R^+ and R^- transition rates."""
    b = 2 * Phi(beta * gamma_grid) - 1
    R_plus = 0.5 * (1 + eta_bar + delta_eta * b)
    R_minus = 0.5 * (1 - eta_bar - delta_eta * b)
    return R_plus, R_minus, b


def master_equation_step(pdf, R_plus, R_minus):
    """
    One step of the master equation.
    p_{n+1}(gamma) = R^+(gamma-1) p_n(gamma-1) + R^-(gamma+1) p_n(gamma+1)
    """
    pdf_new = np.zeros_like(pdf)
    # R^+(gamma-1) * p_n(gamma-1) contributes to p_{n+1}(gamma)
    # i.e., R^+(i) * p_n(i) contributes to p_{n+1}(i+1)
    pdf_new[1:] += R_plus[:-1] * pdf[:-1]
    # R^-(gamma+1) * p_n(gamma+1) contributes to p_{n+1}(gamma)
    # i.e., R^-(i) * p_n(i) contributes to p_{n+1}(i-1)
    pdf_new[:-1] += R_minus[1:] * pdf[1:]
    return pdf_new


def compute_cumulants_from_pdf(pdf, gamma_grid):
    """Compute mean and variance from discrete PDF."""
    mu = np.sum(gamma_grid * pdf)
    var = np.sum((gamma_grid - mu)**2 * pdf)
    # Also compute 3rd and 4th cumulants
    kappa3 = np.sum((gamma_grid - mu)**3 * pdf)
    kappa4 = np.sum((gamma_grid - mu)**4 * pdf) - 3 * var**2
    return mu, var, kappa3, kappa4


def gaussian_cumulant_step(mu, sigma_sq, beta, eta_bar, delta_eta):
    """
    One step of the Gaussian cumulant dynamics.

    mu' = mu + eta_bar + delta_eta * <b>
    sigma'^2 = sigma^2 + 1 - <v>^2 + 2*delta_eta*Cov(gamma, b)

    where:
      beta_tilde = beta / sqrt(1 + beta^2 * sigma^2)
      <b> = 2*Phi(beta_tilde * mu) - 1
      <v> = eta_bar + delta_eta * <b>
      Cov(gamma, b) = 2*beta*sigma^2 / sqrt(1 + beta^2*sigma^2) * phi(beta_tilde * mu)
    """
    sigma = np.sqrt(sigma_sq)
    denom = np.sqrt(1 + beta**2 * sigma_sq)
    beta_tilde = beta / denom

    # Mean of b
    b_mean = 2 * Phi(beta_tilde * mu) - 1

    # Mean of v
    v_mean = eta_bar + delta_eta * b_mean

    # Covariance
    cov_gamma_b = 2 * beta * sigma_sq / denom * phi(beta_tilde * mu)

    # Updates
    mu_new = mu + eta_bar + delta_eta * b_mean
    sigma_sq_new = sigma_sq + 1 - v_mean**2 + 2 * delta_eta * cov_gamma_b

    return mu_new, sigma_sq_new


def run_comparison(m_plus, m_minus, beta, eta_bar, delta_eta, n_steps=5):
    """Run comparison between master equation and Gaussian dynamics."""

    # Grid for master equation (needs to be wide enough)
    gamma_max = int(m_plus + m_minus + 3 * np.sqrt(m_plus + m_minus + n_steps) + n_steps + 10)
    gamma_grid = np.arange(-gamma_max, gamma_max + 1)

    # Initialize
    pdf, mu0, sigma_sq0 = build_initial_pdf(m_plus, m_minus, gamma_grid)
    R_plus, R_minus, b = compute_rates(gamma_grid, beta, eta_bar, delta_eta)

    # Storage for results
    results_master = {'mu': [mu0], 'var': [sigma_sq0], 'kappa3': [0.0], 'kappa4': [0.0]}
    results_gauss = {'mu': [mu0], 'var': [sigma_sq0]}

    # Verify initial cumulants
    mu_check, var_check, k3, k4 = compute_cumulants_from_pdf(pdf, gamma_grid)
    results_master['kappa3'][0] = k3
    results_master['kappa4'][0] = k4

    print(f"Initial state: m_+ = {m_plus}, m_- = {m_minus}")
    print(f"Parameters: beta = {beta}, eta_bar = {eta_bar}, delta_eta = {delta_eta}")
    print(f"\nInitial PDF check: mu = {mu_check:.6f} (expected {mu0}), var = {var_check:.6f} (expected {sigma_sq0})")
    print()

    # Evolution
    mu_g, var_g = mu0, sigma_sq0

    print("=" * 80)
    print(f"{'Step':>4} | {'Master mu':>12} {'Gauss mu':>12} {'diff':>10} | "
          f"{'Master var':>12} {'Gauss var':>12} {'diff':>10}")
    print("=" * 80)

    for step in range(1, n_steps + 1):
        # Master equation step
        pdf = master_equation_step(pdf, R_plus, R_minus)
        mu_m, var_m, k3, k4 = compute_cumulants_from_pdf(pdf, gamma_grid)

        # Gaussian step
        mu_g, var_g = gaussian_cumulant_step(mu_g, var_g, beta, eta_bar, delta_eta)

        # Store
        results_master['mu'].append(mu_m)
        results_master['var'].append(var_m)
        results_master['kappa3'].append(k3)
        results_master['kappa4'].append(k4)
        results_gauss['mu'].append(mu_g)
        results_gauss['var'].append(var_g)

        # Print comparison
        mu_diff = mu_m - mu_g
        var_diff = var_m - var_g
        print(f"{step:>4} | {mu_m:>12.6f} {mu_g:>12.6f} {mu_diff:>10.2e} | "
              f"{var_m:>12.6f} {var_g:>12.6f} {var_diff:>10.2e}")

    print("=" * 80)

    # Print higher cumulants (should be small for Gaussian)
    print("\nHigher cumulants from master equation (should be ~0 for Gaussian):")
    print(f"{'Step':>4} | {'kappa_3':>12} | {'kappa_4':>12}")
    print("-" * 40)
    for step in range(n_steps + 1):
        print(f"{step:>4} | {results_master['kappa3'][step]:>12.4f} | {results_master['kappa4'][step]:>12.4f}")

    return results_master, results_gauss, pdf, gamma_grid


def plot_comparison(results_master, results_gauss, pdf_final, gamma_grid, figname):
    """Plot the comparison results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    steps = np.arange(len(results_master['mu']))

    # Mean comparison
    ax = axes[0, 0]
    ax.plot(steps, results_master['mu'], 'b-o', label='Master equation', markersize=8)
    ax.plot(steps, results_gauss['mu'], 'r--s', label='Gaussian approx', markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean (mu)')
    ax.set_title('Mean evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Variance comparison
    ax = axes[0, 1]
    ax.plot(steps, results_master['var'], 'b-o', label='Master equation', markersize=8)
    ax.plot(steps, results_gauss['var'], 'r--s', label='Gaussian approx', markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Variance (sigma^2)')
    ax.set_title('Variance evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Errors
    ax = axes[1, 0]
    mu_err = np.array(results_master['mu']) - np.array(results_gauss['mu'])
    var_err = np.array(results_master['var']) - np.array(results_gauss['var'])
    ax.plot(steps, mu_err, 'b-o', label='Mean error', markersize=8)
    ax.plot(steps, var_err, 'g-s', label='Variance error', markersize=6)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Error (Master - Gaussian)')
    ax.set_title('Approximation errors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final PDF
    ax = axes[1, 1]
    # Only plot where pdf is significant
    mask = pdf_final > 1e-10
    ax.bar(gamma_grid[mask], pdf_final[mask], width=0.8, alpha=0.7, label='Master equation')
    # Overlay Gaussian
    mu_final = results_gauss['mu'][-1]
    var_final = results_gauss['var'][-1]
    g_cont = np.linspace(gamma_grid[mask].min(), gamma_grid[mask].max(), 200)
    gauss_pdf = norm.pdf(g_cont, loc=mu_final, scale=np.sqrt(var_final))
    # Scale to match discrete (multiply by grid spacing = 1)
    ax.plot(g_cont, gauss_pdf, 'r-', linewidth=2, label='Gaussian approx')
    ax.set_xlabel('gamma')
    ax.set_ylabel('Probability')
    ax.set_title('Final PDF comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f"\nFigure saved to {figname}")


if __name__ == "__main__":
    # Parameters
    m_plus = 10
    m_minus = 10
    beta = 0.3
    eta_plus = 0.6   # p(y=+1|a=+) = (1 + eta_+)/2 = 0.8
    eta_minus = 0.4  # p(y=+1|a=-) = (1 + eta_-)/2 = 0.7

    eta_bar = (eta_plus + eta_minus) / 2      # = 0.5
    delta_eta = (eta_plus - eta_minus) / 2    # = 0.1

    n_steps = 5

    print("=" * 80)
    print("COMPARISON: Master Equation vs Gaussian Cumulant Dynamics")
    print("=" * 80)
    print()

    results_master, results_gauss, pdf_final, gamma_grid = run_comparison(
        m_plus, m_minus, beta, eta_bar, delta_eta, n_steps
    )

    # Save figure
    plot_comparison(results_master, results_gauss, pdf_final, gamma_grid,
                   '../fig/cumulant_comparison.png')

    # Also test with asymmetric initial condition
    print("\n" + "=" * 80)
    print("TEST 2: Asymmetric initial condition (m_+ = 12, m_- = 8)")
    print("=" * 80)
    print()

    results_master2, results_gauss2, pdf_final2, gamma_grid2 = run_comparison(
        12, 8, beta, eta_bar, delta_eta, n_steps
    )

    plot_comparison(results_master2, results_gauss2, pdf_final2, gamma_grid2,
                   '../fig/cumulant_comparison_asymm.png')
