"""
Truncation Rate Error: epsilon_j^K(t)

Measures the instantaneous closure error:
  epsilon_j^K(t) = |Delta_kappa_j^{exact}(t) - Delta_kappa_j^{K-ansatz}(t)|

Where:
- Delta_kappa_j^{exact} = kappa_j(t+1) - kappa_j(t) from exact master equation
- Delta_kappa_j^{K-ansatz} = analytic formula using K-truncated Edgeworth,
  evaluated with the EXACT cumulants kappa_1^0(t), ..., kappa_K^0(t)

This is NOT trajectory error (which accumulates). This measures how well
the K-truncated closure formula predicts the next step, given the true state.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from edgeworth_cumulants import EdgeworthDynamics, ProbitIntegrals


def Phi(x):
    """Standard normal CDF."""
    return norm.cdf(x)


def phi(x):
    """Standard normal PDF."""
    return norm.pdf(x)


def compute_rates(gamma_grid, beta, eta_bar, delta_eta):
    """Compute R^+ and R^- transition rates."""
    b = 2 * Phi(beta * gamma_grid) - 1
    R_plus = 0.5 * (1 + eta_bar + delta_eta * b)
    R_minus = 0.5 * (1 - eta_bar - delta_eta * b)
    return R_plus, R_minus


def master_equation_step(pdf, R_plus, R_minus):
    """One step of the master equation."""
    pdf_new = np.zeros_like(pdf)
    pdf_new[1:] += R_plus[:-1] * pdf[:-1]
    pdf_new[:-1] += R_minus[1:] * pdf[1:]
    return pdf_new


def compute_cumulants_from_pdf(pdf, gamma_grid, max_order=6):
    """Compute cumulants kappa_1 through kappa_{max_order} from discrete PDF."""
    mu = np.sum(gamma_grid * pdf)
    centered = gamma_grid - mu

    # Central moments
    moments = [1.0, 0.0]
    for k in range(2, max_order + 1):
        moments.append(np.sum(centered**k * pdf))

    # Convert to cumulants
    kappas = np.zeros(max_order)
    kappas[0] = mu  # kappa_1 = mean
    if max_order >= 2:
        kappas[1] = moments[2]  # kappa_2 = variance
    if max_order >= 3:
        kappas[2] = moments[3]  # kappa_3
    if max_order >= 4:
        kappas[3] = moments[4] - 3*moments[2]**2  # kappa_4
    if max_order >= 5:
        kappas[4] = moments[5] - 10*moments[3]*moments[2]  # kappa_5
    if max_order >= 6:
        kappas[5] = (moments[6] - 15*moments[4]*moments[2]
                    - 10*moments[3]**2 + 30*moments[2]**3)  # kappa_6

    return kappas


def compute_ansatz_rate(kappas, K, beta, eta_bar, delta_eta):
    """
    Compute Delta_kappa_j for j=1,...,K using K-truncated Edgeworth ansatz.

    Uses the exact cumulants kappas[0:K] as input.
    """
    dynamics = EdgeworthDynamics(K, beta, eta_bar, delta_eta)

    # For K=1, we need to pass sigma_sq separately
    if K == 1:
        sigma_sq = kappas[1] if len(kappas) > 1 else 20.0
        d_kappas = dynamics.cumulant_update(kappas[:K], fixed_sigma_sq=sigma_sq)
    else:
        d_kappas = dynamics.cumulant_update(kappas[:K])

    return d_kappas


def compute_truncation_rate_error(m_plus, m_minus, beta, eta_bar, delta_eta, T, K_values):
    """
    Compute epsilon_j^K(t) for all j and K.

    Returns:
        times: array of time points [0, 1, ..., T-1]
        exact_rates: array (T, max_j) of Delta_kappa_j^{exact}(t)
        ansatz_rates: dict K -> array (T, K) of Delta_kappa_j^{K-ansatz}(t)
        errors: dict K -> array (T, K) of epsilon_j^K(t)
        exact_cumulants: array (T+1, max_j) of kappa_j^0(t)
    """
    max_K = max(K_values)
    max_j = max_K + 2  # Track a couple extra for reference

    # Grid with bounded support for discrete random walker
    gamma_max = m_plus + m_minus + 2 * T + 10
    gamma_grid = np.arange(-gamma_max, gamma_max + 1)

    # Initialize PDF (Gaussian approximation to binomial)
    mu0 = m_plus - m_minus
    sigma0 = np.sqrt(m_plus + m_minus)
    pdf = norm.pdf(gamma_grid, loc=mu0, scale=sigma0)
    pdf /= pdf.sum()

    # Transition rates
    R_plus, R_minus = compute_rates(gamma_grid, beta, eta_bar, delta_eta)

    # Storage
    exact_cumulants = np.zeros((T + 1, max_j))
    exact_rates = np.zeros((T, max_j))
    ansatz_rates = {K: np.zeros((T, K)) for K in K_values}
    errors = {K: np.zeros((T, K)) for K in K_values}

    # Initial cumulants
    kappas_prev = compute_cumulants_from_pdf(pdf, gamma_grid, max_j)
    exact_cumulants[0, :] = kappas_prev

    print(f"Computing truncation rate error...")
    print(f"  Grid: gamma in [{-gamma_max}, {gamma_max}]")
    print(f"  T = {T}, K_values = {K_values}")
    print()

    for t in range(T):
        # Advance master equation
        pdf = master_equation_step(pdf, R_plus, R_minus)

        # New exact cumulants
        kappas_new = compute_cumulants_from_pdf(pdf, gamma_grid, max_j)
        exact_cumulants[t + 1, :] = kappas_new

        # Exact rates
        delta_exact = kappas_new - kappas_prev
        exact_rates[t, :] = delta_exact

        # Ansatz rates for each K
        for K in K_values:
            delta_ansatz = compute_ansatz_rate(kappas_prev, K, beta, eta_bar, delta_eta)
            ansatz_rates[K][t, :len(delta_ansatz)] = delta_ansatz

            # Error
            for j in range(min(K, len(delta_exact))):
                errors[K][t, j] = np.abs(delta_exact[j] - delta_ansatz[j])

        kappas_prev = kappas_new

        if (t + 1) % 10 == 0:
            print(f"  t = {t+1}: mu = {kappas_new[0]:.2f}, sigma^2 = {kappas_new[1]:.2f}")

    times = np.arange(T)
    return times, exact_rates, ansatz_rates, errors, exact_cumulants


def plot_truncation_rate_error(times, exact_rates, ansatz_rates, errors, figname,
                               exact_cumulants=None):
    """Generate plots of truncation rate error (K=2 only)."""
    j_labels = ['mean', 'variance']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: epsilon_1^{K=2}(t) for mean
    ax = axes[0, 0]
    ax.semilogy(times, errors[2][:, 0] + 1e-16, 'b-', label='K=2', linewidth=1.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\epsilon_1^{K=2}(t)$ (mean rate error)')
    ax.set_title(r'Truncation rate error for $\Delta\kappa_1$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: epsilon_2^{K=2}(t) for variance
    ax = axes[0, 1]
    ax.semilogy(times, errors[2][:, 1] + 1e-16, 'r-', label='K=2', linewidth=1.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\epsilon_2^{K=2}(t)$ (variance rate error)')
    ax.set_title(r'Truncation rate error for $\Delta\kappa_2$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: epsilon_j^{K=2}(t) for j=1,2
    ax = axes[1, 0]
    for j in range(2):
        ax.semilogy(times, errors[2][:, j] + 1e-16, '-',
                   label=f'j={j+1} ({j_labels[j]})', linewidth=1.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\epsilon_j^{K=2}(t)$')
    ax.set_title('Truncation rate error for K=2, varying j')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Standardized cumulants (if available) or exact vs ansatz
    ax = axes[1, 1]
    if exact_cumulants is not None and len(exact_cumulants) > 0:
        sigma = np.sqrt(exact_cumulants[:, 1])
        skewness = exact_cumulants[:, 2] / sigma**3
        ex_kurtosis = exact_cumulants[:, 3] / sigma**4
        t_cumulants = np.arange(len(exact_cumulants))
        ax.plot(t_cumulants, skewness, 'b-', label='Skewness', linewidth=1.5)
        ax.plot(t_cumulants, ex_kurtosis, 'r-', label='Excess kurtosis', linewidth=1.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time t')
        ax.set_ylabel('Standardized cumulant')
        ax.set_title('Non-Gaussianity measures')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Fallback: show exact vs ansatz variance rate
        ax.plot(times, exact_rates[:, 1], 'k-', linewidth=2, label='Exact')
        ax.plot(times, ansatz_rates[2][:, 1], 'r--', linewidth=1.5, alpha=0.8, label='K=2 ansatz')
        ax.set_xlabel('Time t')
        ax.set_ylabel(r'$\Delta\kappa_2(t)$')
        ax.set_title('Variance rate: exact vs K=2 ansatz')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f"\nFigure saved to {figname}")


def plot_rates_comparison(times, exact_rates, ansatz_rates, figname):
    """Plot exact vs ansatz rates to visualize the comparison."""
    K_values = sorted(ansatz_rates.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean rate: exact vs ansatz
    ax = axes[0, 0]
    ax.plot(times, exact_rates[:, 0], 'k-', linewidth=2, label='Exact')
    for K in K_values:
        ax.plot(times, ansatz_rates[K][:, 0], '--', alpha=0.7, label=f'K={K}')
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\Delta\kappa_1(t)$')
    ax.set_title('Mean rate: exact vs ansatz')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Variance rate: exact vs ansatz
    ax = axes[0, 1]
    ax.plot(times, exact_rates[:, 1], 'k-', linewidth=2, label='Exact')
    for K in K_values:
        if K >= 2:
            ax.plot(times, ansatz_rates[K][:, 1], '--', alpha=0.7, label=f'K={K}')
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\Delta\kappa_2(t)$')
    ax.set_title('Variance rate: exact vs ansatz')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Exact rates for higher cumulants
    ax = axes[1, 0]
    for j in range(min(4, exact_rates.shape[1])):
        ax.plot(times, exact_rates[:, j], label=f'kappa_{j+1}')
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\Delta\kappa_j^{exact}(t)$')
    ax.set_title('Exact cumulant rates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative error
    ax = axes[1, 1]
    for K in K_values:
        if K >= 2:
            rel_err = np.abs(exact_rates[:, 1] - ansatz_rates[K][:, 1]) / (np.abs(exact_rates[:, 1]) + 1e-10)
            ax.semilogy(times, rel_err + 1e-16, '-', label=f'K={K}')
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$|\epsilon_2^K| / |\Delta\kappa_2^{exact}|$')
    ax.set_title('Relative error in variance rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f"Figure saved to {figname}")


def print_error_summary(times, errors):
    """Print summary of errors at various times."""
    K_values = sorted(errors.keys())
    T = len(times)

    print("\n" + "=" * 70)
    print("TRUNCATION RATE ERROR SUMMARY")
    print("=" * 70)

    for t_frac in [0.1, 0.5, 0.9]:
        t = int(t_frac * T)
        if t >= T:
            t = T - 1
        print(f"\nAt t = {t} ({t_frac*100:.0f}% of T):")
        print(f"{'K':>4} | {'eps_1^K (mean)':>15} | {'eps_2^K (var)':>15}")
        print("-" * 45)
        for K in K_values:
            eps1 = errors[K][t, 0]
            eps2 = errors[K][t, 1] if K >= 2 else float('nan')
            print(f"{K:>4} | {eps1:>15.2e} | {eps2:>15.2e}")


if __name__ == "__main__":
    # Parameters
    m_plus = 10
    m_minus = 10
    beta = 0.3
    eta_plus = 0.6
    eta_minus = 0.4
    eta_bar = (eta_plus + eta_minus) / 2
    delta_eta = (eta_plus - eta_minus) / 2

    T = 20
    K_values = [1, 2, 3, 4]

    # Compute errors
    times, exact_rates, ansatz_rates, errors, exact_cumulants = compute_truncation_rate_error(
        m_plus, m_minus, beta, eta_bar, delta_eta, T, K_values
    )

    # Plot
    plot_truncation_rate_error(times, exact_rates, ansatz_rates, errors,
                               '../fig/truncation_rate_error.png')
    plot_rates_comparison(times, exact_rates, ansatz_rates,
                         '../fig/rates_comparison.png')

    # Summary
    print_error_summary(times, errors)
