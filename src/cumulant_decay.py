"""
Study long-time decay of cumulants in the bandit master equation.

Hypothesis: kappa_j ~ t^{-nu_j} with nu_j being simple rationals (denominator 2 or 3).

We track standardized cumulants (skewness, excess kurtosis, etc.) which should
decay as the distribution approaches Gaussian at long times.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


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
    return R_plus, R_minus


def master_equation_step(pdf, R_plus, R_minus):
    """
    One step of the master equation.
    p_{n+1}(gamma) = R^+(gamma-1) p_n(gamma-1) + R^-(gamma+1) p_n(gamma+1)

    R^+(gamma) is the rate to jump from gamma to gamma+1
    R^-(gamma) is the rate to jump from gamma to gamma-1

    So: from state i, probability R^+(i) to go to i+1, R^-(i) to go to i-1

    We use reflecting boundaries: mass that would leave stays at the boundary.
    """
    pdf_new = np.zeros_like(pdf)
    n = len(pdf)

    # From state i, go to i+1 with prob R^+(i)
    # Contribution to pdf_new[i+1] from pdf[i] * R^+(i)
    pdf_new[1:] += R_plus[:-1] * pdf[:-1]

    # From state i, go to i-1 with prob R^-(i)
    # Contribution to pdf_new[i-1] from pdf[i] * R^-(i)
    pdf_new[:-1] += R_minus[1:] * pdf[1:]

    # Reflecting boundaries: mass that would leave is reflected back
    # At left boundary (i=0): R^-(0)*pdf[0] would go to i=-1, reflect to i=0
    pdf_new[0] += R_minus[0] * pdf[0]
    # At right boundary (i=n-1): R^+(n-1)*pdf[n-1] would go to i=n, reflect to i=n-1
    pdf_new[-1] += R_plus[-1] * pdf[-1]

    return pdf_new


def compute_cumulants_from_pdf(pdf, gamma_grid, max_order=6):
    """Compute cumulants up to given order from discrete PDF."""
    mu = np.sum(gamma_grid * pdf)
    centered = gamma_grid - mu

    # Raw central moments
    moments = [1.0, 0.0]  # m_0 = 1, m_1 = 0
    for k in range(2, max_order + 1):
        moments.append(np.sum(centered**k * pdf))

    # Convert to cumulants
    # kappa_1 = mu
    # kappa_2 = m_2
    # kappa_3 = m_3
    # kappa_4 = m_4 - 3*m_2^2
    # kappa_5 = m_5 - 10*m_3*m_2
    # kappa_6 = m_6 - 15*m_4*m_2 - 10*m_3^2 + 30*m_2^3

    kappa = [0.0, mu, moments[2], moments[3],
             moments[4] - 3*moments[2]**2,
             moments[5] - 10*moments[3]*moments[2],
             moments[6] - 15*moments[4]*moments[2] - 10*moments[3]**2 + 30*moments[2]**3]

    return kappa


def gaussian_cumulant_step(mu, sigma_sq, beta, eta_bar, delta_eta):
    """One step of the Gaussian cumulant dynamics."""
    denom = np.sqrt(1 + beta**2 * sigma_sq)
    beta_tilde = beta / denom
    b_mean = 2 * Phi(beta_tilde * mu) - 1
    v_mean = eta_bar + delta_eta * b_mean
    cov_gamma_b = 2 * beta * sigma_sq / denom * phi(beta_tilde * mu)

    mu_new = mu + eta_bar + delta_eta * b_mean
    sigma_sq_new = sigma_sq + 1 - v_mean**2 + 2 * delta_eta * cov_gamma_b

    return mu_new, sigma_sq_new


def power_law(t, A, nu):
    """Power law: A * t^(-nu)"""
    return A * t**(-nu)


def fit_power_law(t_data, y_data, min_t=None):
    """Fit power law to data, optionally starting from min_t."""
    if min_t is not None:
        mask = t_data >= min_t
        t_fit = t_data[mask]
        y_fit = y_data[mask]
    else:
        t_fit = t_data
        y_fit = y_data

    # Only fit where y > 0
    mask = y_fit > 0
    if np.sum(mask) < 3:
        return None, None, None, None

    t_fit = t_fit[mask]
    y_fit = y_fit[mask]

    # Log-log fit
    log_t = np.log(t_fit)
    log_y = np.log(y_fit)

    # Linear regression in log-log space
    coeffs = np.polyfit(log_t, log_y, 1)
    nu = -coeffs[0]
    A = np.exp(coeffs[1])

    return A, nu, t_fit, y_fit


def run_long_time_simulation(m_plus, m_minus, beta, eta_bar, delta_eta, n_steps):
    """Run simulation and track cumulants."""

    # Need a very wide grid for long times
    # Variance grows roughly linearly, so sigma ~ sqrt(n)
    # Need grid >> 6*sigma_max to avoid boundary effects
    sigma_max_expected = np.sqrt(n_steps + m_plus + m_minus)
    gamma_max = int(10 * sigma_max_expected + n_steps * abs(eta_bar) + 100)
    gamma_grid = np.arange(-gamma_max, gamma_max + 1)
    print(f"Grid: gamma in [{-gamma_max}, {gamma_max}], size = {len(gamma_grid)}")
    print(f"Expected sigma_max ~ {sigma_max_expected:.1f}, grid covers +/- {gamma_max/sigma_max_expected:.1f} sigma")

    # Initialize with Gaussian
    mu0 = m_plus - m_minus
    sigma0 = np.sqrt(m_plus + m_minus)
    pdf = norm.pdf(gamma_grid, loc=mu0, scale=sigma0)
    pdf /= pdf.sum()

    R_plus, R_minus = compute_rates(gamma_grid, beta, eta_bar, delta_eta)

    # Storage - sample at logarithmic intervals
    sample_times = np.unique(np.logspace(0, np.log10(n_steps), 200).astype(int))
    sample_times = sample_times[sample_times <= n_steps]

    results = {
        't': [],
        'kappa': [[] for _ in range(7)],  # kappa_1 through kappa_6
        'kappa_gauss': [[] for _ in range(3)]  # mu, sigma^2 from Gaussian approx
    }

    # Initial
    kappa = compute_cumulants_from_pdf(pdf, gamma_grid)
    results['t'].append(0)
    for j in range(7):
        results['kappa'][j].append(kappa[j] if j < len(kappa) else 0)
    results['kappa_gauss'][0].append(mu0)
    results['kappa_gauss'][1].append(sigma0**2)

    # Gaussian tracking
    mu_g, var_g = float(mu0), float(sigma0**2)

    current_sample_idx = 0
    if sample_times[0] == 0:
        current_sample_idx = 1

    print(f"Running {n_steps} steps...")
    for step in range(1, n_steps + 1):
        # Master equation
        pdf = master_equation_step(pdf, R_plus, R_minus)

        # Gaussian approximation
        mu_g, var_g = gaussian_cumulant_step(mu_g, var_g, beta, eta_bar, delta_eta)

        # Sample at specified times
        if current_sample_idx < len(sample_times) and step == sample_times[current_sample_idx]:
            kappa = compute_cumulants_from_pdf(pdf, gamma_grid)
            results['t'].append(step)
            for j in range(7):
                results['kappa'][j].append(kappa[j] if j < len(kappa) else 0)
            results['kappa_gauss'][0].append(mu_g)
            results['kappa_gauss'][1].append(var_g)
            current_sample_idx += 1

            if step % 1000 == 0:
                prob_mass = pdf.sum()
                print(f"  Step {step}: mu={kappa[1]:.2f}, sigma^2={kappa[2]:.2f}, "
                      f"kappa_3={kappa[3]:.4f}, kappa_4={kappa[4]:.4f}, mass={prob_mass:.6f}")

    # Convert to arrays
    results['t'] = np.array(results['t'])
    for j in range(7):
        results['kappa'][j] = np.array(results['kappa'][j])
    for j in range(3):
        results['kappa_gauss'][j] = np.array(results['kappa_gauss'][j])

    return results


def analyze_and_plot(results, figname):
    """Analyze power-law decay and create plots."""
    t = results['t']

    # Compute standardized cumulants
    sigma = np.sqrt(results['kappa'][2])

    # Standardized cumulants: kappa_j / sigma^j
    # But for decay analysis, look at |kappa_j|
    kappa_3 = np.abs(results['kappa'][3])
    kappa_4 = np.abs(results['kappa'][4])
    kappa_5 = np.abs(results['kappa'][5])
    kappa_6 = np.abs(results['kappa'][6])

    # Compute standardized cumulants: kappa_j / sigma^j
    skewness = np.abs(results['kappa'][3]) / sigma**3
    ex_kurtosis = np.abs(results['kappa'][4]) / sigma**4
    std_kappa_5 = np.abs(results['kappa'][5]) / sigma**5
    std_kappa_6 = np.abs(results['kappa'][6]) / sigma**6

    # Fit power laws (skip early transient)
    min_t_fit = 500

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    cumulants_to_fit = [
        (skewness, 'Skewness |kappa_3/sigma^3|', 'blue'),
        (ex_kurtosis, 'Ex. Kurtosis |kappa_4/sigma^4|', 'red'),
        (std_kappa_5, '|kappa_5/sigma^5|', 'green'),
        (std_kappa_6, '|kappa_6/sigma^6|', 'purple'),
        (kappa_3, '|kappa_3| (raw)', 'orange'),
        (kappa_4, '|kappa_4| (raw)', 'brown'),
    ]

    print("\n" + "=" * 70)
    print("POWER LAW FITS: kappa ~ A * t^(-nu)")
    print("=" * 70)

    for idx, (data, label, color) in enumerate(cumulants_to_fit):
        ax = axes[idx // 3, idx % 3]

        # Filter positive values for log plot
        mask = (t > 0) & (data > 0)
        t_pos = t[mask]
        data_pos = data[mask]

        if len(t_pos) < 10:
            ax.set_title(f'{label}: insufficient data')
            continue

        # Plot data
        ax.loglog(t_pos, data_pos, 'o', color=color, alpha=0.5, markersize=3, label='Data')

        # Fit power law
        A, nu, t_fit, y_fit = fit_power_law(t_pos, data_pos, min_t=min_t_fit)

        if A is not None:
            # Plot fit
            t_line = np.logspace(np.log10(min_t_fit), np.log10(t_pos.max()), 100)
            ax.loglog(t_line, power_law(t_line, A, nu), '--', color='black',
                     linewidth=2, label=f'Fit: nu = {nu:.3f}')

            # Find nearest simple rational
            candidates = [1/3, 1/2, 2/3, 1, 3/2, 2, 5/2, 3]
            nearest = min(candidates, key=lambda x: abs(x - nu))

            print(f"{label:35s}: nu = {nu:.4f}  (nearest rational: {nearest} = {nearest:.4f})")

            ax.set_title(f'{label}\nnu = {nu:.3f} (nearest: {nearest})')
        else:
            ax.set_title(f'{label}: fit failed')

        ax.set_xlabel('t')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f"\nFigure saved to {figname}")

    return


def compare_master_vs_continuum(results, figname):
    """Compare cumulant evolution: master equation vs Gaussian approximation."""
    t = results['t']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mean
    ax = axes[0, 0]
    ax.plot(t, results['kappa'][1], 'b-', label='Master equation', linewidth=1)
    ax.plot(t, results['kappa_gauss'][0], 'r--', label='Gaussian approx', linewidth=1)
    ax.set_xlabel('t')
    ax.set_ylabel('Mean (mu)')
    ax.set_title('Mean evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Variance
    ax = axes[0, 1]
    ax.plot(t, results['kappa'][2], 'b-', label='Master equation', linewidth=1)
    ax.plot(t, results['kappa_gauss'][1], 'r--', label='Gaussian approx', linewidth=1)
    ax.set_xlabel('t')
    ax.set_ylabel('Variance (sigma^2)')
    ax.set_title('Variance evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean error
    ax = axes[1, 0]
    mu_err = results['kappa'][1] - results['kappa_gauss'][0]
    ax.semilogx(t[t>0], mu_err[t>0], 'b-', linewidth=1)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('t')
    ax.set_ylabel('Mean error')
    ax.set_title('Mean: Master - Gaussian')
    ax.grid(True, alpha=0.3)

    # Variance error
    ax = axes[1, 1]
    var_err = results['kappa'][2] - results['kappa_gauss'][1]
    ax.semilogx(t[t>0], var_err[t>0], 'b-', linewidth=1)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('t')
    ax.set_ylabel('Variance error')
    ax.set_title('Variance: Master - Gaussian')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f"Figure saved to {figname}")


if __name__ == "__main__":
    # Parameters - identical arms (delta_eta=0) removes feedback instability
    # but nonzero eta_bar gives drift, nonzero beta gives gamma-dependent rates
    m_plus = 10
    m_minus = 10
    beta = 0.3
    eta_plus = 0.3   # both arms have same reward distribution
    eta_minus = 0.3  # p(y=+1|a) = 0.65 for both
    eta_bar = (eta_plus + eta_minus) / 2      # = 0.3
    delta_eta = (eta_plus - eta_minus) / 2    # = 0

    n_steps = 50000

    print("=" * 70)
    print("LONG-TIME CUMULANT DECAY ANALYSIS")
    print("=" * 70)
    print(f"Parameters: m_+ = {m_plus}, m_- = {m_minus}, beta = {beta}")
    print(f"            eta_bar = {eta_bar}, delta_eta = {delta_eta}")
    print(f"            n_steps = {n_steps}")
    print()

    results = run_long_time_simulation(m_plus, m_minus, beta, eta_bar, delta_eta, n_steps)

    analyze_and_plot(results, '../fig/cumulant_decay.png')
    compare_master_vs_continuum(results, '../fig/cumulant_master_vs_gauss.png')
