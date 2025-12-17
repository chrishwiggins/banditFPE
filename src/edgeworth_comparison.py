"""
Compare master equation cumulants with K-truncated Edgeworth dynamics.

Tests K = 1, 2, 5, 10 and measures:
1. Cumulant trajectory error vs time
2. Error convergence as K increases
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from cumulant_comparison import (
    build_initial_pdf, Phi, phi, compute_rates,
    master_equation_step, compute_cumulants_from_pdf, gaussian_cumulant_step
)
from edgeworth_cumulants import EdgeworthDynamics, ProbitIntegrals


def compute_cumulants_from_pdf_extended(pdf, gamma_grid, max_order=10):
    """Compute cumulants kappa_1 through kappa_{max_order} from discrete PDF."""
    mu = np.sum(gamma_grid * pdf)
    centered = gamma_grid - mu

    # Central moments
    moments = [1.0, 0.0]  # m_0 = 1, m_1 = 0
    for k in range(2, max_order + 1):
        moments.append(np.sum(centered**k * pdf))

    # Convert to cumulants using moment-cumulant relations
    # kappa_1 = mu
    # kappa_2 = m_2
    # kappa_3 = m_3
    # kappa_4 = m_4 - 3*m_2^2
    # kappa_5 = m_5 - 10*m_3*m_2
    # kappa_6 = m_6 - 15*m_4*m_2 - 10*m_3^2 + 30*m_2^3
    # etc.

    kappas = np.zeros(max_order + 1)
    kappas[0] = 0  # placeholder (index 0 unused)
    kappas[1] = mu
    if max_order >= 2:
        kappas[2] = moments[2]
    if max_order >= 3:
        kappas[3] = moments[3]
    if max_order >= 4:
        kappas[4] = moments[4] - 3*moments[2]**2
    if max_order >= 5:
        kappas[5] = moments[5] - 10*moments[3]*moments[2]
    if max_order >= 6:
        kappas[6] = moments[6] - 15*moments[4]*moments[2] - 10*moments[3]**2 + 30*moments[2]**3
    if max_order >= 7:
        kappas[7] = moments[7] - 21*moments[5]*moments[2] - 35*moments[4]*moments[3] + 210*moments[3]*moments[2]**2
    if max_order >= 8:
        kappas[8] = (moments[8] - 28*moments[6]*moments[2] - 56*moments[5]*moments[3]
                    - 35*moments[4]**2 + 420*moments[4]*moments[2]**2
                    + 560*moments[3]**2*moments[2] - 630*moments[2]**4)
    # Higher orders get very complex - truncate here

    return kappas[1:max_order+1]  # Return kappa_1, ..., kappa_{max_order}


def run_master_equation(m_plus, m_minus, beta, eta_bar, delta_eta, n_steps, max_cumulant=10):
    """Run master equation and track cumulants."""
    # Wide grid
    gamma_max = int(m_plus + m_minus + 3 * np.sqrt(m_plus + m_minus + n_steps) + n_steps * abs(eta_bar) + 50)
    gamma_grid = np.arange(-gamma_max, gamma_max + 1)

    # Initialize
    pdf, mu0, sigma_sq0 = build_initial_pdf(m_plus, m_minus, gamma_grid)
    R_plus, R_minus, _ = compute_rates(gamma_grid, beta, eta_bar, delta_eta)

    # Get initial cumulants
    kappas_init = compute_cumulants_from_pdf_extended(pdf, gamma_grid, max_cumulant)

    # Storage
    trajectory = np.zeros((n_steps + 1, max_cumulant))
    trajectory[0, :] = kappas_init

    for step in range(1, n_steps + 1):
        pdf = master_equation_step(pdf, R_plus, R_minus)
        kappas = compute_cumulants_from_pdf_extended(pdf, gamma_grid, max_cumulant)
        trajectory[step, :] = kappas

    return trajectory


def run_edgeworth_K(K, m_plus, m_minus, beta, eta_bar, delta_eta, n_steps):
    """Run K-truncated Edgeworth dynamics."""
    mu0 = m_plus - m_minus
    sigma_sq0 = m_plus + m_minus

    # Initialize cumulants (higher cumulants start at 0 for Gaussian init)
    kappas_init = np.zeros(K)
    kappas_init[0] = mu0
    if K >= 2:
        kappas_init[1] = sigma_sq0

    # Create dynamics
    dynamics = EdgeworthDynamics(K, beta, eta_bar, delta_eta)

    # Evolve (for K=1, pass initial sigma_sq)
    trajectory = dynamics.evolve(kappas_init, n_steps, initial_sigma_sq=sigma_sq0)

    return trajectory


def run_comparison(m_plus, m_minus, beta, eta_bar, delta_eta, n_steps, K_values=[1, 2, 5, 10]):
    """Run comparison for multiple K values."""
    print("=" * 70)
    print("EDGEWORTH-K TRUNCATION COMPARISON")
    print("=" * 70)
    print(f"Parameters: m_+ = {m_plus}, m_- = {m_minus}, beta = {beta}")
    print(f"            eta_bar = {eta_bar}, delta_eta = {delta_eta}")
    print(f"            n_steps = {n_steps}")
    print(f"            K values: {K_values}")
    print()

    # Run master equation
    print("Running master equation...")
    max_K = max(K_values)
    master_traj = run_master_equation(m_plus, m_minus, beta, eta_bar, delta_eta, n_steps, max_K)
    print(f"  Final mean (master): {master_traj[-1, 0]:.4f}")
    print(f"  Final var (master):  {master_traj[-1, 1]:.4f}")
    print()

    # Run Edgeworth for each K
    edgeworth_trajs = {}
    for K in K_values:
        print(f"Running Edgeworth-{K}...")
        traj = run_edgeworth_K(K, m_plus, m_minus, beta, eta_bar, delta_eta, n_steps)
        edgeworth_trajs[K] = traj
        print(f"  Final mean (K={K}): {traj[-1, 0]:.4f}")
        if K >= 2:
            print(f"  Final var (K={K}):  {traj[-1, 1]:.4f}")
        print()

    return master_traj, edgeworth_trajs


def plot_comparison(master_traj, edgeworth_trajs, figname):
    """Plot comparison of cumulant trajectories and errors."""
    n_steps = len(master_traj) - 1
    steps = np.arange(n_steps + 1)
    K_values = sorted(edgeworth_trajs.keys())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = {1: 'red', 2: 'blue', 5: 'green', 10: 'purple'}

    # Mean trajectory
    ax = axes[0, 0]
    ax.plot(steps, master_traj[:, 0], 'k-', linewidth=2, label='Master')
    for K in K_values:
        ax.plot(steps, edgeworth_trajs[K][:, 0], '--', color=colors.get(K, 'gray'),
                label=f'K={K}', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean (kappa_1)')
    ax.set_title('Mean evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Variance trajectory (K >= 2 only)
    ax = axes[0, 1]
    ax.plot(steps, master_traj[:, 1], 'k-', linewidth=2, label='Master')
    for K in K_values:
        if K >= 2:
            ax.plot(steps, edgeworth_trajs[K][:, 1], '--', color=colors.get(K, 'gray'),
                    label=f'K={K}', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Variance (kappa_2)')
    ax.set_title('Variance evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean error
    ax = axes[0, 2]
    for K in K_values:
        err = np.abs(master_traj[:, 0] - edgeworth_trajs[K][:, 0])
        ax.semilogy(steps[1:], err[1:] + 1e-16, '-', color=colors.get(K, 'gray'),
                   label=f'K={K}', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('|Mean error|')
    ax.set_title('Mean error vs time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Variance error
    ax = axes[1, 0]
    for K in K_values:
        if K >= 2:
            err = np.abs(master_traj[:, 1] - edgeworth_trajs[K][:, 1])
            ax.semilogy(steps[1:], err[1:] + 1e-16, '-', color=colors.get(K, 'gray'),
                       label=f'K={K}', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('|Variance error|')
    ax.set_title('Variance error vs time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # kappa_3 from master (should be small if starting Gaussian)
    ax = axes[1, 1]
    ax.plot(steps, master_traj[:, 2], 'k-', linewidth=2, label='kappa_3 (Master)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('kappa_3')
    ax.set_title('Skewness evolution (Master only)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error at final time vs K
    ax = axes[1, 2]
    K_arr = np.array(K_values)
    final_mean_err = np.array([np.abs(master_traj[-1, 0] - edgeworth_trajs[K][-1, 0]) for K in K_values])
    final_var_err = np.array([np.abs(master_traj[-1, 1] - edgeworth_trajs[K][-1, 1]) if K >= 2 else np.nan for K in K_values])

    ax.semilogy(K_arr, final_mean_err + 1e-16, 'bo-', label='Mean error', markersize=8)
    ax.semilogy(K_arr[~np.isnan(final_var_err)], final_var_err[~np.isnan(final_var_err)] + 1e-16,
               'rs-', label='Variance error', markersize=8)
    ax.set_xlabel('K (truncation order)')
    ax.set_ylabel('|Error at final time|')
    ax.set_title('Error convergence in K')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f"Figure saved to {figname}")


def print_error_table(master_traj, edgeworth_trajs, times=[5, 10, 20]):
    """Print error table at specified times."""
    K_values = sorted(edgeworth_trajs.keys())
    n_steps = len(master_traj) - 1

    print("\n" + "=" * 70)
    print("ERROR TABLE: |kappa_j^{master} - kappa_j^{Edgeworth}|")
    print("=" * 70)

    for t in times:
        if t > n_steps:
            continue
        print(f"\nTime t = {t}:")
        print(f"{'K':>4} | {'Mean err':>12} | {'Var err':>12}")
        print("-" * 35)
        for K in K_values:
            mean_err = np.abs(master_traj[t, 0] - edgeworth_trajs[K][t, 0])
            var_err = np.abs(master_traj[t, 1] - edgeworth_trajs[K][t, 1]) if K >= 2 else float('nan')
            print(f"{K:>4} | {mean_err:>12.2e} | {var_err:>12.2e}")


if __name__ == "__main__":
    # Parameters
    m_plus = 10
    m_minus = 10
    beta = 0.3
    eta_plus = 0.6
    eta_minus = 0.4
    eta_bar = (eta_plus + eta_minus) / 2
    delta_eta = (eta_plus - eta_minus) / 2

    n_steps = 20
    K_values = [1, 2, 5]  # Skip K=10 for now (slow)

    # Run comparison
    master_traj, edgeworth_trajs = run_comparison(
        m_plus, m_minus, beta, eta_bar, delta_eta, n_steps, K_values
    )

    # Plot
    plot_comparison(master_traj, edgeworth_trajs, '../fig/edgeworth_comparison.png')

    # Print error table
    print_error_table(master_traj, edgeworth_trajs, times=[10, 25, 50])
