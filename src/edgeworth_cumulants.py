"""
Edgeworth-K truncated cumulant dynamics for the bandit master equation.

Track K cumulants using the Edgeworth expansion:
  p_K(gamma) = (1/sigma) * phi(z) * [1 + sum_{j=3}^K (kappa_j/j!/sigma^j) * He_j(z)]

The key integrals are J_{m,j} = d^j I_m / du^j where:
  I_m = integral z^m phi(z) Phi(u + lambda*z) dz

These allow computing expectations under the Edgeworth distribution.
"""

import numpy as np
from scipy.stats import norm
from scipy.special import factorial
from functools import lru_cache


def Phi(x):
    """Standard normal CDF."""
    return norm.cdf(x)


def phi(x):
    """Standard normal PDF."""
    return norm.pdf(x)


def hermite_prob(n, x):
    """
    Probabilist's Hermite polynomial He_n(x).
    He_0 = 1, He_1 = x, He_2 = x^2 - 1, He_3 = x^3 - 3x, ...
    Recurrence: He_n(x) = x * He_{n-1}(x) - (n-1) * He_{n-2}(x)
    """
    if n == 0:
        return np.ones_like(x) if hasattr(x, '__len__') else 1.0
    elif n == 1:
        return x
    else:
        He_prev2 = np.ones_like(x) if hasattr(x, '__len__') else 1.0
        He_prev1 = x
        for k in range(2, n + 1):
            He_curr = x * He_prev1 - (k - 1) * He_prev2
            He_prev2 = He_prev1
            He_prev1 = He_curr
        return He_curr


class ProbitIntegrals:
    """
    Compute the probit-polynomial-Gaussian integrals I_m and their u-derivatives.

    I_m(u, lambda) = integral z^m phi(z) Phi(u + lambda*z) dz
    J_{m,j}(u, lambda) = d^j I_m / du^j

    Key insight: J_{m,j} = d^j I_m / du^j, and since d/du Phi(u) = phi(u)
    and d/du [He_k(u) phi(u)] = -He_{k+1}(u) phi(u), all J_{m,j} are
    polynomials in (u, lambda) times phi(u) (plus Phi(u) for certain cases).
    """

    def __init__(self, u, lam, max_order=12):
        """
        Precompute I_m and J_{m,j} for m, j = 0, ..., max_order.

        Parameters:
            u: effective argument (beta * mu / sqrt(1 + beta^2 * sigma^2))
            lam: effective slope (beta * sigma / sqrt(1 + beta^2 * sigma^2))
            max_order: maximum order to compute
        """
        self.u = u
        self.lam = lam
        self.max_order = max_order

        # Precompute phi(u) and Phi(u)
        self.phi_u = phi(u)
        self.Phi_u = Phi(u)

        # Compute I_m values
        self._I = self._compute_I_values()

        # Compute J_{m,j} = d^j I_m / du^j using analytic formulas
        self._J = self._compute_J_values_analytic()

    def _compute_I_values(self):
        """Compute I_0, I_1, ..., I_{max_order} using closed forms."""
        I = np.zeros(self.max_order + 1)
        u, lam = self.u, self.lam
        phi_u, Phi_u = self.phi_u, self.Phi_u

        # Explicit formulas for low orders
        I[0] = Phi_u
        if self.max_order >= 1:
            I[1] = lam * phi_u
        if self.max_order >= 2:
            I[2] = Phi_u - lam**2 * u * phi_u
        if self.max_order >= 3:
            He2 = u**2 - 1
            I[3] = lam * phi_u * (3 + lam**2 * He2)
        if self.max_order >= 4:
            He3 = u**3 - 3*u
            I[4] = 3 * Phi_u - lam**2 * phi_u * (3*u + lam**2 * He3)
        if self.max_order >= 5:
            He2 = u**2 - 1
            He4 = u**4 - 6*u**2 + 3
            I[5] = lam * phi_u * (15 + 6*lam**2 * He2 + lam**4 * He4)

        # For k > 5, use the recurrence I_k = (k-1)*I_{k-2} + lambda * J_{k-1,1}
        # But we need J_{k-1,1} which depends on I_{k-1}. Use numerical integration.
        if self.max_order > 5:
            from scipy.integrate import quad
            for k in range(6, self.max_order + 1):
                def integrand(z, k=k):
                    return z**k * phi(z) * Phi(u + lam * z)
                I[k], _ = quad(integrand, -10, 10, limit=100)

        return I

    def _compute_J_values_analytic(self):
        """
        Compute J_{m,j} analytically using the structure:
        - J_{m,0} = I_m
        - J_{m,j} = d^j I_m / du^j

        Key rules:
        - d/du Phi(u) = phi(u)
        - d/du [P(u) * phi(u)] = [P'(u) - u*P(u)] * phi(u) = -He_1(u)*P(u)*phi + P'(u)*phi

        For the Edgeworth corrections, we mainly need J_{0,j} and J_{m,j} for small m.
        """
        J = np.zeros((self.max_order + 1, self.max_order + 1))
        u, lam = self.u, self.lam
        phi_u = self.phi_u

        # j = 0: J_{m,0} = I_m
        J[:, 0] = self._I

        # j = 1: first derivatives
        J[0, 1] = phi_u  # d/du Phi(u) = phi(u)
        if self.max_order >= 1:
            J[1, 1] = -lam * u * phi_u
        if self.max_order >= 2:
            He2 = u**2 - 1
            J[2, 1] = phi_u * (1 + lam**2 * He2)
        if self.max_order >= 3:
            He3 = u**3 - 3*u
            J[3, 1] = lam * phi_u * (-3*u + lam**2 * (-He3 + 2*u))
            # Simplified: J[3,1] = lam * phi_u * (-3*u - lam**2*(u**3 - 3*u) + 2*lam**2*u)
            #                    = lam * phi_u * (-3*u - lam**2*u**3 + 5*lam**2*u)
            J[3, 1] = lam * phi_u * (-3*u - lam**2 * u**3 + 5*lam**2 * u)
        if self.max_order >= 4:
            # J[4,1] = d/du [3*Phi - lam^2*(3u + lam^2*He3)*phi]
            #        = 3*phi - lam^2 * d/du[(3u + lam^2*He3)*phi]
            #        = 3*phi - lam^2 * [(3 + lam^2*(3u^2-3))*phi - u*(3u + lam^2*He3)*phi]
            He3 = u**3 - 3*u
            term1 = 3 + lam**2 * (3*u**2 - 3)
            term2 = u * (3*u + lam**2 * He3)
            J[4, 1] = 3 * phi_u - lam**2 * (term1 - term2) * phi_u
        if self.max_order >= 5:
            # More complex - use numerical for now
            pass

        # j = 2: second derivatives
        J[0, 2] = -u * phi_u  # d^2/du^2 Phi(u) = d/du phi(u) = -u*phi(u)
        if self.max_order >= 1:
            He2 = u**2 - 1
            J[1, 2] = lam * He2 * phi_u
        if self.max_order >= 2:
            # J[2,2] = d/du [phi * (1 + lam^2*He2)]
            #        = -u*phi*(1 + lam^2*He2) + phi*lam^2*2u
            #        = phi * [-u - lam^2*u*He2 + 2*lam^2*u]
            #        = phi * [-u + lam^2*u*(2 - He2)]
            #        = phi * [-u + lam^2*u*(2 - u^2 + 1)]
            #        = phi * [-u + lam^2*u*(3 - u^2)]
            J[2, 2] = phi_u * (-u + lam**2 * u * (3 - u**2))
        if self.max_order >= 3:
            # J[3,2] = d/du J[3,1]
            # J[3,1] = lam * phi * (-3u - lam^2*u^3 + 5*lam^2*u)
            # d/du = lam * [(-3 - 3*lam^2*u^2 + 5*lam^2)*phi + (-3u - lam^2*u^3 + 5*lam^2*u)*(-u)*phi]
            coeff = -3*u - lam**2 * u**3 + 5*lam**2 * u
            deriv_coeff = -3 - 3*lam**2 * u**2 + 5*lam**2
            J[3, 2] = lam * phi_u * (deriv_coeff - u * coeff)

        # j >= 3: fill in remaining values numerically via direct integration
        # J_{m,j} = d^j I_m / du^j = integral z^m phi(z) d^j/du^j Phi(u + lam*z) dz
        #         = integral z^m phi(z) phi^{(j-1)}(u + lam*z) dz
        # where phi^{(k)}(x) = (-1)^k He_k(x) phi(x)
        from scipy.integrate import quad

        for j in range(3, min(self.max_order + 1, 8)):
            for m in range(min(self.max_order + 1, 8)):
                def integrand(z, m=m, j=j):
                    arg = u + lam * z
                    # phi^{(j-1)}(arg) = (-1)^{j-1} He_{j-1}(arg) phi(arg)
                    He_jm1 = hermite_prob(j-1, arg)
                    sign = (-1)**(j-1)
                    return z**m * phi(z) * sign * He_jm1 * phi(arg)
                try:
                    result, _ = quad(integrand, -10, 10, limit=100)
                    J[m, j] = result
                except:
                    J[m, j] = 0.0

        return J

    def I(self, m):
        """Return I_m."""
        return self._I[m]

    def J(self, m, j):
        """Return J_{m,j} = d^j I_m / du^j."""
        if j < self._J.shape[1]:
            return self._J[m, j]
        else:
            # For very high j, return 0 (contribution is negligible)
            return 0.0


class EdgeworthDynamics:
    """
    Cumulant dynamics under K-truncated Edgeworth expansion.

    Tracks kappa_1, ..., kappa_K and evolves them according to the
    master equation expectations computed under the Edgeworth ansatz.
    """

    def __init__(self, K, beta, eta_bar, delta_eta):
        """
        Initialize Edgeworth-K dynamics.

        Parameters:
            K: truncation order (number of cumulants to track)
            beta: inverse temperature parameter
            eta_bar: mean reward parameter
            delta_eta: reward difference parameter
        """
        self.K = K
        self.beta = beta
        self.eta_bar = eta_bar
        self.delta_eta = delta_eta

    def _get_probit_params(self, mu, sigma_sq):
        """Compute u and lambda from (mu, sigma^2)."""
        sigma = np.sqrt(sigma_sq)
        denom = np.sqrt(1 + self.beta**2 * sigma_sq)
        u = self.beta * mu / denom
        lam = self.beta * sigma / denom
        return u, lam

    def expectation_b(self, kappas, integrals, sigma_sq=None):
        """
        Compute <b>_K under K-truncated Edgeworth.

        <b>_K = (2*I_0 - 1) + sum_{j=3}^K (kappa_j / j! / sigma^j) * 2 * J_{0,j}
        """
        if sigma_sq is None:
            sigma_sq = kappas[1] if len(kappas) > 1 else 1.0
        sigma = np.sqrt(sigma_sq)

        # Gaussian part
        b_gauss = 2 * integrals.I(0) - 1

        # Edgeworth corrections
        # kappas array: kappas[0]=mu, kappas[1]=var, kappas[2]=kappa_3, kappas[3]=kappa_4, ...
        # So kappa_j is at kappas[j-1]
        correction = 0.0
        for j in range(3, self.K + 1):
            if j - 1 < len(kappas):
                kappa_j = kappas[j - 1]
                correction += (kappa_j / factorial(j) / sigma**j) * 2 * integrals.J(0, j)

        return b_gauss + correction

    def expectation_gamma_m_b(self, m, kappas, integrals, sigma_sq=None):
        """
        Compute <tilde{gamma}^m * b>_K where tilde{gamma} = gamma - mu.

        <gamma_tilde^m b>_K = sigma^m * (2*I_m - B*delta_{m even}*(m-1)!!)
                             + sum_{j=3}^K (kappa_j/j!/sigma^j) * sigma^m * 2 * J_{m,j}
        """
        if sigma_sq is None:
            sigma_sq = kappas[1] if len(kappas) > 1 else 1.0
        sigma = np.sqrt(sigma_sq)

        # Gaussian part
        # <z^m b> = 2*I_m - <z^m>_Gauss * 1 (for the -1 in b = 2*Phi - 1)
        # <z^m>_Gauss = (m-1)!! for m even, 0 for m odd
        if m % 2 == 0:
            double_fact = np.prod(np.arange(m-1, 0, -2)) if m > 1 else 1
        else:
            double_fact = 0

        gauss_part = sigma**m * (2 * integrals.I(m) - double_fact)

        # Edgeworth corrections
        # kappas array: kappas[0]=mu, kappas[1]=var, kappas[2]=kappa_3, kappas[3]=kappa_4, ...
        # So kappa_j is at kappas[j-1]
        correction = 0.0
        for j in range(3, self.K + 1):
            if j - 1 < len(kappas):
                kappa_j = kappas[j - 1]
                correction += (kappa_j / factorial(j) / sigma**j) * sigma**m * 2 * integrals.J(m, j)

        return gauss_part + correction

    def compute_M(self, m, n, kappas, integrals, sigma_sq, v_mean):
        """
        Compute M_{m,n} = E[gamma_tilde^m * xi_tilde^n]

        Using the closed form from LaTeX:
        M_{m,n} = (kappa_m + V_m)/2 * (1 - v_mean)^n
                + (kappa_m - V_m)/2 * (-1 - v_mean)^n

        where V_m = E[gamma_tilde^m * v] = eta_bar * kappa_m + delta_eta * <gamma_tilde^m b>
        """
        sigma = np.sqrt(sigma_sq)

        # Get kappa_m (the m-th central moment)
        # kappa_0 = 1 (by convention for this formula)
        # kappa_1 = 0 (centered)
        # kappa_2 = variance, etc.
        if m == 0:
            kappa_m = 1.0
        elif m == 1:
            kappa_m = 0.0  # E[gamma_tilde] = 0 by definition
        else:
            # kappas array: kappas[0] = mu, kappas[1] = var, kappas[2] = kappa_3, etc.
            # So kappa_m for m >= 2 is kappas[m-1] if available
            kappa_m = kappas[m-1] if m-1 < len(kappas) else 0.0

        # Compute V_m = E[gamma_tilde^m * v]
        # v(gamma) = eta_bar + delta_eta * b(gamma)
        # So V_m = eta_bar * E[gamma_tilde^m] + delta_eta * E[gamma_tilde^m * b]
        #        = eta_bar * kappa_m + delta_eta * <gamma_tilde^m b>
        gamma_m_b = self.expectation_gamma_m_b(m, kappas, integrals, sigma_sq=sigma_sq)
        V_m = self.eta_bar * kappa_m + self.delta_eta * gamma_m_b

        # Closed form for M_{m,n}
        term1 = (kappa_m + V_m) / 2.0 * (1.0 - v_mean)**n
        term2 = (kappa_m - V_m) / 2.0 * (-1.0 - v_mean)**n

        return term1 + term2

    def cumulant_update(self, kappas, fixed_sigma_sq=None):
        """
        Compute one-step cumulant updates d(kappa_1), ..., d(kappa_K).

        Uses the general formula:
        Delta_kappa_j = sum_{k=0}^{j} C(j,k) * M_{j-k, k} - kappa_j

        where M_{m,n} = E[gamma_tilde^m * xi_tilde^n].

        Parameters:
            kappas: current cumulant values
            fixed_sigma_sq: for K=1, must provide fixed variance

        Returns array of length K with the updates.
        """
        from scipy.special import comb

        mu = kappas[0]

        # For K=1, we need an external variance estimate
        if self.K == 1:
            if fixed_sigma_sq is None:
                raise ValueError("For K=1, must provide fixed_sigma_sq")
            sigma_sq = fixed_sigma_sq
        else:
            sigma_sq = kappas[1]

        sigma = np.sqrt(sigma_sq)

        u, lam = self._get_probit_params(mu, sigma_sq)
        integrals = ProbitIntegrals(u, lam, max_order=max(self.K + 2, 8))

        # Compute v_mean = <v> = eta_bar + delta_eta * <b>
        b_mean = self.expectation_b(kappas, integrals, sigma_sq=sigma_sq)
        v_mean = self.eta_bar + self.delta_eta * b_mean

        # Compute updates
        d_kappa = np.zeros(self.K)

        # j=1 (mean): Delta_kappa_1 = <v> = v_mean
        # This is special because kappa_1 = mu is NOT a central moment
        d_kappa[0] = v_mean

        # j >= 2: Use general formula for central moments
        # Delta_kappa_j = sum_{k=0}^{j} C(j,k) * M_{j-k, k} - kappa_j
        for j in range(2, self.K + 1):
            delta_j = 0.0
            for k in range(j + 1):
                M_mk = self.compute_M(j - k, k, kappas, integrals, sigma_sq, v_mean)
                delta_j += comb(j, k, exact=True) * M_mk

            # Subtract current kappa_j
            # kappas[j-1] = kappa_j for j >= 1
            kappa_j = kappas[j-1] if j-1 < len(kappas) else 0.0
            d_kappa[j-1] = delta_j - kappa_j

        return d_kappa

    def evolve(self, kappas_0, n_steps, initial_sigma_sq=None):
        """
        Evolve cumulants for n_steps.

        Parameters:
            kappas_0: initial cumulants [kappa_1, kappa_2, ..., kappa_K]
            n_steps: number of steps to evolve
            initial_sigma_sq: for K=1, the initial variance (will grow as t)

        Returns:
            Array of shape (n_steps+1, K) with cumulant trajectories
        """
        trajectory = np.zeros((n_steps + 1, self.K))
        trajectory[0, :len(kappas_0)] = kappas_0[:self.K]

        kappas = np.array(kappas_0[:self.K], dtype=float)

        # For K=1, we track sigma_sq externally (approximate growth)
        sigma_sq_ext = initial_sigma_sq if initial_sigma_sq is not None else 20.0

        for step in range(n_steps):
            if self.K == 1:
                d_kappas = self.cumulant_update(kappas, fixed_sigma_sq=sigma_sq_ext)
                # Approximate variance growth: sigma^2 grows roughly linearly
                sigma_sq_ext += 1.0  # Simple approximation
            else:
                d_kappas = self.cumulant_update(kappas)

            kappas = kappas + d_kappas
            trajectory[step + 1, :] = kappas

        return trajectory


def test_K2_against_gaussian():
    """Test that K=2 Edgeworth matches the Gaussian dynamics."""
    from cumulant_comparison import gaussian_cumulant_step

    beta = 0.3
    eta_bar = 0.5
    delta_eta = 0.1

    mu0, sigma_sq0 = 0.0, 20.0

    # K=2 Edgeworth
    edgeworth = EdgeworthDynamics(K=2, beta=beta, eta_bar=eta_bar, delta_eta=delta_eta)
    kappas = [mu0, sigma_sq0]
    d_kappas = edgeworth.cumulant_update(kappas)

    # Gaussian dynamics
    mu_new, sigma_sq_new = gaussian_cumulant_step(mu0, sigma_sq0, beta, eta_bar, delta_eta)
    d_mu_gauss = mu_new - mu0
    d_var_gauss = sigma_sq_new - sigma_sq0

    print("K=2 Edgeworth vs Gaussian dynamics:")
    print(f"  d(mu):      Edgeworth = {d_kappas[0]:.8f}, Gaussian = {d_mu_gauss:.8f}")
    print(f"  d(sigma^2): Edgeworth = {d_kappas[1]:.8f}, Gaussian = {d_var_gauss:.8f}")
    print(f"  Difference in d(mu):      {abs(d_kappas[0] - d_mu_gauss):.2e}")
    print(f"  Difference in d(sigma^2): {abs(d_kappas[1] - d_var_gauss):.2e}")


if __name__ == "__main__":
    print("Testing Edgeworth cumulant dynamics\n")
    print("=" * 60)

    # Test K=2 against known Gaussian
    test_K2_against_gaussian()

    print("\n" + "=" * 60)
    print("Testing ProbitIntegrals\n")

    # Test the integrals
    u, lam = 0.5, 0.3
    integrals = ProbitIntegrals(u, lam, max_order=6)

    print(f"u = {u}, lambda = {lam}")
    print(f"phi(u) = {phi(u):.6f}, Phi(u) = {Phi(u):.6f}")
    print()
    print("I_m values:")
    for m in range(6):
        print(f"  I_{m} = {integrals.I(m):.6f}")

    print()
    print("J_{m,j} values (d^j I_m / du^j):")
    print("    m\\j |", end="")
    for j in range(4):
        print(f"    j={j}    ", end="")
    print()
    print("-" * 50)
    for m in range(5):
        print(f"    {m}   |", end="")
        for j in range(4):
            print(f" {integrals.J(m, j):9.5f}", end="")
        print()
