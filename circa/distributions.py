"""Value and premium-reward distributions used throughout the paper.

An agent's total value V is split into a deployment value v^d = (1 - lambda) V and a premium
value v^p = lambda V, with lambda ~ U(0, 1/2). An agent with V < p_eps can never afford to
comply, so the premium CDF F is taken over participating agents (V conditioned on V >= p_eps).
The piecewise closed forms below have a kink at v^p = p_eps / 2 and are valid on [0, 1/2].
"""

import numpy as np

LAMBDA_MAX = 0.5  # lambda ~ U(0, LAMBDA_MAX); every closed form below assumes this value


def _branches(z, p_eps):
    """Return z as an array, the lower-branch mask z <= p_eps/2, and z clipped into the upper branch."""
    z = np.asarray(z, dtype=float)
    return z, z <= p_eps / 2, np.maximum(z, p_eps / 2)


class ValueDistribution:
    name: str
    label: str

    def sample(self, rng, size):
        raise NotImplementedError

    def value_cdf(self, v):
        raise NotImplementedError

    def value_quantile(self, u):
        raise NotImplementedError

    def pdf(self, z, p_eps):
        """Density f of v^p for participating agents."""
        raise NotImplementedError

    def cdf(self, z, p_eps):
        """CDF F of v^p for participating agents."""
        raise NotImplementedError

    def cdf_integral(self, z, p_eps):
        """int_0^z F(t) dt."""
        raise NotImplementedError

    def sample_participating(self, rng, size, p_eps):
        """Draw V conditioned on V >= p_eps by inverse-CDF sampling."""
        return self.value_quantile(rng.uniform(self.value_cdf(p_eps), 1, size))


class Uniform(ValueDistribution):
    name, label = "uniform", "Uniform"

    def sample(self, rng, size):
        return rng.uniform(0, 1, size)

    def value_cdf(self, v):
        return np.clip(v, 0, 1)

    def value_quantile(self, u):
        return np.asarray(u, dtype=float)

    def pdf(self, z, p_eps):
        z, lower, u = _branches(z, p_eps)
        return 2 * np.where(lower, np.log(p_eps), np.log(2 * u)) / (p_eps - 1)

    def cdf(self, z, p_eps):
        z, lower, u = _branches(z, p_eps)
        return np.where(lower, 2 * z * np.log(p_eps), 2 * u * (np.log(2 * u) - 1) + p_eps) / (p_eps - 1)

    def cdf_integral(self, z, p_eps):
        z, lower, u = _branches(z, p_eps)
        upper = (4 * u**2 * (2 * np.log(2 * u) - 3) + 8 * p_eps * u - p_eps**2) / 8
        return np.where(lower, z**2 * np.log(p_eps), upper) / (p_eps - 1)


class Beta22(ValueDistribution):
    name, label = "beta", "Beta(2, 2)"

    def sample(self, rng, size):
        return rng.beta(2, 2, size)

    def value_cdf(self, v):
        v = np.clip(v, 0, 1)
        return 3 * v**2 - 2 * v**3

    def value_quantile(self, u):
        # Trigonometric root of 3v^2 - 2v^3 = u that lies in [0, 1].
        return 0.5 + np.cos((np.arccos(1 - 2 * np.asarray(u, dtype=float)) + 4 * np.pi) / 3)

    def _mass(self, p_eps):
        return 1 - self.value_cdf(p_eps)

    def pdf(self, z, p_eps):
        z, lower, u = _branches(z, p_eps)
        return 6 * np.where(lower, (1 - p_eps) ** 2, (1 - 2 * u) ** 2) / self._mass(p_eps)

    def cdf(self, z, p_eps):
        z, lower, u = _branches(z, p_eps)
        upper = 2 * u * (4 * u**2 - 6 * u + 3) + p_eps**2 * (2 * p_eps - 3)
        return np.where(lower, 6 * z * (1 - p_eps) ** 2, upper) / self._mass(p_eps)

    def cdf_integral(self, z, p_eps):
        z, lower, u = _branches(z, p_eps)
        upper = (2 * u**4 - 4 * u**3 + 3 * u**2 + u * p_eps**2 * (2 * p_eps - 3)
                 + p_eps**3 / 2 - 3 * p_eps**4 / 8)
        return np.where(lower, 3 * z**2 * (1 - p_eps) ** 2, upper) / self._mass(p_eps)


DISTRIBUTIONS = {dist.name: dist for dist in (Uniform(), Beta22())}
