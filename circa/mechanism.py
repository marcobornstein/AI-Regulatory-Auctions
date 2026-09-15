"""Equilibrium quantities of Circa and the Reserve Thresholding baseline (paper, Section 5)."""

import numpy as np
from scipy import integrate

from .distributions import LAMBDA_MAX


def optimal_bid(v_p, p_eps, dist):
    """Nash bid b* = p_eps + v^p F(v^p) - int_0^{v^p} F(z) dz (Theorem 5), capped at the maximum bid 1."""
    return np.minimum(1.0, p_eps + v_p * dist.cdf(v_p, p_eps) - dist.cdf_integral(v_p, p_eps))


def equilibrium_utility(v_d, v_p, p_eps, dist):
    """Expected utility of bidding b* against an equilibrium rival, who is outbid with probability F(v^p)."""
    return v_d - optimal_bid(v_p, p_eps, dist) + v_p * dist.cdf(v_p, p_eps)


def realized_utility(bid, rival_bid, v_d, v_p, p_eps):
    """Utility of one pairing (Eq. 6): -b below p_eps, otherwise v^d - b plus v^p for outbidding the rival."""
    return np.where(bid < p_eps, -bid, v_d - bid + v_p * (bid > rival_bid))


def participates(v_d, v_p, p_eps, dist):
    """An agent participates iff its equilibrium utility v^d - p_eps + int_0^{v^p} F(z) dz is positive."""
    return v_d + dist.cdf_integral(v_p, p_eps) > p_eps


def expected_bid(p_eps, dist):
    """Expected participating bid E[b*] = p_eps + int_0^{1/2} z f(z) (1 - F(z)) dz (Proposition 1)."""
    integrand = lambda z: float(z * dist.pdf(z, p_eps) * (1 - dist.cdf(z, p_eps)))
    # v^p is at most LAMBDA_MAX; split at the density's kink for accuracy.
    lower, _ = integrate.quad(integrand, 0, p_eps / 2)
    upper, _ = integrate.quad(integrand, p_eps / 2, LAMBDA_MAX)
    return p_eps + lower + upper


def participation_rates(p_eps, dist, n_lambda=2001, iters=50):
    """Fraction of all agents (V ~ dist, lambda ~ U(0, 1/2)) participating under Circa and Reserve Thresholding.

    Both participation conditions are increasing in V, so for each lambda on a grid the threshold
    value is found by bisection (Circa) or in closed form (Reserve Thresholding: V (1 - lambda) >= p_eps),
    and P(V > threshold) is averaged over lambda.
    """
    lam = np.linspace(0, LAMBDA_MAX, n_lambda)
    lo, hi = np.zeros_like(lam), np.ones_like(lam)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = participates(mid * (1 - lam), mid * lam, p_eps, dist)
        lo, hi = np.where(ok, lo, mid), np.where(ok, mid, hi)
    circa = 1 - dist.value_cdf(hi)
    reserve = 1 - dist.value_cdf(np.minimum(1, p_eps / (1 - lam)))
    average = lambda rate: integrate.trapezoid(rate, lam) / LAMBDA_MAX
    return average(circa), average(reserve)
