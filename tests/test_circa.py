"""Check the closed forms against the paper's corollaries, numerical integration, and simulation."""

import numpy as np
import pytest
from scipy import integrate

from circa import DISTRIBUTIONS, LAMBDA_MAX, expected_bid, optimal_bid, participates, participation_rates
from experiments.equilibrium import SCALES, mean_deviation_utility

by_dist = pytest.mark.parametrize("dist", DISTRIBUTIONS.values(), ids=DISTRIBUTIONS.keys())
by_p_eps = pytest.mark.parametrize("p_eps", [0.01, 0.25, 0.5, 0.75, 0.99])


def integral(f, z, p_eps):
    """int_0^z f, split at the kink p_eps / 2."""
    kink = min(z, p_eps / 2)
    return integrate.quad(f, 0, kink)[0] + (integrate.quad(f, kink, z)[0] if z > kink else 0.0)


def participating_premiums(dist, p_eps, n, seed=0):
    rng = np.random.default_rng(seed)
    return dist.sample_participating(rng, n, p_eps) * rng.uniform(0, LAMBDA_MAX, n)


@by_dist
def test_value_quantile_inverts_cdf(dist):
    v = np.linspace(0, 1, 101)
    np.testing.assert_allclose(dist.value_quantile(dist.value_cdf(v)), v, atol=1e-7)


@by_dist
@by_p_eps
def test_closed_forms_are_consistent(dist, p_eps):
    assert integral(lambda t: dist.pdf(t, p_eps), LAMBDA_MAX, p_eps) == pytest.approx(1, abs=1e-8)
    for z in np.linspace(0.01, LAMBDA_MAX, 12):
        assert dist.cdf(z, p_eps) == pytest.approx(integral(lambda t: dist.pdf(t, p_eps), z, p_eps), abs=1e-8)
        assert dist.cdf_integral(z, p_eps) == pytest.approx(integral(lambda t: dist.cdf(t, p_eps), z, p_eps), abs=1e-8)


@by_dist
@pytest.mark.parametrize("p_eps", [0.25, 0.5, 0.75])
def test_cdf_matches_simulation(dist, p_eps):
    v_p = participating_premiums(dist, p_eps, 400_000)
    z = np.linspace(0.02, 0.48, 24)
    # DKW bound at n = 400k gives sup-error < 0.0031 with probability 99.9%.
    np.testing.assert_allclose(dist.cdf(z, p_eps), (v_p[:, None] <= z).mean(axis=0), atol=0.005)


@by_dist
@by_p_eps
def test_optimal_bid_matches_paper_corollaries(dist, p_eps):
    v = np.linspace(0, LAMBDA_MAX, 201)
    u = np.maximum(v, p_eps / 2)
    if dist.name == "uniform":  # Corollary 1
        lower = p_eps + v**2 * np.log(p_eps) / (p_eps - 1)
        upper = p_eps + (8 * u**2 * (np.log(2 * u) - 0.5) + p_eps**2) / (8 * (p_eps - 1))
    else:  # Corollary 2
        mass = 1 - (3 * p_eps**2 - 2 * p_eps**3)
        lower = p_eps + 3 * v**2 * (p_eps**2 - 2 * p_eps + 1) / mass
        upper = p_eps + (8 * u**2 * (6 * u**2 - 8 * u + 3) + p_eps**3 * (3 * p_eps - 4)) / (8 * mass)
    paper = np.minimum(1, np.where(v <= p_eps / 2, lower, upper))
    bids = optimal_bid(v, p_eps, dist)
    np.testing.assert_allclose(bids, paper, rtol=1e-10)
    assert np.all(bids[1:] > p_eps)


@by_dist
@pytest.mark.parametrize("p_eps", [0.25, 0.5, 0.75])
def test_expected_bid_matches_simulation(dist, p_eps):
    simulated = optimal_bid(participating_premiums(dist, p_eps, 400_000), p_eps, dist).mean()
    assert expected_bid(p_eps, dist) == pytest.approx(simulated, abs=2e-3)


@by_dist
@pytest.mark.parametrize("p_eps", [0.1, 0.4, 0.7, 0.95])
def test_participation_rates_match_simulation(dist, p_eps):
    rng, n = np.random.default_rng(1), 400_000
    values, lambdas = dist.sample(rng, n), rng.uniform(0, LAMBDA_MAX, n)
    circa, reserve = participation_rates(p_eps, dist)
    assert circa == pytest.approx(participates(values * (1 - lambdas), values * lambdas, p_eps, dist).mean(), abs=5e-3)
    assert reserve == pytest.approx((values * (1 - lambdas) >= p_eps).mean(), abs=5e-3)
    assert circa >= reserve


@by_dist
@pytest.mark.parametrize("p_eps", [0.25, 0.75])
def test_optimal_bid_is_best_response(dist, p_eps):
    utility = mean_deviation_utility(dist, p_eps, 50_000, np.random.default_rng(2))
    at_optimum = utility[np.flatnonzero(np.isclose(SCALES, 1))[0]]
    assert at_optimum >= utility.max() - 2e-3
    assert at_optimum > utility[0] and at_optimum > utility[-1]
