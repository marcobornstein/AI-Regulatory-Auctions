"""Figure 2: Monte Carlo validation that the Circa bid b* is a Nash equilibrium.

Each trial draws two agents and is kept only if both participate. Agent 0 then bids a scaled
version of b* (50%-150%) while agent 1 bids b*; agent 0's average utility should peak at b*.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from circa import DISTRIBUTIONS, LAMBDA_MAX, equilibrium_utility, optimal_bid, realized_utility
from circa.plotting import LABEL_FONT, add_output_args, finish

SCALES = np.linspace(0.5, 1.5, 101)


def mean_deviation_utility(dist, p_eps, n_trials, rng, scales=SCALES, batch=50_000):
    """Average utility of agent 0 bidding scale * b* against a rival bidding b*, over n_trials kept trials."""
    total, done = np.zeros_like(scales), 0
    while done < n_trials:
        values = dist.sample(rng, (batch, 2))
        lambdas = rng.uniform(0, LAMBDA_MAX, (batch, 2))
        v_p, v_d = values * lambdas, values * (1 - lambdas)
        keep = np.all(equilibrium_utility(v_d, v_p, p_eps, dist) > 0, axis=1)
        keep &= np.cumsum(keep) <= n_trials - done
        bids = optimal_bid(v_p[keep], p_eps, dist)
        total += realized_utility(bids[:, :1] * scales, bids[:, 1:], v_d[keep, :1], v_p[keep, :1], p_eps).sum(axis=0)
        done += keep.sum()
    return total / n_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", choices=DISTRIBUTIONS, default="uniform", help="distribution of agent values V")
    parser.add_argument("--p-eps", type=float, default=0.75, help="compliance price threshold")
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    add_output_args(parser)
    args = parser.parse_args()

    utility = mean_deviation_utility(DISTRIBUTIONS[args.dist], args.p_eps, args.trials, np.random.default_rng(args.seed))

    fig, ax = plt.subplots()
    ax.axvline(0, color="r", linestyle=":", label="Optimal Bid")
    ax.plot(100 * (SCALES - 1), utility)
    ax.set_xlim(-50, 50)
    ax.set_xlabel("Percent Deviated from Optimal Bid (%)", **LABEL_FONT)
    ax.set_ylabel("Average Agent Utility", **LABEL_FONT)
    finish(fig, ax, args.out_dir / f"equilibrium_{args.dist}_pe{args.p_eps}.png", args.show, loc="best")


if __name__ == "__main__":
    main()
