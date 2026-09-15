"""Figure 4: expected participating bid (compliance) under Circa vs. Reserve Thresholding."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from circa import DISTRIBUTIONS, expected_bid
from circa.plotting import LABEL_FONT, add_output_args, finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", choices=DISTRIBUTIONS, default="uniform", help="distribution of agent values V")
    add_output_args(parser)
    args = parser.parse_args()
    dist = DISTRIBUTIONS[args.dist]

    p_eps = np.linspace(0.05, 0.95, 400)
    bids = np.array([expected_bid(p, dist) for p in p_eps])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(p_eps, bids, color="steelblue", linewidth=2, label=f"Circa Bid ({dist.label})")
    ax.plot(p_eps, p_eps, color="gray", linewidth=1, linestyle="--", label=r"Reserve Thresholding Bid ($p_\epsilon$)")
    ax.fill_between(p_eps, p_eps, bids, alpha=0.15, color="steelblue", label="Bid Surplus")
    ax.set_xlabel(r"Epsilon Price Threshold $p_\epsilon$", **LABEL_FONT)
    ax.set_ylabel(r"Agent Expected Bid $\mathbb{E}[\hat{b}_i^*]$", **LABEL_FONT)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    finish(fig, ax, args.out_dir / f"expected_bid_{args.dist}.png", args.show)


if __name__ == "__main__":
    main()
