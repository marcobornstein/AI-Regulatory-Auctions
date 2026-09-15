"""Figure 3: agent participation rate under Circa vs. Reserve Thresholding across compliance prices."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from circa import DISTRIBUTIONS, participation_rates
from circa.plotting import LABEL_FONT, add_output_args, finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", choices=DISTRIBUTIONS, default="uniform", help="distribution of agent values V")
    add_output_args(parser)
    args = parser.parse_args()

    p_eps = np.linspace(0.01, 1, 99, endpoint=False)
    circa, reserve = 100 * np.array([participation_rates(p, DISTRIBUTIONS[args.dist]) for p in p_eps]).T

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(p_eps, circa, color="steelblue", linewidth=2, label=r"Circa: $u_i(b^*) > 0$")
    ax.plot(p_eps, reserve, color="gray", linewidth=2, linestyle="--",
            label=r"Reserve Thresholding: $v_i^d \geq p_\epsilon$")
    ax.fill_between(p_eps, reserve, circa, alpha=0.2, color="steelblue", label="Participation Surplus")
    ax.set_xlabel(r"Epsilon Price $p_\epsilon$ Threshold", **LABEL_FONT)
    ax.set_ylabel("Agent Participation Rate (%)", **LABEL_FONT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    finish(fig, ax, args.out_dir / f"participation_{args.dist}.png", args.show, loc="best")


if __name__ == "__main__":
    main()
