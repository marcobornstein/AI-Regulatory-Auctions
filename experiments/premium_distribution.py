"""Appendix: check the closed-form PDF and CDF of the premium value v^p against simulation."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from circa import DISTRIBUTIONS, LAMBDA_MAX
from circa.plotting import LABEL_FONT, add_output_args, finish

# Legend labels for the (lower, upper) branches of the closed forms.
LABELS = {
    "uniform": {
        "pdf": (r"$f_{v}(v_i^p) = \frac{2\ln(p_\epsilon)}{p_\epsilon - 1}$",
                r"$f_{v}(v_i^p) = \frac{2\ln(2v_i^p)}{p_\epsilon - 1}$"),
        "cdf": (r"$F_{v}(v_i^p) = \frac{2v_i^p\ln(p_\epsilon)}{p_\epsilon - 1}$",
                r"$F_{v}(v_i^p) = \frac{2v_i^p(\ln(2v_i^p)-1) + p_\epsilon}{p_\epsilon - 1}$"),
    },
    "beta": {
        "pdf": (r"$f_{v}(v_i^p) = \frac{6(p_\epsilon^2 - 2p_\epsilon + 1)}{1 - F_\beta(p_\epsilon)}$",
                r"$f_{v}(v_i^p) = \frac{6(4(v_i^p)^2 - 4v_i^p + 1)}{1 - F_\beta(p_\epsilon)}$"),
        "cdf": (r"$F_{v}(v_i^p) = \frac{6v_i^p(p_\epsilon^2 - 2p_\epsilon + 1)}{1 - F_\beta(p_\epsilon)}$",
                r"$F_{v}(v_i^p) = \frac{2v_i^p(4(v_i^p)^2 - 6v_i^p + 3) + p_\epsilon^2(2p_\epsilon - 3)}{1 - F_\beta(p_\epsilon)}$"),
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", choices=DISTRIBUTIONS, default="uniform", help="distribution of agent values V")
    parser.add_argument("--p-eps", type=float, default=0.25, help="compliance price threshold")
    parser.add_argument("--samples", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=0)
    add_output_args(parser)
    args = parser.parse_args()
    dist, p_eps, rng = DISTRIBUTIONS[args.dist], args.p_eps, np.random.default_rng(args.seed)

    v_p = dist.sample_participating(rng, args.samples, p_eps) * rng.uniform(0, LAMBDA_MAX, args.samples)
    density, edges = np.histogram(v_p, bins=500, range=(0, LAMBDA_MAX), density=True)
    z = np.linspace(0, LAMBDA_MAX, 1001)
    branches = (z <= p_eps / 2, z >= p_eps / 2)

    panels = {
        "pdf": (dist.pdf, (edges[:-1] + edges[1:]) / 2, density, ("-", "--"), "Probability Density"),
        "cdf": (dist.cdf, edges[1:], np.cumsum(density * np.diff(edges)), ("--", ":"), "Probability"),
    }
    for kind, (closed_form, x, simulated, styles, ylabel) in panels.items():
        fig, ax = plt.subplots()
        ax.plot(x, simulated, color="red", label=f"Simulated {kind.upper()}")
        for mask, style, label in zip(branches, styles, LABELS[args.dist][kind]):
            ax.plot(z[mask], closed_form(z[mask], p_eps), style, color="blue", label=label)
        ax.set_xlabel("Premium Compensation Value $v_i^p$", **LABEL_FONT)
        ax.set_ylabel(ylabel, **LABEL_FONT)
        ax.set_xlim(0, LAMBDA_MAX)
        ax.set_ylim(0, 1.05 * dist.pdf(0, p_eps) if kind == "pdf" else 1.025)
        finish(fig, ax, args.out_dir / f"premium_{kind}_{args.dist}_pe{p_eps}.png", args.show)


if __name__ == "__main__":
    main()
