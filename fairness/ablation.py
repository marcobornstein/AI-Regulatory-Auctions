"""Figure 5 and Table 2: fit the price-of-safety curve M to the FairFace results.

Each point averages ten seeds of the test metrics at the best-validation-accuracy epoch (the last
row of each results CSV). Safety is inverse equalized odds and cost is the minority share of the
training data, both normalized to a maximum of 1.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_results(results_dir=HERE / "results"):
    """One row per run: minority share in percent, seed, and final test metrics."""
    rows = []
    for path in sorted(results_dir.glob("*.csv")):
        _, per_min, seed = path.stem.split("-")
        metrics = pd.read_csv(path).iloc[-1].drop(["name", "epoch"]).astype(float)
        rows.append({"minority_pct": round(100 * float(per_min)), "seed": int(seed), **metrics.to_dict()})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, default=HERE.parent / "figures" / "fairness_ablation.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    err_odd = load_results(args.results_dir).groupby("minority_pct")["err_odd"].mean()
    print("Mean equalized odds by minority share (%):", err_odd.round(2).to_string(), sep="\n")

    safety = 1 / err_odd.to_numpy()
    safety = safety / safety.max()
    cost = err_odd.index.to_numpy() / err_odd.index.max()
    fit = np.poly1d(np.polyfit(safety, cost, 2))
    x = np.linspace(safety.min(), safety.max(), 50)

    fig, ax = plt.subplots()
    ax.scatter(safety, cost)
    ax.plot(x, fit(x), "r", label="$M$: Price of Safety")
    ax.set_xlabel(r"Safety Level $\epsilon$", fontsize=15, fontweight="bold")
    ax.set_ylabel("Cost", fontsize=15, fontweight="bold")
    ax.legend(loc="best", fontsize=15)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"saved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
