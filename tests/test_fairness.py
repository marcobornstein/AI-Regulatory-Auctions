"""The table in fairness/README.md must match the saved training results."""

import re

import pytest

pytest.importorskip("pandas")
from fairness.ablation import HERE, load_results


def test_readme_table_matches_results():
    results = load_results()
    assert results.groupby("minority_pct").size().eq(10).all(), "expected ten seeds per minority share"
    means = results.groupby("minority_pct")["err_odd"].mean()
    table = re.findall(r"^\|\s*(\d+)%\s*\|\s*([\d.]+)\s*\|", (HERE / "README.md").read_text(), flags=re.M)
    assert [int(pct) for pct, _ in table] == list(means.index)
    for pct, value in table:
        # The published table truncates (e.g. 22.3156 -> 22.31) rather than rounds.
        assert float(value) == pytest.approx(means[int(pct)], abs=0.01)
