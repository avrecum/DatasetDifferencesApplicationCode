"""Render a paper table and plot from completed, frozen real-data analyses."""

import argparse
from pathlib import Path

from preference_diff.io import read_json, write_json, atomic_text


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--imagereward", type=Path, required=True)
    p.add_argument("--pickapic", type=Path, required=True)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    left = read_json(args.imagereward / "analysis/joint_concept_score.json")["metrics"][
        0
    ]
    right = read_json(args.pickapic / "analysis/joint_concept_score.json")["metrics"][0]
    rows = [
        ("ImageReward all pairs", "ImageReward all pairs", left),
        ("Pick-a-Pic", "Pick-a-Pic", right),
    ]
    for mode, label in (("primary", "all pairs"), ("best_vs_rest", "best/rest")):
        transfer = read_json(args.comparison / mode / "frozen_list_transfer.json")
        rows.extend(
            [
                (
                    f"ImageReward {label}",
                    "Pick-a-Pic",
                    transfer["imagereward_to_pickapic"]["joint_metrics"][0],
                ),
                (
                    "Pick-a-Pic",
                    f"ImageReward {label}",
                    transfer["pickapic_to_imagereward"]["joint_metrics"][0],
                ),
            ]
        )
    write_json(
        args.output / "combined_results.json",
        [
            dict(discovery=source, evaluation=target, **metric)
            for source, target, metric in rows
        ],
    )
    latex = [
        "% Additional frozen signed-concept-mean diagnostic, not MLLM discrimination.",
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Discovery & Evaluation & Test prompts & Concordance & 95\% CI \\",
        r"\midrule",
    ]
    for source, target, m in rows:
        latex.append(
            f"{source} & {target} & {m['counts']['prompts']} & "
            f"{100 * m['raw_concordance']:.1f}\\% & "
            f"[{100 * m['concordance_ci_low']:.1f}, {100 * m['concordance_ci_high']:.1f}]\\% \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    atomic_text(args.output / "combined_results_table.tex", "\n".join(latex) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(rows))
    point = np.array([100 * r[2]["raw_concordance"] for r in rows])
    low = np.array([100 * r[2]["concordance_ci_low"] for r in rows])
    high = np.array([100 * r[2]["concordance_ci_high"] for r in rows])
    ax.hlines(y, low, high, color="navy")
    ax.plot(point, y, "o", color="navy")
    ax.axvline(50, color="grey", linestyle="--")
    ax.set_yticks(y, [f"{source} → {target}" for source, target, _ in rows])
    ax.invert_yaxis()
    ax.set(
        xlim=(30, 85),
        xlabel="Prompt-balanced paired concordance (%)",
        title="Frozen concept score: discovery → held-out evaluation",
    )
    fig.text(
        0.5,
        0.01,
        "95% component-bootstrap intervals; not simultaneous. Pick-a-Pic: 33 early-archive test prompts.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"combined_concordance.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
