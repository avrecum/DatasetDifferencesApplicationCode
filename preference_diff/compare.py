"""Descriptive cross-dataset agreement and frozen-list transfer."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from .io import read_json, read_rows, write_json, write_csv
from .scoring import open_scores
from .statistics import aggregate, evaluate_frozen


def _identities(manifest, event_ids):
    es = [e for e in manifest["events"] if e["event_id"] in event_ids]
    ids = {i for e in es for i in e["candidate_ids"]}
    ims = [i for i in manifest["images"] if i["image_id"] in ids]
    return (
        {e["normalized_prompt_group_id"] for e in es},
        {i[k] for i in ims for k in ("content_hash", "pixel_hash") if i.get(k)},
    )


def compare(
    left_manifest,
    right_manifest,
    left_scores,
    right_scores,
    left_analysis,
    right_analysis,
    output,
    bootstrap=2000,
    seed=42,
    top_k=10,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    left = open_scores(left_scores)
    right = open_scores(right_scores)
    if left[3] != right[3]:
        raise ValueError(
            "Cross-dataset comparison requires the same ordered vocabulary"
        )
    # Device and image identities differ; encoder/preprocessing/scoring must agree.
    for key in (
        "model_name",
        "model_revision",
        "bank_revision",
        "patch_budget",
        "dtype",
        "implementation",
        "text_chunk_size",
        "image_batch_size",
        "library_versions",
        "bank",
    ):
        if left[4]["configuration"].get(key) != right[4]["configuration"].get(key):
            raise ValueError(f"Incompatible cross-dataset scoring configuration: {key}")
    la, ra = Path(left_analysis), Path(right_analysis)
    lt, rt = read_rows(la / "all_concepts.csv"), read_rows(ra / "all_concepts.csv")

    def number(x):
        return float(x) if x not in ("", None) else np.nan

    effects = {
        split: (
            np.array([number(r[f"{split}_effect"]) for r in lt]),
            np.array([number(r[f"{split}_effect"]) for r in rt]),
        )
        for split in ("discovery", "test")
    }
    joined = []
    for j, word in enumerate(left[3]):
        joined.append(
            dict(
                concept_id=j,
                concept=word,
                **{
                    f"{side}_{split}_effect": float(effects[split][k][j])
                    for split in effects
                    for k, side in enumerate(("imagereward", "pickapic"))
                },
            )
        )
    write_csv(output / "joined_concepts.csv", joined)
    stats = {}
    for split, (a, b) in effects.items():
        finite = np.isfinite(a) & np.isfinite(b)
        rho = (
            float(spearmanr(a[finite], b[finite]).statistic)
            if finite.sum() > 1 and np.std(a[finite]) > 0 and np.std(b[finite]) > 0
            else None
        )
        stats[split] = dict(
            signed_rank_correlation=rho,
            available_concepts=int(finite.sum()),
            direction_agreement=float(np.mean(np.sign(a[finite]) == np.sign(b[finite])))
            if finite.any()
            else None,
        )
    lcan = read_json(la / "frozen_candidates.json")["candidates"]
    rcan = read_json(ra / "frozen_candidates.json")["candidates"]
    discoveries_available = all(
        np.isfinite(values).any() for values in effects["discovery"]
    )
    overlap = {}
    for side in ("preferred", "rejected"):
        a = {
            c["concept_id"]
            for c in lcan
            if c["side"] == side and c["discovery_rank"] <= top_k
        }
        b = {
            c["concept_id"]
            for c in rcan
            if c["side"] == side and c["discovery_rank"] <= top_k
        }
        overlap[side] = dict(
            left_size=len(a),
            right_size=len(b),
            intersection=len(a & b) if discoveries_available else None,
            jaccard=len(a & b) / len(a | b)
            if discoveries_available and a | b
            else None,
            status="available"
            if discoveries_available
            else "unavailable: one or both datasets lack discovery observations",
        )
    stats["top_k_overlap"] = overlap
    transfer = {}
    for (
        name,
        source_manifest,
        target_manifest,
        source_dir,
        target_dir,
        target_cache,
        candidates,
    ) in (
        ("imagereward_to_pickapic", left_manifest, right_manifest, la, ra, right, lcan),
        ("pickapic_to_imagereward", right_manifest, left_manifest, ra, la, left, rcan),
    ):
        source_cs = read_rows(source_dir / "comparisons.jsonl")
        target_cs = [
            c
            for c in read_rows(target_dir / "comparisons.jsonl")
            if c["analysis_split"] == "test"
        ]
        source_ids = {
            c["event_id"] for c in source_cs if c["analysis_split"] == "discovery"
        }
        prompts, hashes = _identities(source_manifest, source_ids)
        bad_events = []
        bad_clusters = set()
        target_events = {c["event_id"] for c in target_cs}
        for e in target_manifest["events"]:
            if e["event_id"] not in target_events:
                continue
            p, h = _identities(target_manifest, {e["event_id"]})
            if p & prompts or h & hashes:
                bad_events.append(e["event_id"])
                bad_clusters.add(e["dependency_cluster_id"])
        retained = [
            c for c in target_cs if c["dependency_cluster_id"] not in bad_clusters
        ]
        tolerance = read_json(target_dir / "analysis_summary.json")["tie_tolerance"]
        a = aggregate(
            target_cache[0],
            target_cache[1],
            retained,
            [c["concept_id"] for c in candidates],
            tolerance,
        )
        transfer[name] = dict(
            metrics=evaluate_frozen(candidates, a, bootstrap, seed),
            counts=a["counts"],
            duplicate_events=bad_events,
            excluded_components=sorted(bad_clusters),
            note="Recipient test components overlapping donor discovery prompts or known image hashes excluded; candidate list and direction frozen on donor discovery.",
        )
    all_left = {e["event_id"] for e in left_manifest["events"] if e["sampled"]}
    all_right = {e["event_id"] for e in right_manifest["events"] if e["sampled"]}
    lp, lh = _identities(left_manifest, all_left)
    rp, rh = _identities(right_manifest, all_right)
    stats["cross_dataset_duplicate_audit"] = dict(
        shared_normalized_prompts=len(lp & rp),
        shared_hashes=len(lh & rh),
        hash_scope="resolved pilot images",
    )
    stats["interpretation"] = (
        "Descriptive agreement; different candidate pools/protocols. Direction agreement differs from magnitude agreement. Significance in one dataset alone is not a between-dataset difference test."
    )
    write_json(output / "comparison_summary.json", stats)
    write_json(output / "frozen_list_transfer.json", transfer)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    a, b = effects["test"]
    finite = np.isfinite(a) & np.isfinite(b)
    ax.scatter(a[finite], b[finite], s=3, alpha=0.2, rasterized=True)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set(
        xlabel="ImageReward held-out preferred − rejected",
        ylabel="Pick-a-Pic held-out preferred − rejected",
        title="Signed concept-score effects (descriptive)",
    )
    if not finite.any():
        ax.text(
            0.5,
            0.5,
            "Real held-out comparison unavailable",
            ha="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()
    fig.savefig(output / "effect_scatter.png", dpi=180)
    fig.savefig(output / "effect_scatter.pdf")
    plt.close(fig)
    return stats
