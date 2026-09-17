"""Frozen-feature generator checks, conditional on valid complete events."""

from collections import defaultdict

import numpy as np

from .statistics import aggregate, evaluate_frozen


def generator_comparison_strata(comparisons, images):
    """Partition pairs, then restore equal event weights within each subset.

    Inputs must already come from complete valid events. This is a conditional
    comparison analysis, not permission to analyze events with missing images.
    """
    grouped = defaultdict(lambda: defaultdict(dict))
    for c in comparisons:
        winner, loser = (
            images[c[k]]["generation_metadata"].get("model_id")
            for k in ("preferred_image_id", "rejected_image_id")
        )
        if winner is None or loser is None:
            continue
        names = ["same_model" if winner == loser else "cross_model"]
        if winner == loser:
            names.append(f"same_model:{winner}")
        for name in names:
            key = (c["preferred_image_id"], c["rejected_image_id"])
            previous = grouped[name][c["event_id"]].get(key)
            if previous is not None and previous != c:
                raise ValueError("Conflicting duplicate comparison")
            grouped[name][c["event_id"]][key] = c
    return {
        name: [
            dict(c, within_event_weight=1 / len(cs))
            for eid, cs in sorted(events.items())
            for c in cs.values()
        ]
        for name, events in sorted(grouped.items())
    }


def frozen_joint_scores(scores, candidates):
    columns = [c["concept_id"] for c in candidates]
    directions = np.array([c["direction"] for c in candidates])
    joint = (np.asarray(scores[:, columns], float) @ directions / len(columns))[:, None]
    candidate = [
        dict(
            concept_id=0,
            concept="frozen signed concept mean",
            direction=1,
            side="preferred",
            discovery_rank=1,
            discovery_effect=float(
                np.mean([c["direction"] * c["discovery_effect"] for c in candidates])
            ),
        )
    ]
    return joint, candidate


def generator_sensitivity(
    scores,
    index,
    comparisons,
    images,
    candidates,
    tolerance=0.0,
    bootstrap=2000,
    seed=42,
):
    rows = {}
    if candidates:
        joint, joint_candidate = frozen_joint_scores(scores, candidates)
        columns = [c["concept_id"] for c in candidates]
        test = [c for c in comparisons if c["analysis_split"] == "test"]
        for name, cs in generator_comparison_strata(test, images).items():
            a = aggregate(scores, index, cs, columns, tolerance)
            j = aggregate(joint, index, cs, [0], tolerance)
            rows[name] = dict(
                counts=a["counts"],
                concepts=evaluate_frozen(candidates, a, bootstrap, seed),
                joint=evaluate_frozen(joint_candidate, j, bootstrap, seed)[0],
            )
    return dict(
        method="Restrict comparisons from complete valid test events to known same- or cross-generator pairs; renormalize comparisons within each retained event, then events within prompts, then prompts. Omit events with no eligible pair.",
        caveat="These conditional estimands have different observation sets; they do not isolate causal generator effects. Primary discovery features/directions are frozen. No new feature selection.",
        unknown_models="Excluded from these sensitivities only; retained in primary analysis.",
        strata=rows,
    )
