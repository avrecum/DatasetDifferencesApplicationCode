"""Prompt-balanced estimands and cluster uncertainty; no threshold fitting."""

from __future__ import annotations

from collections import defaultdict
import numpy as np


def concordance(difference, tolerance=0.0):
    if tolerance < 0:
        raise ValueError("Tie tolerance must be nonnegative")
    d = np.asarray(difference)
    return (d > tolerance).astype(float) + 0.5 * (np.abs(d) <= tolerance)


def pooled_auc(positive, negative, positive_weights=None, negative_weights=None):
    """Tie-correct weighted marginal AUROC, not paired concordance. NaN if degenerate."""
    a, b = np.asarray(positive, float), np.asarray(negative, float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("AUROC expects one concept at a time")
    wa = (
        np.ones(len(a))
        if positive_weights is None
        else np.asarray(positive_weights, float)
    )
    wb = (
        np.ones(len(b))
        if negative_weights is None
        else np.asarray(negative_weights, float)
    )
    if wa.shape != a.shape or wb.shape != b.shape or (wa < 0).any() or (wb < 0).any():
        raise ValueError("Invalid AUROC weights")
    if not all(np.isfinite(x).all() for x in (a, b, wa, wb)):
        raise ValueError("AUROC cannot include invalid scores")
    if wa.sum() == 0 or wb.sum() == 0:
        return float("nan")
    order = np.argsort(b, kind="stable")
    b, wb = b[order], wb[order]
    cumulative = np.r_[0.0, np.cumsum(wb)]
    left, right = np.searchsorted(b, a, "left"), np.searchsorted(b, a, "right")
    return float(
        np.sum(wa * (0.5 * (cumulative[left] + cumulative[right])))
        / (wa.sum() * wb.sum())
    )


def hierarchy(comparisons):
    """Deduplicate exact comparison copies; reject ambiguous/conflicting duplicates."""
    unique, event_info = {}, {}
    for c in comparisons:
        key = (c["event_id"], c["preferred_image_id"], c["rejected_image_id"])
        if key in unique and unique[key] != c:
            raise ValueError("Conflicting duplicate comparison")
        unique[key] = c
    groups = defaultdict(lambda: defaultdict(list))
    clusters = {}
    for c in unique.values():
        prompt = c["prompt_group_id"]
        info = (prompt, c["analysis_split"], c["dependency_cluster_id"])
        if c["event_id"] in event_info and event_info[c["event_id"]] != info:
            raise ValueError("Inconsistent event identity")
        event_info[c["event_id"]] = info
        if prompt in clusters and clusters[prompt] != c["dependency_cluster_id"]:
            raise ValueError("A prompt spans dependence components")
        clusters[prompt] = c["dependency_cluster_id"]
        groups[prompt][c["event_id"]].append(c)
    weights = []
    for prompt in sorted(groups):
        for eid in sorted(groups[prompt]):
            cs = groups[prompt][eid]
            if not np.allclose(
                [c["within_event_weight"] for c in cs], 1 / len(cs), atol=1e-12, rtol=0
            ):
                raise ValueError(
                    "Within-event comparisons must have equal normalized weights"
                )
            for c in cs:
                weights.append((c, 1 / (len(groups) * len(groups[prompt]) * len(cs))))
    return groups, clusters, weights


def aggregate(scores, row_index, comparisons, columns=None, tie_tolerance=0.0):
    groups, clusters, weighted = hierarchy(comparisons)
    columns = (
        np.arange(scores.shape[1]) if columns is None else np.asarray(columns, int)
    )
    ids = sorted(groups)
    n, v = len(ids), len(columns)
    preferred, rejected, ordering = [np.empty((n, v), np.float64) for _ in range(3)]
    for pidx, p in enumerate(ids):
        em, lm, om = [], [], []
        for eid in sorted(groups[p]):
            cs = groups[p][eid]
            wi = [row_index[c["preferred_image_id"]] for c in cs]
            li = [row_index[c["rejected_image_id"]] for c in cs]
            w, rejected_value = (
                np.asarray(scores[np.ix_(wi, columns)], float),
                np.asarray(scores[np.ix_(li, columns)], float),
            )
            if not np.isfinite(w).all() or not np.isfinite(rejected_value).all():
                raise ValueError(
                    "Invalid image scores reached statistics; exclude complete events first"
                )
            em.append(w.mean(axis=0))
            lm.append(rejected_value.mean(axis=0))
            om.append(concordance(w - rejected_value, tie_tolerance).mean(axis=0))
        preferred[pidx], rejected[pidx], ordering[pidx] = (
            np.mean(em, axis=0),
            np.mean(lm, axis=0),
            np.mean(om, axis=0),
        )
    diffs = preferred - rejected

    def avg(x):
        return x.mean(axis=0) if n else np.full(v, np.nan)

    def std(x):
        return x.std(axis=0, ddof=1) if n > 1 else np.full(v, np.nan)

    return dict(
        effect=avg(diffs),
        preferred_mean=avg(preferred),
        rejected_mean=avg(rejected),
        prompt_effect_sd=std(diffs),
        preferred_prompt_sd=std(preferred),
        rejected_prompt_sd=std(rejected),
        concordance=avg(ordering),
        prompt_effects=diffs,
        prompt_concordance=ordering,
        prompt_ids=ids,
        cluster_ids=[clusters[p] for p in ids],
        counts=dict(
            prompts=n,
            events=sum(len(g) for g in groups.values()),
            comparisons=len(weighted),
            components=len(set(clusters.values())),
            unique_images=len(
                {
                    c[k]
                    for c, _ in weighted
                    for k in ("preferred_image_id", "rejected_image_id")
                }
            ),
        ),
    )


def cluster_bootstrap(prompt_values, cluster_ids, resamples=2000, seed=42):
    """Sample components, then divide summed prompt values by sampled prompt counts.

    Unequal component sizes must not silently become equal component weights.
    With fewer than two independent components, intervals are unavailable.
    """
    values = np.asarray(prompt_values, float)
    if values.ndim != 2 or len(values) != len(cluster_ids):
        raise ValueError(
            "Expected prompt-by-concept matrix with one cluster per prompt"
        )
    if resamples < 1:
        raise ValueError("Bootstrap count must be positive")
    keys = sorted(set(cluster_ids))
    v = values.shape[1]
    if len(keys) < 2:
        return dict(
            low=np.full(v, np.nan),
            high=np.full(v, np.nan),
            resamples=resamples,
            independent_clusters=len(keys),
            status="unavailable: fewer than two clusters",
        )
    labels = np.array(cluster_ids)
    sums = np.stack([values[labels == c].sum(axis=0) for c in keys])
    sizes = np.array([np.sum(labels == c) for c in keys])
    rng = np.random.default_rng(seed)
    samples = np.empty((resamples, v))
    for b in range(resamples):
        counts = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))
        samples[b] = counts @ sums / (counts @ sizes)
    lo, hi = np.quantile(samples, [0.025, 0.975], axis=0)
    return dict(
        low=lo,
        high=hi,
        resamples=resamples,
        independent_clusters=len(keys),
        status="per-concept percentile intervals; not simultaneous",
    )


def select_candidates(effects, vocab, top_k=10):
    effects = np.asarray(effects)
    result = []
    for direction, side in ((1, "preferred"), (-1, "rejected")):
        indices = np.flatnonzero(np.isfinite(effects) & (direction * effects > 0))
        indices = indices[np.argsort(-direction * effects[indices], kind="stable")][
            :top_k
        ]
        for rank, idx in enumerate(indices, 1):
            result.append(
                dict(
                    concept_id=int(idx),
                    concept=vocab[idx],
                    side=side,
                    direction=direction,
                    discovery_rank=rank,
                    discovery_effect=float(effects[idx]),
                )
            )
    return result


def evaluate_frozen(candidates, aggregation, resamples=2000, seed=42):
    """Aggregation columns MUST be in frozen candidate order; directions never reselected."""
    if aggregation["prompt_effects"].shape[1] != len(candidates):
        raise ValueError("Frozen candidate/aggregation alignment mismatch")
    effect_ci = cluster_bootstrap(
        aggregation["prompt_effects"], aggregation["cluster_ids"], resamples, seed
    )
    order_ci = cluster_bootstrap(
        aggregation["prompt_concordance"], aggregation["cluster_ids"], resamples, seed
    )
    out = []
    for j, candidate in enumerate(candidates):
        raw = float(aggregation["concordance"][j])
        lo, hi = float(order_ci["low"][j]), float(order_ci["high"][j])
        direction = candidate["direction"]
        out.append(
            dict(
                **candidate,
                heldout_effect=float(aggregation["effect"][j]),
                effect_ci_low=float(effect_ci["low"][j]),
                effect_ci_high=float(effect_ci["high"][j]),
                raw_concordance=raw,
                concordance_ci_low=lo,
                concordance_ci_high=hi,
                direction_adjusted_concordance=raw if direction == 1 else 1 - raw,
                adjusted_concordance_ci_low=lo if direction == 1 else 1 - hi,
                adjusted_concordance_ci_high=hi if direction == 1 else 1 - lo,
                counts=aggregation["counts"],
                interval_status=effect_ci["status"],
            )
        )
    return out
