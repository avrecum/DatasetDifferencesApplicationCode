"""Frozen discovery, held-out validation and explicitly labeled sensitivities."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from .adapters import construct_comparisons
from .io import digest, read_json, write_json, write_csv, write_jsonl
from .scoring import open_scores
from .splits import assert_no_leakage, components
from .statistics import aggregate, evaluate_frozen, select_candidates
from .sensitivity import frozen_joint_scores, generator_sensitivity


METRICS = (
    "effect",
    "preferred_mean",
    "rejected_mean",
    "prompt_effect_sd",
    "preferred_prompt_sd",
    "rejected_prompt_sd",
    "concordance",
)


def analyze(
    manifest,
    score_dir,
    output,
    top_k=10,
    bootstrap=2000,
    seed=42,
    tie_tolerance=0.0,
    mode=None,
    concept_chunk=256,
    diversity_correlation=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scores, index, valid, vocab, cache_meta, state = open_scores(score_dir)
    assert_no_leakage(manifest["events"], manifest["images"])
    valid_ids = {iid for iid, row in index.items() if valid[row]}
    comparisons, excluded = construct_comparisons(manifest, valid_ids, mode)
    write_jsonl(output / "comparisons.jsonl", comparisons)
    split_cs = {
        s: [c for c in comparisons if c["analysis_split"] == s]
        for s in ("discovery", "validation", "test")
    }
    arrays = {s: {k: np.full(len(vocab), np.nan) for k in METRICS} for s in split_cs}
    counts = {}
    # Only [prompt, concept_chunk] is live; never [all images, patches, vocabulary].
    # Discovery is completed and candidate artifact committed before held-out scores are aggregated.
    for start in range(0, len(vocab), concept_chunk):
        stop = min(start + concept_chunk, len(vocab))
        result = aggregate(
            scores, index, split_cs["discovery"], np.arange(start, stop), tie_tolerance
        )
        counts["discovery"] = result["counts"]
        for metric in METRICS:
            arrays["discovery"][metric][start:stop] = result[metric]
    candidates = select_candidates(arrays["discovery"]["effect"], vocab, top_k)
    selection_settings = dict(
        seed=seed,
        top_k=top_k,
        tie_tolerance=tie_tolerance,
        mode=mode,
        score_fingerprint=cache_meta["fingerprint"],
        comparisons_hash=digest(comparisons),
        selection="signed discovery effect; strictly positive/negative only; stable vocabulary-index tie break",
    )
    frozen = dict(
        settings=selection_settings,
        candidates=candidates,
        artifact_hash=digest([selection_settings, candidates]),
    )
    frozen_path = output / "frozen_candidates.json"
    if frozen_path.exists() and read_json(frozen_path) != frozen:
        raise ValueError(
            "Frozen candidates differ from prior run; choose a fresh analysis directory"
        )
    write_json(frozen_path, frozen)
    if diversity_correlation is not None:
        if not 0 < diversity_correlation <= 1:
            raise ValueError("Diversity correlation must be in (0,1]")
        discovery_ids = sorted(
            {
                index[c[k]]
                for c in split_cs["discovery"]
                for k in ("preferred_image_id", "rejected_image_id")
            }
        )
        kept = []
        suppressed = []
        for candidate in candidates:
            current = np.asarray(scores[discovery_ids, candidate["concept_id"]], float)
            parent = None
            correlation = None
            for previous in kept:
                if previous["side"] != candidate["side"]:
                    continue
                other = np.asarray(scores[discovery_ids, previous["concept_id"]], float)
                corr = (
                    float(np.corrcoef(current, other)[0, 1])
                    if len(current) > 1 and current.std() > 0 and other.std() > 0
                    else 0.0
                )
                if corr >= diversity_correlation:
                    parent = previous["concept_id"]
                    correlation = corr
                    break
            if parent is None:
                kept.append(candidate)
            else:
                suppressed.append(
                    dict(
                        **candidate,
                        retained_concept_id=parent,
                        discovery_activation_correlation=correlation,
                    )
                )
        write_json(
            output / "presentation_candidates.json",
            dict(
                method="greedy within-side discovery-image activation correlation; not Residual-OMP",
                threshold=diversity_correlation,
                retained=kept,
                suppressed=suppressed,
                scope="presentation only; primary unmodified frozen list retained",
            ),
        )
    # No changes to candidate identity, direction, ranking or threshold after this boundary.
    for split in ("validation", "test"):
        for start in range(0, len(vocab), concept_chunk):
            stop = min(start + concept_chunk, len(vocab))
            result = aggregate(
                scores, index, split_cs[split], np.arange(start, stop), tie_tolerance
            )
            counts[split] = result["counts"]
            for metric in METRICS:
                arrays[split][metric][start:stop] = result[metric]
    table = []
    selected = {c["concept_id"] for c in candidates}
    for j, word in enumerate(vocab):
        row = dict(concept_id=j, concept=word, selected_on_discovery=j in selected)
        for split in arrays:
            row.update(
                {
                    f"{split}_{metric}": float(arrays[split][metric][j])
                    for metric in METRICS
                }
            )
        table.append(row)
    write_csv(output / "all_concepts.csv", table)
    columns = [c["concept_id"] for c in candidates]
    heldout = aggregate(scores, index, split_cs["test"], columns, tie_tolerance)
    evaluated = evaluate_frozen(candidates, heldout, bootstrap, seed)
    if candidates:
        # A transparent frozen feature ensemble; no fitted weights or test tuning.
        directions = np.array([c["direction"] for c in candidates])
        joint_scores = (
            np.asarray(scores[:, columns], float) @ directions / len(columns)
        )[:, None]
        joint = aggregate(joint_scores, index, split_cs["test"], [0], tie_tolerance)
        joint_candidate = [
            dict(
                concept_id=0,
                concept="frozen signed concept mean",
                side="preferred",
                direction=1,
                discovery_rank=1,
                discovery_effect=float(
                    np.mean(arrays["discovery"]["effect"][columns] * directions)
                ),
            )
        ]
        write_json(
            output / "joint_concept_score.json",
            dict(
                rule="equal mean of discovery-direction-signed selected scores; no fitted weights",
                candidate_hash=frozen["artifact_hash"],
                metrics=evaluate_frozen(joint_candidate, joint, bootstrap, seed),
            ),
        )
    write_json(
        output / "top_concepts.json",
        dict(
            candidates=evaluated,
            counts=counts,
            frozen_artifact_hash=frozen["artifact_hash"],
            bootstrap_resamples=bootstrap,
            evaluation_split="test",
            all_vocabulary_test_fields="exploratory; no reselection",
            multiplicity="Per-concept intervals for frozen candidates, no simultaneous guarantee or full-vocabulary significance claim",
        ),
    )
    summary = dict(
        dataset=manifest["audit"]["dataset"],
        synthetic=manifest.get("provenance", {}).get("synthetic", False),
        construction_mode=mode
        or (
            "all_strict_pairs"
            if manifest["audit"]["dataset"] == "imagereward"
            else "winner_vs_rest"
        ),
        counts=counts,
        vocabulary_size=len(vocab),
        score_fingerprint=cache_meta["fingerprint"],
        frozen_artifact_hash=frozen["artifact_hash"],
        excluded_invalid_events=excluded,
        tie_tolerance=tie_tolerance,
        bootstrap_resamples=bootstrap,
        seed=seed,
        patch_budget=cache_meta["configuration"].get("patch_budget"),
        status="held-out results available"
        if counts["test"]["prompts"]
        else "held-out results unavailable: no valid test groups",
        inference="associations; similarity scores are not calibrated concept presence probabilities",
    )
    write_json(output / "analysis_summary.json", summary)
    write_json(
        output / "dataset_audit.json",
        dict(**manifest["audit"], analysis_counts=counts, scoring_exclusions=excluded),
    )
    confounder_audit(
        manifest,
        comparisons,
        scores,
        index,
        state,
        candidates,
        output,
        tie_tolerance,
        bootstrap,
        seed,
    )
    return summary


def _stratify_complete_events(comparisons, events, images, state, index):
    by_event = defaultdict(list)
    for c in comparisons:
        by_event[c["event_id"]].append(c)
    strata = defaultdict(list)
    for eid, cs in by_event.items():
        e = events[eid]
        ims = [images[i] for i in e["candidate_ids"]]
        models = [i["generation_metadata"].get("model_id") for i in ims]
        if all(m is not None for m in models):
            label = "same_model" if len(set(models)) == 1 else "mixed_models"
            strata[label].extend(cs)
            if label == "same_model":
                strata[f"generator:{models[0]}"].extend(cs)
        patches = [
            state["patch_geometry"]
            .get(str(index[i["image_id"]]), {})
            .get("valid_patches")
            for i in ims
        ]
        if all(p is not None for p in patches) and len(set(patches)) == 1:
            strata["equal_patch_count_event"].extend(cs)
        ratios = [i.get("aspect_ratio") for i in ims]
        if all(r is not None for r in ratios) and max(ratios) - min(ratios) < 1e-8:
            strata["equal_aspect_ratio_event"].extend(cs)
        sizes = [(i.get("width"), i.get("height")) for i in ims]
        if (
            all(w is not None and h is not None for w, h in sizes)
            and len(set(sizes)) == 1
        ):
            strata["equal_resolution_event"].extend(cs)
        # ImageReward category; never infer categories from prompt wording.
        categories = {i["generation_metadata"].get("classification") for i in ims}
        if len(categories) == 1 and None not in categories:
            strata[f"category:{next(iter(categories))}"].extend(cs)
    return strata


def confounder_audit(
    manifest,
    comparisons,
    scores,
    index,
    state,
    candidates,
    output,
    tolerance,
    bootstrap,
    seed,
):
    images = {i["image_id"]: i for i in manifest["images"]}
    events = {e["event_id"]: e for e in manifest["events"]}
    active_events = {c["event_id"] for c in comparisons}
    active_images = {i for e in active_events for i in events[e]["candidate_ids"]}
    annotators = Counter(
        str(events[e]["annotator_id"])
        for e in active_events
        if events[e]["annotator_id"] is not None
    )
    patches = [
        state["patch_geometry"].get(str(index[i]), {}).get("valid_patches")
        for i in active_images
    ]

    def distribution(xs):
        known = [x for x in xs if x is not None]
        return dict(
            known=len(known),
            missing=len(xs) - len(known),
            min=min(known) if known else None,
            max=max(known) if known else None,
            quantiles=np.quantile(known, [0.25, 0.5, 0.75]).tolist() if known else [],
        )

    audit = dict(
        annotators_available=len(annotators),
        repeated_annotators=sum(n > 1 for n in annotators.values()),
        max_events_per_annotator=max(annotators.values(), default=0),
        patch_count=distribution(patches),
        width=distribution([images[i].get("width") for i in active_images]),
        height=distribution([images[i].get("height") for i in active_images]),
        aspect_ratio=distribution(
            [images[i].get("aspect_ratio") for i in active_images]
        ),
        generation_models=dict(
            Counter(
                images[i]["generation_metadata"].get("model_id") or "unavailable"
                for i in active_images
            )
        ),
        settings={
            k: distribution(
                [images[i]["generation_metadata"].get(k) for i in active_images]
            )
            for k in ("gs", "steps")
        },
        note="Prompt matching does not eliminate generator/user confounding; subgroup estimates change the target population. Categories only when provided.",
    )
    selected_columns = [c["concept_id"] for c in candidates]
    rows = []
    for name, cs in _stratify_complete_events(
        comparisons, events, images, state, index
    ).items():
        test = [c for c in cs if c["analysis_split"] == "test"]
        a = aggregate(scores, index, test, selected_columns, tolerance)
        for row in evaluate_frozen(candidates, a, bootstrap, seed):
            rows.append(dict(stratum=name, **row))
    write_json(Path(output) / "confounder_audit.json", audit)
    write_json(Path(output) / "stratified_sensitivity.json", rows)
    write_json(
        Path(output) / "generator_pair_sensitivity.json",
        generator_sensitivity(
            scores, index, comparisons, images, candidates, tolerance, bootstrap, seed
        ),
    )
    if annotators:
        # Merge prompts/image components connected by a user. Conservative one-way
        # cluster sensitivity, with the ORIGINAL prompt-weighted point estimand.
        mapping = components(manifest["events"], manifest["images"], annotators=True)
        cs = [
            dict(c, dependency_cluster_id=mapping[c["event_id"]])
            for c in comparisons
            if c["analysis_split"] == "test"
        ]
        a = aggregate(scores, index, cs, selected_columns, tolerance)
        joint_metrics = []
        if candidates:
            joint, joint_candidate = frozen_joint_scores(scores, candidates)
            j = aggregate(joint, index, cs, [0], tolerance)
            joint_metrics = evaluate_frozen(joint_candidate, j, bootstrap, seed)
        write_json(
            Path(output) / "annotator_component_sensitivity.json",
            dict(
                method="union prompt/image components sharing annotators; recompute original prompt estimand",
                metrics=evaluate_frozen(candidates, a, bootstrap, seed),
                joint_metrics=joint_metrics,
                counts=a["counts"],
                caveat="May collapse to very few clusters; unavailable intervals are retained.",
            ),
        )


def analyze_with_sensitivities(manifest, score_dir, output, **kwargs):
    result = analyze(manifest, score_dir, output, **kwargs)
    if manifest["audit"]["dataset"] == "imagereward":
        for mode in ("best_vs_rest", "best_vs_worst"):
            analyze(
                manifest,
                score_dir,
                Path(output) / "sensitivity" / mode,
                mode=mode,
                **kwargs,
            )
    return result
