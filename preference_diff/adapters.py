"""Canonical preference events: labels belong to judgments, never permanently to images."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import PurePosixPath
import unicodedata
import uuid

from .io import clean, digest


def normalize_prompt(text):
    """NFC and whitespace only; preserve case, punctuation, spelling and word order."""
    if not isinstance(text, str):
        return None
    return " ".join(unicodedata.normalize("NFC", text).split())


def prompt_id(text):
    return digest(normalize_prompt(text))


def _image(dataset, revision, uid, row, source):
    identity = uid
    if dataset == "imagereward":
        try:
            identity = str(uuid.UUID(PurePosixPath(source).stem))
        except (ValueError, TypeError):
            identity = uid
    return dict(
        dataset=dataset,
        source_revision=revision,
        image_id=f"{dataset}:{uid}",
        source_image_uid=identity,
        original_path_or_url=source,
        local_path=row.get("local_path"),
        content_hash=None,
        pixel_hash=None,
        original_prompt=row.get("prompt"),
        generation_metadata=clean(row),
        status="unresolved",
        error=None,
    )


def _event(dataset, revision, uid, split, prompt, candidates, raw, annotator=None):
    return dict(
        dataset=dataset,
        source_revision=revision,
        event_id=f"{dataset}:{uid}",
        original_split=split,
        annotation_group_id=str(uid),
        original_prompt=prompt,
        normalized_prompt_group_id=prompt_id(prompt),
        annotator_id=annotator,
        candidate_ids=candidates,
        raw_preference_annotation=clean(raw),
        construction_mode="all_strict_pairs"
        if dataset == "imagereward"
        else "winner_vs_rest",
        exclusion_reason=None,
        analysis_split=None,
        dependency_cluster_id=None,
        sampled=False,
    )


def _audit(dataset, rows, images, events, duplicate_rows=0, **extra):
    return dict(
        dataset=dataset,
        input_rows=rows,
        unique_images=len(images),
        events=len(events),
        duplicate_rows=duplicate_rows,
        schema_version=1,
        prompt_normalization="Unicode NFC + collapse whitespace; case/punctuation retained",
        exclusions=dict(
            Counter(e["exclusion_reason"] for e in events if e["exclusion_reason"])
        ),
        **extra,
    )


def missingness(rows):
    fields = sorted({k for row in rows for k in row})
    return {
        k: dict(
            null=sum(r.get(k) is None for r in rows),
            blank_string=sum(
                isinstance(r.get(k), str) and not r[k].strip() for r in rows
            ),
        )
        for k in fields
    }


def imagereward(rows, revision="local", mode="all_strict_pairs"):
    """Use explicit (split,prompt_id); repeated copies do not create extra evidence."""
    rows = clean(rows)
    unique, duplicate_rows = {}, 0
    for row in rows:
        key = digest(row)
        duplicate_rows += key in unique
        unique[key] = row
    groups = defaultdict(list)
    for row in unique.values():
        if row.get("prompt_id") is None:
            raise ValueError("ImageReward row lacks explicit prompt_id")
        groups[
            (
                row.get("original_split", row.get("split", "train")),
                str(row["prompt_id"]),
            )
        ].append(row)
    images, events = {}, []
    conflicting_images = set()
    for (split, group), records in sorted(groups.items()):
        candidates, ranks, expected = [], {}, set()
        reason = None
        for r in records:
            source = r.get("image_path")
            if not isinstance(source, str) or not source:
                reason = "missing_image_identity"
                continue
            im = _image("imagereward", revision, source, r, source)
            # Annotation fields may vary between events; image identity remains the path.
            iid = im["image_id"]
            if iid in images and normalize_prompt(
                images[iid]["original_prompt"]
            ) != normalize_prompt(r.get("prompt")):
                conflicting_images.add(iid)
            images.setdefault(iid, im)
            if iid in candidates:
                reason = "conflicting_duplicate_image_annotation"
            candidates.append(iid)
            try:
                rank = float(r["rank"])
                if not math.isfinite(rank) or rank != int(rank):
                    raise ValueError()
                ranks[iid] = int(rank)
                count = float(r["image_amount_in_total"])
                if not math.isfinite(count) or count != int(count) or count < 2:
                    raise ValueError()
                expected.add(int(count))
            except (KeyError, ValueError, TypeError):
                reason = "invalid_rank_or_candidate_count"
        if len(expected) != 1 or (
            expected and len(set(candidates)) != next(iter(expected))
        ):
            reason = reason or "incomplete_annotated_group"
        prompts = {normalize_prompt(r.get("prompt")) for r in records}
        if len(prompts) != 1 or None in prompts:
            reason = "inconsistent_group_prompt"
        ev = _event(
            "imagereward",
            revision,
            f"{split}:{group}",
            split,
            records[0].get("prompt"),
            sorted(set(candidates)),
            {"ranks": ranks, "original_rows": records},
        )
        ev.update(
            construction_mode=mode,
            expected_candidates=sorted(expected),
            observed_candidates=len(set(candidates)),
            exclusion_reason=reason,
        )
        if not reason and len(set(ranks.values())) < 2:
            ev["exclusion_reason"] = "no_strict_rank_pairs"
        events.append(ev)
    for ev in events:
        if conflicting_images.intersection(ev["candidate_ids"]):
            ev["exclusion_reason"] = "conflicting_image_prompt_metadata"
    audit = _audit(
        "imagereward",
        len(rows),
        images,
        events,
        duplicate_rows,
        fields=sorted({k for r in rows for k in r}),
        missingness=missingness(rows),
        observed_candidate_count_distribution=dict(
            Counter(e["observed_candidates"] for e in events)
        ),
        groups_with_rank_ties=sum(
            len(set(e["raw_preference_annotation"]["ranks"].values()))
            < len(e["candidate_ids"])
            for e in events
        ),
        rank_direction="smaller rank is preferred",
        incomplete_policy="exclude whole event",
    )
    return dict(
        images=list(images.values()),
        events=events,
        comparisons=[],
        audit=audit,
        provenance={"dataset_revision": revision},
    )


def pickapic(
    rankings, image_rows, revision="local", image_revision="local", cohort="matched"
):
    """The linked four-candidate release, joined by UID (not the pairwise release)."""
    image_rows, rankings = clean(image_rows), clean(rankings)
    images, conflicts, duplicate_images = {}, set(), 0
    for row in image_rows:
        uid = row.get("image_uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError("PickaPic image lacks image_uid")
        im = _image("pickapic", image_revision, uid, row, row.get("url"))
        iid = im["image_id"]
        if iid in images:
            if images[iid]["generation_metadata"] != im["generation_metadata"]:
                conflicts.add(iid)
            else:
                duplicate_images += 1
        else:
            images[iid] = im
    event_rows, event_conflicts, duplicate_events = {}, set(), 0
    for row in rankings:
        if row.get("ranking_id") is None:
            raise ValueError("PickaPic ranking lacks ranking_id")
        uid = str(row["ranking_id"])
        if uid in event_rows:
            if row == event_rows[uid]:
                duplicate_events += 1
            else:
                event_conflicts.add(uid)
        else:
            event_rows[uid] = row
    events = []
    for uid, row in sorted(event_rows.items()):
        raw_candidates = [row.get(f"image_{i}_uid") for i in range(1, 5)]
        candidates = [f"pickapic:{c}" for c in raw_candidates if c is not None]
        winner = row.get("best_image_uid")
        reason = None
        if uid in event_conflicts:
            reason = "conflicting_duplicate_event"
        elif winner is None or str(winner).strip().lower() in (
            "",
            "none",
            "null",
            "nan",
        ):
            reason = "invalid_or_none_selection"
        elif len(set(candidates)) != 4 or len(candidates) != 4:
            reason = "invalid_candidate_set"
        elif winner not in raw_candidates:
            reason = "winner_not_a_candidate"
        elif any(c not in images for c in candidates):
            reason = "missing_candidate_metadata"
        elif conflicts.intersection(candidates):
            reason = "conflicting_image_metadata"
        matched, negative_compatible = False, False
        if all(c in images for c in candidates) and len(candidates) == 4:
            normalized = normalize_prompt(row.get("prompt"))
            matched = normalized is not None and all(
                normalize_prompt(images[c]["original_prompt"]) == normalized
                for c in candidates
            )
            negatives = [
                images[c]["generation_metadata"].get("negative_prompt")
                for c in candidates
            ]
            # All unknown is auditable; mixed known/unknown is unverifiable and excluded.
            negative_compatible = all(n is None for n in negatives) or (
                all(n is not None for n in negatives)
                and len({normalize_prompt(n) for n in negatives}) == 1
            )
            if cohort == "matched" and not matched:
                reason = reason or "candidate_prompt_mismatch"
            if cohort == "matched" and not negative_compatible:
                reason = reason or "incompatible_or_partly_missing_negative_prompts"
        ev = _event(
            "pickapic",
            revision,
            uid,
            row.get("original_split", "train"),
            row.get("prompt"),
            candidates,
            row,
            row.get("user_id"),
        )
        ev.update(
            winner_image_id=f"pickapic:{winner}",
            exclusion_reason=reason,
            expected_candidates=4,
            observed_candidates=len(candidates),
            prompts_match=matched,
            negative_prompts_compatible=negative_compatible,
            cohort=cohort,
        )
        events.append(ev)
    appearances = Counter(c for e in events for c in e["candidate_ids"])
    audit = _audit(
        "pickapic",
        len(rankings),
        images,
        events,
        duplicate_events,
        image_fields=sorted({k for r in image_rows for k in r}),
        ranking_fields=sorted({k for r in rankings for k in r}),
        missingness={
            "images": missingness(image_rows),
            "rankings": missingness(rankings),
        },
        duplicate_image_rows=duplicate_images,
        conflicting_image_ids=len(conflicts),
        reused_candidate_images=sum(n > 1 for n in appearances.values()),
        prompt_mismatch_events=sum(not e["prompts_match"] for e in events),
        negative_prompt_incompatible_events=sum(
            not e["negative_prompts_compatible"] for e in events
        ),
        cohort=cohort,
        incomplete_policy="exclude whole event",
    )
    return dict(
        images=list(images.values()),
        events=events,
        comparisons=[],
        audit=audit,
        provenance={"dataset_revision": revision, "image_revision": image_revision},
    )


def event_pairs(event, mode=None):
    mode = mode or event["construction_mode"]
    ids = event["candidate_ids"]
    if event["dataset"] == "pickapic":
        w = event["winner_image_id"]
        return [(w, c) for c in ids if c != w]
    ranks = event["raw_preference_annotation"]["ranks"]
    lo, hi = min(ranks.values()), max(ranks.values())
    if mode not in ("all_strict_pairs", "best_vs_rest", "best_vs_worst"):
        raise ValueError(f"Unknown comparison mode: {mode}")
    return [
        (w, rejected_value)
        for w in ids
        for rejected_value in ids
        if ranks[w] < ranks[rejected_value]
        and (mode == "all_strict_pairs" or ranks[w] == lo)
        and (mode != "best_vs_worst" or ranks[rejected_value] == hi)
    ]


def construct_comparisons(manifest, valid_ids=None, mode=None):
    """Exclude entire events with any invalid candidate, including tied unused candidates."""
    rows, excluded = [], []
    for e in manifest["events"]:
        if (
            e["exclusion_reason"]
            or not e.get("sampled", True)
            or e.get("analysis_split") == "quarantine"
        ):
            continue
        if valid_ids is not None and not set(e["candidate_ids"]).issubset(valid_ids):
            excluded.append(
                {"event_id": e["event_id"], "reason": "incomplete_valid_scored_event"}
            )
            continue
        pairs = event_pairs(e, mode)
        for w, rejected_value in pairs:
            rows.append(
                dict(
                    event_id=e["event_id"],
                    preferred_image_id=w,
                    rejected_image_id=rejected_value,
                    within_event_weight=1 / len(pairs),
                    prompt_group_id=e["normalized_prompt_group_id"],
                    analysis_split=e["analysis_split"],
                    dependency_cluster_id=e["dependency_cluster_id"],
                    construction_mode=mode or e["construction_mode"],
                )
            )
    return rows, excluded
