"""Dependence components and deterministic, auditable partitions."""

from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from .io import digest


class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, k):
        if self.parent[k] != k:
            self.parent[k] = self.find(self.parent[k])
        return self.parent[k]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def components(events, images, annotators=False):
    uf = UnionFind(e["event_id"] for e in events)
    by_image = {i["image_id"]: i for i in images}
    seen = {}
    for e in events:
        keys = [("prompt", e["normalized_prompt_group_id"])]
        for iid in e["candidate_ids"]:
            keys.append(("image", iid))
            im = by_image.get(iid, {})
            if im.get("source_image_uid"):
                keys.append(("source_uid", im.get("dataset"), im["source_image_uid"]))
            for field in ("content_hash", "pixel_hash"):
                value = by_image.get(iid, {}).get(field)
                if value:
                    keys.append((field, value))
        if annotators and e.get("annotator_id") is not None:
            keys.append(("annotator", str(e["annotator_id"])))
        for key in keys:
            if key in seen:
                uf.union(e["event_id"], seen[key])
            else:
                seen[key] = e["event_id"]
    groups = defaultdict(list)
    for e in events:
        groups[uf.find(e["event_id"])].append(e["event_id"])
    result = {}
    for ids in groups.values():
        component_id = digest(sorted(ids))
        result.update((eid, component_id) for eid in ids)
    return result


def assign_splits(manifest, seed=42, policy="auto"):
    events = manifest["events"]
    mapping = components(events, manifest["images"])
    grouped = defaultdict(list)
    for e in events:
        e["dependency_cluster_id"] = mapping[e["event_id"]]
        grouped[e["dependency_cluster_id"]].append(e)
    dataset = manifest["audit"]["dataset"]
    official = policy == "official" or (policy == "auto" and dataset == "imagereward")
    if dataset == "imagereward" and not official:
        raise ValueError("ImageReward primary analysis requires official splits")
    if dataset == "pickapic" and official:
        raise ValueError("Linked PickaPic has only train; use grouped policy")
    quarantined = []
    if official:
        aliases = {"train": "discovery", "validation": "validation", "test": "test"}
        for cid, es in grouped.items():
            source_splits = {e["original_split"] for e in es}
            split = (
                aliases.get(next(iter(source_splits)))
                if len(source_splits) == 1
                else None
            )
            if split is None:
                split = "quarantine"
                quarantined.extend(e["event_id"] for e in es)
            for e in es:
                e["analysis_split"] = split
    else:
        # Components containing no eligible event do not consume split quotas.
        ids = sorted(
            c for c, es in grouped.items() if any(not e["exclusion_reason"] for e in es)
        )
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
        n = len(ids)
        # Largest-remainder allocation over independent units (not images/events).
        target = np.array([0.70, 0.15, 0.15]) * n
        sizes = np.floor(target).astype(int)
        for i in np.argsort(-(target - sizes), kind="stable")[: n - sizes.sum()]:
            sizes[i] += 1
        assignment = {}
        start = 0
        for split, size in zip(("discovery", "validation", "test"), sizes):
            assignment.update({c: split for c in ids[start : start + size]})
            start += size
        for cid, es in grouped.items():
            for e in es:
                e["analysis_split"] = assignment.get(cid, "quarantine")
    manifest["audit"]["split_audit"] = dict(
        policy="official; quarantine entire cross-boundary component"
        if official
        else "seeded PCG64 shuffled sorted component IDs, largest-remainder 70/15/15",
        seed=seed,
        components=len(grouped),
        quarantined_event_ids=quarantined,
        event_counts=dict(
            Counter(e["analysis_split"] for e in events if not e["exclusion_reason"])
        ),
        component_counts={
            s: len(
                {
                    e["dependency_cluster_id"]
                    for e in events
                    if e["analysis_split"] == s and not e["exclusion_reason"]
                }
            )
            for s in ("discovery", "validation", "test", "quarantine")
        },
        content_hash_scope="decoded/downloaded images only; metadata identity and prompts checked across full metadata",
    )
    assert_no_leakage(events, manifest["images"])


def sample_components(manifest, limit=500, seed=42):
    """Cap distinct prompts without breaking components; stratify 70/15/15 for pilot."""
    events = manifest["events"]
    grouped = defaultdict(list)
    for e in events:
        e["sampled"] = False
        if not e["exclusion_reason"] and e["analysis_split"] != "quarantine":
            grouped[(e["analysis_split"], e["dependency_cluster_id"])].append(e)
    quotas = dict(zip(("discovery", "validation", "test"), [limit, 0, 0]))
    if limit:
        quotas["validation"] = int(round(limit * 0.15))
        quotas["test"] = int(round(limit * 0.15))
        quotas["discovery"] = limit - quotas["validation"] - quotas["test"]
    counts = Counter()
    skipped = []
    for (split, cid), es in sorted(
        grouped.items(), key=lambda kv: digest([seed, kv[0]])
    ):
        size = len({e["normalized_prompt_group_id"] for e in es})
        if not limit or counts[split] + size <= quotas[split]:
            for e in es:
                e["sampled"] = True
            counts[split] += size
        else:
            skipped.append(cid)
    manifest["audit"]["sampling"] = dict(
        seed=seed,
        prompt_limit=limit,
        algorithm="hash-order whole components within fixed splits; 70/15/15 prompt caps; no replenishment after image failure",
        selected_prompt_counts=dict(counts),
        skipped_components=len(skipped),
    )


def restrict_available_cohort(manifest, image_ids, provenance):
    """Declare archive coverage before sampling, retaining all dependency links.

    This changes the target population to complete events covered by the named
    archive. It never removes candidates from an event or substitutes a winner.
    Splits must already have been assigned on the full original metadata.
    """
    image_ids = set(image_ids)
    known = {i["image_id"] for i in manifest["images"]}
    if image_ids - known:
        raise ValueError("Availability list contains unknown canonical image IDs")
    excluded, retained = [], []
    for e in manifest["events"]:
        if "analysis_split" not in e:
            raise ValueError(
                "Assign full-metadata dependency splits before availability filtering"
            )
        if e["exclusion_reason"]:
            continue
        if not set(e["candidate_ids"]) <= image_ids:
            e["exclusion_reason"] = "outside_available_archive_cohort"
            excluded.append(e["event_id"])
        else:
            retained.append(e)
    manifest["audit"]["availability_cohort"] = dict(
        provenance=provenance,
        available_image_ids=len(image_ids),
        excluded_event_ids=excluded,
        retained_events=len(retained),
        retained_prompts=len({e["normalized_prompt_group_id"] for e in retained}),
        target_population="Complete eligible original events covered by the declared archive; not the full dataset",
        split_policy="Keep assignments and dependency components from full original metadata",
    )
    manifest["audit"]["exclusions"] = dict(
        Counter(
            e["exclusion_reason"] for e in manifest["events"] if e["exclusion_reason"]
        )
    )


def assert_no_leakage(events, images):
    mapping = components(events, images)
    seen = defaultdict(set)
    for e in events:
        if e.get("analysis_split") not in (None, "quarantine"):
            seen[mapping[e["event_id"]]].add(e["analysis_split"])
    if any(len(s) > 1 for s in seen.values()):
        raise ValueError("Prompt/image/content dependency crosses analysis splits")


def quarantine_new_hash_conflicts(manifest):
    """After downloads, keep original split assignments and quarantine new cross-split links."""
    mapping = components(manifest["events"], manifest["images"])
    splits = defaultdict(set)
    for e in manifest["events"]:
        splits[mapping[e["event_id"]]].add(e["analysis_split"])
    bad = {k for k, v in splits.items() if len(v) > 1}
    excluded = []
    for e in manifest["events"]:
        e["dependency_cluster_id"] = mapping[e["event_id"]]
        if e["dependency_cluster_id"] in bad:
            e["analysis_split"] = "quarantine"
            e["sampled"] = False
            excluded.append(e["event_id"])
    manifest["audit"]["post_download_hash_quarantine"] = excluded
    assert_no_leakage(manifest["events"], manifest["images"])
