import copy
import numpy as np
import pytest

from preference_diff.adapters import (
    imagereward,
    pickapic,
    event_pairs,
    construct_comparisons,
)
from preference_diff.splits import (
    assign_splits,
    sample_components,
    assert_no_leakage,
    quarantine_new_hash_conflicts,
    restrict_available_cohort,
)
from preference_diff.statistics import (
    aggregate,
    pooled_auc,
    concordance,
    cluster_bootstrap,
    select_candidates,
    evaluate_frozen,
)


def ir_rows(ranks=(1, 1, 2, 3), split="train", group="p", prompt="A cat"):
    return [
        dict(
            prompt_id=group,
            prompt=prompt,
            rank=r,
            image_path=f"{group}/{i}.webp",
            image_amount_in_total=len(ranks),
            original_split=split,
        )
        for i, r in enumerate(ranks)
    ]


def pap_rows():
    ims = [
        dict(
            image_uid=str(i),
            prompt="cat",
            negative_prompt=None,
            url=f"https://example.org/{i}",
        )
        for i in range(5)
    ]
    event = dict(
        ranking_id=1,
        prompt="cat",
        user_id=7,
        best_image_uid="0",
        **{f"image_{i + 1}_uid": str(i) for i in range(4)},
    )
    return ims, event


def active(m):
    assign_splits(m)
    sample_components(m, 0)
    return construct_comparisons(m)[0]


def test_declared_archive_cohort_keeps_full_candidates_and_existing_dependencies():
    ims, event = pap_rows()
    second = dict(event, ranking_id=2, image_4_uid="4")
    manifest = pickapic([event, second], ims)
    assign_splits(manifest)
    before = [
        (e["analysis_split"], e["dependency_cluster_id"], list(e["candidate_ids"]))
        for e in manifest["events"]
    ]
    restrict_available_cohort(
        manifest, {f"pickapic:{i}" for i in range(4)}, {"archive": "pinned"}
    )
    assert [
        (e["analysis_split"], e["dependency_cluster_id"], e["candidate_ids"])
        for e in manifest["events"]
    ] == before
    assert manifest["events"][0]["exclusion_reason"] is None
    assert (
        manifest["events"][1]["exclusion_reason"] == "outside_available_archive_cohort"
    )
    sample_components(manifest, 0)
    pairs, _ = construct_comparisons(manifest)
    assert len(pairs) == 3
    assert manifest["audit"]["availability_cohort"]["retained_events"] == 1
    with pytest.raises(ValueError, match="unknown"):
        restrict_available_cohort(manifest, {"pickapic:absent"}, {})


def test_rank_direction_ties_and_modes():
    e = imagereward(ir_rows())["events"][0]
    assert len(event_pairs(e)) == 5
    assert len(event_pairs(e, "best_vs_rest")) == 4
    assert len(event_pairs(e, "best_vs_worst")) == 2
    assert all(
        e["raw_preference_annotation"]["ranks"][w]
        < e["raw_preference_annotation"]["ranks"][rejected_value]
        for w, rejected_value in event_pairs(e)
    )
    assert (
        imagereward(ir_rows()[:-1])["events"][0]["exclusion_reason"]
        == "incomplete_annotated_group"
    )
    assert (
        imagereward(ir_rows((2, 2)))["events"][0]["exclusion_reason"]
        == "no_strict_rank_pairs"
    )


@pytest.mark.parametrize(
    "winner,reason",
    [
        ("none", "invalid_or_none_selection"),
        (None, "invalid_or_none_selection"),
        ("99", "winner_not_a_candidate"),
    ],
)
def test_invalid_pickapic(winner, reason):
    ims, e = pap_rows()
    e["best_image_uid"] = winner
    assert pickapic([e], ims)["events"][0]["exclusion_reason"] == reason


def test_pickapic_prompt_negative_membership_and_repeated_labels():
    ims, e = pap_rows()
    other = dict(e, ranking_id=2, best_image_uid="1")
    m = pickapic([e, other, e], ims)
    assert m["audit"]["duplicate_rows"] == 1 and len(m["events"]) == 2
    assert ("pickapic:0", "pickapic:1") in event_pairs(m["events"][0])
    assert ("pickapic:1", "pickapic:0") in event_pairs(m["events"][1])
    ims[0]["prompt"] = "another cat"
    assert (
        pickapic([e], ims)["events"][0]["exclusion_reason"]
        == "candidate_prompt_mismatch"
    )
    assert pickapic([e], ims, cohort="event")["events"][0]["exclusion_reason"] is None
    ims[0]["prompt"] = "cat"
    ims[0]["negative_prompt"] = "blur"
    assert "negative" in pickapic([e], ims)["events"][0]["exclusion_reason"]
    assert (
        pickapic([dict(e, image_4_uid="0")], ims)["events"][0]["exclusion_reason"]
        == "invalid_candidate_set"
    )
    assert (
        pickapic([e], ims[1:])["events"][0]["exclusion_reason"]
        == "missing_candidate_metadata"
    )


def test_prompt_event_weighting_duplicates_and_identity():
    rows = (
        ir_rows((1, 2), group="a", prompt="same")
        + ir_rows((1, 2), group="b", prompt="same")
        + ir_rows((1, 2, 3), group="c", prompt="different")
    )
    m = imagereward(rows + rows)
    cs = active(m)
    idx = {im["image_id"]: i for i, im in enumerate(m["images"])}
    x = np.array([[1], [0], [0.5], [0], [0], [0], [0]], float)
    a = aggregate(x, idx, cs)
    assert a["effect"][0] == 0.375
    np.testing.assert_allclose(a["effect"], a["preferred_mean"] - a["rejected_mean"])
    np.testing.assert_allclose(a["effect"], aggregate(x, idx, cs + cs)["effect"])
    valid = set(idx) - {m["images"][0]["image_id"]}
    kept, excluded = construct_comparisons(m, valid)
    assert len(excluded) == 1 and all(c["event_id"] != cs[0]["event_id"] for c in kept)


def test_tie_correct_auc_concordance_and_degenerate():
    assert pooled_auc([1, 1], [1, 1]) == 0.5
    assert pooled_auc([0, 1], [1, 2]) == 0.125
    assert pooled_auc([0, 1], [0, 1], [1, 3], [3, 1]) == 0.75
    assert np.isnan(pooled_auc([], [1]))
    np.testing.assert_array_equal(
        concordance(np.array([0, 0.01, -0.01]), 0.02), [0.5, 0.5, 0.5]
    )
    with pytest.raises(ValueError):
        pooled_auc([np.nan], [1])


def test_dependency_components_prompt_image_hash_official_quarantine():
    m = imagereward(
        ir_rows((1, 2), group="a", prompt="repeat", split="train")
        + ir_rows((1, 2), group="b", prompt="repeat", split="test")
    )
    assign_splits(m)
    assert all(e["analysis_split"] == "quarantine" for e in m["events"])
    m = imagereward(
        ir_rows((1, 2), group="a", split="train")
        + ir_rows((1, 2), group="b", prompt="other", split="test")
    )
    assign_splits(m)
    m["images"][0]["content_hash"] = "shared"
    m["images"][2]["content_hash"] = "shared"
    with pytest.raises(ValueError):
        assert_no_leakage(m["events"], m["images"])
    quarantine_new_hash_conflicts(m)
    assert all(e["analysis_split"] == "quarantine" for e in m["events"])


def test_bootstrap_reproducible_and_prompt_estimand():
    x = np.array([[0.0], [0.0], [0.0], [1.0]])
    a = cluster_bootstrap(x, ["a", "a", "a", "b"], 2000, 8)
    b = cluster_bootstrap(x, ["a", "a", "a", "b"], 2000, 8)
    np.testing.assert_array_equal(a["low"], b["low"])
    # Independent manual resampling oracle checks unequal-size component denominator.
    rng = np.random.default_rng(8)
    vals = []
    for _ in range(2000):
        draw = rng.integers(0, 2, 2)
        nb = (draw == 1).sum()
        vals.append(nb / (3 * (2 - nb) + nb))
    np.testing.assert_allclose(
        [a["low"][0], a["high"][0]], np.quantile(vals, [0.025, 0.975])
    )
    # Seed 1 draws one large and one small component: estimator must be 1/4, not 1/2.
    single = cluster_bootstrap(x, ["a", "a", "a", "b"], 1, 1)
    assert single["low"][0] == 0.25
    assert np.isnan(cluster_bootstrap(x, ["a"] * 4)["low"][0])


def test_synthetic_sign_null_and_frozen_direction():
    m = imagereward(
        ir_rows((1, 2), group="a") + ir_rows((1, 2), group="b", prompt="dog")
    )
    cs = active(m)
    idx = {im["image_id"]: i for i, im in enumerate(m["images"])}
    x = np.array([[1, 0, 0.5], [0, 1, 0.5], [1, 0, 0.5], [0, 1, 0.5]])
    a = aggregate(x, idx, cs)
    candidates = select_candidates(a["effect"], ["positive", "negative", "null"])
    assert [c["concept"] for c in candidates] == ["positive", "negative"]
    reverse = aggregate(1 - x, idx, cs, [c["concept_id"] for c in candidates])
    evaluated = evaluate_frozen(candidates, reverse, 30)
    assert evaluated[0]["direction"] == 1 and evaluated[0]["heldout_effect"] < 0
    assert evaluated[0]["direction_adjusted_concordance"] == 0
    assert (
        select_candidates(
            aggregate(np.ones_like(x), idx, cs)["effect"], ["a", "b", "c"]
        )
        == []
    )


def test_grouped_reproducibility_and_components_not_events():
    ims, e = pap_rows()
    events = [dict(e, ranking_id=k) for k in range(10)]
    m = pickapic(events, ims)
    assign_splits(m)
    assert len({e["analysis_split"] for e in m["events"]}) == 1
    assert len({e["dependency_cluster_id"] for e in m["events"]}) == 1
    m2 = copy.deepcopy(m)
    assign_splits(m2)
    assert m2["events"] == m["events"]


def test_original_uuid_identity_connects_different_archive_paths():
    rows = ir_rows((1, 2), group="a", split="train") + ir_rows(
        (1, 2), group="b", prompt="other", split="test"
    )
    uid = "12345678-1234-1234-1234-123456789abc.webp"
    rows[0]["image_path"] = "images/train/train_1/" + uid
    rows[2]["image_path"] = "images/test/test_1/" + uid
    m = imagereward(rows)
    assign_splits(m)
    assert all(e["analysis_split"] == "quarantine" for e in m["events"])
