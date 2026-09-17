import copy

import numpy as np

from preference_diff.sensitivity import (
    generator_comparison_strata,
    generator_sensitivity,
)
from preference_diff.statistics import aggregate


def fixture():
    images = {
        str(i): {"generation_metadata": {"model_id": m}}
        for i, m in enumerate(("A", "A", "B", "B"))
    }
    cs = [
        dict(
            event_id="e",
            prompt_group_id="p",
            analysis_split="test",
            dependency_cluster_id="d",
            preferred_image_id="0",
            rejected_image_id=str(i),
            within_event_weight=1 / 3,
        )
        for i in (1, 2, 3)
    ]
    return images, cs


def test_generator_strata_renormalize_complete_event_pairs_without_mutation():
    images, cs = fixture()
    original = copy.deepcopy(cs)
    strata = generator_comparison_strata(cs + cs, images)
    assert cs == original
    assert len(strata["same_model"]) == 1
    assert len(strata["cross_model"]) == 2
    assert [c["within_event_weight"] for c in strata["same_model"]] == [1]
    assert [c["within_event_weight"] for c in strata["cross_model"]] == [0.5, 0.5]
    scores = np.array([[1], [2], [0], [0.5]])
    index = {str(i): i for i in range(4)}
    assert aggregate(scores, index, strata["same_model"])["effect"][0] == -1
    assert aggregate(scores, index, strata["cross_model"])["effect"][0] == 0.75


def test_generator_sensitivity_freezes_directions_and_excludes_unknown_only_here():
    images, cs = fixture()
    images["3"]["generation_metadata"]["model_id"] = None
    candidate = dict(
        concept_id=0,
        concept="x",
        direction=-1,
        side="rejected",
        discovery_rank=1,
        discovery_effect=-0.1,
    )
    result = generator_sensitivity(
        np.array([[1], [2], [0], [0.5]]),
        {str(i): i for i in range(4)},
        cs,
        images,
        [candidate],
        bootstrap=10,
    )
    same, cross = result["strata"]["same_model"], result["strata"]["cross_model"]
    assert same["joint"]["raw_concordance"] == 1
    assert cross["joint"]["raw_concordance"] == 0
    assert cross["counts"]["comparisons"] == 1
    assert cross["concepts"][0]["direction"] == -1
    assert len(cs) == 3
    assert generator_sensitivity(np.zeros((4, 1)), {}, cs, images, [])["strata"] == {}
