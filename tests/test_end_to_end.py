import json

from preference_diff.cli import main
from preference_diff.io import read_json


def test_offline_end_to_end_and_compare(tmp_path):
    for ds in ("imagereward", "pickapic"):
        main(
            [
                "fixture",
                "--dataset",
                ds,
                "--output",
                str(tmp_path / ds),
                "--bootstrap",
                "40",
            ]
        )
        a = tmp_path / ds / "analysis"
        for file in (
            "all_concepts.csv",
            "frozen_candidates.json",
            "top_concepts.json",
            "gallery.html",
            "annotation_template.csv",
            "report.md",
            "results_table.tex",
            "paper_subsection.tex",
            "joint_concept_score.json",
        ):
            assert (a / file).is_file()
        result = read_json(a / "top_concepts.json")
        assert result["candidates"][0]["concept"] == "known_positive"
        assert result["candidates"][0]["heldout_effect"] > 0
        assert "SYNTHETIC" in (a / "report.md").read_text()
    main(
        [
            "compare",
            "--left",
            str(tmp_path / "imagereward"),
            "--right",
            str(tmp_path / "pickapic"),
            "--output",
            str(tmp_path / "comparison"),
            "--bootstrap",
            "40",
        ]
    )
    assert (tmp_path / "comparison/primary/effect_scatter.png").exists()
    transfer = read_json(tmp_path / "comparison/primary/frozen_list_transfer.json")
    # Shared synthetic prompts are audited/excluded from transfer, never claimed independent.
    assert transfer["imagereward_to_pickapic"]["duplicate_events"]


def test_null_fixture_has_no_selected_features(tmp_path):
    main(["fixture", "--output", str(tmp_path), "--null-fixture", "--bootstrap", "20"])
    assert read_json(tmp_path / "analysis/top_concepts.json")["candidates"] == []
    # Strict JSON: no nonstandard NaN tokens in reports, even without candidate evidence.
    for path in tmp_path.rglob("*.json"):
        json.loads(
            path.read_text(),
            parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)),
        )


def test_missing_images_are_unavailable_not_cross_dataset_disagreement(tmp_path):
    from preference_diff.fixture import build_fixture
    from preference_diff.io import write_json

    main(["fixture", "--output", str(tmp_path / "imagereward"), "--bootstrap", "20"])
    right = tmp_path / "pickapic"
    build_fixture(right, "pickapic")
    state = read_json(right / "scores/state.json")
    state["valid"] = [False] * len(state["valid"])
    write_json(right / "scores/state.json", state)
    main(["analyze", "--output", str(right), "--bootstrap", "20"])
    main(
        [
            "compare",
            "--left",
            str(tmp_path / "imagereward"),
            "--right",
            str(right),
            "--output",
            str(tmp_path / "comparison"),
            "--bootstrap",
            "20",
        ]
    )
    result = read_json(tmp_path / "comparison/primary/comparison_summary.json")
    assert result["test"]["signed_rank_correlation"] is None
    for overlap in result["top_k_overlap"].values():
        assert overlap["intersection"] is None
        assert overlap["jaccard"] is None
        assert overlap["status"].startswith("unavailable")


def test_discovery_artifact_rejects_changed_top_k_and_keeps_test_order(tmp_path):
    import pytest

    main(["fixture", "--output", str(tmp_path), "--bootstrap", "20"])
    before = read_json(tmp_path / "analysis/frozen_candidates.json")
    main(
        [
            "analyze",
            "--output",
            str(tmp_path),
            "--bootstrap",
            "20",
            "--diversity-correlation",
            "0.95",
        ]
    )
    assert read_json(tmp_path / "analysis/frozen_candidates.json") == before
    assert (tmp_path / "analysis/presentation_candidates.json").is_file()
    with pytest.raises(ValueError, match="Frozen candidates"):
        main(
            ["analyze", "--output", str(tmp_path), "--top-k", "1", "--bootstrap", "20"]
        )
