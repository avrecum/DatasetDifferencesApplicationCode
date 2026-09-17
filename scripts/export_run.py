"""Copy compact, reviewable run artifacts into the repository, excluding raw data/scores.

Usage: python scripts/export_run.py RUN_ROOT REPORT_ROOT [--gallery]
Large full-vocabulary tables remain in RUN_ROOT and are indexed by SHA256.
"""

import argparse
from pathlib import Path
import shutil

from preference_diff.io import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--gallery", action="store_true")
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    compact = {
        "analysis_summary.json",
        "dataset_audit.json",
        "top_concepts.json",
        "frozen_candidates.json",
        "joint_concept_score.json",
        "replication_summary.json",
        "confounder_audit.json",
        "stratified_sensitivity.json",
        "annotator_component_sensitivity.json",
        "report.md",
        "results_table.tex",
        "paper_subsection.tex",
        "presentation_candidates.json",
        "bank_template_diagnostic.json",
        "discrimination_summary.csv",
        "discrimination_summary.png",
        "discrimination_summary.pdf",
        "concept_effects.png",
        "concept_effects.pdf",
    }
    for path in (args.run / "analysis").rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(args.run / "analysis")
        gallery = (
            "gallery_images" in relative.parts
            or "heatmaps" in relative.parts
            or path.name
            in ("gallery.html", "annotation_template.csv", "gallery_selection.json")
        )
        if path.name in compact or (args.gallery and gallery):
            target = args.report / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for path in (args.run / "scores").glob("*.json"):
        if path.name in (
            "cache.json",
            "bank_audit.json",
            "text_mode_audit.json",
            "reference_validation.json",
            "reuse_audit.json",
            "distributed_execution.json",
        ):
            target = args.report / "scoring" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
    for name in ("score_config.json", "gpu_batch_reference_validation.json"):
        if (args.run / name).exists():
            shutil.copy2(args.run / name, args.report / name)
    index = []
    for path in sorted(args.run.rglob("*")):
        if path.is_file() and path.suffix in (
            ".json",
            ".jsonl",
            ".csv",
            ".f32",
            ".npy",
            ".tex",
            ".md",
        ):
            index.append(
                dict(
                    path=str(path.resolve()),
                    relative_path=str(path.relative_to(args.run)),
                    bytes=path.stat().st_size,
                    sha256=file_hash(path),
                )
            )
    write_json(
        args.report / "artifact_index.json",
        dict(
            run_root=str(args.run.resolve()),
            files=index,
            note="Raw manifests, scores and full vocabulary tables remain at original run paths; this export copies compact review artifacts only.",
        ),
    )
    # Preserve the existing generated report and prepend navigation explaining full tables.
    report = args.report / "report.md"
    if report.exists():
        original = str((args.run / "analysis").resolve())
        report.write_text(
            f"> Compact export. Full tables and raw data are indexed in [artifact_index.json](artifact_index.json) and remain at `{original}`. Relative full-table links refer to that original directory.\n\n"
            + report.read_text()
        )


if __name__ == "__main__":
    main()
