from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

from .adapters import imagereward, pickapic, construct_comparisons
from .analysis import analyze_with_sensitivities
from .io import (
    load_manifest,
    read_json,
    read_rows,
    save_manifest,
    write_json,
    file_hash,
)
from .scoring import ScoreConfig, score_manifest
from .sources import (
    default_cache,
    hub_file,
    inspect_sources,
    load_metadata,
    resolve_images,
    prefetch_model,
)
from .splits import (
    assign_splits,
    sample_components,
    quarantine_new_hash_conflicts,
    restrict_available_cohort,
)


def parser():
    p = argparse.ArgumentParser(
        description="Fine-grained concepts associated with recorded human preferences"
    )
    p.add_argument(
        "command",
        choices=[
            "inspect",
            "prepare",
            "score",
            "analyze",
            "compare",
            "report",
            "run",
            "fixture",
            "prefetch",
        ],
    )
    p.add_argument(
        "--dataset", choices=["imagereward", "pickapic"], default="imagereward"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--sources",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs/sources.lock.json",
    )
    p.add_argument("--hf-home", type=Path, default=default_cache())
    p.add_argument("--manifest", type=Path)
    p.add_argument("--score-dir", type=Path)
    p.add_argument("--analysis-dir", type=Path)
    p.add_argument(
        "--metadata",
        type=Path,
        help="Local metadata JSON/JSONL/parquet; ImageReward rows need original_split",
    )
    p.add_argument("--image-metadata", type=Path, help="Local PickaPic UID table")
    p.add_argument("--local-image-root", type=Path)
    p.add_argument(
        "--available-image-ids",
        type=Path,
        help="Explicit archive-covered cohort JSON: canonical image_ids plus source provenance; require complete events",
    )
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument(
        "--group-limit",
        type=int,
        default=500,
        help="Distinct prompt cap; 0=full population, preserves components",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--comparison-mode",
        choices=["all_strict_pairs", "best_vs_rest", "best_vs_worst"],
        default="all_strict_pairs",
    )
    p.add_argument("--cohort", choices=["matched", "event"], default="matched")
    p.add_argument(
        "--split-policy", choices=["auto", "official", "grouped"], default="auto"
    )
    p.add_argument("--download-workers", type=int, default=4)
    p.add_argument("--download-retries", type=int, default=2)
    p.add_argument("--download-timeout", type=int, default=30)
    p.add_argument("--bank-dir", type=Path)
    p.add_argument(
        "--bank-diagnostic",
        type=Path,
        help="Optional independently run text-bank diagnostic to include in reporting",
    )
    p.add_argument("--model-name")
    p.add_argument("--model-revision")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    p.add_argument("--image-batch-size", type=int, default=1)
    p.add_argument("--text-chunk-size", type=int, default=512)
    p.add_argument("--preset", choices=["pilot", "reproduction"], default="pilot")
    p.add_argument("--patch-budget", type=int)
    p.add_argument("--reference-atol", type=float, default=2e-4)
    p.add_argument("--reference-rtol", type=float, default=2e-4)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--tie-tolerance", type=float, default=0.0)
    p.add_argument(
        "--diversity-correlation",
        type=float,
        help="Optional presentation-only discovery activation correlation filter",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--reuse-scores",
        type=Path,
        action="append",
        default=[],
        help="Read-only source score cache for identical-content reuse; repeatable",
    )
    p.add_argument("--examples-per-kind", type=int, default=2)
    p.add_argument(
        "--heatmaps",
        type=int,
        default=0,
        help="Number of gallery concepts to map; requires encoder/GPU",
    )
    p.add_argument("--left", type=Path, help="ImageReward run root for compare")
    p.add_argument("--right", type=Path, help="PickaPic run root for compare")
    p.add_argument("--null-fixture", action="store_true")
    return p


def bank_directory(args, lock):
    if args.bank_dir:
        return args.bank_dir
    root = args.hf_home / "preference_diff/bank" / lock["bank"]["revision"]
    root.mkdir(parents=True, exist_ok=True)
    for filename in ("embeddings.pt", "vocab.txt"):
        source = hub_file(lock["bank"], filename, args.hf_home)
        if not (root / filename).exists():
            (root / filename).symlink_to(source)
    return root


def prepare(args, lock, manifest_dir):
    availability_settings = (
        {"available_image_ids_sha256": file_hash(args.available_image_ids)}
        if args.available_image_ids
        else {}
    )
    if args.resume and (manifest_dir / "dataset_audit.json").exists():
        m = load_manifest(manifest_dir)
        old = m["provenance"].get("prepare_settings")
        new = dict(
            seed=args.seed,
            group_limit=args.group_limit,
            cohort=args.cohort,
            mode=args.comparison_mode,
            split_policy=args.split_policy,
            sources=lock,
            **availability_settings,
        )
        if old != new:
            raise ValueError("Prepare configuration changed; use a fresh run root")
    else:
        if args.metadata:
            rows = read_rows(args.metadata)
            ims = read_rows(args.image_metadata) if args.image_metadata else None
        else:
            rows, ims = load_metadata(args.dataset, lock, args.hf_home)
        if args.dataset == "imagereward":
            m = imagereward(rows, lock["imagereward"]["revision"], args.comparison_mode)
        else:
            if ims is None:
                raise ValueError(
                    "PickaPic requires the separate image UID metadata table"
                )
            m = pickapic(
                rows,
                ims,
                lock["pickapic_rankings"]["revision"],
                lock["pickapic_images"]["revision"],
                args.cohort,
            )
        assign_splits(m, args.seed, args.split_policy)
        if args.available_image_ids:
            available = read_json(args.available_image_ids)
            restrict_available_cohort(
                m,
                available["image_ids"],
                {
                    "sha256": availability_settings["available_image_ids_sha256"],
                    "sources": available.get("sources"),
                },
            )
        sample_components(m, args.group_limit, args.seed)
        m["provenance"].update(
            sources=lock,
            synthetic=False,
            prepare_settings=dict(
                seed=args.seed,
                group_limit=args.group_limit,
                cohort=args.cohort,
                mode=args.comparison_mode,
                split_policy=args.split_policy,
                sources=lock,
                **availability_settings,
            ),
        )
        save_manifest(manifest_dir, m)
    if not args.metadata_only:
        resolve_images(
            m,
            lock,
            args.hf_home,
            args.download_workers,
            args.download_retries,
            args.download_timeout,
            args.local_image_root,
        )
        quarantine_new_hash_conflicts(m)
    m["comparisons"], _ = construct_comparisons(m)
    save_manifest(manifest_dir, m)
    return m


def main(argv=None):
    args = parser().parse_args(argv)
    rank = int(os.environ.get("RANK", 0))
    if int(os.environ.get("WORLD_SIZE", 1)) > 1 and args.command not in (
        "score",
        "report",
    ):
        raise ValueError(
            "torchrun is supported for score and report; run CPU analysis once"
        )
    if args.group_limit < 0 or args.top_k < 0 or args.bootstrap < 1:
        raise ValueError("Invalid nonpositive run bounds")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.manifest or args.output / "manifest"
    score_dir = args.score_dir or args.output / "scores"
    analysis_dir = args.analysis_dir or args.output / "analysis"
    lock = read_json(args.sources)
    if args.command == "prefetch":
        prefetch_model(lock["model"], args.hf_home)
        bank_directory(args, lock)
        return
    if args.command == "inspect":
        inspect_sources(lock, args.hf_home, args.output)
        return
    if args.command == "fixture":
        from .fixture import build_fixture

        m = build_fixture(args.output, args.dataset, args.null_fixture)
        analyze_with_sensitivities(
            m,
            score_dir,
            analysis_dir,
            top_k=args.top_k,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        from .reporting import report

        report(m, score_dir, analysis_dir, args.examples_per_kind, args.seed)
        return
    if args.command == "compare":
        from .compare import compare

        if not args.left or not args.right:
            raise ValueError("compare requires --left and --right run roots")
        for mode in ("primary", "best_vs_rest"):
            la = (
                args.left / "analysis"
                if mode == "primary"
                else args.left / "analysis/sensitivity/best_vs_rest"
            )
            compare(
                load_manifest(args.left / "manifest"),
                load_manifest(args.right / "manifest"),
                args.left / "scores",
                args.right / "scores",
                la,
                args.right / "analysis",
                args.output / mode,
                args.bootstrap,
                args.seed,
                args.top_k,
            )
        return
    if args.command in ("prepare", "run"):
        m = prepare(args, lock, manifest_dir)
        if args.command == "prepare" or args.metadata_only:
            return
    else:
        m = load_manifest(manifest_dir)
    if args.command in ("score", "run"):
        patch_budget = args.patch_budget or (
            16384 if args.preset == "reproduction" else 1024
        )
        config = ScoreConfig(
            model_name=args.model_name or lock["model"]["repo"],
            model_revision=args.model_revision or lock["model"]["revision"],
            bank_revision=lock["bank"]["revision"],
            patch_budget=patch_budget,
            dtype=args.dtype,
            device=args.device,
            image_batch_size=args.image_batch_size,
            text_chunk_size=args.text_chunk_size,
            preset=args.preset,
            reference_atol=args.reference_atol,
            reference_rtol=args.reference_rtol,
        )
        score_manifest(
            m,
            bank_directory(args, lock),
            score_dir,
            config,
            args.hf_home,
            args.resume,
            args.reuse_scores,
        )
        if rank == 0:
            write_json(args.output / "score_config.json", asdict(config))
        if args.command == "score":
            return
    if args.command in ("analyze", "run"):
        analyze_with_sensitivities(
            m,
            score_dir,
            analysis_dir,
            top_k=args.top_k,
            bootstrap=args.bootstrap,
            seed=args.seed,
            tie_tolerance=args.tie_tolerance,
            diversity_correlation=args.diversity_correlation,
        )
        if args.command == "analyze":
            return
    if args.command in ("report", "run"):
        from .reporting import report
        from .parallel import context, barrier

        rank, _ = context()
        if rank == 0:
            report(
                m,
                score_dir,
                analysis_dir,
                args.examples_per_kind,
                args.seed,
                bank_diagnostic=args.bank_diagnostic,
            )
        barrier()
        if args.heatmaps:
            from .heatmaps import heatmaps

            heatmaps(
                m,
                score_dir,
                analysis_dir,
                bank_directory(args, lock),
                args.hf_home,
                args.heatmaps,
            )


if __name__ == "__main__":
    main()
