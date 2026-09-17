"""Actual observed pairs, audit templates and paper text grounded in saved outputs."""

from __future__ import annotations

import html
from pathlib import Path
import numpy as np
from PIL import Image

from .io import atomic_text, digest, read_json, read_rows, write_csv, write_json
from .scoring import open_scores


def _public_metadata(image):
    keys = (
        "model_id",
        "negative_prompt",
        "gs",
        "steps",
        "scheduler_cls",
        "classification",
    )
    return {k: image["generation_metadata"].get(k) for k in keys}


def gallery(manifest, score_dir, analysis_dir, examples_per_kind=2, seed=42):
    root = Path(analysis_dir)
    scores, index, valid, vocab, meta, state = open_scores(score_dir)
    candidates = read_json(root / "frozen_candidates.json")["candidates"]
    comparisons = [
        c
        for c in read_rows(root / "comparisons.jsonl")
        if c["analysis_split"] == "test"
    ]
    events = {e["event_id"]: e for e in manifest["events"]}
    images = {i["image_id"]: i for i in manifest["images"]}
    records, sections = [], []
    thumbs = root / "gallery_images"
    thumbs.mkdir(exist_ok=True)

    def thumbnail(iid):
        im = images[iid]
        name = digest(iid)[:24] + ".jpg"
        path = thumbs / name
        if not path.exists() and im.get("local_path"):
            try:
                with Image.open(im["local_path"]) as source:
                    source = source.convert("RGB")
                    source.thumbnail((384, 384))
                    source.save(path, quality=85)
            except OSError:
                return ""
        return f"gallery_images/{name}" if path.exists() else ""

    for candidate in candidates:
        j, direction = candidate["concept_id"], candidate["direction"]
        pairs = []
        for c in comparisons:
            w, rejected_value = c["preferred_image_id"], c["rejected_image_id"]
            delta = float(scores[index[w], j] - scores[index[rejected_value], j])
            pairs.append((direction * delta, digest([seed, j, c]), c, delta))
        by_kind = {
            "supportive": sorted(
                (p for p in pairs if p[0] > 0), key=lambda p: (-p[0], p[1])
            ),
            "counterexample": sorted(
                (p for p in pairs if p[0] < 0), key=lambda p: (p[0], p[1])
            ),
            "random_audit": sorted(pairs, key=lambda p: p[1]),
        }
        cards = []
        for kind, ordered in by_kind.items():
            used = set()
            for _, _, c, delta in ordered:
                if c["prompt_group_id"] in used:
                    continue
                used.add(c["prompt_group_id"])
                w, rejected_value = c["preferred_image_id"], c["rejected_image_id"]
                e = events[c["event_id"]]
                row = dict(
                    concept_id=j,
                    concept=candidate["concept"],
                    direction=direction,
                    selection=kind,
                    event_id=c["event_id"],
                    prompt=e["original_prompt"],
                    prompt_group_id=c["prompt_group_id"],
                    preferred_image_id=w,
                    rejected_image_id=rejected_value,
                    preferred_score=float(scores[index[w], j]),
                    rejected_score=float(scores[index[rejected_value], j]),
                    difference=delta,
                    preferred_thumbnail=thumbnail(w),
                    rejected_thumbnail=thumbnail(rejected_value),
                    preferred_metadata=_public_metadata(images[w]),
                    rejected_metadata=_public_metadata(images[rejected_value]),
                    human_review_status="unverified",
                    concept_present_preferred="",
                    concept_present_rejected="",
                    reviewer_notes="",
                )
                records.append(row)

                def esc(v):
                    return html.escape(str(v))

                cards.append(
                    f'<article><b>{esc(kind)}</b><p>{esc(row["prompt"])}</p><div class="pair">'
                    + "".join(
                        f'<figure><img src="{esc(row[side + "_thumbnail"])}"><figcaption>{side}: {row[side + "_score"]:.4f}<br>{esc(row[side + "_metadata"])}</figcaption></figure>'
                        for side in ("preferred", "rejected")
                    )
                    + f"</div><p>Preferred − rejected: {delta:+.4f}; event {esc(c['event_id'])}. Unverified.</p></article>"
                )
                if len(used) >= examples_per_kind:
                    break
        sections.append(
            f"<section><h2>{html.escape(candidate['concept'])} ({candidate['side']} direction frozen on discovery)</h2>"
            + "".join(cards)
            + "</section>"
        )
    title = (
        "SYNTHETIC OFFLINE FIXTURE"
        if manifest.get("provenance", {}).get("synthetic")
        else manifest["audit"]["dataset"]
    )
    page = '<!doctype html><meta charset="utf-8"><title>Preference concept audit</title><style>body{font:16px system-ui;max-width:1100px;margin:2rem auto}article{border-top:1px solid #ccc;padding:1rem}.pair{display:flex}figure{width:48%;margin:1%}img{max-width:100%}figcaption{overflow-wrap:anywhere}p{overflow-wrap:anywhere}</style>'
    page += f"<h1>{html.escape(title)} — unverified visual audit</h1><p>Held-out observed pairs. Supportive/counterexamples are extremes in the frozen direction, one per distinct prompt; random audit uses a seeded hash ordering. Examples are illustrative, not typicality or causal evidence. The CSV is a blank human annotation template.</p>"
    page += (
        "".join(sections)
        if records
        else "<p>No eligible held-out image pairs. Qualitative results unavailable.</p>"
    )
    atomic_text(root / "gallery.html", page)
    write_csv(
        root / "annotation_template.csv",
        records,
        fields=list(records[0])
        if records
        else ["concept", "event_id", "human_review_status"],
    )
    write_json(
        root / "gallery_selection.json",
        dict(
            seed=seed,
            rule="extreme supporting/counterexample and hash-random; distinct prompt per concept/kind",
            examples=records,
        ),
    )
    return records


def _fmt(value, digits=4):
    return (
        "unavailable"
        if value is None or not np.isfinite(value)
        else f"{value:.{digits}f}"
    )


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(c, c) for c in str(text))


def report(
    manifest,
    score_dir,
    analysis_dir,
    examples_per_kind=2,
    seed=42,
    bank_diagnostic=None,
):
    root = Path(analysis_dir)
    summary = read_json(root / "analysis_summary.json")
    top = read_json(root / "top_concepts.json")
    cache = read_json(Path(score_dir) / "cache.json")
    diagnostic_path = root / "bank_template_diagnostic.json"
    if bank_diagnostic:
        diagnostic = read_json(bank_diagnostic)
        cfg = cache["configuration"]
        if diagnostic["model"]["model_revision"] != cfg.get("model_revision") or any(
            diagnostic["bank_audit"][key] != cfg.get("bank", {}).get(key)
            for key in ("embedding_sha256", "vocab_sha256", "vocabulary_order_hash")
        ):
            raise ValueError(
                "Bank diagnostic is incompatible with this score configuration"
            )
        write_json(diagnostic_path, diagnostic)
    gallery(manifest, score_dir, root, examples_per_kind, seed)
    plot_discrimination(root, summary, top)
    title = (
        "SYNTHETIC FIXTURE — NOT EMPIRICAL RESULTS"
        if summary["synthetic"]
        else "Recorded human-preference pilot"
    )
    counts = summary["counts"]
    lines = [
        f"# {title}: {summary['dataset']}",
        "",
        f"Status: {summary['status']}.",
        "",
        "## Method",
        "",
        "Frozen FG-CLIP normalized text/patch cosine similarity, maximum over real patches. The full supplied bank is the candidate set. Positive effects mean higher scores in preferred images. Scores are not probabilities of concept presence.",
        "",
        f"Construction: `{summary['construction_mode']}`. Equal weight per comparison within event, per event within normalized prompt, and per prompt. Matched preferred/rejected means have exactly the same difference as the weighted paired effect. Pair retention enables dependence-aware inference; pairing alone does not change that algebraic identity.",
        "",
        "Discovery selects strictly positive/negative effects and freezes identity, direction and ordering before test aggregation. No held-out reselection. Whole incomplete or failed events are excluded. Splits preserve normalized prompts and known image/content dependencies; cross-official-split components are quarantined. Content-hash coverage is limited to resolved images.",
        "",
        f"Patch budget: {summary['patch_budget']}; vocabulary: {summary['vocabulary_size']}. The 1,024-patch pilot differs from the supplied fg_clip.py 16,384-patch setting. Frozen score configuration: `{cache['fingerprint']}`.",
        "",
        f"95% per-concept percentile cluster-bootstrap intervals use {summary['bootstrap_resamples']} resamples, seed {summary['seed']}. Unequal-size components are resampled while preserving the prompt-weighted estimand. Fewer than two clusters gives unavailable intervals. These intervals are not simultaneous or tests of significance over the vocabulary.",
        "",
        "## Actual observation counts",
        "",
        "| Split | Prompts | Events | Comparisons | Components | Unique images |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s, c in counts.items():
        lines.append(
            f"| {s} | {c['prompts']} | {c['events']} | {c['comparisons']} | {c['components']} | {c['unique_images']} |"
        )
    lines += [
        "",
        "## Frozen candidates in discovery order",
        "",
        "| Concept | Side | Discovery Δ | Held-out Δ [95% CI] | Raw paired concordance |",
        "|---|---|---:|---|---:|",
    ]
    tex = [
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Concept & Side & Discovery $\Delta$ & Test $\Delta$ & 95\% CI \\",
        r"\midrule",
    ]
    for row in top["candidates"]:
        lines.append(
            f"| {row['concept'].replace('|', '/')} | {row['side']} | {_fmt(row['discovery_effect'])} | {_fmt(row['heldout_effect'])} [{_fmt(row['effect_ci_low'])}, {_fmt(row['effect_ci_high'])}] | {_fmt(row['raw_concordance'])} |"
        )
        tex.append(
            f"{latex_escape(row['concept'])} & {row['side']} & {_fmt(row['discovery_effect'])} & {_fmt(row['heldout_effect'])} & [{_fmt(row['effect_ci_low'])}, {_fmt(row['effect_ci_high'])}] \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    if not top["candidates"]:
        lines.append(
            "No discovery candidates could be selected; empirical concept results are unavailable."
        )
    lines += [
        "",
        "## Limits and artifacts",
        "",
        "Small pilots measure feasibility, not a publication-ready sample size. Intervals can be unstable with few components. Concept detections and galleries remain **unverified by humans**. Supportive extremes, counterexamples and a random audit are shown separately. Concepts can encode generator, resolution, prompt or user effects; matching does not identify why annotators chose images. A failed held-out effect is retained as a result.",
        "",
        "Bank generation metadata is absent. Inspect `bank_audit.json` and `text_mode_audit.json` in the score directory before interpreting dense/text compatibility. Model/source revisions, vocabulary/embedding hashes, dtype, processor and score settings are recorded in `cache.json`. No MLLM verification, Residual-OMP, masking extension, reward training, or SigLIP baseline is claimed.",
        "",
        "[All signed concept effects](all_concepts.csv), [frozen candidates](frozen_candidates.json), [held-out metrics](top_concepts.json), [dataset audit](dataset_audit.json), [paired gallery](gallery.html), [annotation template](annotation_template.csv), [confounder audit](confounder_audit.json), [stratified sensitivities](stratified_sensitivity.json), [LaTeX table](results_table.tex).",
        "",
        "[Overall discrimination plot](discrimination_summary.png), [discrimination table](discrimination_summary.csv), and [discovery-versus-held-out concept effects](concept_effects.png). The overall score combines concepts frozen on discovery; 0.5 is the paired equal-ordering reference.",
        "",
        "Inspect the separate ImageReward `sensitivity/best_vs_rest` and `sensitivity/best_vs_worst` analyses when available. Model/patch-count stratifications use frozen primary candidates. Annotator-component sensitivity preserves prompt weights and may have too few clusters for intervals.",
    ]
    mode_path = Path(score_dir) / "text_mode_audit.json"
    if mode_path.exists():
        modes = read_json(mode_path)["settings"]
        lines += [
            "",
            "## Measured text-bank compatibility",
            "",
            "Sample re-encoding cosine agreement with the supplied bank: "
            + ", ".join(
                f"{r['walk_type']} mean={r['mean_cosine']:.4f}, minimum={r['min_cosine']:.4f}"
                for r in modes
            )
            + ".",
            "These measurements do not establish the original encoding recipe. If agreement is weak, treat concept interpretations as provisional until the bank's model/template provenance is confirmed; dimensional compatibility alone is insufficient.",
        ]
    if (root / "joint_concept_score.json").exists():
        joint = read_json(root / "joint_concept_score.json")
        metric = joint["metrics"][0]
        lines += [
            "",
            "## Joint discovery-selected concept score",
            "",
            "Equal mean of discovery-direction-adjusted concept scores; no fitted reward model. Values below use the same prompt-balanced test observations.",
            "",
            "| Metric | Estimate | 95% component-bootstrap interval |",
            "|---|---:|---|",
            f"| Preferred-minus-rejected similarity | {_fmt(metric['heldout_effect'])} | [{_fmt(metric['effect_ci_low'])}, {_fmt(metric['effect_ci_high'])}] |",
            f"| Paired concordance | {_fmt(metric['raw_concordance'])} | [{_fmt(metric['concordance_ci_low'])}, {_fmt(metric['concordance_ci_high'])}] |",
            "",
            "Concordance 0.5 is the equal-ordering reference; 0.6 means 60% weighted ordering credit, counting ties as half. An effect of +0.02 means +0.02 cosine-similarity units, not two percentage points of concept prevalence. Confidence intervals describe sampling uncertainty conditional on this scorer and cohort; they do not verify concept presence.",
        ]
    if diagnostic_path.exists():
        diagnostic = read_json(diagnostic_path)
        best = max(diagnostic["settings"], key=lambda r: r["mean_cosine"])
        lines += [
            "",
            "## Extended text-bank diagnostic",
            "",
            f"An independently run diagnostic tested {len(diagnostic['settings'])} fixed mode/template/mask combinations on {len(diagnostic['words'])} vocabulary entries, without preference labels. Closest was `{best['walk_type']}` with template `{best['template']}`: mean cosine **{best['mean_cosine']:.4f}**, minimum **{best['minimum_cosine']:.4f}**, same-row nearest match fraction **{best['same_row_nearest_fraction']:.3f}** over the full bank.",
            "This supports compatibility with templated dense-text embeddings more strongly than the raw-word check. It does not identify the exact original generation recipe, establish detector accuracy, or replace the bank. [Complete diagnostic](bank_template_diagnostic.json).",
        ]
    atomic_text(root / "report.md", "\n".join(lines) + "\n")
    atomic_text(root / "results_table.tex", "% " + title + "\n" + "\n".join(tex) + "\n")
    test = counts["test"]
    empirical = (
        f"This engineering pilot retained {test['prompts']} held-out prompts in {test['components']} dependence components for {latex_escape(summary['dataset'])}. Table entries report observed effects and per-concept intervals; visual interpretations await human review."
        if test["prompts"] and not summary["synthetic"]
        else "Empirical human-preference results are unavailable in this artifact; synthetic outputs are only implementation checks."
    )
    paragraph = (
        r"\mypara{Visual concepts associated with human preference}"
        + "\n"
        + r"We apply frozen FG-CLIP2 dense image features and the full supplied concept bank to human-ranked alternatives within each dataset. For every concept, we average preferred-minus-rejected maximum patch similarities within annotation events, then within normalized prompts, then equally across prompts. ImageReward uses all strict ranked pairs; the linked four-candidate Pick-a-Pic release uses the selected image against each non-selected alternative. We freeze concepts and directions on discovery data and evaluate held-out prompts, preserving prompt and image dependencies. We report paired concordance and component-bootstrap uncertainty, with best-versus-rest and generator/patch-count sensitivities. These are associations with recorded choices, not calibrated prevalence estimates or causal explanations. "
        + empirical
        + "\n"
    )
    if test["prompts"] and not summary["synthetic"] and top["candidates"]:
        evaluated = [c for c in top["candidates"] if c["heldout_effect"] is not None]
        replicated = sum(c["direction"] * c["heldout_effect"] > 0 for c in evaluated)
        names = [c["concept"] for c in top["candidates"][:3]]
        paragraph += f"The first discovery-ranked terms were {', '.join(latex_escape(n) for n in names)}. Of {len(evaluated)} frozen candidates, {replicated} retained their discovery direction in held-out effects; this descriptive count is not a multiplicity-adjusted significance claim. "
        joint_path = root / "joint_concept_score.json"
        if joint_path.exists():
            metrics = read_json(joint_path)["metrics"][0]
            paragraph += f"The equal signed mean of the selected concepts achieved held-out paired concordance {_fmt(metrics['raw_concordance'], 3)} (95\\% component-bootstrap interval [{_fmt(metrics['concordance_ci_low'], 3)}, {_fmt(metrics['concordance_ci_high'], 3)}]). "
        paragraph += "These pilot findings remain provisional: visual concept presence is unverified, the supplied text-bank encoding provenance is unresolved, and the sample is small.\n"
        write_json(
            root / "replication_summary.json",
            dict(
                selected_evaluated=len(evaluated),
                same_direction=replicated,
                different_or_zero_direction=len(evaluated) - replicated,
                discovery_examples=names,
                interpretation="descriptive direction replication, not significance or verified semantic presence",
            ),
        )
    atomic_text(root / "paper_subsection.tex", paragraph)
    return summary


def plot_discrimination(root, summary, top):
    """Readable figures from already frozen results; no selection or fitting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    rows = []
    directories = [root] + sorted((root / "sensitivity").glob("*"))
    for directory in directories:
        path = directory / "joint_concept_score.json"
        if not path.exists():
            continue
        info = read_json(directory / "analysis_summary.json")
        for metric in read_json(path)["metrics"]:
            rows.append(
                dict(
                    construction=info["construction_mode"],
                    paired_concordance=metric["raw_concordance"],
                    ci_low=metric["concordance_ci_low"],
                    ci_high=metric["concordance_ci_high"],
                    signed_effect=metric["heldout_effect"],
                    prompts=metric["counts"]["prompts"],
                    components=metric["counts"]["components"],
                    comparisons=metric["counts"]["comparisons"],
                    synthetic=summary["synthetic"],
                )
            )
    write_csv(
        root / "discrimination_summary.csv",
        rows,
        fields=[
            "construction",
            "paired_concordance",
            "ci_low",
            "ci_high",
            "signed_effect",
            "prompts",
            "components",
            "comparisons",
            "synthetic",
        ],
    )
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, row in enumerate(rows):
        value = row["paired_concordance"]
        if value is None:
            continue
        ax.scatter([value], [i], color="navy", zorder=3)
        if row["ci_low"] is not None:
            ax.hlines(i, row["ci_low"], row["ci_high"], color="navy", lw=2)
        ax.annotate(
            f"{value:.3f}; {row['prompts']} prompts",
            (value, i),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.axvline(0.5, ls="--", color="grey", label="Equal-ordering reference (0.5)")
    ax.set(
        xlim=(0, 1),
        ylim=(-0.5, max(0.5, len(rows) - 0.5)),
        yticks=list(range(len(rows))),
        yticklabels=[r["construction"].replace("_", " ") for r in rows],
        xlabel="Held-out prompt-balanced paired concordance",
    )
    ax.invert_yaxis()
    if not rows:
        ax.text(
            0.5,
            0.5,
            "Unavailable: no frozen concept score",
            transform=ax.transAxes,
            ha="center",
        )
    label = "SYNTHETIC CHECK" if summary["synthetic"] else summary["dataset"]
    ax.set_title(f"{label}: frozen concept-score discrimination (95% intervals)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "discrimination_summary.png", dpi=180)
    fig.savefig(root / "discrimination_summary.pdf")
    plt.close(fig)
    candidates = top["candidates"]
    fig, ax = plt.subplots(figsize=(8, max(3, len(candidates) * 0.3 + 1.5)))
    for i, c in enumerate(candidates):
        ax.scatter(
            [c["discovery_effect"]],
            [i],
            marker="x",
            color="grey",
            label="Discovery" if i == 0 else None,
        )
        if c["heldout_effect"] is not None:
            ax.scatter(
                [c["heldout_effect"]],
                [i],
                color="navy",
                label="Held-out" if i == 0 else None,
            )
            if c["effect_ci_low"] is not None:
                ax.hlines(
                    i, c["effect_ci_low"], c["effect_ci_high"], color="navy", lw=1.4
                )
    ax.axvline(0, color="grey", ls="--")
    ax.set(
        yticks=list(range(len(candidates))),
        yticklabels=[c["concept"] for c in candidates],
        xlabel="Preferred minus rejected cosine-similarity score",
        title=f"{label}: concepts in frozen discovery order",
    )
    ax.invert_yaxis()
    if candidates:
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "Unavailable: no selected concepts",
            transform=ax.transAxes,
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(root / "concept_effects.png", dpi=180)
    fig.savefig(root / "concept_effects.pdf")
    plt.close(fig)
