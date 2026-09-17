# Recorded human-preference pilot: imagereward

Status: held-out results available.

## Method

Frozen FG-CLIP normalized text/patch cosine similarity, maximum over real patches. The full supplied bank is the candidate set. Positive effects mean higher scores in preferred images. Scores are not probabilities of concept presence.

Construction: `best_vs_worst`. Equal weight per comparison within event, per event within normalized prompt, and per prompt. Matched preferred/rejected means have exactly the same difference as the weighted paired effect. Pair retention enables dependence-aware inference; pairing alone does not change that algebraic identity.

Discovery selects strictly positive/negative effects and freezes identity, direction and ordering before test aggregation. No held-out reselection. Whole incomplete or failed events are excluded. Splits preserve normalized prompts and known image/content dependencies; cross-official-split components are quarantined. Content-hash coverage is limited to resolved images.

Patch budget: 1024; vocabulary: 54030. The 1,024-patch pilot differs from the supplied fg_clip.py 16,384-patch setting. Frozen score configuration: `5b806a087a0fb296639f66cfdaf44d920f0463f8dc187a092e4af198dcc1b36f`.

95% per-concept percentile cluster-bootstrap intervals use 2000 resamples, seed 42. Unequal-size components are resampled while preserving the prompt-weighted estimand. Fewer than two clusters gives unavailable intervals. These intervals are not simultaneous or tests of significance over the vocabulary.

## Actual observation counts

| Split | Prompts | Events | Comparisons | Components | Unique images |
|---|---:|---:|---:|---:|---:|
| discovery | 337 | 346 | 685 | 337 | 976 |
| validation | 75 | 75 | 145 | 75 | 209 |
| test | 75 | 75 | 152 | 75 | 213 |

## Frozen candidates in discovery order

| Concept | Side | Discovery Δ | Held-out Δ [95% CI] | Raw paired concordance |
|---|---|---:|---|---:|
| lip-syncing | preferred | 0.0089 | 0.0048 [-0.0010, 0.0108] | 0.4667 |
| lip-smacking | preferred | 0.0084 | 0.0043 [-0.0023, 0.0111] | 0.4667 |
| stare | preferred | 0.0083 | 0.0055 [0.0006, 0.0102] | 0.5433 |
| smirk | preferred | 0.0081 | 0.0066 [0.0019, 0.0115] | 0.5667 |
| wide-eyed | preferred | 0.0080 | 0.0049 [-0.0015, 0.0116] | 0.4700 |
| eyebrow-raising | preferred | 0.0077 | 0.0040 [-0.0016, 0.0098] | 0.5233 |
| big-eyed | preferred | 0.0074 | 0.0044 [-0.0015, 0.0107] | 0.5267 |
| physiognomy | preferred | 0.0073 | 0.0047 [-0.0007, 0.0106] | 0.5167 |
| serious-looking | preferred | 0.0070 | 0.0060 [0.0019, 0.0106] | 0.6233 |
| staring | preferred | 0.0070 | 0.0047 [0.0003, 0.0092] | 0.5333 |
| tarmac | rejected | -0.0068 | -0.0056 [-0.0125, 0.0008] | 0.4000 |
| shoe | rejected | -0.0065 | -0.0079 [-0.0154, -0.0007] | 0.4033 |
| ankle-deep | rejected | -0.0060 | -0.0093 [-0.0156, -0.0032] | 0.3800 |
| footprint | rejected | -0.0060 | -0.0058 [-0.0125, 0.0006] | 0.4567 |
| hand-shaped | rejected | -0.0060 | -0.0053 [-0.0117, 0.0009] | 0.4100 |
| bullet-shaped | rejected | -0.0059 | -0.0046 [-0.0101, 0.0004] | 0.4533 |
| trampling | rejected | -0.0058 | -0.0088 [-0.0156, -0.0019] | 0.3733 |
| snowfield | rejected | -0.0058 | -0.0012 [-0.0071, 0.0048] | 0.4733 |
| floor | rejected | -0.0058 | -0.0022 [-0.0101, 0.0056] | 0.4967 |
| fibular | rejected | -0.0057 | -0.0081 [-0.0133, -0.0031] | 0.3300 |

## Limits and artifacts

Small pilots measure feasibility, not a publication-ready sample size. Intervals can be unstable with few components. Concept detections and galleries remain **unverified by humans**. Supportive extremes, counterexamples and a random audit are shown separately. Concepts can encode generator, resolution, prompt or user effects; matching does not identify why annotators chose images. A failed held-out effect is retained as a result.

Bank generation metadata is absent. Inspect `bank_audit.json` and `text_mode_audit.json` in the score directory before interpreting dense/text compatibility. Model/source revisions, vocabulary/embedding hashes, dtype, processor and score settings are recorded in `cache.json`. No MLLM verification, Residual-OMP, masking extension, reward training, or SigLIP baseline is claimed.

[All signed concept effects](all_concepts.csv), [frozen candidates](frozen_candidates.json), [held-out metrics](top_concepts.json), [dataset audit](dataset_audit.json), [paired gallery](gallery.html), [annotation template](annotation_template.csv), [confounder audit](confounder_audit.json), [stratified sensitivities](stratified_sensitivity.json), [LaTeX table](results_table.tex).

[Overall discrimination plot](discrimination_summary.png), [discrimination table](discrimination_summary.csv), and [discovery-versus-held-out concept effects](concept_effects.png). The overall score combines concepts frozen on discovery; 0.5 is the paired equal-ordering reference.

Inspect the separate ImageReward `sensitivity/best_vs_rest` and `sensitivity/best_vs_worst` analyses when available. Model/patch-count stratifications use frozen primary candidates. Annotator-component sensitivity preserves prompt weights and may have too few clusters for intervals.

## Measured text-bank compatibility

Sample re-encoding cosine agreement with the supplied bank: box mean=0.6770, minimum=0.4235, short mean=0.3944, minimum=0.2389, long mean=0.3626, minimum=0.2034.
These measurements do not establish the original encoding recipe. If agreement is weak, treat concept interpretations as provisional until the bank's model/template provenance is confirmed; dimensional compatibility alone is insufficient.

## Joint discovery-selected concept score

Equal mean of discovery-direction-adjusted concept scores; no fitted reward model. Values below use the same prompt-balanced test observations.

| Metric | Estimate | 95% component-bootstrap interval |
|---|---:|---|
| Preferred-minus-rejected similarity | 0.0054 | [0.0024, 0.0087] |
| Paired concordance | 0.5967 | [0.4900, 0.6967] |

Concordance 0.5 is the equal-ordering reference; 0.6 means 60% weighted ordering credit, counting ties as half. An effect of +0.02 means +0.02 cosine-similarity units, not two percentage points of concept prevalence. Confidence intervals describe sampling uncertainty conditional on this scorer and cohort; they do not verify concept presence.

## Extended text-bank diagnostic

An independently run diagnostic tested 66 fixed mode/template/mask combinations on 32 vocabulary entries, without preference labels. Closest was `box` with template `a photo of a {}.`: mean cosine **0.9656**, minimum **0.8721**, same-row nearest match fraction **1.000** over the full bank.
This supports compatibility with templated dense-text embeddings more strongly than the raw-word check. It does not identify the exact original generation recipe, establish detector accuracy, or replace the bank. [Complete diagnostic](bank_template_diagnostic.json).
