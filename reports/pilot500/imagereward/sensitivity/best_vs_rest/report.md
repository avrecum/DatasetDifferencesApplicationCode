# Recorded human-preference pilot: imagereward

Status: held-out results available.

## Method

Frozen FG-CLIP normalized text/patch cosine similarity, maximum over real patches. The full supplied bank is the candidate set. Positive effects mean higher scores in preferred images. Scores are not probabilities of concept presence.

Construction: `best_vs_rest`. Equal weight per comparison within event, per event within normalized prompt, and per prompt. Matched preferred/rejected means have exactly the same difference as the weighted paired effect. Pair retention enables dependence-aware inference; pairing alone does not change that algebraic identity.

Discovery selects strictly positive/negative effects and freezes identity, direction and ordering before test aggregation. No held-out reselection. Whole incomplete or failed events are excluded. Splits preserve normalized prompts and known image/content dependencies; cross-official-split components are quarantined. Content-hash coverage is limited to resolved images.

Patch budget: 1024; vocabulary: 54030. The 1,024-patch pilot differs from the supplied fg_clip.py 16,384-patch setting. Frozen score configuration: `5b806a087a0fb296639f66cfdaf44d920f0463f8dc187a092e4af198dcc1b36f`.

95% per-concept percentile cluster-bootstrap intervals use 2000 resamples, seed 42. Unequal-size components are resampled while preserving the prompt-weighted estimand. Fewer than two clusters gives unavailable intervals. These intervals are not simultaneous or tests of significance over the vocabulary.

## Actual observation counts

| Split | Prompts | Events | Comparisons | Components | Unique images |
|---|---:|---:|---:|---:|---:|
| discovery | 337 | 346 | 2495 | 337 | 2136 |
| validation | 75 | 75 | 521 | 75 | 458 |
| test | 75 | 75 | 516 | 75 | 442 |

## Frozen candidates in discovery order

| Concept | Side | Discovery Δ | Held-out Δ [95% CI] | Raw paired concordance |
|---|---|---:|---|---:|
| party-building | preferred | 0.0054 | -0.0051 [-0.0096, -0.0007] | 0.4275 |
| building | preferred | 0.0052 | -0.0042 [-0.0085, 0.0003] | 0.4776 |
| portico | preferred | 0.0051 | -0.0028 [-0.0071, 0.0018] | 0.4402 |
| cabin | preferred | 0.0049 | -0.0026 [-0.0065, 0.0016] | 0.4435 |
| guesthouse | preferred | 0.0048 | -0.0038 [-0.0076, -0.0001] | 0.4328 |
| outbuilding | preferred | 0.0047 | -0.0036 [-0.0075, 0.0002] | 0.4598 |
| log-cabin | preferred | 0.0047 | -0.0030 [-0.0071, 0.0009] | 0.4424 |
| city-dwelling | preferred | 0.0047 | -0.0026 [-0.0065, 0.0014] | 0.4785 |
| community-building | preferred | 0.0047 | -0.0030 [-0.0068, 0.0009] | 0.4775 |
| house | preferred | 0.0047 | -0.0026 [-0.0069, 0.0017] | 0.4406 |
| hand-shaped | rejected | -0.0042 | -0.0029 [-0.0074, 0.0017] | 0.4488 |
| backsaw | rejected | -0.0042 | -0.0002 [-0.0056, 0.0047] | 0.5302 |
| handgrip | rejected | -0.0039 | -0.0009 [-0.0058, 0.0038] | 0.5158 |
| tarmac | rejected | -0.0038 | -0.0053 [-0.0105, -0.0003] | 0.4186 |
| machete | rejected | -0.0036 | -0.0017 [-0.0068, 0.0035] | 0.5047 |
| watchband | rejected | -0.0035 | -0.0008 [-0.0065, 0.0050] | 0.4984 |
| handsaw | rejected | -0.0035 | -0.0027 [-0.0081, 0.0021] | 0.4943 |
| clasp | rejected | -0.0035 | -0.0013 [-0.0057, 0.0036] | 0.4542 |
| pocketknife | rejected | -0.0034 | -0.0016 [-0.0057, 0.0024] | 0.4861 |
| grenade | rejected | -0.0034 | -0.0039 [-0.0085, 0.0006] | 0.4214 |

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
| Preferred-minus-rejected similarity | -0.0006 | [-0.0027, 0.0016] |
| Paired concordance | 0.4840 | [0.4173, 0.5535] |

Concordance 0.5 is the equal-ordering reference; 0.6 means 60% weighted ordering credit, counting ties as half. An effect of +0.02 means +0.02 cosine-similarity units, not two percentage points of concept prevalence. Confidence intervals describe sampling uncertainty conditional on this scorer and cohort; they do not verify concept presence.

## Extended text-bank diagnostic

An independently run diagnostic tested 66 fixed mode/template/mask combinations on 32 vocabulary entries, without preference labels. Closest was `box` with template `a photo of a {}.`: mean cosine **0.9656**, minimum **0.8721**, same-row nearest match fraction **1.000** over the full bank.
This supports compatibility with templated dense-text embeddings more strongly than the raw-word check. It does not identify the exact original generation recipe, establish detector accuracy, or replace the bank. [Complete diagnostic](bank_template_diagnostic.json).
