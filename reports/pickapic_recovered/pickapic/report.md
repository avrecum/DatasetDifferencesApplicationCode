> Compact export. Full tables and raw data are indexed in [artifact_index.json](artifact_index.json) and remain at `/e/project1/taco-vlm/avrecum/DatasetDifferencesApplicationsCode/runs/pickapic_recovered/pickapic/analysis`. Relative full-table links refer to that original directory.

# Recorded human-preference pilot: pickapic

Status: held-out results available.

**Cohort restriction:** Complete eligible original events covered by the declared archive; not the full dataset. Archive coverage retained 573 metadata-eligible events across 330 prompts before decoding/hash checks. All displayed candidates are required. Original human labels, candidates and full-metadata dependency splits are preserved. This availability-selected subset need not represent the full dataset.

## Method

Frozen FG-CLIP normalized text/patch cosine similarity, maximum over real patches. The full supplied bank is the candidate set. Positive effects mean higher scores in preferred images. Scores are not probabilities of concept presence.

Construction: `winner_vs_rest`. Equal weight per comparison within event, per event within normalized prompt, and per prompt. Matched preferred/rejected means have exactly the same difference as the weighted paired effect. Pair retention enables dependence-aware inference; pairing alone does not change that algebraic identity.

Discovery selects strictly positive/negative effects and freezes identity, direction and ordering before test aggregation. No held-out reselection. Whole incomplete or failed events are excluded. Splits preserve normalized prompts and known image/content dependencies; cross-official-split components are quarantined. Content-hash coverage is limited to resolved images.

Patch budget: 1024; vocabulary: 54030. The 1,024-patch pilot differs from the supplied fg_clip.py 16,384-patch setting. Frozen score configuration: `8badb97d707266a3b54e1c920d92060285523dc67fe1cca01201d968de41284b`.

95% per-concept percentile cluster-bootstrap intervals use 2000 resamples, seed 42. Unequal-size components are resampled while preserving the prompt-weighted estimand. Fewer than two clusters gives unavailable intervals. These intervals are not simultaneous or tests of significance over the vocabulary.

## Actual observation counts

| Split | Prompts | Events | Comparisons | Components | Unique images |
|---|---:|---:|---:|---:|---:|
| discovery | 262 | 482 | 1446 | 170 | 1714 |
| validation | 35 | 47 | 141 | 32 | 177 |
| test | 33 | 44 | 132 | 31 | 162 |

## Frozen candidates in discovery order

| Concept | Side | Discovery Δ | Held-out Δ [95% CI] | Raw paired concordance |
|---|---|---:|---|---:|
| lip | preferred | 0.0080 | -0.0032 [-0.0093, 0.0026] | 0.4596 |
| lip-syncing | preferred | 0.0072 | -0.0042 [-0.0106, 0.0022] | 0.4419 |
| lip-smacking | preferred | 0.0070 | -0.0040 [-0.0116, 0.0032] | 0.5051 |
| nose | preferred | 0.0068 | -0.0043 [-0.0125, 0.0028] | 0.4848 |
| mouth | preferred | 0.0065 | -0.0002 [-0.0068, 0.0064] | 0.4975 |
| nostril | preferred | 0.0061 | -0.0052 [-0.0133, 0.0019] | 0.4192 |
| mouthpart | preferred | 0.0059 | -0.0008 [-0.0067, 0.0046] | 0.4848 |
| nosebleed | preferred | 0.0057 | -0.0059 [-0.0150, 0.0021] | 0.4444 |
| lipstick | preferred | 0.0056 | -0.0060 [-0.0122, -0.0002] | 0.4040 |
| mouth-to-mouth | preferred | 0.0054 | -0.0003 [-0.0060, 0.0057] | 0.5379 |
| pavement | rejected | -0.0079 | -0.0108 [-0.0263, 0.0034] | 0.4571 |
| kickstand | rejected | -0.0078 | -0.0013 [-0.0134, 0.0114] | 0.5000 |
| driveway | rejected | -0.0076 | -0.0084 [-0.0225, 0.0050] | 0.4318 |
| sidewalk | rejected | -0.0074 | -0.0035 [-0.0157, 0.0088] | 0.4722 |
| doormat | rejected | -0.0073 | -0.0039 [-0.0116, 0.0043] | 0.4545 |
| carpeting | rejected | -0.0073 | -0.0032 [-0.0111, 0.0049] | 0.4874 |
| carpet | rejected | -0.0072 | -0.0042 [-0.0128, 0.0048] | 0.4672 |
| carpeted | rejected | -0.0071 | -0.0050 [-0.0134, 0.0037] | 0.4015 |
| bollard | rejected | -0.0070 | -0.0016 [-0.0135, 0.0107] | 0.4343 |
| paved | rejected | -0.0070 | -0.0099 [-0.0235, 0.0031] | 0.4470 |

## Limits and artifacts

Small pilots measure feasibility, not a publication-ready sample size. Intervals can be unstable with few components. Concept detections and galleries remain **unverified by humans**. Supportive extremes, counterexamples and a random audit are shown separately. Concepts can encode generator, resolution, prompt or user effects; matching does not identify why annotators chose images. A failed held-out effect is retained as a result.

Bank generation metadata is absent. Inspect `bank_audit.json` and `text_mode_audit.json` in the score directory before interpreting dense/text compatibility. Model/source revisions, vocabulary/embedding hashes, dtype, processor and score settings are recorded in `cache.json`. No MLLM verification, Residual-OMP, masking extension, reward training, or SigLIP baseline is claimed.

[All signed concept effects](all_concepts.csv), [frozen candidates](frozen_candidates.json), [held-out metrics](top_concepts.json), [dataset audit](dataset_audit.json), [paired gallery](gallery.html), [annotation template](annotation_template.csv), [confounder audit](confounder_audit.json), [stratified sensitivities](stratified_sensitivity.json), [LaTeX table](results_table.tex).

[Overall discrimination plot](discrimination_summary.png), [discrimination table](discrimination_summary.csv), and [discovery-versus-held-out concept effects](concept_effects.png). The overall score combines concepts frozen on discovery; 0.5 is the paired equal-ordering reference.

Inspect the separate ImageReward `sensitivity/best_vs_rest` and `sensitivity/best_vs_worst` analyses when available. Model/patch-count stratifications use frozen primary candidates. [Generator-pair sensitivities](generator_pair_sensitivity.json) restrict comparisons within complete events and renormalize event/prompt weights; this changes the conditional observation set. Annotator-component sensitivity preserves prompt weights and may have too few clusters for intervals.

## Measured text-bank compatibility

Sample re-encoding cosine agreement with the supplied bank: box mean=0.6770, minimum=0.4235, short mean=0.3944, minimum=0.2389, long mean=0.3626, minimum=0.2034.
These measurements do not establish the original encoding recipe. If agreement is weak, treat concept interpretations as provisional until the bank's model/template provenance is confirmed; dimensional compatibility alone is insufficient.

## Joint discovery-selected concept score

Additional application diagnostic: equal mean of discovery-direction-adjusted concept scores; no fitted reward model. This differs from the draft paper's MLLM discrimination evaluation. Values below use the same prompt-balanced test observations.

| Metric | Estimate | 95% component-bootstrap interval |
|---|---:|---|
| Preferred-minus-rejected similarity | 0.0009 | [-0.0033, 0.0049] |
| Paired concordance | 0.4924 | [0.3750, 0.6111] |

Concordance 0.5 is the equal-ordering reference; 0.6 means 60% weighted ordering credit, counting ties as half. An effect of +0.02 means +0.02 cosine-similarity units, not two percentage points of concept prevalence. Confidence intervals describe sampling uncertainty conditional on this scorer and cohort; they do not verify concept presence.

## Extended text-bank diagnostic

An independently run diagnostic tested 66 fixed mode/template/mask combinations on 32 vocabulary entries, without preference labels. Closest was `box` with template `a photo of a {}.`: mean cosine **0.9656**, minimum **0.8721**, same-row nearest match fraction **1.000** over the full bank.
This supports compatibility with templated dense-text embeddings more strongly than the raw-word check. It does not identify the exact original generation recipe, establish detector accuracy, or replace the bank. [Complete diagnostic](bank_template_diagnostic.json).
