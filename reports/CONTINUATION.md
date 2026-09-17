# Human-preference application: 500-prompt continuation

The ImageReward pilot now covers 500 sampled prompts and has completed real frozen FG-CLIP scoring, held-out evaluation, both ranking sensitivities, galleries and four patch maps. The primary frozen concept score obtains **55.1% prompt-balanced paired concordance, 95% CI 49.9–59.9%**, on 75 held-out prompts. Its interval includes the 50% reference. **Pick-a-Pic results and cross-dataset replication remain unavailable** because no complete eligible events with accessible original images were recovered.

## Review the results

- [Generated methods/results report](pilot500/imagereward/report.md), [discrimination plot](pilot500/imagereward/discrimination_summary.png), [all twenty frozen candidates and intervals](pilot500/imagereward/top_concepts.json), [discovery versus held-out effects](pilot500/imagereward/concept_effects.png).
- [Paired-image gallery](pilot500/imagereward/gallery.html), [blank human annotation CSV](pilot500/imagereward/annotation_template.csv), [four patch maps](pilot500/imagereward/heatmaps/index.json). These are **human-unverified**.
- [Best versus rest](pilot500/imagereward/sensitivity/best_vs_rest/report.md), [best versus worst](pilot500/imagereward/sensitivity/best_vs_worst/report.md), [geometry/category sensitivity metrics](pilot500/imagereward/stratified_sensitivity.json).
- [LaTeX concept table](pilot500/imagereward/results_table.tex), [construction table](pilot500/results_overview.tex), [proposed application subsection](pilot500/human_preference_application.tex).
- [Machine-readable execution summary](continuation_summary.json), [full artifact paths and hashes](pilot500/imagereward/artifact_index.json), [unavailable cross-dataset comparison](pilot500/comparison/primary/comparison_summary.json).

The full 54,030-concept signed tables, explicit image/event/comparison manifests, raw FP32 score matrix, row mapping and validity masks remain under `runs/pilot500/imagereward/`. The repository includes compact review artifacts. The earlier [100-prompt run](HANDOFF.md) is preserved; the larger sample overlaps it and is **not an independent replication**. The sample limit changed, but the scorer, full vocabulary, split algorithm, feature-selection rule, weighting and primary estimand did not.

## What the numbers mean

The joint score is the equal average of the twenty discovery-selected concept scores after applying their discovery-frozen signs. It has no fitted weights. Concordance asks whether this score ranks the human-preferred member above its less-preferred alternative, counting ties as half. Comparisons first share equal weight within their event, events within their normalized prompt, and then prompts share equal weight. This is a within-preference ordering measure, not pooled dataset classification or a calibrated probability of preference.

| Construction | Test prompts | Test comparisons | Joint concordance [95% CI] | Preferred-minus-rejected joint effect [95% CI] |
|---|---:|---:|---|---|
| All strict rank pairs (primary) | 75 | 1,043 | 0.5513 [0.4993, 0.5993] | +0.002243 [+0.000622, +0.003845] |
| Best versus rest | 75 | 516 | 0.4840 [0.4173, 0.5535] | See sensitivity report |
| Best versus worst | 75 | 152 | 0.5967 [0.4900, 0.6967] | See sensitivity report |

Each construction has its own discovery-frozen candidates and uses the same image scores. These are specified sensitivity analyses, not a contest to choose the best test-set result. None of the concordance intervals excludes 0.5. The primary **mean signed effect** interval excludes zero, but mean score margins and the fraction of correctly ordered pairs are different quantities; a positive mean can coexist with uncertain ordering performance.

Nineteen of twenty primary candidates retain their discovery effect direction on test. This is descriptive direction agreement, **not nineteen significant or visually verified concepts**. Returned preferred-side terms include `lip-syncing`, `lip-smacking`, `stare`, `wide-eyed` and `smirk`; rejected-side terms include `tarmac`, `hand-shaped`, `pavement`, `dance-floor` and `asphalt`. All twenty remain in discovery order, including `strabismus`, whose effect reverses. No test-set replacement was made. Individual intervals are not simultaneous confidence guarantees.

An effect of +0.002243 means that the signed joint score averages 0.002243 cosine-similarity units higher for preferred images under the specified weights. It does not measure concept prevalence, causal influence or the gain from adding an attribute to an image. The [visual spot-check note](pilot500/imagereward/visual_spot_check.json) records an example where the `stare` map highlights architectural regions in science-fiction interiors. That assistant inspection is not human ground truth and illustrates why semantic verification is still necessary.

## Actual cohort and execution

Metadata sampling selected **500 distinct prompts / 510 ranking events / 3,119 image IDs**. Eighty-three image IDs failed decoding; whole affected events were excluded, without replacing candidates or replenishing groups. The resulting cohort has **487 prompts / 496 events / 3,036 valid image IDs**. Identical decoded content reduces scoring to **3,029 valid unique image rows**; 576 previously committed rows were reused under exact configuration/content checks and 2,453 were newly inferred.

| Split | Prompts | Events | Strict comparisons | Image IDs |
|---|---:|---:|---:|---:|
| Discovery | 337 | 346 | 5,255 | 2,136 |
| Validation | 75 | 75 | 1,122 | 458 |
| Test | 75 | 75 | 1,043 | 442 |

Official ImageReward split boundaries and prompt/image dependency quarantine remain fixed. No new hash conflict was found after this download. Test has 75 independent prompt/image components. Bootstrap uses 2,000 component resamples, seed 42, retaining prompt weights. Source-wide metadata contains 55,247 images and 8,878 ranking groups; 136,892 strict pairs reproduce the published loader total before split exclusions. Lower rank is preferred. Overall/alignment/fidelity ratings do not redefine labels.

Patch counts are 968–1,024 (median 1,024). The equal-patch-count sensitivity retains 74 of 75 test events / 1,025 comparisons and keeps discovery-selected concepts fixed. The source lacks generator/annotator IDs, so model and annotator confounding cannot be separated. Prompt matching does not make these estimates causal.

| Job | Actual work and result |
|---|---|
| 1863775 | Four-GPU real-image equivalence: 22 images × all 54,030 concepts; maximum score error **0** against the single-GPU cache. Completed 0:0 in 2:33, also running the initial template diagnostic. |
| 1863840 | 500-prompt ImageReward scoring, analysis and four maps. Completed 0:0 in **10:28**. Four workers received 778 unique rows each, including unavailable rows; new inference counts were **608 / 613 / 606 / 626**. |
| 1863913 | Corrected version-2 text-bank template/mask diagnostic, 66 fixed settings across four GPUs. Completed 0:0 in 2:46. |
| 1864083 | Independent four-GPU batch-16 recomputation of all 3,029 valid unique image rows (3,036 image IDs), all 54,030 concepts. Completed 0:0 in 5:13. Full-bank maximum absolute score error **8.3447e-7** versus the primary reference; all 83 invalid-image statuses agree. |

The primary run used FP32, batch one, the official dense-feature API, 1,024 patches and 512-concept chunks. This pilot differs explicitly from the supplied 16,384-patch reproduction setting. All GPU work ran in Slurm allocations. Per-worker audits and measured telemetry are exported under `pilot500/imagereward/scoring/`. Every GPU performed work, but the [whole-job utilization averages](source_inspection/pilot500_gpu_utilization.json) were low (about 1.6–2.1%), reflecting initialization, file/cache handling, small inference batches, CPU analysis and shutdown. Balanced assignment alone is not evidence of efficient GPU saturation. The continuation adds bounded parallel file checks, batched cache commits and two-batch CPU prefetch; the separate batch verification audit records the tested larger-batch path.

The [larger-batch verification](pilot500/batch16_verification/full_bank_equivalence.json) recomputed every available pilot image with batch 16 / text chunks 2,048 in separate caches. Per-worker inference took 26.3–28.2 seconds for 757–758 images, compared with 74.9–80.5 seconds for 606–626 new images in the primary job. This is an observed throughput improvement, not a controlled microbenchmark; cache warmth and I/O also differ. Sampled GPU peaks were 68%, 99%, 85% and 100%; initialization and merging still reduce whole-job averages. Fresh cluster jobs use this validated batch setting on all four GPUs. The scientific tables retain the original scores and frozen candidate lists. [Per-worker timing and reference checks](pilot500/batch16_verification/distributed_execution.json) and [measured utilization](pilot500/batch16_verification/gpu_utilization.json) make the remaining overhead visible.

CPU verification after those changes: **29 tests passed, one GPU-specific test skipped on the login node**, plus Ruff checks passed. The actual GPU equivalence checks above are separate executed evidence, not inferred from skipped tests. Tests cover rank/tie/completeness logic, four-candidate parsing, recurring image roles, hierarchical weights, failures/padding, tie-correct metrics, split dependencies, frozen candidates, cluster bootstrap, cache invalidation/reuse, distributed partition/merge, bounded prefetch, known-sign/null fixtures and offline reports.

## Bank compatibility and remaining data blocker

The unchanged bank contains **54,030 × 768** finite normalized embeddings, not a claimed COCA-20k subset. The original bare-word diagnostic was weak. The expanded diagnostic tests 32 evenly spaced terms against eleven fixed templates, three documented modes and two actual attention-mask settings. The nearest recipe is **box mode, `a photo of a {}.`, without attention mask**: mean cosine **0.965598**, minimum **0.872059**, correct nearest bank row for all 32 terms. This provides stronger compatibility evidence; exact generation provenance and semantic detector validity remain unresolved. Explicit mask handling matters: the tokenizer's default omits `attention_mask`, and the masked settings gave very different embeddings. Version 2 corrects that diagnostic distinction. No embeddings or production scores were replaced.

The original linked Pick-a-Pic metadata has 25,355 four-candidate events, 109,356 image rows, and **13,085** strictly eligible prompt-matched events. Invalid/none selections and mismatched candidates remain excluded. All 119 smoke image URLs returned 403. Further recovery inspected two original-author first-day archives: 959 UIDs overlap the original image table, but **zero** eligible events have all four candidates. A separate accessible pairwise UID index had no overlap; exact-UID queries to another public archive timed out. Timeouts are not proof of absence. See [first-day coverage](source_inspection/pickapic_firstday_coverage.json), [recovery schemas/revisions](source_inspection/pickapic_recovery_schemas.json), [UID-index check](source_inspection/pickapic_pairwise_uid_overlap.json), and [viewer query outcomes](source_inspection/pickapic_viewer_availability.json). No pairwise labels or partial-event fallback were used. Original-UID-verifiable image bytes are still needed for empirical Pick-a-Pic/transfer results.

## Reproduce or extend

From the repository root, reproduce the primary 500-prompt configuration and reuse the earlier compatible rows:

```bash
export HF_HOME="$PROJECT/avrecum/.hf"
.venv/bin/python -m preference_diff prepare --dataset imagereward \
  --output runs/pilot500/imagereward --group-limit 500 --seed 42 --resume --download-workers 4
mkdir -p reports/logs
PREFERENCE_IMAGE_BATCH=1 PREFERENCE_TEXT_CHUNK=512 \
  sbatch --time=01:00:00 scripts/imagereward_pilot.sbatch \
  runs/pilot500/imagereward runs/pilot100/imagereward/scores
```

Rerun the bounded template diagnostic with `sbatch scripts/bank_diagnostics.sbatch`. Recompute the whole 500-prompt image cache with the larger batch and compare full-bank scores using `sbatch scripts/batch_benchmark.sbatch`. Compatible completed caches resume without rescoring valid rows; incompatible numerical configurations require separate score roots.

The full experiment commands are in [README — Full experiment](../README.md#full-experiment). Use `--group-limit 0`, a fresh sufficiently large run directory, and original-UID images for Pick-a-Pic. Metadata/downloads run before requesting a GPU node. Both datasets must use identical model/bank/preprocessing settings. The scheduler requests all four GPUs and starts four workers; no additional training or paid APIs are involved. Full-population results, human semantic annotations and cross-dataset replication have not been obtained here.
