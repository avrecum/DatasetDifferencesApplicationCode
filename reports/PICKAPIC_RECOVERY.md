# Original Pick-a-Pic recovery and evaluation

The original four-candidate release is now evaluated. An archive published by the original author contains verifiable copies of 6,568 original image UIDs, covering 573 complete eligible ranking events across 330 prompts. All 2,053 images required by those events were recovered, decoded and scored successfully. This is an availability-selected early-period cohort; it need not represent the full release.

The Pick-a-Pic discovery-selected combined score achieved **49.2% held-out paired concordance (95% interval 37.5–61.1%)**. These data do not establish above-chance discrimination for that score. A separately frozen ImageReward best-versus-rest list transferred better: **64.6% (53.3–75.8%)** on the same Pick-a-Pic test cohort. Both results are retained, together with the other transfer directions and construction sensitivities. We did not replace failed Pick-a-Pic candidates with features selected on its test set.

## Source recovery and identity checks

The successful source is [yuvalkirstain/images_first_day](https://huggingface.co/datasets/yuvalkirstain/images_first_day/tree/4f1114880da57a4d7c53d53189d5fa7f19c57527), pinned at `4f1114880da57a4d7c53d53189d5fa7f19c57527`. Its name does not contain “PickaPic”; enumerating the author's datasets found it. Eleven parquet shards contain 6,916 rows with embedded images and original image URLs, prompts, model IDs and generation metadata. The [coverage audit](recovery_search/images_first_day_coverage.json) records schemas, UID locations and revisions; the [extraction audit](recovery_search/image_extraction.json) records archive and recovered-image hashes.

The original `PickaPic-images` and `PickaPic-rankings` repositories remain the sole sources of image identity metadata and human labels. No preference labels from a later pairwise release are used. We match original URL UUIDs, normalized generation prompts and available negative-prompt, model, numeric image-ID, seed and index fields. Extracted bytes must decode, and subsequent loads must match their recorded SHA256. This establishes identifier/metadata agreement with the original release; inaccessible original AWS bytes cannot be independently compared byte-for-byte. The original AWS URLs still return 403 in this environment.

The containing parquet files require **5,024,119,623 bytes** in HF_HOME; only the **2,053 required image files / 1,439,361,558 bytes** were extracted. There were no extraction, decoding or inference failures and no new cross-split hash conflicts. All events retain four candidates, one selected winner and three winner-versus-rest comparisons. No missing candidate or winner was replaced.

Search evidence also explains why other copies were not used:

| Archive inspected | Coverage of original image identities |
|---|---|
| `yuvalkirstain/images_first_day`, all 11 shards | 6,568 original UIDs; 573 complete eligible original events |
| `yuvalkirstain/PickaPic-ft-ranked`, pinned current/historical versions | 3,748 original UIDs each, but no complete eligible original events |
| `pickapic-anonymous/pickapic_v1`, all 411 shards | 656,480 distinct observed UIDs; zero overlap with original image UIDs |
| `liuhuohuo2/pick-a-pic-v2`, bounded 16-shard scan | Zero overlap in the inspected shards; uninspected shards were not ruled out |

Pinned revisions and scan scope are in [recovery_search](recovery_search/). `combined_original_archive_coverage.json` is an intermediate audit of the unsuccessful earlier archive combination, before the successful `images_first_day` scan. It is not the final coverage result.

## Cohort and execution

The original metadata contains 25,355 ranking events and 109,356 images; 13,085 events satisfy the strict metadata cohort before image availability. The recovered 573 events date from January 9–10, 2023. `--available-image-ids` explicitly restricts eligible events to complete archive coverage. Full-metadata prompt/image dependency components and their fixed split assignments are computed **before** this restriction. The availability declaration participates in the prepare/resume fingerprint. All 330 available prompts fit the requested 500-prompt cap.

| Split | Prompts | Events | Comparisons | Dependency components | Unique images |
|---|---:|---:|---:|---:|---:|
| Discovery | 262 | 482 | 1,446 | 170 | 1,714 |
| Validation | 35 | 47 | 141 | 32 | 177 |
| Test | 33 | 44 | 132 | 31 | 162 |

The archive restriction changes the realized split proportions; it does not trigger a new split chosen for these results. Repeated image identities and independent judgments remain in the manifests. Statistical weights average comparisons within events, events within prompts, then prompts equally.

Slurm job **1866021** completed with exit **0:0** in **6:16**, using all four GPUs on `jpbo-025-42`. Worker image counts were **514 / 513 / 513 / 513**, with no failed rows. Each worker checked the adapted batched dense-feature path against the official single-image FP32 API on four real images and 128 concepts; the maximum observed error was **1.1921e-7**. The previous full-bank ImageReward equivalence test covers the same batching implementation. The run used FP32, batch 16, 2,048-concept chunks, the unchanged **54,030 × 768** bank and **1,024 valid patches** per image. This is the pilot setting, not the supplied 16,384-patch reproduction preset.

All four workers performed inference in 20.3–22.6 seconds each after initialization. [Execution records](pickapic_recovered/pickapic/scoring/execution_summary.json) and [measured GPU utilization](pickapic_recovered/pickapic/scoring/gpu_utilization.json) are retained. Whole-job utilization was low, averaging 0.6–1.9%, because initialization, cache work, CPU analysis and report generation dominate this small workload. The telemetry shows work on every GPU, not continuous saturation.

## What was found

The preferred-side discovery list is dominated by face-related words: *lip, lip-syncing, lip-smacking, nose, mouth, nostril, mouthpart, nosebleed, lipstick,* and *mouth-to-mouth*. Every one has a negative held-out effect. The rejected-side list—*pavement, kickstand, driveway, sidewalk, doormat, carpeting, carpet, carpeted, bollard,* and *paved*—retains negative held-out effects, but all ten individual intervals include zero. Thus **10/20 directions persist**, without evidence that the selected descriptions reliably explain preference in this cohort. These redundant terms also show a limitation of unmodified top-k selection.

The signed ensemble averages the 20 concepts after multiplying each by its discovery direction. It has held-out mean preferred-minus-rejected score **+0.000882**, interval **[−0.003292, +0.004889]**, and concordance **0.492424**. Concordance evaluates a selected image against one non-selected alternative using the same hierarchical weights, with ties worth half. Its equal-ordering reference is **50%**, not the 25% baseline for choosing one winner among four. The effect is in cosine-similarity units, not concept-presence percentage points. This ensemble is an additional application diagnostic; it is not the draft's MLLM discrimination metric.

The primary evidence is in the [frozen concept table](pickapic_recovered/pickapic/top_concepts.json), [concept plot](pickapic_recovered/pickapic/concept_effects.png) and [generated report](pickapic_recovered/pickapic/report.md). [Paired examples](pickapic_recovered/pickapic/gallery.html) include supportive extremes, counterexamples and seeded random audit pairs from distinct prompts. [Four patch maps](pickapic_recovered/pickapic/heatmaps/index.json) use the actual processor geometry and identify maximal patches. The [annotation template](pickapic_recovered/pickapic/annotation_template.csv) remains blank: these visual interpretations have **not been verified by humans**.

## Cross-dataset comparison

Comparison uses the already verified batch-16 ImageReward cache and the new Pick-a-Pic cache, with identical model/bank revisions, preprocessing, dtype, patch budget and chunk/batch settings. ImageReward's candidate identities, directions and ordering match its original single-image primary run for all three constructions; [configuration audit](recovery_search/imagereward_comparison_configuration.json). The original ImageReward reports remain unchanged. No shared normalized prompt or known image hash was found between these cohorts, and no recipient test component was removed by the transfer overlap audit.

| Concepts selected on | Held-out evaluation | Paired concordance | 95% component interval |
|---|---|---:|---:|
| ImageReward all-strict-pairs | ImageReward all-strict-pairs | 55.1% | 49.9–59.9% |
| Pick-a-Pic winner-versus-rest | Pick-a-Pic winner-versus-rest | 49.2% | 37.5–61.1% |
| ImageReward all-strict-pairs | Pick-a-Pic winner-versus-rest | 55.8% | 43.5–67.7% |
| Pick-a-Pic winner-versus-rest | ImageReward all-strict-pairs | 51.5% | 46.6–56.1% |
| ImageReward best-versus-rest | Pick-a-Pic winner-versus-rest | 64.6% | 53.3–75.8% |
| Pick-a-Pic winner-versus-rest | ImageReward best-versus-rest | 54.5% | 47.4–61.5% |

All donor lists and directions are frozen using donor discovery data. The stronger best-versus-rest transfer is a secondary finding, not permission to select the best-performing protocol on recipient test data. ImageReward best-versus-rest's own held-out concordance is 48.4% [41.7–55.4%]. These per-analysis intervals have no simultaneous guarantee across the comparisons or concept families. The [combined plot](pickapic_recovered/combined_concordance.png) shows every row above with its interval.

Across all 54,030 concepts, held-out signed effect rank correlation is **−0.0442** for ImageReward all-strict-pairs versus Pick-a-Pic; it is **−0.0345** using ImageReward best-versus-rest. Discovery correlation is 0.1864 and 0.0938 respectively. The primary discovery lists share two preferred terms (*lip-syncing, lip-smacking*) and one rejected term (*pavement*); best-versus-rest top-ten lists have no same-side overlap. Shared discovery terms therefore do not establish held-out replication. See [primary comparison](pickapic_recovered/comparison/primary/comparison_summary.json), [effect scatter](pickapic_recovered/comparison/primary/effect_scatter.png), [transfer](pickapic_recovered/comparison/primary/frozen_list_transfer.json), and [best-versus-rest transfer](pickapic_recovered/comparison/best_vs_rest/frozen_list_transfer.json).

## Confounders and uncertainty

The cohort has two known generators: 1,050 Stable Diffusion 2.1 images and 1,003 ProtoGen X3.4 images. All images are square, with 1,024 valid scored patches; unequal patch opportunities do not explain differences within this recovered cohort. Original widths range from 512 to 768 pixels. There are 115 annotators, 63 with repeated events. Joining full-metadata prompt/image components that share an annotator leaves just **nine test clusters**. For Pick-a-Pic's own ensemble, this conservative sensitivity gives 49.2% [42.6–72.2%]. Nine-cluster uncertainty is unstable and does not address every possible dependence.

Pair-restricted sensitivities use only complete valid events, retain frozen features, and renormalize weights after restricting to same- or cross-model comparisons. Prompts/events without an eligible pair leave that conditional estimand. Pick-a-Pic's own ensemble gives **48.2% [35.9–59.7%]** on 31 prompts with same-model pairs, and **66.7% [33.3–100.0%]** on ten prompts with cross-model pairs. The latter is too imprecise to establish a pattern.

For the ImageReward best-versus-rest transfer, same-model pairs give **64.2% [52.3–75.6%]**. Splitting them by generator gives **47.1% [32.3–63.0%]** for Stable Diffusion (17 prompts) and **76.0% [59.4–89.6%]** for ProtoGen (16 prompts). The annotator-component transfer interval is **60.8–85.2%**, still based on only nine clusters. These subgroup estimates use different prompt/comparison sets, and are not a direct test of generator effects or a causal explanation. Generator, user, prompt and candidate-pool differences remain plausible contributors. Detailed conditional counts and all concept metrics are retained in the sensitivity/transfer JSONs.

The unchanged bank's exact generation recipe remains unknown. The fixed template diagnostic gives mean cosine 0.9656 for box mode with `a photo of a {}.` and no attention mask, with all 32 probes nearest to their own bank rows. This supports compatibility, but not calibrated detection of every word or phrase.

## Code, checks and reproduction

New recovery scripts index pinned parquet UID/URL columns and extract verified image bytes. Preparation supports an explicit complete-event archive cohort; scoring verifies retrieval hashes. Frozen generator-pair, annotator and transfer diagnostics are added without changing primary candidates. Component hashing now computes one hash per component rather than repeatedly hashing the same member list. Reporting distinguishes the signed ensemble from the paper's MLLM metric.

**36 CPU tests passed; one GPU-only test was skipped on the login node. Ruff passed.** The real four-GPU checks above ran separately and passed. New regression coverage includes archive UID/metadata checks, coverage-cache resume/revision changes, changed recovered bytes, dependency-preserving cohort restriction, conditional pair-weight normalization, frozen sensitivity directions and transfer counts. Existing offline end-to-end and null-fixture tests also passed.

The [README recovery commands](../README.md#recover-and-evaluate-the-original-pick-a-pic-images) reproduce metadata preparation, extraction and the four-GPU run. The finished cohort and matching ImageReward cache can be compared again with:

```bash
.venv/bin/python -m preference_diff compare \
  --left runs/batch16_verification/imagereward \
  --right runs/pickapic_recovered/pickapic \
  --output runs/pickapic_recovered/comparison
.venv/bin/python scripts/report_recovered_comparison.py \
  --imagereward runs/batch16_verification/imagereward \
  --pickapic runs/pickapic_recovered/pickapic \
  --comparison runs/pickapic_recovered/comparison \
  --output reports/pickapic_recovered
```

For a new matching ImageReward pilot, prepare a fresh `--group-limit 500` run and use `sbatch scripts/dataset_pilot.sbatch RUN_ROOT imagereward`; its default batch/chunk configuration matches this Pick-a-Pic run. Full original-release execution commands remain in [README](../README.md#full-experiment), conditional on recovering the remaining original Pick-a-Pic images. Running with `--group-limit 0` on this archive cannot add prompts beyond the 330 already evaluated.

The [artifact index](pickapic_recovered/pickapic/artifact_index.json) locates raw manifests, the score matrix/validity mask, and all-concept signed CSVs under `runs/pickapic_recovered/`; they remain outside Git. Compact results, galleries, provenance and LaTeX tables are committed. The [combined paper subsection](pickapic_recovered/application_subsection.tex) and [results table](pickapic_recovered/combined_results_table.tex) contain observed results only. Remaining limitations are incomplete coverage of the original release, few held-out groups, unresolved exact bank provenance and absent human verification of concept presence.
