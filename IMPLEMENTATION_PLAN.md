# Human-preference application implementation

1. Inspect the supplied manuscript/scripts (pasted in the assignment), pinned upstream dataset metadata, model code, and complete text bank. This initially empty workspace has no unrelated workflow to modify.
2. Implement explicit images, events and comparisons; complete-group adapters; conservative prompt matching; identity/content dependency components; official/grouped splits; deterministic bounded sampling.
3. Implement safe normalized bank loading, masked dense scoring, official API reference checks, stable on-disk scores and resumable fingerprints. GPU work runs through Slurm.
4. Test prompt/event/comparison weighting, signed effects, concordance, ties, frozen discovery, component bootstrap, failure handling and leakage using offline fixtures before any real scoring.
5. Produce held-out tables, sensitivities, transfer comparisons, reproducible paired galleries, patch similarity maps, annotation templates and generated LaTeX/report artifacts. Run a bounded real-data pilot when data/resources permit; report blockers and unavailable results explicitly.

Primary estimand: average preferred-minus-rejected similarity over comparisons, then events, then distinct normalized prompts. No reward-model training, generated images, concept prevalence claims, or causal claims.
