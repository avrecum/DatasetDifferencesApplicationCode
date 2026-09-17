# Working in this repository

- On this JUPITER allocation, use all four GPUs on every allocated node for GPU work. Request `--gres=gpu:4` and launch one worker per GPU through the supplied `torchrun` scheduler scripts. Record per-GPU work counts and utilization. Run downloads and CPU-only checks without requesting a GPU node; never start GPU work on a login node.
- Preserve the original human annotations and the discovery/test boundary. Do not substitute the pairwise Pick-a-Pic release for the specified four-candidate rankings. Keep unavailable results explicit.
- Keep source revisions, bank ordering, preprocessing and score fingerprints fixed within an analysis. Do not replace the supplied bank or silently change patch budgets. Numerical optimization needs a real-image equivalence check.
- Store large files under the configured HF cache or ignored run directories. Keep compact review artifacts and exact reproduction commands in the repository.
