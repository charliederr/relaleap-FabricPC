# 2026-07-02 Codex Singbuntu24 Work Log

## Log

- Read the relevant project instructions and confirmed the working boundary:
  all implementation belongs in `relaleap-FabricPC`, while `FabricPC` remains
  read-only unless explicit permission is granted later.
- Inspected `relaleap`, `columnarCL-fabricPC-experiments`, and `FabricPC`.
  Identified the Phase 0 port surface as the smoke model, run command,
  comparison command, artifact checker, and checked configs.
- Confirmed `/home/ni/repos/fpc/py3/bin/python` resolves `fabricpc` from
  `/home/ni/repos/fpc/FabricPC/fabricpc/__init__.py`.
- Started the downstream Phase 0 port by adding package, config, test, and
  documentation files under `relaleap-FabricPC`.
- Added `relaleap_fabricpc.fabricpc_nodes.ResidualColumnsNode`, a local downstream FabricPC `NodeBase` subclass for sparse residual columns.
- Added a JAX/Optax Phase 0 smoke implementation with frozen base parameters, residual-only updates, HEP settling, support audits, and invariant reporting.
- Added `run`, `compare`, and `check_artifacts` command modules.
- Replaced the placeholder README with environment, run, comparison, artifact check, and test instructions.
- Added focused tests for imports, FabricPC resolution, residual behavior, runner artifacts, comparison output, and baseline writing.
- First pytest run selected the CUDA backend and failed during JAX startup because cuDNN could not initialize; added local CPU-default JAX setup with user override through `JAX_PLATFORMS`.
- Reran the focused pytest suite after the JAX CPU-default fix; all 10 tests passed.
- Ran the default Phase 0 comparison command; it completed with status `ok`, verdict `pass`, 12/12 model invariants passing, 9/9 artifact invariants passing, and accepted HEP alpha `0.25`.
- Ran the artifact checker against `results/comparisons/phase0`; it reported status `pass`.
- Added `.gitignore` entries for Python caches, generated comparison results, and generated baselines.
- Added `scripts/run_phase0_gpu.sh`, a short wrapper that sets `JAX_PLATFORMS=gpu`, runs the default Phase 0 comparison, and checks artifacts.
- Removed the `.gitignore` file added earlier so generated `results/` artifacts can be tracked if desired.
- Updated the README with the GPU wrapper invocation and optional output-directory argument.
- Updated `scripts/run_phase0_gpu.sh` to unset `JAX_PLATFORMS` and set `RELALEAP_FABRICPC_JAX_AUTO=1`, so JAX chooses the backend automatically instead of preserving a stale platform value.

## GPU Sanity Burn Script

- Added `scripts/jax_gpu_burn.py` and `scripts/run_jax_gpu_burn.sh` for a direct 60-second JAX matmul workload that does not set `JAX_PLATFORMS` or import the Phase 0 package setup.
- The script prints JAX backend/device selection and keeps a dense jitted matmul loop active so GPU use can be checked from a separate `nvidia-smi` process.

## Phase 0 Artifact Parity Slice

- Ported the source RelaLeap comparison artifact contract into `relaleap_fabricpc.experiments.compare`: baseline schema version 3, stable baseline mismatch fields, fixed top-level comparison metrics schema, and richer notes with HEP/support sections.
- Ported the source artifact checker behavior into `relaleap_fabricpc.experiments.check_artifacts`, including child run artifact validation and baseline-reference reporting.
- Added focused tests for comparison parity, config inventory compatibility, and the checked FabricPC Phase 0 baseline.
- Mirrored 92 source Phase 0 YAML configs into `configs/` and created `baselines/phase0_fabricpc_comparison.json` from the passing FabricPC comparison.
- Verified `/home/ni/repos/fpc/py3/bin/python -m pytest` passes with 19 tests, then reran the default comparison and artifact checker successfully.
- JAX GPU visibility was confirmed by the user (`gpu`, `CudaDevice(id=0)`); stopped backend-probe work and returned to porting/artifact parity.
- Added `scripts/run_phase0_experiment.sh` for one-off config runs and `scripts/run_phase0_comparison.sh` for default or custom comparison experiments.
