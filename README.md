# relaleap-FabricPC

This repository contains the RelaLeap Phase 0 harness ported to JAX and the
local FabricPC checkout.

The source project is `/home/ni/repos/fpc/relaleap`. This port keeps its own
JAX/FabricPC support code under `relaleap-FabricPC/relaleap_fabricpc` and does
not modify `/home/ni/repos/fpc/FabricPC`.

## Environment

Use the shared virtual environment from the parent workspace:

```bash
cd /home/ni/repos/fpc/relaleap-FabricPC
/home/ni/repos/fpc/py3/bin/python -c "import fabricpc; print(fabricpc.__file__)"
```

The printed path should be:

```text
/home/ni/repos/fpc/FabricPC/fabricpc/__init__.py
```

The local commands default to the CPU JAX backend because this workspace can
have CUDA wheels even when the active driver cannot initialize cuDNN. To force a
different backend, set `JAX_PLATFORMS` before launching Python.

No package installation is required if you run commands from this repository
root. For editable development, users may install this package themselves with:

```bash
/home/ni/repos/fpc/py3/bin/python -m pip install -e .
```

## Run One Phase 0 Config

```bash
cd /home/ni/repos/fpc/relaleap-FabricPC
/home/ni/repos/fpc/py3/bin/python -m relaleap_fabricpc.experiments.run \
  --config configs/char_smoke.yaml \
  --out results/runs/char_smoke
```

Each run writes:

- `summary.json`
- `metrics.csv`
- `notes.md`
- `config.yaml`

The summary includes the Phase 0 model invariants:

- zero-initialized residual columns preserve base logits;
- frozen base parameters remain unchanged during residual training;
- HEP alpha `0.0` matches ordinary residual inference;
- residual parameters update during residual training.

For a shorter experiment wrapper, run:

```bash
scripts/run_phase0_experiment.sh
```

By default this runs `configs/char_smoke_hep_support_stress_clipped.yaml` and
writes under `results/runs/`. Pass a config and output directory explicitly:

```bash
scripts/run_phase0_experiment.sh \
  configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml \
  results/runs/token_contextual_router_trial
```

For a checked comparison wrapper, run:

```bash
scripts/run_phase0_comparison.sh
```

By default this runs the default Phase 0 comparison and checks it against
`baselines/phase0_fabricpc_comparison.json`. Pass an output directory first,
then two or more configs to compare a custom set:

```bash
scripts/run_phase0_comparison.sh \
  results/comparisons/support_stress_trial \
  configs/char_smoke_hep_support_stress_clipped.yaml \
  configs/char_smoke_hep_support_stress_temporal_clipped.yaml
```

## Run The Phase 0 Comparison On GPU

Use the short GPU wrapper when you want to run the fuller check yourself on a
machine where JAX can initialize CUDA:

```bash
cd /home/ni/repos/fpc/relaleap-FabricPC
scripts/run_phase0_gpu.sh
```

Pass an output directory as the first argument if needed:

```bash
scripts/run_phase0_gpu.sh results/comparisons/phase0_gpu_my_run
```

The script unsets `JAX_PLATFORMS`, disables this repo's CPU default, lets JAX choose the backend automatically, runs the default Phase 0 comparison, and then checks the generated artifacts.

## Run The Phase 0 Comparison

```bash
cd /home/ni/repos/fpc/relaleap-FabricPC
/home/ni/repos/fpc/py3/bin/python -m relaleap_fabricpc.experiments.compare \
  --out results/comparisons/phase0
```

By default this runs:

- `configs/char_smoke.yaml`
- `configs/char_smoke_pc.yaml`
- `configs/char_smoke_hep.yaml`

The comparison writes `summary.json`, `metrics.csv`, and `notes.md` under the
comparison directory. The verdict passes only when all child runs are `ok`, all
Phase 0 model invariants pass, and all required artifacts exist. The top-level
comparison artifacts now follow the source RelaLeap Phase 0 schema, including a
fixed comparison metrics CSV, HEP alpha acceptance fields, and child-run
artifact checks.

The source Phase 0 config inventory is mirrored under `configs/`; the default
comparison stays small, but larger char/token, support-stress, contextual-router,
and objective-gate configs are available for explicit `--config` runs.

A checked FabricPC baseline is available at:

```text
baselines/phase0_fabricpc_comparison.json
```

To refresh that baseline from a known-good comparison:

```bash
/home/ni/repos/fpc/py3/bin/python -m relaleap_fabricpc.experiments.compare \
  --out results/comparisons/phase0 \
  --baseline-out baselines/phase0_fabricpc_comparison.json
```

To compare a run against that baseline:

```bash
/home/ni/repos/fpc/py3/bin/python -m relaleap_fabricpc.experiments.compare \
  --out results/comparisons/phase0_check \
  --baseline-reference baselines/phase0_fabricpc_comparison.json
```

## Check Existing Artifacts

```bash
/home/ni/repos/fpc/py3/bin/python -m relaleap_fabricpc.experiments.check_artifacts \
  --comparison-dir results/comparisons/phase0
```

With the checked FabricPC baseline:

```bash
/home/ni/repos/fpc/py3/bin/python -m relaleap_fabricpc.experiments.check_artifacts \
  --comparison-dir results/comparisons/phase0 \
  --baseline-reference baselines/phase0_fabricpc_comparison.json
```

## Tests

```bash
cd /home/ni/repos/fpc/relaleap-FabricPC
/home/ni/repos/fpc/py3/bin/python -m pytest
```

## Implementation Notes

- `relaleap_fabricpc.fabricpc_nodes.ResidualColumnsNode` is a downstream
  FabricPC `NodeBase` subclass for the sparse residual adapter.
- `relaleap_fabricpc.smoke` implements the Phase 0 run in JAX/Optax and uses
  the same residual-column parameterization as the FabricPC node.
- The tiny base model is frozen during residual training. Only residual-column
  parameters are updated.
- Numeric values are not expected to match the PyTorch source baseline exactly.
  The preserved contract is the command surface, artifact schema, and invariant
  semantics.
