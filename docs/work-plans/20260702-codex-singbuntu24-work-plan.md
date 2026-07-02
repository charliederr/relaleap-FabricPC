# 2026-07-02 Codex Singbuntu24 Work Plan

## Objective

Port the RelaLeap Phase 0 harness into `relaleap-FabricPC` so it runs against
the local `FabricPC` checkout without modifying `FabricPC`.

## Current Plan

1. Create an installable `relaleap_fabricpc` package in `relaleap-FabricPC`.
2. Add downstream FabricPC-compatible residual-column node code in this repo.
3. Port the Phase 0 smoke model and invariants to JAX/Optax.
4. Port the config-driven run command, comparison command, and artifact checker.
5. Add Phase 0 configs equivalent to the source `relaleap` smoke configs.
6. Add focused tests for imports, FabricPC resolution, residual invariants,
   runner artifacts, and comparison verdicts.
7. Update `README.md` with setup, run, compare, and verification instructions.
8. Run tests and a smoke command using `/home/ni/repos/fpc/py3/bin/python`.
9. Fix issues found by verification, keeping all fixes inside `relaleap-FabricPC`.

## Boundaries

- Do not edit `/home/ni/repos/fpc/FabricPC`.
- Do not edit `/home/ni/repos/fpc/relaleap`.
- Do not install Python packages. If a dependency is missing, ask the user.
- Keep custom JAX/FabricPC support code inside `relaleap-FabricPC`.

## Intended First Gate

The first usable gate is:

```bash
/home/ni/repos/fpc/py3/bin/python -m relaleap_fabricpc.experiments.compare \
  --out results/comparisons/phase0
```

The comparison should write `summary.json`, `metrics.csv`, and `notes.md`, and
the verdict should pass when all Phase 0 model and artifact invariants pass.

## Current Status

- Package, configs, command modules, README, and focused tests have been added.
- Verification passed for focused tests, the default Phase 0 comparison, and artifact checking.
- Source-compatible comparison artifacts, baseline schema v3, child artifact checks, mirrored configs, and a FabricPC baseline have been added.
- Remaining work is to continue porting beyond Phase 0 or deepen FabricPC integration as requested.

## Follow-up Adjustment

- Add a strict GPU wrapper script for user-run artifact generation.
- Leave `results/` unignored so GPU artifacts can be committed when useful.

## Completed GPU Confirmation Note

- The user confirmed JAX selects `gpu` and sees `CudaDevice(id=0)` in the local environment.
- No further backend-probe work is planned; continue with RelaLeap-to-FabricPC porting and artifact verification.

## Current Porting Slice

- Keep the Phase 0 default comparison small and passing on CPU.
- Use `baselines/phase0_fabricpc_comparison.json` as the checked FabricPC baseline for artifact drift checks.
- Treat the mirrored 92 source configs as available run surfaces; validate broad config compatibility with lightweight tests rather than running every large config by default.
