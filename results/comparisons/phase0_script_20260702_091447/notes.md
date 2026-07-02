# RelaLeap Objective Comparison

Command-driven comparison of Phase 0 residual objectives.

- Status: `ok`
- Verdict: `pass`
- Phase 0 invariants: `12` checked, passed `True`
- Artifact invariants: `9` checked, passed `True`
- HEP alpha acceptance: `accepted`
- Loss scale note: Residual objectives may use different loss scales; compare each trajectory against its own initial loss.

## Runs

| Experiment | Objective | Pinned | Stress | Status | Initial loss | Final loss | Delta | Ratio | HEP clip | Support change | Pinned-vs-repicked |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| char_smoke | supervised_ce | False | False | ok | 4.01388359 | 3.92986178 | -0.08402181 | 0.97906720 |  | 0.00000000 | 0.00000000 |
| char_smoke_pc | pc_logit_mse | False | False | ok | 0.02952804 | 0.02937417 | -0.00015387 | 0.99478902 |  | 0.00000000 | 0.00000000 |
| char_smoke_hep | supervised_ce | False | False | ok | 4.01388359 | 3.92986178 | -0.08402181 | 0.97906720 |  | 0.00000000 | 0.00000000 |

## Artifacts

- `char_smoke`: `results/comparisons/phase0_script_20260702_091447/runs/char_smoke`
- `char_smoke_pc`: `results/comparisons/phase0_script_20260702_091447/runs/char_smoke_pc`
- `char_smoke_hep`: `results/comparisons/phase0_script_20260702_091447/runs/char_smoke_hep`

## HEP Alpha Sweeps

- `char_smoke_hep`: alpha 0.0: loss 3.92986178, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.91077828, delta 0.06316346, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.89246535, delta 0.12632635, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.85813832, delta 0.25265265, support-change 0.00000000, pinned-vs-repicked 0.00000000

## Verdict

- Best HEP alpha by loss: `1.0` in `char_smoke_hep` with loss `3.85813832` and ordinary-logit delta `0.25265265`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `0.25` in `char_smoke_hep` with loss improvement `0.01908350` and ordinary-logit delta `0.06316346`
