# char_smoke_hep

RelaLeap FabricPC Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- JAX backend: `cpu`
- CUDA available: `False`
- FabricPC file: `/home/ni/repos/fpc/FabricPC/fabricpc/__init__.py`
- Final smoke loss: `3.92986178`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `False`
- Base loss: `4.013883590698242`
- Residual training steps: `10`
- Residual final loss: `3.9298617839813232`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.9298617839813232`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.910778284072876`, max ordinary-logit delta `0.06316345930099487`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.892465353012085`, max ordinary-logit delta `0.12632635235786438`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.8581383228302`, max ordinary-logit delta `0.252652645111084`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
