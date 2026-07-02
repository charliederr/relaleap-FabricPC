"""Minimal config-driven Phase 0 experiment runner."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

from relaleap_fabricpc.jax_setup import configure_jax

configure_jax()

import jax

from relaleap_fabricpc.smoke import run_phase0_smoke


def _load_backend_info() -> dict[str, Any]:
    try:
        import fabricpc
    except Exception as exc:  # pragma: no cover - environment dependent
        fabricpc_file = None
        fabricpc_error = repr(exc)
    else:
        fabricpc_file = fabricpc.__file__
        fabricpc_error = None

    backend = jax.default_backend()
    return {
        "jax_available": True,
        "jax_version": jax.__version__,
        "backend": backend,
        "device": backend,
        "cuda_available": backend == "gpu",
        "jax_devices": [str(device) for device in jax.devices()],
        "fabricpc_file": fabricpc_file,
        "fabricpc_error": fabricpc_error,
    }


def _read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ModuleNotFoundError:
        return _read_simple_yaml(text)


def _read_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, sep, value = raw_line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def run(config_path: Path, out_dir: Path) -> dict[str, Any]:
    config = _read_config(config_path)
    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 1))
    max_steps = int(run_cfg.get("max_steps", 10))
    experiment_id = str(run_cfg.get("experiment_id", "smoke"))

    out_dir.mkdir(parents=True, exist_ok=True)
    backend_info = _load_backend_info()
    start = time.time()

    phase0: dict[str, Any] | None = None
    rows: list[dict[str, Any]]
    status = "ok"
    error: str | None = None
    try:
        phase0_result = run_phase0_smoke(config)
        phase0 = phase0_result.to_summary()
        rows = _build_phase0_rows(
            phase0_result.to_metric_rows(),
            max_steps=max_steps,
            seed=seed,
            experiment_id=experiment_id,
            device=backend_info["device"],
        )
        if not all(phase0_result.invariants.values()):
            status = "failed"
            error = "Phase 0 invariant failure"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        rows = _build_failed_rows(
            max_steps=max_steps,
            seed=seed,
            experiment_id=experiment_id,
            device=backend_info["device"],
            error=error,
        )

    for row in rows:
        row["status"] = status

    _write_metrics(out_dir / "metrics.csv", rows)
    final_smoke_loss = _final_loss(rows)
    summary = {
        "experiment_id": experiment_id,
        "seed": seed,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "status": status,
        "error": error,
        "final_smoke_loss": final_smoke_loss,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "phase0": phase0,
        **backend_info,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", experiment_id, summary)

    required = config.get("outputs", {})
    artifact_invariants = {
        "summary_json": not required.get("require_summary_json", True)
        or (out_dir / "summary.json").is_file(),
        "metrics_csv": not required.get("require_metrics_csv", True)
        or (out_dir / "metrics.csv").is_file(),
        "notes_md": not required.get("require_notes_md", True)
        or (out_dir / "notes.md").is_file(),
    }
    summary["artifact_invariants"] = artifact_invariants
    if not all(artifact_invariants.values()):
        summary["status"] = "failed"
        summary["error"] = summary["error"] or "Required artifact missing"
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_phase0_rows(
    metric_rows: list[dict[str, Any]],
    *,
    max_steps: int,
    seed: int,
    experiment_id: str,
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in metric_rows:
        if int(source["step"]) > max_steps:
            continue
        rows.append(
            {
                "step": source["step"],
                "seed": seed,
                "experiment_id": experiment_id,
                "phase": source["phase"],
                "residual_objective": source["residual_objective"],
                "base_loss": _format_metric(source["base_loss"]),
                "residual_loss": _format_metric(source["residual_loss"]),
                "zero_init_loss": _format_metric(source["zero_init_loss"]),
                "residual_parameter_delta": _format_metric(
                    source["residual_parameter_delta"]
                ),
                "max_zero_init_logit_delta": _format_metric(
                    source["max_zero_init_logit_delta"]
                ),
                "max_hep_alpha0_logit_delta": _format_metric(
                    source["max_hep_alpha0_logit_delta"]
                ),
                "hep_alpha": _format_optional_metric(source.get("hep_alpha")),
                "hep_loss": _format_optional_metric(source.get("hep_loss")),
                "max_hep_logit_delta_from_ordinary": _format_optional_metric(
                    source.get("max_hep_logit_delta_from_ordinary")
                ),
                "hep_support_change_fraction": _format_optional_metric(
                    source.get("hep_support_change_fraction")
                ),
                "hep_pinned_vs_repicked_logit_delta": _format_optional_metric(
                    source.get("hep_pinned_vs_repicked_logit_delta")
                ),
                "device": device,
                "error": "",
            }
        )
    return rows


def _build_failed_rows(
    *,
    max_steps: int,
    seed: int,
    experiment_id: str,
    device: str,
    error: str,
) -> list[dict[str, Any]]:
    return [
        {
            "step": 0,
            "seed": seed,
            "experiment_id": experiment_id,
            "phase": "failed",
            "residual_objective": "",
            "base_loss": "",
            "residual_loss": "",
            "zero_init_loss": "",
            "residual_parameter_delta": "",
            "max_zero_init_logit_delta": "",
            "max_hep_alpha0_logit_delta": "",
            "hep_alpha": "",
            "hep_loss": "",
            "max_hep_logit_delta_from_ordinary": "",
            "hep_support_change_fraction": "",
            "hep_pinned_vs_repicked_logit_delta": "",
            "device": device,
            "error": error,
            "max_steps": max_steps,
        }
    ]


def _format_metric(value: Any) -> str:
    return f"{float(value):.8f}"


def _format_optional_metric(value: Any) -> str:
    if value in {"", None}:
        return ""
    return _format_metric(value)


def _final_loss(rows: list[dict[str, Any]]) -> float | None:
    for row in reversed(rows):
        value = row.get("residual_loss")
        if value not in {"", None}:
            return float(value)
    return None


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, experiment_id: str, summary: dict[str, Any]) -> None:
    phase0 = summary.get("phase0") or {}
    invariants = phase0.get("invariants") or {}
    hep_sweep = phase0.get("hep_alpha_sweep") or []
    invariant_lines = [
        f"- {name}: `{value}`" for name, value in sorted(invariants.items())
    ]
    if not invariant_lines:
        invariant_lines = ["- Phase 0 invariants: `not run`"]
    hep_lines = [
        (
            f"- alpha `{entry['alpha']}`: loss `{entry['loss']}`, "
            f"max ordinary-logit delta `{entry['max_logit_delta_from_ordinary']}`, "
            "support-change fraction "
            f"`{entry.get('support_change_fraction', 'not measured')}`, "
            "pinned-vs-repicked delta "
            f"`{entry.get('pinned_vs_repicked_logit_delta', 'not measured')}`"
        )
        for entry in hep_sweep
    ]
    if not hep_lines:
        hep_lines = ["- HEP alpha sweep: `not configured`"]

    path.write_text(
        "\n".join(
            [
                f"# {experiment_id}",
                "",
                "RelaLeap FabricPC Phase 0 smoke run.",
                "",
                f"- Status: `{summary['status']}`",
                f"- Error: `{summary['error'] or 'none'}`",
                f"- JAX backend: `{summary['backend']}`",
                f"- CUDA available: `{summary['cuda_available']}`",
                f"- FabricPC file: `{summary.get('fabricpc_file')}`",
                f"- Final smoke loss: `{summary['final_smoke_loss']}`",
                f"- Residual objective: `{phase0.get('residual_objective', 'not run')}`",
                f"- Pinned support: `{phase0.get('pinned_support', 'not run')}`",
                f"- Support stress: `{phase0.get('support_stress', 'not run')}`",
                f"- Base loss: `{phase0.get('base_loss', 'not run')}`",
                f"- Residual training steps: `{phase0.get('training_steps', 'not run')}`",
                f"- Residual final loss: `{phase0.get('post_step_loss', 'not run')}`",
                "",
                "## Invariants",
                "",
                *invariant_lines,
                "",
                "## HEP Alpha Sweep",
                "",
                *hep_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RelaLeap FabricPC config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", default=Path("results/runs/smoke"), type=Path)
    args = parser.parse_args()
    summary = run(args.config, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
