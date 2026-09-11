#!/usr/bin/env python3
"""Reproduce the registered multi-seed mechanism and baseline evidence.

The script discovers only eligible seeds from the three declared V15
namespaces, keeps the seed as the independent unit, and never edits source
acceptance artifacts. Existing completed outputs are reused unless ``--force``
is supplied.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(".")
PROJECT = ROOT / "rig-energy-forecasting"
PAPER = ROOT / "paper/experiments"
RESULTS = PAPER / "results"


DECLARED = [
    ("v15_generator_first_reserve", [20261011, 20261012]),
    ("paper_v15_extension_holdouts_v2", [20261014, 20261015]),
    ("paper_v15_extension_holdouts_v3", list(range(20261016, 20261028))),
]


def eligible_roots() -> list[Path]:
    roots: list[Path] = []
    v3_count = 0
    for namespace, seeds in DECLARED:
        for seed in seeds:
            root = PROJECT / "artifacts" / namespace / f"seed_{seed}"
            eligibility = root / "eligibility.json"
            result = root / "result.json"
            if not eligibility.exists():
                continue
            record = json.loads(eligibility.read_text(encoding="utf-8"))
            if not bool(record.get("eligible")):
                continue
            if not result.exists():
                raise RuntimeError(f"eligible seed lacks result: {root}")
            if namespace.endswith("v3"):
                if v3_count >= 6:
                    raise RuntimeError("v3 contains more than the registered six eligible seeds")
                v3_count += 1
            roots.append(root)
    if v3_count != 6:
        raise RuntimeError(f"registered v3 extension is incomplete: {v3_count}/6 eligible seeds")
    if len(roots) != 10:
        raise RuntimeError(f"expected 10 total eligible holdouts, found {len(roots)}")
    return roots


def run(command: list[str], output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"REUSE {output}", flush=True)
        return
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not output.exists():
        raise RuntimeError(f"command did not produce {output}")


def process_seed(root: Path, force: bool, force_causal: bool) -> None:
    seed = root.name
    history = RESULTS / "shared_initial_histories" / f"{seed}.json"
    causal = RESULTS / "independent_causal_full_milp_history_matched" / f"{seed}.json"
    risk = RESULTS / "risk_signal_ablation" / seed / "risk_ablation_result.json"
    uncertainty = RESULTS / "uncertainty_envelope_ablation" / seed / "uncertainty_ablation_result.json"
    run(
        [sys.executable, str(PAPER / "export_shared_initial_histories.py"), "--artifact-root", str(root), "--output", str(history)],
        history,
        force,
    )
    run(
        [sys.executable, str(PAPER / "run_independent_causal_full_milp.py"), "--artifact-root", str(root), "--config", str(root / "resolved_dispatch_config.yaml"), "--initial-history", str(history), "--output", str(causal)],
        causal,
        force or force_causal,
    )
    run(
        [sys.executable, str(PAPER / "run_risk_signal_ablation.py"), "--artifact-root", str(root), "--output-root", str(risk.parent)],
        risk,
        force,
    )
    run(
        [sys.executable, str(PAPER / "run_no_uncertainty_ablation.py"), "--artifact-root", str(root), "--output-root", str(uncertainty.parent)],
        uncertainty,
        force,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-causal", action="store_true")
    args = parser.parse_args()
    roots = eligible_roots()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process_seed, root, args.force, args.force_causal)
            for root in roots
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    command = [
        sys.executable,
        str(PAPER / "summarize_extended_holdouts.py"),
    ]
    for root in roots:
        command.extend(["--seed-root", str(root)])
    command.extend(
        [
            "--causal-root",
            str(RESULTS / "independent_causal_full_milp_history_matched"),
            "--risk-ablation-root",
            str(RESULTS / "risk_signal_ablation"),
            "--uncertainty-ablation-root",
            str(RESULTS / "uncertainty_envelope_ablation"),
            "--output",
            str(RESULTS / "extended_holdout_analysis.json"),
        ]
    )
    subprocess.run(command, cwd=ROOT, check=True)
    overlap_command = [
        sys.executable,
        str(PAPER / "summarize_risk_action_overlap.py"),
    ]
    for root in roots:
        overlap_command.extend(["--seed-root", str(root)])
    overlap_command.extend(
        [
            "--risk-ablation-root",
            str(RESULTS / "risk_signal_ablation"),
            "--output",
            str(RESULTS / "risk_action_overlap.json"),
            "--figure",
            str(RESULTS / "risk_action_overlap.png"),
        ]
    )
    subprocess.run(overlap_command, cwd=ROOT, check=True)
    print(json.dumps({"eligible_seeds": [root.name for root in roots], "summary": str(RESULTS / "extended_holdout_analysis.json")}))


if __name__ == "__main__":
    main()
