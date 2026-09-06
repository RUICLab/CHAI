#!/usr/bin/env python3
"""Cheap deterministic checks required before a six-benchmark LLM campaign."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.constraint_gatekeeper import filter_feasible_techniques
from lib.hardware_profile import load_hardware_profile
from lib.lsp import lsp_patcher
from lib.technique_registry import TECHNIQUE_REGISTRY
from utils.knob_metadata import KnobMetadataError, validate_plan_entries
from utils.source_functions import recover_source_entities


APPS = (
    "ar-iclib",
    "sobel-iclib",
    "stringsearch-iclib",
    "fft-iclib",
    "lqi-iclib",
    "bc-iclib",
)


def _run(command, cwd):
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True)


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return default


def preflight(repo_root, hardware_path):
    report = {
        "operator_catalog": "checkmate_plus_10",
        "operator_ids": list(range(1, 31)),
        "hardware": hardware_path,
        "apps": {},
        "errors": [],
    }
    profile = load_hardware_profile(hardware_path)
    catalog_ids = set(range(1, 31))
    if not catalog_ids.issubset(TECHNIQUE_REGISTRY):
        report["errors"].append("registry does not contain every T1-T30 operator")

    for app in APPS:
        app_source = repo_root / "benchmark_applications" / app
        app_report = {
            "native_build": False,
            "main_initialization": {},
            "source_targets": [],
            "external_or_unresolved_pdg_nodes": [],
            "cached_plan_valid": None,
            "cached_plan_error": None,
            "cached_plan_compile": None,
            "hardware_feasible_ids": [],
        }
        report["apps"][app] = app_report

        with tempfile.TemporaryDirectory(prefix=f"chai_preflight_{app}_") as work:
            work_path = Path(work)
            shutil.copytree(app_source, work_path, dirs_exist_ok=True)
            _run(["make", "clean"], work_path)
            build = _run(["make", "main"], work_path)
            app_report["native_build"] = build.returncode == 0
            if build.returncode != 0:
                app_report["build_error"] = (build.stdout + build.stderr)[-4000:]
                report["errors"].append(f"{app}: pristine native build failed")

        topology_path = repo_root / "llm-prerun" / app / "dependency_graphs" / "topological_order.json"
        topology = _read_json(topology_path, [])
        source_files = [
            str(path.relative_to(app_source))
            for path in app_source.rglob("*")
            if path.suffix in {".c", ".cpp"}
        ]
        recovered = recover_source_entities(topology, source_files, str(app_source))
        source_names = {entry["functionName"] for entry in recovered}
        app_report["source_targets"] = sorted(source_names)
        app_report["external_or_unresolved_pdg_nodes"] = sorted(
            set(topology) - source_names - {"main"}
        )

        plan_path = repo_root / "llm-prerun" / app / "approximated_functions" / "apx_all.json"
        if plan_path.exists():
            plan = _read_json(plan_path, None)
            try:
                validate_plan_entries(plan)
                missing = sorted(
                    entry.get("functionName")
                    for entry in plan
                    if entry.get("functionName") not in source_names
                )
                if missing:
                    raise KnobMetadataError(
                        f"cached targets have no local source definition: {missing}"
                    )
                app_report["cached_plan_valid"] = True

                with tempfile.TemporaryDirectory(prefix=f"chai_plan_{app}_") as work:
                    work_path = Path(work)
                    shutil.copytree(app_source, work_path, dirs_exist_ok=True)
                    plan_copy = work_path / "apx_all.json"
                    plan_copy.write_text(json.dumps(plan, indent=2), encoding="utf-8")
                    try:
                        lsp_patcher(str(work_path), plan_copy.name)
                        _run(["make", "clean"], work_path)
                        plan_build = _run(["make", "main"], work_path)
                        app_report["cached_plan_compile"] = plan_build.returncode == 0
                        if plan_build.returncode != 0:
                            app_report["cached_plan_build_error"] = (
                                plan_build.stdout + plan_build.stderr
                            )[-4000:]
                            report["errors"].append(
                                f"{app}: cached approximation plan does not compile"
                            )
                    except Exception as error:
                        app_report["cached_plan_compile"] = False
                        app_report["cached_plan_build_error"] = str(error)
                        report["errors"].append(
                            f"{app}: cached approximation patch failed: {error}"
                        )
            except (KnobMetadataError, TypeError) as error:
                app_report["cached_plan_valid"] = False
                app_report["cached_plan_error"] = str(error)
                report["errors"].append(f"{app}: cached plan invalid: {error}")

        feasible, _ = filter_feasible_techniques(app, profile)
        app_report["hardware_feasible_ids"] = sorted(set(feasible) & catalog_ids)
        if not app_report["hardware_feasible_ids"]:
            report["errors"].append(f"{app}: MSP430 has no feasible T1-T30 operators")

        for variant, extra_args in (
            ("operator_only", []),
            ("chai_msp430", ["--hardware", hardware_path]),
        ):
            command = [
                sys.executable,
                str(repo_root / "main.py"),
                "--bm_name", app,
                "--operator_catalog", "checkmate_plus_10",
                "--preflight_only",
                *extra_args,
            ]
            initialized = _run(command, repo_root)
            app_report["main_initialization"][variant] = {
                "ok": initialized.returncode == 0,
                "tail": (initialized.stdout + initialized.stderr)[-2000:],
            }
            if initialized.returncode != 0:
                report["errors"].append(
                    f"{app}/{variant}: real main.py initialization failed"
                )

    report["ok"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", default="config/hardware_msp430.json")
    parser.add_argument("--output", default="outputs/preflight_six_benchmarks.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    hardware_path = str((repo_root / args.hardware).resolve())
    report = preflight(repo_root, hardware_path)
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for app, result in report["apps"].items():
        state = "PASS" if result["native_build"] else "FAIL"
        print(
            f"{state} {app}: build={result['native_build']}, "
            f"source_targets={len(result['source_targets'])}, "
            f"unresolved={result['external_or_unresolved_pdg_nodes']}, "
            f"cached_plan={result['cached_plan_valid']}, "
            f"plan_compile={result['cached_plan_compile']}, "
            f"main_init={all(item['ok'] for item in result['main_initialization'].values())}"
        )
    print(f"Preflight report: {output_path}")
    if not report["ok"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

