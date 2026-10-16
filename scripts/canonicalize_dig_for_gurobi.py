#!/usr/bin/env python
"""Canonicalize DIG-MILP LP files and validate them with Gurobi.

DIG's historical writer emits constraint terms such as ``+1.0x1+1.0x2``.
SCIP accepts that dialect, while Gurobi either rejects long tokens or silently
interprets a short expression as a variable name.  This tool preserves the
model that SCIP reads, writes it back through PySCIPOpt's canonical LP writer,
and checks that Gurobi sees the same variable/constraint counts and objective
sense.

Raw generated files are never modified.  By default, repaired files are
written to sibling directories named ``gurobi_fixed_eta0.01`` and so on.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROBLEMS = ("mis", "setcover", "ca", "cfl", "mik")
ETAS = ("eta0.01", "eta0.05", "eta0.10", "eta0.20")
INSTANCE_SUFFIXES = (".lp", ".mps", ".lp.gz", ".mps.gz")

_GUROBI_ENV = None


@dataclass(frozen=True)
class ConversionTask:
    group_id: str
    problem: str
    eta: str
    source_variant: str
    source_path: str
    output_path: str
    overwrite: bool
    validate_gurobi: bool


def eta_aliases(eta: str, prefix: str = "eta") -> Tuple[str, ...]:
    suffix = eta[3:] if eta.startswith("eta") else eta
    aliases = [f"{prefix}{suffix}"]
    if suffix in {"0.10", "0.20"}:
        aliases.append(f"{prefix}{suffix[:-1]}")
    return tuple(aliases)


def resolve_eta_dir(base_dir: Path, eta: str, prefix: str = "eta") -> Path:
    for alias in eta_aliases(eta, prefix=prefix):
        candidate = base_dir / alias
        if candidate.is_dir():
            return candidate
    return base_dir / eta_aliases(eta, prefix=prefix)[0]


def source_dir(source_root: Path, problem: str, eta: str) -> Tuple[Path, str]:
    base_dir = source_root / problem
    if problem == "ca" and eta == "eta0.20":
        return resolve_eta_dir(base_dir, eta, prefix="fixed_eta"), "fixed_sense"
    return resolve_eta_dir(base_dir, eta), "raw"


def output_dir(output_root: Path, problem: str, eta: str, prefix: str) -> Path:
    suffix = eta[3:] if eta.startswith("eta") else eta
    return output_root / problem / f"{prefix}{suffix}"


def instance_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and any(path.name.endswith(suffix) for suffix in INSTANCE_SUFFIXES)
    )


def initialize_worker(validate_gurobi: bool) -> None:
    global _GUROBI_ENV
    if not validate_gurobi:
        return
    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    _GUROBI_ENV = env


def scip_signature(path: Path) -> Dict[str, object]:
    from pyscipopt import Model

    model = Model()
    model.hideOutput()
    model.setIntParam("parallel/maxnthreads", 1)
    model.readProblem(str(path))
    signature = {
        "vars": int(model.getNVars()),
        "conss": int(model.getNConss()),
        "nonzeros": int(sum(len(model.getValsLinear(cons)) for cons in model.getConss())),
        "sense": str(model.getObjectiveSense()).lower(),
    }
    return {"model": model, "signature": signature}


def gurobi_signature(path: Path) -> Dict[str, object]:
    import gurobipy as gp

    if _GUROBI_ENV is None:
        raise RuntimeError("Gurobi validation environment was not initialized")
    model = gp.read(str(path), env=_GUROBI_ENV)
    model.setParam("Threads", 1)
    signature = {
        "vars": int(model.NumVars),
        "conss": int(model.NumConstrs),
        "nonzeros": int(model.NumNZs),
        "sense": "minimize" if int(model.ModelSense) == 1 else "maximize",
    }
    model.dispose()
    return signature


def canonicalize_one(task: ConversionTask) -> Dict[str, object]:
    started = time.monotonic()
    source_path = Path(task.source_path)
    output_path = Path(task.output_path)
    result: Dict[str, object] = {
        "group_id": task.group_id,
        "problem": task.problem,
        "eta": task.eta,
        "source_variant": task.source_variant,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "status": "error",
        "error": "",
        "source_vars": "",
        "source_conss": "",
        "source_nonzeros": "",
        "source_sense": "",
        "gurobi_vars": "",
        "gurobi_conss": "",
        "gurobi_nonzeros": "",
        "gurobi_sense": "",
        "source_size_bytes": source_path.stat().st_size if source_path.exists() else "",
        "output_size_bytes": "",
        "elapsed_sec": "",
    }
    temp_path: Optional[Path] = None
    try:
        payload = scip_signature(source_path)
        model = payload["model"]
        source_sig = payload["signature"]
        result.update(
            source_vars=source_sig["vars"],
            source_conss=source_sig["conss"],
            source_nonzeros=source_sig["nonzeros"],
            source_sense=source_sig["sense"],
        )

        if output_path.exists() and not task.overwrite:
            candidate_path = output_path
            status = "existing"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = output_path.with_name(
                f".{output_path.stem}.{os.getpid()}.tmp.lp"
            )
            if temp_path.exists():
                temp_path.unlink()
            model.writeProblem(str(temp_path), trans=False, genericnames=False, verbose=False)
            candidate_path = temp_path
            status = "converted"

        if task.validate_gurobi:
            gurobi_sig = gurobi_signature(candidate_path)
            result.update(
                gurobi_vars=gurobi_sig["vars"],
                gurobi_conss=gurobi_sig["conss"],
                gurobi_nonzeros=gurobi_sig["nonzeros"],
                gurobi_sense=gurobi_sig["sense"],
            )
            mismatches = []
            for key in ("vars", "conss", "nonzeros", "sense"):
                if source_sig[key] != gurobi_sig[key]:
                    mismatches.append(
                        f"{key}: SCIP={source_sig[key]} Gurobi={gurobi_sig[key]}"
                    )
            if mismatches:
                raise ValueError("signature mismatch after canonicalization: " + "; ".join(mismatches))

        if temp_path is not None:
            os.replace(temp_path, output_path)
            temp_path = None
        result["output_size_bytes"] = output_path.stat().st_size
        result["status"] = status
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    finally:
        result["elapsed_sec"] = round(time.monotonic() - started, 6)
    return result


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_tasks(args: argparse.Namespace) -> Tuple[List[ConversionTask], List[Dict[str, object]]]:
    tasks: List[ConversionTask] = []
    groups: List[Dict[str, object]] = []
    for problem in args.problems:
        for eta in args.etas:
            src_dir, source_variant = source_dir(args.source_root, problem, eta)
            dst_dir = output_dir(args.output_root, problem, eta, args.output_prefix)
            files = instance_files(src_dir)
            selected = files[: args.limit] if args.limit is not None else files
            group_id = f"dig_milp_{problem}_{eta.replace('.', '_')}_gurobi_fixed"
            groups.append(
                {
                    "group_id": group_id,
                    "problem": problem,
                    "eta": eta,
                    "source_variant": source_variant,
                    "source_dir": str(src_dir),
                    "output_dir": str(dst_dir),
                    "source_count": len(files),
                    "selected_count": len(selected),
                }
            )
            for path in selected:
                tasks.append(
                    ConversionTask(
                        group_id=group_id,
                        problem=problem,
                        eta=eta,
                        source_variant=source_variant,
                        source_path=str(path),
                        output_path=str(dst_dir / path.name),
                        overwrite=args.overwrite,
                        validate_gurobi=args.validate_gurobi,
                    )
                )
    return tasks, groups


def summarize_groups(
    groups: Sequence[Dict[str, object]],
    results: Sequence[Dict[str, object]],
    expected_count: int,
    limited: bool,
) -> List[Dict[str, object]]:
    by_group: Dict[str, List[Dict[str, object]]] = {}
    for result in results:
        by_group.setdefault(str(result["group_id"]), []).append(result)
    summaries: List[Dict[str, object]] = []
    for group in groups:
        group_results = by_group.get(str(group["group_id"]), [])
        output_count = len(instance_files(Path(str(group["output_dir"]))))
        errors = sum(result["status"] == "error" for result in group_results)
        selected_count = int(group["selected_count"])
        target_count = selected_count if limited else expected_count
        row = dict(group)
        row.update(
            converted_count=sum(result["status"] == "converted" for result in group_results),
            existing_count=sum(result["status"] == "existing" for result in group_results),
            error_count=errors,
            output_count=output_count,
            status=(
                "ok"
                if errors == 0
                and len(group_results) == selected_count
                and output_count >= target_count
                and int(group["source_count"]) >= target_count
                else "incomplete"
            ),
        )
        summaries.append(row)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path("/mnt/data1/luoyidong/benchmark_data/data/generated/dig-milp")
    parser.add_argument("--source-root", type=Path, default=default_root)
    parser.add_argument("--output-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="gurobi_fixed_eta")
    parser.add_argument("--problems", nargs="+", choices=PROBLEMS, default=list(PROBLEMS))
    parser.add_argument("--etas", nargs="+", choices=ETAS, default=list(ETAS))
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(validate_gurobi=True)
    parser.add_argument(
        "--validate-gurobi",
        dest="validate_gurobi",
        action="store_true",
        help="Read every repaired file with Gurobi and compare its signature with SCIP (default).",
    )
    parser.add_argument(
        "--no-validate-gurobi",
        dest="validate_gurobi",
        action="store_false",
        help="Skip Gurobi validation (not recommended for benchmark outputs).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("outputs/dig_gurobi_fix/20260731"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 150:
        raise ValueError("--workers must be between 1 and 150")
    if args.expected_count < 1:
        raise ValueError("--expected-count must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    tasks, groups = build_tasks(args)
    print(f"groups={len(groups)} files={len(tasks)} workers={args.workers}")
    results: List[Dict[str, object]] = []
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(args.validate_gurobi,),
    ) as executor:
        futures = [executor.submit(canonicalize_one, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 100 == 0 or completed == len(futures):
                errors = sum(row["status"] == "error" for row in results)
                elapsed = time.monotonic() - started
                print(
                    f"completed={completed}/{len(futures)} errors={errors} "
                    f"elapsed_sec={elapsed:.1f}",
                    flush=True,
                )

    results.sort(key=lambda row: (str(row["group_id"]), str(row["source_path"])))
    group_rows = summarize_groups(
        groups,
        results,
        expected_count=args.expected_count,
        limited=args.limit is not None,
    )
    write_csv(args.report_dir / "canonicalization_files.csv", results)
    write_csv(args.report_dir / "canonicalization_groups.csv", group_rows)

    summary = {
        "groups": len(group_rows),
        "complete_groups": sum(row["status"] == "ok" for row in group_rows),
        "incomplete_groups": sum(row["status"] != "ok" for row in group_rows),
        "files": len(results),
        "converted_files": sum(row["status"] == "converted" for row in results),
        "existing_files": sum(row["status"] == "existing" for row in results),
        "error_files": sum(row["status"] == "error" for row in results),
        "validated_with_gurobi": bool(args.validate_gurobi),
        "threads_per_gurobi_model": 1,
        "workers": args.workers,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "output_prefix": args.output_prefix,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "canonicalization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not args.allow_incomplete and (summary["error_files"] or summary["incomplete_groups"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
