#!/usr/bin/env python
"""Diagnose G2-CFL constraints that conflict with variable bounds.

This is a parser-only audit: it does not optimize the instances.  A row is
intrinsically infeasible when its best possible activity, computed solely from
the variable bounds, still cannot satisfy the row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

import gurobipy as gp


INSTANCE_PATTERNS = ("*.lp", "*.mps", "*.lp.gz", "*.mps.gz")


def discover_instances(directory: Path, limit: int | None) -> list[Path]:
    files: list[Path] = []
    for pattern in INSTANCE_PATTERNS:
        files.extend(directory.glob(pattern))
    files = sorted(set(files))
    return files[:limit] if limit is not None else files


def finite_bound(value: float) -> bool:
    return math.isfinite(value) and abs(value) < gp.GRB.INFINITY


def row_activity_bounds(model: gp.Model, constraint: gp.Constr) -> tuple[float, float, int]:
    row = model.getRow(constraint)
    min_activity = 0.0
    max_activity = 0.0
    for index in range(row.size()):
        variable = row.getVar(index)
        coefficient = row.getCoeff(index)
        lower, upper = variable.LB, variable.UB
        min_bound = lower if coefficient >= 0 else upper
        max_bound = upper if coefficient >= 0 else lower
        if not finite_bound(min_bound):
            min_activity = -math.inf
        elif math.isfinite(min_activity):
            min_activity += coefficient * min_bound
        if not finite_bound(max_bound):
            max_activity = math.inf
        elif math.isfinite(max_activity):
            max_activity += coefficient * max_bound
    return min_activity, max_activity, row.size()


def row_violation(constraint: gp.Constr, min_activity: float, max_activity: float) -> float:
    rhs = constraint.RHS
    if constraint.Sense == gp.GRB.LESS_EQUAL:
        return max(0.0, min_activity - rhs)
    if constraint.Sense == gp.GRB.GREATER_EQUAL:
        return max(0.0, rhs - max_activity)
    return max(0.0, min_activity - rhs, rhs - max_activity)


def analyze_instance(task: tuple[str, str]) -> dict:
    dataset, path_text = task
    path = Path(path_text)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
        model = gp.read(str(path), env=env)
        model.Params.Threads = 1
        impossible: list[tuple[str, int, float, float, float, str]] = []
        for constraint in model.getConstrs():
            min_activity, max_activity, degree = row_activity_bounds(model, constraint)
            violation = row_violation(constraint, min_activity, max_activity)
            if violation > 1e-7:
                impossible.append(
                    (
                        constraint.ConstrName,
                        degree,
                        min_activity,
                        max_activity,
                        constraint.RHS,
                        constraint.Sense,
                    )
                )
        worst = max(
            impossible,
            key=lambda item: row_violation_from_values(item[2], item[3], item[4], item[5]),
            default=None,
        )
        worst_violation = (
            row_violation_from_values(worst[2], worst[3], worst[4], worst[5]) if worst else 0.0
        )
        return {
            "dataset": dataset,
            "instance": path.name,
            "num_variables": model.NumVars,
            "num_constraints": model.NumConstrs,
            "intrinsically_infeasible_rows": len(impossible),
            "worst_row": worst[0] if worst else "",
            "worst_row_degree": worst[1] if worst else "",
            "worst_min_activity": worst[2] if worst else "",
            "worst_max_activity": worst[3] if worst else "",
            "worst_rhs": worst[4] if worst else "",
            "worst_sense": worst[5] if worst else "",
            "worst_bound_violation": worst_violation,
            "error": "",
        }
    except Exception as exc:  # Preserve per-file evidence instead of aborting the audit.
        return {
            "dataset": dataset,
            "instance": path.name,
            "num_variables": "",
            "num_constraints": "",
            "intrinsically_infeasible_rows": "",
            "worst_row": "",
            "worst_row_degree": "",
            "worst_min_activity": "",
            "worst_max_activity": "",
            "worst_rhs": "",
            "worst_sense": "",
            "worst_bound_violation": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        env.dispose()


def row_violation_from_values(
    min_activity: float, max_activity: float, rhs: float, sense: str
) -> float:
    if sense == gp.GRB.LESS_EQUAL:
        return max(0.0, min_activity - rhs)
    if sense == gp.GRB.GREATER_EQUAL:
        return max(0.0, rhs - max_activity)
    return max(0.0, min_activity - rhs, rhs - max_activity)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Iterable[dict]) -> list[dict]:
    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    summary = []
    for dataset, dataset_rows in sorted(by_dataset.items()):
        valid = [row for row in dataset_rows if not row["error"]]
        row_counts = [int(row["intrinsically_infeasible_rows"]) for row in valid]
        violations = [float(row["worst_bound_violation"]) for row in valid]
        summary.append(
            {
                "dataset": dataset,
                "instances": len(dataset_rows),
                "parsed_instances": len(valid),
                "parse_errors": len(dataset_rows) - len(valid),
                "instances_with_intrinsically_infeasible_row": sum(value > 0 for value in row_counts),
                "ratio_with_intrinsically_infeasible_row": (
                    sum(value > 0 for value in row_counts) / len(valid) if valid else ""
                ),
                "mean_intrinsically_infeasible_rows": mean(row_counts) if row_counts else "",
                "median_intrinsically_infeasible_rows": median(row_counts) if row_counts else "",
                "max_intrinsically_infeasible_rows": max(row_counts) if row_counts else "",
                "mean_worst_bound_violation": mean(violations) if violations else "",
                "max_worst_bound_violation": max(violations) if violations else "",
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("NAME", "DIRECTORY"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=32)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.processes <= 150:
        raise ValueError("--processes must be between 1 and 150")
    tasks: list[tuple[str, str]] = []
    for name, directory_text in args.dataset:
        directory = Path(directory_text)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        tasks.extend((name, str(path)) for path in discover_instances(directory, args.limit))
    if not tasks:
        raise ValueError("No instances found")

    with ProcessPoolExecutor(max_workers=min(args.processes, len(tasks))) as executor:
        rows = list(executor.map(analyze_instance, tasks, chunksize=4))
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "instance_bound_audit.csv", rows)
    write_csv(args.output_dir / "dataset_bound_audit.csv", summary)
    report = {
        "audit_type": "parser_only_variable_bound_activity_check",
        "optimization_performed": False,
        "threads_per_model": 1,
        "processes": min(args.processes, len(tasks)),
        "datasets": summary,
    }
    (args.output_dir / "diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
