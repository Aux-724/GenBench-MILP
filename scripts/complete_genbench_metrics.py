#!/usr/bin/env python
"""Complete missing GenBench-MILP metrics for a comparison manifest.

The script is intentionally resumable: each dataset/pair is skipped when the
expected evidence file already exists.  It also treats the historical
``milp-evolve/is`` outputs as aliases for the current ``milp-evolve/mis`` rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INSTANCE_EXTENSIONS = (".lp", ".mps", ".mps.gz", ".lp.gz")


def sanitize_id(value: str) -> str:
    import re

    value = value.replace("/", "_").replace(".", "_").replace("-", "_")
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_instance_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and any(path.name.endswith(ext) for ext in INSTANCE_EXTENSIONS)
    )


def generated_dataset_id(row: dict) -> str:
    return sanitize_id(f"generated_{row['method']}_{row['problem']}_{row['eta']}_{row['generated_variant']}")


def original_dataset_id(row: dict) -> str:
    return sanitize_id(f"original_{row['original_problem']}")


def dataset_aliases(dataset_id: str) -> List[str]:
    aliases = [dataset_id]
    if dataset_id.startswith("generated_milp_evolve_mis_"):
        aliases.append(dataset_id.replace("generated_milp_evolve_mis_", "generated_milp_evolve_is_", 1))
    return aliases


def single_metric_dirs(output_base: Path, dataset_id: str) -> List[Path]:
    return [output_base / "single_metrics" / alias for alias in dataset_aliases(dataset_id)]


def first_existing_file(directory: Path, patterns: Sequence[str]) -> Optional[Path]:
    if not directory.exists():
        return None
    for pattern in patterns:
        matches = sorted(path for path in directory.rglob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def first_existing_file_in_dirs(directories: Sequence[Path], patterns: Sequence[str]) -> Optional[Path]:
    for directory in directories:
        found = first_existing_file(directory, patterns)
        if found:
            return found
    return None


def solving_time_evidence(output_base: Path, dataset_id: str) -> Optional[Path]:
    dirs: List[Path] = []
    for single_out in single_metric_dirs(output_base, dataset_id):
        dirs.append(single_out / "solving_time")
        dirs.extend(sorted(path for path in single_out.glob("solving_time_*") if path.is_dir()))
    return first_existing_file_in_dirs(dirs, ["solving_time_results.json"])


def branching_evidence(output_base: Path, dataset_id: str) -> Optional[Path]:
    return first_existing_file_in_dirs(
        [single_out / "branching_nodes" for single_out in single_metric_dirs(output_base, dataset_id)],
        ["branching_nodes_results.json"],
    )


def solver_info_evidence(output_base: Path, dataset_id: str) -> Optional[Path]:
    return first_existing_file_in_dirs(
        [single_out / "solver_info" for single_out in single_metric_dirs(output_base, dataset_id)],
        ["solver_info.csv"],
    )


def eta_dir_name(eta: str) -> str:
    return eta.replace(".", "_")


def pair_behavior_dirs(output_base: Path, row: dict) -> List[Path]:
    dirs = [Path(row["output_dir"]) / "solver_behavior"]
    if row["method"] == "milp-evolve" and row["problem"] == "mis":
        suffix = row["eta"] if row["generated_variant"] == "raw" else f"{row['eta']}_{row['generated_variant']}"
        dirs.append(
            output_base
            / "pairs"
            / sanitize_id(row["method"])
            / "is"
            / sanitize_id(suffix)
            / "solver_behavior"
        )
    return dirs


def behavior_evidence(output_base: Path, row: dict, kind: str) -> Optional[Path]:
    patterns = {
        "rootgap": ["rootgap_comparison_results.csv"],
        "cutplane": ["cutplane_pca_results_distances.csv"],
        "heuristic": ["heuristic_comparison_results.csv"],
    }[kind]
    return first_existing_file_in_dirs(pair_behavior_dirs(output_base, row), patterns)


def solve_one(args: tuple) -> dict:
    path, time_limit, threads = args
    start = time.monotonic()
    model = None
    env = None
    result = {
        "instance_name": Path(path).name,
        "instance_path": path,
        "solving_time": None,
        "obj": None,
        "status_name": "ERROR",
        "is_feasible": False,
        "is_error": True,
        "num_nodes": None,
        "num_sols": None,
        "wall_time": 0.0,
        "error": "",
    }
    try:
        import gurobipy as gp

        status_map = {
            gp.GRB.OPTIMAL: "OPTIMAL",
            gp.GRB.INFEASIBLE: "INFEASIBLE",
            gp.GRB.UNBOUNDED: "UNBOUNDED",
            gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
            gp.GRB.TIME_LIMIT: "TIME_LIMIT",
            gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
            gp.GRB.LOADED: "LOADED",
            gp.GRB.INTERRUPTED: "INTERRUPTED",
            gp.GRB.NUMERIC: "NUMERIC",
            gp.GRB.NODE_LIMIT: "NODE_LIMIT",
            gp.GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
            gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            gp.GRB.CUTOFF: "CUTOFF",
        }
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.read(path, env=env)
        if time_limit is not None:
            model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", threads)
        model.optimize()

        status = status_map.get(model.Status, f"UNKNOWN({model.Status})")
        is_feasible = model.Status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL) or (
            model.Status == gp.GRB.TIME_LIMIT and model.SolCount > 0
        )
        obj = None
        if is_feasible:
            try:
                obj = float(model.ObjVal)
            except Exception:
                obj = None
        result.update(
            {
                "solving_time": float(model.Runtime),
                "obj": obj,
                "status_name": status,
                "is_feasible": bool(is_feasible),
                "is_error": False,
                "num_nodes": float(model.NodeCount),
                "num_sols": int(model.SolCount),
            }
        )
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass
        if env is not None:
            try:
                env.dispose()
            except Exception:
                pass
    result["wall_time"] = time.monotonic() - start
    return result


def run_hardness_dataset(
    output_base: Path,
    dataset_id: str,
    instances_dir: Path,
    time_limit: int,
    threads: int,
    processes: int,
    max_instances: Optional[int],
    skip_existing: bool,
) -> None:
    if skip_existing and solving_time_evidence(output_base, dataset_id) and branching_evidence(output_base, dataset_id):
        print(f"[hardness] skip existing {dataset_id}")
        return

    files = iter_instance_files(instances_dir)
    if max_instances is not None:
        files = files[:max_instances]
    if not files:
        raise RuntimeError(f"No instance files found for {dataset_id}: {instances_dir}")

    print(f"[hardness] {dataset_id}: solving {len(files)} instances with processes={processes}, threads={threads}")
    started = time.monotonic()
    args = [(str(path), time_limit, threads) for path in files]
    results: List[dict] = []
    with ProcessPoolExecutor(max_workers=min(processes, len(args))) as executor:
        futures = [executor.submit(solve_one, item) for item in args]
        for idx, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if idx % 100 == 0 or idx == len(futures):
                print(f"[hardness] {dataset_id}: {idx}/{len(futures)} complete", flush=True)

    total_time = time.monotonic() - started
    solving_times = [float(row["solving_time"]) for row in results if row.get("solving_time") is not None]
    solved_count = len(solving_times)
    stats = {
        "total_count": len(results),
        "solved_count": solved_count,
        "mean_time": float(sum(solving_times) / solved_count) if solved_count else 0.0,
        "median_time": float(sorted(solving_times)[solved_count // 2]) if solved_count else 0.0,
        "min_time": float(min(solving_times)) if solved_count else 0.0,
        "max_time": float(max(solving_times)) if solved_count else 0.0,
        "std_time": float((sum((x - (sum(solving_times) / solved_count)) ** 2 for x in solving_times) / solved_count) ** 0.5)
        if solved_count
        else 0.0,
        "feasible_count": sum(1 for row in results if row.get("is_feasible")),
        "infeasible_count": sum(1 for row in results if not row.get("is_feasible") and not row.get("is_error")),
        "error_count": sum(1 for row in results if row.get("is_error")),
    }

    single_out = output_base / "single_metrics" / dataset_id
    solving_out = single_out / f"solving_time_completed_{time.strftime('%Y%m%d_%H%M%S')}"
    solving_out.mkdir(parents=True, exist_ok=True)
    (solving_out / "solving_time_results.json").write_text(
        json.dumps({"results": results, "stats": stats}, indent=2),
        encoding="utf-8",
    )

    valid_results = [row for row in results if row.get("num_nodes") is not None]
    node_counts = [float(row["num_nodes"]) for row in valid_results]
    status_counts: Dict[str, int] = {}
    for row in valid_results:
        status = str(row.get("status_name") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

    dataset_name = Path(instances_dir).name
    branching_stats = {
        "dataset_name": dataset_name,
        "total_instances": len(results),
        "valid_instances": len(valid_results),
        "total_nodes": float(sum(node_counts)) if node_counts else 0.0,
        "mean_nodes": float(sum(node_counts) / len(node_counts)) if node_counts else 0.0,
        "median_nodes": float(sorted(node_counts)[len(node_counts) // 2]) if node_counts else 0.0,
        "max_nodes": float(max(node_counts)) if node_counts else 0.0,
        "min_nodes": float(min(node_counts)) if node_counts else 0.0,
        "std_nodes": float((sum((x - (sum(node_counts) / len(node_counts))) ** 2 for x in node_counts) / len(node_counts)) ** 0.5)
        if node_counts
        else 0.0,
        "status_distribution": status_counts,
        "total_processing_time": total_time,
        "average_processing_time": total_time / len(results) if results else 0.0,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "parameters": {
            "time_limit": time_limit,
            "parallel": True,
            "processes": min(processes, len(args)),
            "threads": threads,
            "max_instances": max_instances,
        },
        "instance_details": [
            {
                "instance_name": row["instance_name"],
                "node_count": row.get("num_nodes"),
                "status": row.get("status_name"),
                "solve_time": row.get("solving_time"),
                "obj_val": row.get("obj") if row.get("is_feasible") else None,
                "num_solutions": row.get("num_sols"),
                "error": row.get("error", ""),
            }
            for row in results
        ],
    }
    branching_out = single_out / "branching_nodes" / f"branching_nodes_{dataset_name}"
    branching_out.mkdir(parents=True, exist_ok=True)
    (branching_out / "branching_nodes_results.json").write_text(json.dumps(branching_stats, indent=2), encoding="utf-8")
    write_csv(
        branching_out / "instance_details.csv",
        branching_stats["instance_details"],
        ["instance_name", "node_count", "status", "solve_time", "obj_val", "num_solutions", "error"],
    )


def run_command(command: Sequence[str], cwd: Path, dry_run: bool) -> None:
    print("[cmd]", " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def run_solver_info_dataset(
    output_base: Path,
    dataset_id: str,
    instances_dir: Path,
    time_limit: int,
    threads: int,
    processes: int,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    if skip_existing and solver_info_evidence(output_base, dataset_id):
        print(f"[solver_info] skip existing {dataset_id}")
        return
    out = output_base / "single_metrics" / dataset_id / "solver_info"
    out.mkdir(parents=True, exist_ok=True)
    command = [
        "conda",
        "run",
        "-n",
        "b4milp",
        "python",
        "scripts/evaluate_solver.py",
        f"paths.instances_dir={instances_dir}",
        f"paths.output_dir={out}",
        f"solve.time_limit={time_limit}",
        f"solve.threads={threads}",
        "solve.parallel.enabled=true",
        f"solve.parallel.processes={processes}",
        "hydra.run.dir=${paths.output_dir}/experiment_${now:%Y%m%d_%H%M%S}",
    ]
    run_command(command, PROJECT_ROOT, dry_run)


def run_behavior_pair(
    output_base: Path,
    row: dict,
    original_solver_info: Path,
    generated_solver_info: Path,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    pair_dir = Path(row["output_dir"]) / "solver_behavior"
    name1 = original_dataset_id(row)
    name2 = generated_dataset_id(row)
    pair_dir.mkdir(parents=True, exist_ok=True)

    if not (skip_existing and behavior_evidence(output_base, row, "rootgap")):
        run_command(
            [
                "conda",
                "run",
                "-n",
                "b4milp",
                "python",
                "scripts/compare_solver_rootgap.py",
                f"paths.csv_file1={original_solver_info}",
                f"paths.csv_file2={generated_solver_info}",
                f"paths.output_dir={pair_dir / 'rootgap'}",
                f"comparison.name1={name1}",
                f"comparison.name2={name2}",
                "output.result_filename=rootgap_comparison_results.csv",
                "visualization.enabled=false",
                "hydra.run.dir=${paths.output_dir}",
            ],
            PROJECT_ROOT,
            dry_run,
        )
    else:
        print(f"[behavior] skip rootgap {row['comparison_id']}")

    if not (skip_existing and behavior_evidence(output_base, row, "cutplane")):
        run_command(
            [
                "conda",
                "run",
                "-n",
                "b4milp",
                "python",
                "scripts/compare_solver_cutplane.py",
                f"paths.csv_file1={original_solver_info}",
                f"paths.csv_file2={generated_solver_info}",
                f"paths.output_dir={pair_dir / 'cutplane'}",
                f"comparison.name1={name1}",
                f"comparison.name2={name2}",
                "output.result_filename=cutplane_pca_results.csv",
                "visualization.enabled=false",
            ],
            PROJECT_ROOT,
            dry_run,
        )
    else:
        print(f"[behavior] skip cutplane {row['comparison_id']}")

    if not (skip_existing and behavior_evidence(output_base, row, "heuristic")):
        run_command(
            [
                "conda",
                "run",
                "-n",
                "b4milp",
                "python",
                "scripts/compare_solver_heur.py",
                f"paths.csv_file1={original_solver_info}",
                f"paths.csv_file2={generated_solver_info}",
                f"paths.output_dir={pair_dir / 'heuristic'}",
                f"comparison.name1={name1}",
                f"comparison.name2={name2}",
                "output.result_filename=heuristic_comparison_results.csv",
                "visualization.enabled=false",
            ],
            PROJECT_ROOT,
            dry_run,
        )
    else:
        print(f"[behavior] skip heuristic {row['comparison_id']}")


def selected_rows(rows: Sequence[dict], args: argparse.Namespace) -> List[dict]:
    result = [row for row in rows if row.get("scope") == "main"]
    if args.methods:
        methods = set(args.methods.split(","))
        result = [row for row in result if row["method"] in methods]
    if args.problems:
        problems = set(args.problems.split(","))
        result = [row for row in result if row["problem"] in problems]
    if args.etas:
        etas = set(args.etas.split(","))
        result = [row for row in result if row["eta"] in etas]
    if args.limit:
        result = result[: args.limit]
    return result


def dataset_map(rows: Sequence[dict], include_originals: bool = True) -> Dict[str, Path]:
    data: Dict[str, Path] = {}
    for row in rows:
        if include_originals:
            data[original_dataset_id(row)] = Path(row["original_dir"])
        data[generated_dataset_id(row)] = Path(row["generated_dir"])
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", type=Path, default=PROJECT_ROOT / "outputs/genbench_milp_comparison/20260505")
    parser.add_argument("--tasks", default="hardness,solver_info,solver_behavior")
    parser.add_argument("--methods", default="")
    parser.add_argument("--problems", default="")
    parser.add_argument("--etas", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--processes", type=int, default=64)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument(
        "--generated-only",
        action="store_true",
        help="Run single-dataset metrics only for generated rows; reuse original evidence for pair comparisons.",
    )
    parser.add_argument(
        "--reference-output-base",
        type=Path,
        help="Fallback output tree containing reusable original solver_info evidence.",
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads * args.processes > 150:
        raise SystemExit(f"Refusing to run {args.processes} processes with {args.threads} threads (>150 solver threads)")
    output_base = args.output_base.resolve()
    manifest = read_csv(output_base / "manifest_main.csv")
    rows = selected_rows(manifest, args)
    tasks = {task.strip() for task in args.tasks.split(",") if task.strip()}
    max_instances = args.max_instances or None

    datasets = dataset_map(rows, include_originals=not args.generated_only)
    if "hardness" in tasks:
        for dataset_id, instances_dir in sorted(datasets.items()):
            run_hardness_dataset(
                output_base,
                dataset_id,
                instances_dir,
                args.time_limit,
                args.threads,
                args.processes,
                max_instances,
                args.skip_existing,
            )

    if "solver_info" in tasks:
        for dataset_id, instances_dir in sorted(datasets.items()):
            run_solver_info_dataset(
                output_base,
                dataset_id,
                instances_dir,
                args.time_limit,
                args.threads,
                args.processes,
                args.skip_existing,
                args.dry_run,
            )

    if "solver_behavior" in tasks:
        for row in rows:
            original_info = solver_info_evidence(output_base, original_dataset_id(row))
            if not original_info and args.reference_output_base:
                original_info = solver_info_evidence(
                    args.reference_output_base.resolve(),
                    original_dataset_id(row),
                )
            generated_info = solver_info_evidence(output_base, generated_dataset_id(row))
            if not original_info or not generated_info:
                print(f"[behavior] missing solver_info for {row['comparison_id']}: {original_info}, {generated_info}")
                continue
            run_behavior_pair(output_base, row, original_info, generated_info, args.skip_existing, args.dry_run)


if __name__ == "__main__":
    main()
