#!/usr/bin/env python
"""Build and run the GenBench-MILP full generated-vs-original comparison.

The default tasks create the comparison manifest, input audit, and a command
plan for solver-heavy metrics. Add ``tasks=[manifest,audit,commands,similarity]``
to compute structural similarity for all main comparison rows. Add
``tasks=[status]`` to audit which expected metric outputs are present.
"""

import csv
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.feasible_ratio import FeasibleRatioMetric
from metrics.instance_similarity import FEATURES, compute_features, compute_jsdiv


ETA_VALUES = ("eta0.01", "eta0.05", "eta0.10", "eta0.20")
INSTANCE_EXTENSIONS = (".lp", ".mps", ".mps.gz", ".lp.gz")
METHOD_PROBLEM_GRID = {
    "acm-milp": {
        "mis": ("mis", "independent_set", "MIS and independent_set are the same problem type."),
        "setcover": ("setcover", "set_cover", ""),
        "ca": ("ca", "combinatorial_auction", ""),
        "cfl": ("cfl", "capacitated_facility_location", ""),
        "mik": ("mik", "mik", "Original MIK is MPS while generated MIK is LP; record format caveat."),
    },
    "dig-milp": {
        "mis": ("mis", "independent_set", "MIS and independent_set are the same problem type."),
        "setcover": ("setcover", "set_cover", ""),
        "ca": ("ca", "combinatorial_auction", ""),
        "cfl": ("cfl", "capacitated_facility_location", ""),
        "mik": ("mik", "mik", "Original MIK is MPS while generated MIK is LP; record format caveat."),
    },
    "g2milp": {
        "mis": ("mis", "independent_set", "MIS and independent_set are the same problem type."),
        "setcover": ("setcover", "set_cover", ""),
        "ca": ("ca", "combinatorial_auction", ""),
        "cfl": ("cfl", "capacitated_facility_location", ""),
        "mik": ("mik", "mik", "Original MIK is MPS while generated MIK is LP; record format caveat."),
    },
    "milp-evolve": {
        "mis": ("is", "independent_set", "IS/MIS and independent_set are the same problem type."),
        "setcover": ("setcover", "set_cover", ""),
        "ca": ("ca", "combinatorial_auction", ""),
        "cfl": ("cfl", "capacitated_facility_location", ""),
        "mik": ("mik", "mik", "Original MIK is MPS while generated MIK is LP; record format caveat."),
    },
    "milp-studio": {
        "mis": ("mis", "independent_set", "MIS and independent_set are the same problem type. Local reproduction from the paper."),
        "setcover": ("setcover", "set_cover", "Local reproduction from the paper."),
        "ca": ("ca", "combinatorial_auction", "Local reproduction from the paper."),
        "cfl": ("cfl", "capacitated_facility_location", "Local reproduction from the paper."),
        "mik": ("mik", "mik", "Local reproduction from the paper. Original MIK is MPS while generated MIK is LP; record format caveat."),
    },
    "milp-fbgen": {
        "mis": ("mis", "independent_set", "MIS and independent_set are the same problem type. Local reproduction from the paper."),
        "setcover": ("setcover", "set_cover", "Local reproduction from the paper."),
        "ca": ("ca", "combinatorial_auction", "Local reproduction from the paper."),
        "cfl": ("cfl", "capacitated_facility_location", "Local reproduction from the paper."),
        "mik": ("mik", "mik", "Local reproduction from the paper. Original MIK is MPS while generated MIK is LP; record format caveat."),
    },
}


@dataclass
class ManifestRow:
    comparison_id: str
    scope: str
    method: str
    problem: str
    generated_problem: str
    eta: str
    generated_variant: str
    original_problem: str
    original_dir: str
    generated_dir: str
    output_dir: str
    original_file_ext: str
    generated_file_ext: str
    n_original: int
    n_generated: int
    notes: str


def sanitize_id(value: str) -> str:
    value = value.replace("/", "_").replace(".", "_").replace("-", "_")
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def canonical_eta(dirname: str) -> Optional[str]:
    match = re.search(r"eta0\.(\d+)", dirname)
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 1:
        digits = f"{digits}0"
    return f"eta0.{digits}"


def eta_aliases(eta: str, prefix: str = "eta") -> List[str]:
    suffix = eta.replace("eta", "")
    aliases = [f"{prefix}{suffix}"]
    if suffix in ("0.10", "0.20"):
        aliases.append(f"{prefix}{suffix[:-1]}")
    return aliases


def resolve_eta_dir(base_dir: Path, eta: str, prefix: str = "eta") -> Path:
    for alias in eta_aliases(eta, prefix=prefix):
        candidate = base_dir / alias
        if candidate.exists():
            return candidate
    # Return the canonical path so missing directories are visible in the audit.
    return base_dir / eta_aliases(eta, prefix=prefix)[0]


def iter_instance_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    files: List[Path] = []
    for child in directory.iterdir():
        if child.is_file() and any(child.name.endswith(ext) for ext in INSTANCE_EXTENSIONS):
            files.append(child)
    return sorted(files)


def file_ext_summary(files: Sequence[Path]) -> str:
    counts: Dict[str, int] = {}
    for file in files:
        ext = ".mps.gz" if file.name.endswith(".mps.gz") else ".lp.gz" if file.name.endswith(".lp.gz") else file.suffix
        counts[ext] = counts.get(ext, 0) + 1
    return ";".join(f"{ext}:{counts[ext]}" for ext in sorted(counts)) if counts else ""


def comparison_output_dir(output_base: Path, method: str, problem: str, eta: str, variant: str) -> Path:
    suffix = f"{eta}_{variant}" if variant != "raw" else eta
    return output_base / "pairs" / sanitize_id(method) / sanitize_id(problem) / sanitize_id(suffix)


def make_row(
    output_base: Path,
    data_root: Path,
    method: str,
    problem: str,
    generated_problem: str,
    eta: str,
    generated_dir: Path,
    original_problem: str,
    variant: str = "raw",
    scope: str = "main",
    notes: str = "",
) -> ManifestRow:
    original_dir = data_root / "original" / original_problem
    original_files = iter_instance_files(original_dir)
    generated_files = iter_instance_files(generated_dir)
    comparison_id = sanitize_id(f"{method}_{problem}_{eta}_{variant}")
    return ManifestRow(
        comparison_id=comparison_id,
        scope=scope,
        method=method,
        problem=problem,
        generated_problem=generated_problem,
        eta=eta,
        generated_variant=variant,
        original_problem=original_problem,
        original_dir=str(original_dir),
        generated_dir=str(generated_dir),
        output_dir=str(comparison_output_dir(output_base, method, problem, eta, variant)),
        original_file_ext=file_ext_summary(original_files),
        generated_file_ext=file_ext_summary(generated_files),
        n_original=len(original_files),
        n_generated=len(generated_files),
        notes=notes,
    )


def build_manifest(data_root: Path, output_base: Path) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    generated_root = data_root / "generated"

    def add_grid(
        method: str,
        problem: str,
        generated_problem: str,
        original_problem: str,
        notes: str = "",
    ) -> None:
        base = generated_root / method / generated_problem
        for eta in ETA_VALUES:
            rows.append(
                make_row(
                    output_base=output_base,
                    data_root=data_root,
                    method=method,
                    problem=problem,
                    generated_problem=generated_problem,
                    eta=eta,
                    generated_dir=resolve_eta_dir(base, eta),
                    original_problem=original_problem,
                    notes=notes,
                )
            )

    for method, problems in METHOD_PROBLEM_GRID.items():
        for problem, (generated_problem, original_problem, notes) in problems.items():
            if method == "dig-milp":
                dig_base = generated_root / "dig-milp" / generated_problem
                for eta in ETA_VALUES:
                    fixed_notes = (
                        "Canonical PySCIPOpt re-export validated with Gurobi. "
                        "Raw DIG output is preserved outside the main score table."
                    )
                    if problem == "ca" and eta == "eta0.20":
                        fixed_notes += " Source is the fixed-sense CA companion dataset."
                    if notes:
                        fixed_notes = f"{notes} {fixed_notes}"
                    rows.append(
                        make_row(
                            output_base,
                            data_root,
                            "dig-milp",
                            problem,
                            generated_problem,
                            eta,
                            resolve_eta_dir(dig_base, eta, prefix="gurobi_fixed_eta"),
                            original_problem,
                            variant="gurobi_fixed",
                            notes=fixed_notes,
                        )
                    )
            else:
                add_grid(method, problem, generated_problem, original_problem, notes)

    dig_ca_base = generated_root / "dig-milp" / "ca"
    rows.append(
        make_row(
            output_base,
            data_root,
            "dig-milp",
            "ca",
            "ca",
            "eta0.20",
            resolve_eta_dir(dig_ca_base, "eta0.20"),
            "combinatorial_auction",
            variant="raw",
            scope="audit_only",
            notes="Raw DIG-MILP CA eta0.20 is retained for pre/post-fix audit only, not the main score table.",
        )
    )
    for eta in ("eta0.01", "eta0.05", "eta0.10"):
        rows.append(
            make_row(
                output_base,
                data_root,
                "dig-milp",
                "ca",
                "ca",
                eta,
                resolve_eta_dir(dig_ca_base, eta, prefix="fixed_eta"),
                "combinatorial_auction",
                variant="fixed",
                scope="audit_only",
                notes="Fixed DIG-MILP CA companion directory audited but not used in the main table for this eta.",
            )
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(output_base: Path, rows: Sequence[ManifestRow]) -> None:
    data = [asdict(row) for row in rows]
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    write_csv(output_base / "manifest.csv", data, fieldnames)
    write_csv(output_base / "manifest_main.csv", [r for r in data if r["scope"] == "main"], fieldnames)
    write_csv(output_base / "manifest_audit_only.csv", [r for r in data if r["scope"] != "main"], fieldnames)

    summary = {
        "total_rows": len(rows),
        "main_rows": sum(1 for row in rows if row.scope == "main"),
        "audit_only_rows": sum(1 for row in rows if row.scope != "main"),
        "methods": sorted({row.method for row in rows if row.scope == "main"}),
        "eta_values": list(ETA_VALUES),
    }
    (output_base / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Wrote manifest with %s main rows to %s", summary["main_rows"], output_base / "manifest.csv")


def select_readability_samples(files: Sequence[Path], sample_count: int) -> List[Path]:
    if not files or sample_count <= 0:
        return []
    if len(files) <= sample_count:
        return list(files)
    indices = sorted({0, len(files) // 2, len(files) - 1})
    next_index = 1
    while len(indices) < sample_count and next_index < len(files) - 1:
        indices.append(next_index)
        next_index += 1
    return [files[i] for i in sorted(indices[:sample_count])]


def read_with_pyscipopt(instance_path: Path) -> Tuple[bool, str, Optional[dict]]:
    try:
        from pyscipopt import Model

        model = Model()
        model.hideOutput()
        model.setIntParam("parallel/maxnthreads", 1)
        model.readProblem(str(instance_path))
        signature = {
            "vars": int(model.getNVars()),
            "conss": int(model.getNConss()),
            "nonzeros": int(sum(len(model.getValsLinear(cons)) for cons in model.getConss())),
            "sense": str(model.getObjectiveSense()).lower(),
        }
        return True, ";".join(f"{key}={value}" for key, value in signature.items()), signature
    except Exception as exc:
        return False, str(exc), None


def read_with_gurobi(instance_path: Path) -> Tuple[bool, str, Optional[dict]]:
    try:
        import gurobipy as gp

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("Threads", 1)
            env.start()
            model = gp.read(str(instance_path), env=env)
            model.setParam("Threads", 1)
            signature = {
                "vars": int(model.NumVars),
                "conss": int(model.NumConstrs),
                "nonzeros": int(model.NumNZs),
                "sense": "minimize" if int(model.ModelSense) == 1 else "maximize",
            }
            model.dispose()
        return True, ";".join(f"{key}={value}" for key, value in signature.items()), signature
    except Exception as exc:
        return False, str(exc), None


def read_with_configured_solvers(instance_path: Path, reader: str) -> Tuple[bool, str]:
    reader = reader.lower()
    if reader not in {"pyscipopt", "gurobi", "both"}:
        raise ValueError(f"Unsupported audit.reader={reader!r}; use pyscipopt, gurobi, or both")

    messages: List[str] = []
    signatures: Dict[str, dict] = {}
    checks = []
    if reader in {"pyscipopt", "both"}:
        checks.append(("PySCIPOpt", read_with_pyscipopt))
    if reader in {"gurobi", "both"}:
        checks.append(("Gurobi", read_with_gurobi))

    for name, check in checks:
        ok, message, signature = check(instance_path)
        messages.append(f"{name}[{message}]")
        if not ok or signature is None:
            return False, "; ".join(messages)
        signatures[name] = signature

    if reader == "both":
        mismatches = [
            key
            for key in ("vars", "conss", "nonzeros", "sense")
            if signatures["PySCIPOpt"][key] != signatures["Gurobi"][key]
        ]
        if mismatches:
            messages.append("signature_mismatch=" + ",".join(mismatches))
            return False, "; ".join(messages)
    return True, "; ".join(messages)


def audit_directory(dataset_id: str, directory: Path, cfg: DictConfig) -> Tuple[dict, List[dict], List[dict]]:
    files = iter_instance_files(directory)
    exists = directory.exists()
    sizes = [file.stat().st_size for file in files]
    small_threshold = int(cfg.audit.small_file_bytes)
    small_files = [
        {"dataset_id": dataset_id, "path": str(file), "size_bytes": file.stat().st_size}
        for file in files
        if file.stat().st_size < small_threshold
    ]
    empty_count = sum(1 for size in sizes if size == 0)

    readability_rows: List[dict] = []
    reader = str(cfg.audit.reader).lower()
    for sample in select_readability_samples(files, int(cfg.audit.readability_samples)):
        ok, message = read_with_configured_solvers(sample, reader)
        readability_rows.append(
            {
                "dataset_id": dataset_id,
                "path": str(sample),
                "reader": reader,
                "readable": ok,
                "message": message,
            }
        )

    ext_counts: Dict[str, int] = {}
    for file in files:
        ext = ".mps.gz" if file.name.endswith(".mps.gz") else ".lp.gz" if file.name.endswith(".lp.gz") else file.suffix
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    audit_row = {
        "dataset_id": dataset_id,
        "path": str(directory),
        "exists": exists,
        "instance_count": len(files),
        "ext_counts": json.dumps(ext_counts, sort_keys=True),
        "small_file_count": len(small_files),
        "empty_file_count": empty_count,
        "min_size_bytes": min(sizes) if sizes else "",
        "max_size_bytes": max(sizes) if sizes else "",
        "readability_samples": len(readability_rows),
        "readability_failures": sum(1 for row in readability_rows if not row["readable"]),
    }
    return audit_row, small_files, readability_rows


def dataset_entries_for_audit(rows: Sequence[ManifestRow]) -> Dict[str, Path]:
    entries: Dict[str, Path] = {}
    for row in rows:
        original_key = sanitize_id(f"original_{row.original_problem}")
        entries[original_key] = Path(row.original_dir)
        scope_suffix = "audit" if row.scope != "main" else "main"
        generated_key = sanitize_id(f"{scope_suffix}_{row.method}_{row.problem}_{row.eta}_{row.generated_variant}")
        entries[generated_key] = Path(row.generated_dir)
    return entries


def run_audit(output_base: Path, rows: Sequence[ManifestRow], cfg: DictConfig) -> None:
    audit_rows: List[dict] = []
    small_rows: List[dict] = []
    readability_rows: List[dict] = []
    for dataset_id, directory in sorted(dataset_entries_for_audit(rows).items()):
        audit_row, small_files, sample_rows = audit_directory(dataset_id, directory, cfg)
        audit_rows.append(audit_row)
        small_rows.extend(small_files)
        readability_rows.extend(sample_rows)

    audit_fields = [
        "dataset_id",
        "path",
        "exists",
        "instance_count",
        "ext_counts",
        "small_file_count",
        "empty_file_count",
        "min_size_bytes",
        "max_size_bytes",
        "readability_samples",
        "readability_failures",
    ]
    write_csv(output_base / "input_audit.csv", audit_rows, audit_fields)
    write_csv(output_base / "small_files.csv", small_rows, ["dataset_id", "path", "size_bytes"])
    readability_fields = ["dataset_id", "path", "reader", "readable", "message"]
    write_csv(output_base / "readability_samples.csv", readability_rows, readability_fields)
    failures = [row for row in readability_rows if not row["readable"]]
    write_csv(output_base / "readability_failures.csv", failures, readability_fields)

    summary = {
        "datasets_audited": len(audit_rows),
        "missing_dirs": sum(1 for row in audit_rows if not row["exists"]),
        "dirs_with_small_files": sum(1 for row in audit_rows if row["small_file_count"]),
        "readability_failures": len(failures),
        "main_readability_failures": sum(
            not str(row["dataset_id"]).startswith("audit_") for row in failures
        ),
        "audit_only_readability_failures": sum(
            str(row["dataset_id"]).startswith("audit_") for row in failures
        ),
        "reader": str(cfg.audit.reader).lower(),
    }
    (output_base / "input_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Wrote input audit for %s unique directories", len(audit_rows))


def shell_quote(path: str) -> str:
    return "'" + path.replace("'", "'\"'\"'") + "'"


def dataset_entries_for_main_metrics(rows: Sequence[ManifestRow]) -> Dict[str, Path]:
    entries: Dict[str, Path] = {}
    for row in rows:
        if row.scope != "main":
            continue
        entries[sanitize_id(f"original_{row.original_problem}")] = Path(row.original_dir)
        entries[sanitize_id(f"generated_{row.method}_{row.problem}_{row.eta}_{row.generated_variant}")] = Path(row.generated_dir)
    return entries


def dataset_entry_row_index(rows: Sequence[ManifestRow]) -> Dict[str, ManifestRow]:
    index: Dict[str, ManifestRow] = {}
    for row in rows:
        if row.scope != "main":
            continue
        index[sanitize_id(f"generated_{row.method}_{row.problem}_{row.eta}_{row.generated_variant}")] = row
    return index


def command_row(task: str, dataset_id: str, output_dir: Path, command: str, notes: str = "") -> dict:
    return {
        "task": task,
        "dataset_id": dataset_id,
        "output_dir": str(output_dir),
        "command": command,
        "notes": notes,
    }


def write_command_plan(output_base: Path, rows: Sequence[ManifestRow], cfg: DictConfig) -> None:
    commands: List[dict] = []
    max_jobs = min(int(cfg.solve.max_parallel_jobs), 150)
    threads = int(cfg.solve.threads)
    time_limit = int(cfg.solve.time_limit)
    entries = select_dataset_entries(
        dataset_entries_for_main_metrics(rows),
        cfg,
        row_index=dataset_entry_row_index(rows),
    )

    for dataset_id, directory in sorted(entries.items()):
        single_out = output_base / "single_metrics" / dataset_id
        dir_arg = shell_quote(str(directory))
        commands.append(
            command_row(
                "feasibility",
                dataset_id,
                single_out / "feasibility",
                " ".join(
                    [
                        "conda run -n b4milp python scripts/calculate_feasibility.py",
                        f"single.dir={dir_arg}",
                        f"single.output_dir={shell_quote(str(single_out / 'feasibility'))}",
                        f"time_limit={time_limit}",
                        f"threads={threads}",
                        "parallel=true",
                        f"n_jobs={max_jobs}",
                    ]
                ),
            )
        )
        commands.append(
            command_row(
                "solving_time",
                dataset_id,
                single_out / "solving_time",
                " ".join(
                    [
                        "conda run -n b4milp python scripts/calculate_instance_solving_time.py",
                        f"paths.instances_dir={dir_arg}",
                        f"paths.output_dir={shell_quote(str(single_out / 'solving_time'))}",
                        f"solve.time_limit={time_limit}",
                        f"solve.threads={threads}",
                    ]
                ),
            )
        )
        commands.append(
            command_row(
                "branching_nodes",
                dataset_id,
                single_out / "branching_nodes",
                " ".join(
                    [
                        "conda run -n b4milp python scripts/calculate_branching_nodes.py",
                        f"paths.instances_dir={dir_arg}",
                        f"paths.output_dir={shell_quote(str(single_out / 'branching_nodes'))}",
                        f"solve.time_limit={time_limit}",
                        f"solve.threads={threads}",
                        "solve.parallel.enabled=true",
                        f"solve.parallel.processes={max_jobs}",
                    ]
                ),
            )
        )
        commands.append(
            command_row(
                "gurobi_solver_info",
                dataset_id,
                single_out / "solver_info",
                " ".join(
                    [
                        "conda run -n b4milp python scripts/evaluate_solver.py",
                        f"paths.instances_dir={dir_arg}",
                        f"paths.output_dir={shell_quote(str(single_out / 'solver_info'))}",
                        f"solve.time_limit={time_limit}",
                        f"solve.threads={threads}",
                    ]
                ),
                "evaluate_solver.py creates a timestamped experiment subdirectory under paths.output_dir.",
            )
        )

    for row in select_rows(rows, cfg):
        pair_dir = Path(row.output_dir)
        commands.append(
            command_row(
                "similarity",
                row.comparison_id,
                pair_dir / "similarity",
                " ".join(
                    [
                        "conda run -n b4milp python scripts/compare_similarity.py",
                        f"paths.set1_dir={shell_quote(row.original_dir)}",
                        f"paths.set2_dir={shell_quote(row.generated_dir)}",
                        f"paths.output_dir={shell_quote(str(pair_dir / 'similarity'))}",
                        f"compute.num_workers={int(cfg.similarity.num_workers)}",
                        f"compute.num_samples={int(cfg.similarity.num_samples)}",
                    ]
                ),
            )
        )
        commands.append(
            command_row(
                "solver_behavior_compare",
                row.comparison_id,
                pair_dir / "solver_behavior",
                "",
                "Run after solver_info CSVs exist: compare_solver_rootgap.py, compare_solver_cutplane.py, and compare_solver_heur.py with the original/generated solver_info.csv paths.",
            )
        )

    write_csv(output_base / "command_plan.csv", commands, ["task", "dataset_id", "output_dir", "command", "notes"])
    logging.info("Wrote command plan with %s rows", len(commands))


def feature_cache_key(directory: Path) -> str:
    digest = hashlib.sha1(str(directory).encode("utf-8")).hexdigest()[:12]
    return f"{sanitize_id(directory.name)}_{digest}"


def load_or_compute_features(directory: Path, cache_dir: Path, cfg: DictConfig) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = iter_instance_files(directory)
    cache_key = feature_cache_key(directory)
    feature_path = cache_dir / f"{cache_key}.npz"
    meta_path = cache_dir / f"{cache_key}.json"

    if bool(cfg.similarity.reuse_cache) and feature_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("directory") == str(directory) and meta.get("instance_count") == len(files):
            logging.info("Reusing feature cache for %s", directory)
            return np.load(feature_path)["features"]

    logging.info("Computing structural features for %s (%s files)", directory, len(files))
    feature_vectors = compute_features(str(directory), num_workers=int(cfg.similarity.num_workers))
    if not feature_vectors:
        raise RuntimeError(f"No structural features extracted from {directory}")
    features = np.vstack(feature_vectors)
    np.savez_compressed(feature_path, features=features)
    meta = {
        "directory": str(directory),
        "instance_count": len(files),
        "feature_count": int(features.shape[0]),
        "feature_names": list(FEATURES.keys()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return features


def plot_similarity_heatmap(feature_similarities: Dict[str, float], output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    features = list(feature_similarities.keys())
    similarities = list(feature_similarities.values())
    plt.figure(figsize=(12, 4))
    sns.heatmap(
        [similarities],
        annot=True,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        xticklabels=features,
        yticklabels=["Similarity"],
        cbar_kws={"label": "Similarity score"},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_similarity(output_base: Path, rows: Sequence[ManifestRow], cfg: DictConfig) -> None:
    cache_dir = output_base / "feature_cache"
    summary_rows: List[dict] = []
    np.random.seed(int(cfg.similarity.seed))

    selected_rows = select_rows(rows, cfg)
    for row in selected_rows:
        pair_dir = Path(row.output_dir) / "similarity"
        pair_dir.mkdir(parents=True, exist_ok=True)
        try:
            original_features = load_or_compute_features(Path(row.original_dir), cache_dir, cfg)
            generated_features = load_or_compute_features(Path(row.generated_dir), cache_dir, cfg)
            score, feature_scores = compute_jsdiv(
                list(original_features),
                list(generated_features),
                num_samples=int(cfg.similarity.num_samples),
            )
            result = {
                "comparison_id": row.comparison_id,
                "method": row.method,
                "problem": row.problem,
                "eta": row.eta,
                "generated_variant": row.generated_variant,
                "overall_similarity": score,
                "feature_similarities": feature_scores,
                "meta": {
                    "set1_dir": row.original_dir,
                    "set2_dir": row.generated_dir,
                    "set1_count": row.n_original,
                    "set2_count": row.n_generated,
                    "set1_feature_count": int(original_features.shape[0]),
                    "set2_feature_count": int(generated_features.shape[0]),
                    "num_samples": int(cfg.similarity.num_samples),
                    "notes": row.notes,
                },
            }
            (pair_dir / "similarity_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            if bool(cfg.similarity.save_plots):
                plot_similarity_heatmap(
                    feature_scores,
                    pair_dir / "similarity_heatmap.png",
                    f"{row.comparison_id} similarity: {score:.4f}",
                )
            summary_row = {
                "comparison_id": row.comparison_id,
                "method": row.method,
                "problem": row.problem,
                "eta": row.eta,
                "generated_variant": row.generated_variant,
                "overall_similarity": score,
                "n_original": row.n_original,
                "n_generated": row.n_generated,
                "status": "ok",
                "error": "",
            }
            summary_row.update({f"sim_{key}": value for key, value in feature_scores.items()})
            summary_rows.append(summary_row)
        except Exception as exc:
            logging.exception("Similarity failed for %s", row.comparison_id)
            summary_rows.append(
                {
                    "comparison_id": row.comparison_id,
                    "method": row.method,
                    "problem": row.problem,
                    "eta": row.eta,
                    "generated_variant": row.generated_variant,
                    "overall_similarity": "",
                    "n_original": row.n_original,
                    "n_generated": row.n_generated,
                    "status": "error",
                    "error": str(exc),
                }
            )

    base_fields = [
        "comparison_id",
        "method",
        "problem",
        "eta",
        "generated_variant",
        "overall_similarity",
        "n_original",
        "n_generated",
        "status",
        "error",
    ]
    feature_fields = [f"sim_{key}" for key in FEATURES.keys()]
    write_csv(output_base / "similarity_summary.csv", summary_rows, base_fields + feature_fields)
    logging.info("Wrote structural similarity summary for %s main comparisons", len(summary_rows))


def nonempty_selection(values: Iterable[str]) -> List[str]:
    return [str(value) for value in values if str(value)]


def select_dataset_entries(
    entries: Dict[str, Path],
    cfg: DictConfig,
    row_index: Optional[Dict[str, ManifestRow]] = None,
) -> Dict[str, Path]:
    dataset_ids = set(nonempty_selection(cfg.selection.dataset_ids))
    methods = set(nonempty_selection(cfg.selection.methods))
    problems = set(nonempty_selection(cfg.selection.problems))
    etas = set(nonempty_selection(cfg.selection.etas))
    selected_items = [(dataset_id, directory) for dataset_id, directory in sorted(entries.items())]
    if dataset_ids:
        selected_items = [(dataset_id, directory) for dataset_id, directory in selected_items if dataset_id in dataset_ids]
    if row_index and (methods or problems or etas):
        filtered_items = []
        for dataset_id, directory in selected_items:
            row = row_index.get(dataset_id)
            if row is None:
                # Keep shared original datasets only when the caller did not
                # request method/problem/eta filtering.
                continue
            if methods and row.method not in methods:
                continue
            if problems and row.problem not in problems:
                continue
            if etas and row.eta not in etas:
                continue
            filtered_items.append((dataset_id, directory))
        selected_items = filtered_items

    limit = cfg.selection.limit
    if limit is not None:
        selected_items = selected_items[: int(limit)]
    return dict(selected_items)


def select_rows(rows: Sequence[ManifestRow], cfg: DictConfig) -> List[ManifestRow]:
    selected = [row for row in rows if row.scope == "main"]
    comparison_ids = set(nonempty_selection(cfg.selection.comparison_ids))
    methods = set(nonempty_selection(cfg.selection.methods))
    problems = set(nonempty_selection(cfg.selection.problems))
    etas = set(nonempty_selection(cfg.selection.etas))

    if comparison_ids:
        selected = [row for row in selected if row.comparison_id in comparison_ids]
    if methods:
        selected = [row for row in selected if row.method in methods]
    if problems:
        selected = [row for row in selected if row.problem in problems]
    if etas:
        selected = [row for row in selected if row.eta in etas]

    limit = cfg.selection.limit
    if limit is not None:
        selected = selected[: int(limit)]
    return selected


def write_feasibility_outputs(output_dir: Path, dataset_id: str, directory: Path, result: dict, cfg: DictConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "dataset_id": dataset_id,
        "instances_dir": str(directory),
        "time_limit": int(cfg.solve.time_limit),
        "threads": int(cfg.solve.threads),
        "n_jobs": min(int(cfg.feasibility.n_jobs), 150),
        "result": result,
    }
    (output_dir / "feasibility_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    instance_rows = result.get("details", {}).get("instance_results", [])
    if instance_rows:
        base_fields = ["instance", "path", "feasible", "status", "status_code", "runtime_sec", "sol_count", "error"]
        extra_fields = sorted({key for row in instance_rows for key in row.keys()} - set(base_fields))
        write_csv(output_dir / "feasibility_instances.csv", instance_rows, base_fields + extra_fields)


FEASIBILITY_SUMMARY_FIELDS = [
    "dataset_id",
    "instances_dir",
    "total_count",
    "feasible_count",
    "feasible_ratio_percent",
    "timeout_count",
    "error_count",
    "status_counts",
    "error_counts",
    "status",
    "error",
    "result_json",
]


def summarize_feasibility_result(dataset_id: str, directory: Path, output_dir: Path, result: dict, status: str, error: str = "") -> dict:
    details = result.get("details", {}) if result else {}
    return {
        "dataset_id": dataset_id,
        "instances_dir": str(directory),
        "total_count": details.get("total_count", ""),
        "feasible_count": details.get("feasible_count", ""),
        "feasible_ratio_percent": result.get("value", "") if result else "",
        "timeout_count": details.get("timeout_count", ""),
        "error_count": details.get("error_count", ""),
        "status_counts": json.dumps(details.get("status_counts", {}), sort_keys=True),
        "error_counts": json.dumps(details.get("error_counts", {}), sort_keys=True),
        "status": status,
        "error": error,
        "result_json": str(output_dir / "feasibility_results.json"),
    }


def feasibility_run_status(result: dict, base_status: str) -> str:
    details = result.get("details", {}) if result else {}
    error_count = int(details.get("error_count", 0) or 0)
    timeout_count = int(details.get("timeout_count", 0) or 0)
    if error_count:
        return f"{base_status}_with_solver_errors"
    if timeout_count:
        return f"{base_status}_with_timeouts"
    return base_status


def load_existing_feasibility(output_dir: Path) -> Optional[dict]:
    result_path = output_dir / "feasibility_results.json"
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        logging.info("Ignoring legacy feasibility result without schema_version=2: %s", result_path)
        return None
    return payload.get("result")


def load_feasibility_summary(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            dataset_id = row.get("dataset_id", "")
            if dataset_id:
                rows[dataset_id] = {field: row.get(field, "") for field in FEASIBILITY_SUMMARY_FIELDS}
        return rows


def write_feasibility_summary(output_base: Path, summary_by_dataset: Dict[str, dict]) -> None:
    ordered_rows = [summary_by_dataset[key] for key in sorted(summary_by_dataset)]
    write_csv(output_base / "feasibility_summary.csv", ordered_rows, FEASIBILITY_SUMMARY_FIELDS)


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
        evidence = first_existing_file(directory, patterns)
        if evidence:
            return evidence
    return None


def metric_status_row(task: str, target_id: str, status: str, evidence: str = "", notes: str = "") -> dict:
    return {
        "task": task,
        "target_id": target_id,
        "status": status,
        "evidence": evidence,
        "notes": notes,
    }


def feasibility_status(output_dir: Path) -> Tuple[str, str, str]:
    result_path = output_dir / "feasibility_results.json"
    if not result_path.exists():
        return "missing", "", ""
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return "error", str(result_path), f"Could not parse JSON: {exc}"
    if payload.get("schema_version") != 2:
        return "legacy_invalid", str(result_path), "Missing schema_version=2; rerun with the current feasibility code."
    result = payload.get("result", {})
    details = result.get("details", {})
    if int(details.get("error_count", 0) or 0):
        return "done_with_solver_errors", str(result_path), json.dumps(details.get("error_counts", {}), sort_keys=True)
    if int(details.get("timeout_count", 0) or 0):
        return "done_with_timeouts", str(result_path), json.dumps(details.get("status_counts", {}), sort_keys=True)
    return "done", str(result_path), ""


def dataset_id_aliases(dataset_id: str) -> List[str]:
    aliases = [dataset_id]
    if dataset_id.startswith("generated_milp_evolve_mis_"):
        aliases.append(dataset_id.replace("generated_milp_evolve_mis_", "generated_milp_evolve_is_", 1))
    return aliases


def single_metric_dirs(output_base: Path, dataset_id: str) -> List[Path]:
    return [output_base / "single_metrics" / alias for alias in dataset_id_aliases(dataset_id)]


def first_feasibility_status(output_base: Path, dataset_id: str) -> Tuple[str, str, str]:
    for single_out in single_metric_dirs(output_base, dataset_id):
        status, evidence, notes = feasibility_status(single_out / "feasibility")
        if status != "missing":
            if single_out.name != dataset_id:
                notes = f"{notes}; reused alias output {single_out.name}".strip("; ")
            return status, evidence, notes
    return "missing", "", ""


def pair_behavior_dirs(output_base: Path, row: ManifestRow) -> List[Path]:
    dirs = [Path(row.output_dir) / "solver_behavior"]
    if row.method == "milp-evolve" and row.problem == "mis":
        suffix = f"{row.eta}_{row.generated_variant}" if row.generated_variant != "raw" else row.eta
        dirs.append(
            output_base
            / "pairs"
            / sanitize_id(row.method)
            / "is"
            / sanitize_id(suffix)
            / "solver_behavior"
        )
    return dirs


def write_metric_status(output_base: Path, rows: Sequence[ManifestRow]) -> None:
    status_rows: List[dict] = []

    for dataset_id in sorted(dataset_entries_for_main_metrics(rows)):
        single_outs = single_metric_dirs(output_base, dataset_id)

        status, evidence, notes = first_feasibility_status(output_base, dataset_id)
        status_rows.append(metric_status_row("feasibility", dataset_id, status, evidence, notes))

        solving_time_dirs = []
        for single_out in single_outs:
            solving_time_dirs.append(single_out / "solving_time")
            solving_time_dirs.extend(sorted(path for path in single_out.glob("solving_time_*") if path.is_dir()))
        solving_time = first_existing_file_in_dirs(solving_time_dirs, ["*.csv", "*.json", "*.txt"])
        status_rows.append(
            metric_status_row(
                "solving_time",
                dataset_id,
                "done" if solving_time else "missing",
                str(solving_time) if solving_time else "",
                "reused alias output" if solving_time and dataset_id not in str(solving_time) else "",
            )
        )

        branching_nodes = first_existing_file_in_dirs(
            [single_out / "branching_nodes" for single_out in single_outs],
            ["branching_nodes_results.json", "*.json", "*.csv", "*.txt"],
        )
        status_rows.append(
            metric_status_row(
                "branching_nodes",
                dataset_id,
                "done" if branching_nodes else "missing",
                str(branching_nodes) if branching_nodes else "",
                "reused alias output" if branching_nodes and dataset_id not in str(branching_nodes) else "",
            )
        )

        solver_info = first_existing_file_in_dirs(
            [single_out / "solver_info" for single_out in single_outs],
            ["solver_info.csv"],
        )
        status_rows.append(
            metric_status_row(
                "gurobi_solver_info",
                dataset_id,
                "done" if solver_info else "missing",
                str(solver_info) if solver_info else "",
                "reused alias output" if solver_info and dataset_id not in str(solver_info) else "",
            )
        )

    for row in rows:
        if row.scope != "main":
            continue
        pair_dir = Path(row.output_dir)

        similarity = pair_dir / "similarity" / "similarity_results.json"
        status_rows.append(
            metric_status_row(
                "similarity",
                row.comparison_id,
                "done" if similarity.exists() else "missing",
                str(similarity) if similarity.exists() else "",
            )
        )

        for task, patterns in [
            ("solver_behavior_rootgap", ["*rootgap*.json", "*rootgap*.csv"]),
            ("solver_behavior_cutplane", ["*cutplane*.json", "*cutplane*.csv"]),
            ("solver_behavior_heuristic", ["*heur*.json", "*heur*.csv"]),
        ]:
            evidence = first_existing_file_in_dirs(pair_behavior_dirs(output_base, row), patterns)
            status_rows.append(
                metric_status_row(
                    task,
                    row.comparison_id,
                    "done" if evidence else "missing",
                    str(evidence) if evidence else "",
                    "reused alias output" if evidence and str(pair_dir) not in str(evidence) else "",
                )
            )

    write_csv(output_base / "metric_status.csv", status_rows, ["task", "target_id", "status", "evidence", "notes"])
    summary: Dict[str, Dict[str, int]] = {}
    for row in status_rows:
        task_counts = summary.setdefault(row["task"], {})
        task_counts[row["status"]] = task_counts.get(row["status"], 0) + 1
    (output_base / "metric_status_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Wrote metric status with %s rows", len(status_rows))


def run_feasibility(output_base: Path, rows: Sequence[ManifestRow], cfg: DictConfig) -> None:
    entries = select_dataset_entries(dataset_entries_for_main_metrics(rows), cfg, dataset_entry_row_index(rows))
    summary_by_dataset = load_feasibility_summary(output_base / "feasibility_summary.csv")
    n_jobs = min(int(cfg.feasibility.n_jobs), 150)

    for dataset_id, directory in entries.items():
        output_dir = output_base / "single_metrics" / dataset_id / "feasibility"
        existing = load_existing_feasibility(output_dir) if bool(cfg.feasibility.skip_existing) else None
        if existing is not None:
            logging.info("Reusing feasibility result for %s", dataset_id)
            summary_by_dataset[dataset_id] = summarize_feasibility_result(
                dataset_id,
                directory,
                output_dir,
                existing,
                feasibility_run_status(existing, "cached"),
            )
            continue

        files = iter_instance_files(directory)
        max_instances = cfg.feasibility.max_instances
        if max_instances is not None:
            files = files[: int(max_instances)]

        logging.info(
            "Running feasibility for %s (%s files, time_limit=%s, threads=%s, n_jobs=%s)",
            dataset_id,
            len(files),
            cfg.solve.time_limit,
            cfg.solve.threads,
            n_jobs,
        )

        metric = FeasibleRatioMetric(
            time_limit=int(cfg.solve.time_limit),
            threads=int(cfg.solve.threads),
            parallel=True,
            n_jobs=n_jobs,
        )

        try:
            result = metric.calculate([str(file) for file in files])
            write_feasibility_outputs(output_dir, dataset_id, directory, result, cfg)
            summary_by_dataset[dataset_id] = summarize_feasibility_result(
                dataset_id,
                directory,
                output_dir,
                result,
                feasibility_run_status(result, "ok"),
            )
        except Exception as exc:
            logging.exception("Feasibility failed for %s", dataset_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "feasibility_error.txt").write_text(str(exc), encoding="utf-8")
            summary_by_dataset[dataset_id] = summarize_feasibility_result(dataset_id, directory, output_dir, {}, "error", str(exc))

        write_feasibility_summary(output_base, summary_by_dataset)

    logging.info("Wrote feasibility summary for %s datasets", len(summary_by_dataset))


@hydra.main(config_path="../configs", config_name="genbench_comparison", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=getattr(logging, cfg.logging.level), format=cfg.logging.format)
    logging.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    data_root = Path(cfg.data_root).expanduser().resolve()
    output_base = (Path(cfg.output_root).expanduser() / str(cfg.date_tag)).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    rows = build_manifest(data_root, output_base)
    tasks = set(cfg.tasks)

    if "manifest" in tasks:
        write_manifest(output_base, rows)
    if "audit" in tasks:
        run_audit(output_base, rows, cfg)
    if "commands" in tasks:
        write_command_plan(output_base, rows, cfg)
    if "similarity" in tasks:
        run_similarity(output_base, rows, cfg)
    if "feasibility" in tasks:
        run_feasibility(output_base, rows, cfg)
    if "status" in tasks:
        write_metric_status(output_base, rows)

    logging.info("GenBench comparison tasks completed. Output: %s", output_base)


if __name__ == "__main__":
    main()
