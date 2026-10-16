#!/usr/bin/env python
"""Summarize GenBench-MILP comparison evidence into CSV and LaTeX."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sanitize_id(value: str) -> str:
    value = value.replace("/", "_").replace(".", "_").replace("-", "_")
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_id(row: dict) -> str:
    return sanitize_id(f"generated_{row['method']}_{row['problem']}_{row['eta']}_{row['generated_variant']}")


def original_id(row: dict) -> str:
    return sanitize_id(f"original_{row['original_problem']}")


def dataset_aliases(identifier: str) -> List[str]:
    aliases = [identifier]
    if identifier.startswith("generated_milp_evolve_mis_"):
        aliases.append(identifier.replace("generated_milp_evolve_mis_", "generated_milp_evolve_is_", 1))
    return aliases


def single_dirs(output_base: Path, identifier: str) -> List[Path]:
    return [output_base / "single_metrics" / alias for alias in dataset_aliases(identifier)]


def pair_dir(output_base: Path, row: dict) -> Path:
    suffix = row["eta"] if row["generated_variant"] == "raw" else f"{row['eta']}_{row['generated_variant']}"
    return (
        output_base
        / "pairs"
        / sanitize_id(row["method"])
        / sanitize_id(row["problem"])
        / sanitize_id(suffix)
    )


def first_file(directory: Path, patterns: Sequence[str]) -> Optional[Path]:
    if not directory.exists():
        return None
    for pattern in patterns:
        matches = sorted(path for path in directory.rglob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def first_file_in_dirs(directories: Sequence[Path], patterns: Sequence[str]) -> Optional[Path]:
    for directory in directories:
        found = first_file(directory, patterns)
        if found:
            return found
    return None


def load_json(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def has_value(value) -> bool:
    return value not in (None, "")


def fmt_num(value: Optional[float], digits: int = 3, missing: str = "--") -> str:
    if value is None or not math.isfinite(value):
        return missing
    return f"{value:.{digits}f}"


def fmt_pct(value: Optional[float], digits: int = 1, missing: str = "--") -> str:
    if value is None or not math.isfinite(value):
        return missing
    return f"{value:.{digits}f}"


def solving_time_path(output_base: Path, identifier: str) -> Optional[Path]:
    dirs: List[Path] = []
    for base in single_dirs(output_base, identifier):
        dirs.append(base / "solving_time")
        dirs.extend(sorted(path for path in base.glob("solving_time_*") if path.is_dir()))
    return first_file_in_dirs(dirs, ["solving_time_results.json"])


def branching_path(output_base: Path, identifier: str) -> Optional[Path]:
    return first_file_in_dirs([base / "branching_nodes" for base in single_dirs(output_base, identifier)], ["branching_nodes_results.json"])


def solver_info_path(output_base: Path, identifier: str) -> Optional[Path]:
    return first_file_in_dirs([base / "solver_info" for base in single_dirs(output_base, identifier)], ["solver_info.csv"])


def pair_behavior_dirs(output_base: Path, row: dict) -> List[Path]:
    dirs = [pair_dir(output_base, row) / "solver_behavior"]
    if row["method"] == "milp-evolve" and row["problem"] == "mis":
        suffix = row["eta"] if row["generated_variant"] == "raw" else f"{row['eta']}_{row['generated_variant']}"
        dirs.append(output_base / "pairs" / sanitize_id(row["method"]) / "is" / sanitize_id(suffix) / "solver_behavior")
    return dirs


def behavior_file(output_base: Path, row: dict, kind: str) -> Optional[Path]:
    patterns = {
        "rootgap": ["rootgap_comparison_results.csv"],
        "cutplane": ["cutplane_pca_results_distances.csv"],
        "heuristic": ["heuristic_comparison_results.csv"],
    }[kind]
    return first_file_in_dirs(pair_behavior_dirs(output_base, row), patterns)


def first_from_bases(output_bases: Sequence[Path], finder, *args) -> Optional[Path]:
    for output_base in output_bases:
        found = finder(output_base, *args)
        if found:
            return found
    return None


def stat_from_comparison_csv(path: Optional[Path], row_label: str, column_index: int = 1) -> Optional[float]:
    rows = read_csv(path) if path else []
    for row in rows:
        if row.get("Statistic") == row_label:
            values = list(row.values())
            if len(values) > column_index:
                return to_float(values[column_index])
    return None


def distance_from_cutplane(path: Optional[Path], pc: str) -> Optional[float]:
    rows = read_csv(path) if path else []
    for row in rows:
        if row.get("Principal_Component") == pc:
            return to_float(row.get("Wasserstein_Distance"))
    return None


def collect_row(
    output_bases: Sequence[Path],
    row: dict,
    similarity_by_id: Dict[str, dict],
    feasibility_by_id: Dict[str, dict],
) -> dict:
    did = dataset_id(row)
    sim = similarity_by_id.get(row["comparison_id"], {})
    feas = {}
    for alias in dataset_aliases(did):
        candidate = feasibility_by_id.get(alias)
        if candidate and has_value(candidate.get("feasible_ratio_percent")):
            feas = candidate
            break
        if candidate and not feas:
            feas = feasibility_by_id[alias]

    solve_path = first_from_bases(output_bases, solving_time_path, did)
    branch_evidence = first_from_bases(output_bases, branching_path, did)
    rootgap = first_from_bases(output_bases, behavior_file, row, "rootgap")
    cutplane = first_from_bases(output_bases, behavior_file, row, "cutplane")
    heuristic = first_from_bases(output_bases, behavior_file, row, "heuristic")
    similarity_evidence = first_from_bases(
        output_bases,
        lambda base, item: first_file(pair_dir(base, item) / "similarity", ["similarity_results.json"]),
        row,
    )
    generated_solver_info = first_from_bases(output_bases, solver_info_path, did)
    original_solver_info = first_from_bases(output_bases, solver_info_path, original_id(row))

    solve = load_json(solve_path).get("stats", {})
    branch = load_json(branch_evidence)

    generated_col = did
    original_col = original_id(row)

    notes = row.get("notes", "")

    return {
        "comparison_id": row["comparison_id"],
        "method": row["method"],
        "problem": row["problem"],
        "eta": row["eta"],
        "n_original": row["n_original"],
        "n_generated": row["n_generated"],
        "similarity": sim.get("overall_similarity", ""),
        "feasible_percent": feas.get("feasible_ratio_percent", ""),
        "feasibility_status": feas.get("status", ""),
        "timeout_count": feas.get("timeout_count", ""),
        "solver_error_count": feas.get("error_count", ""),
        "mean_solve_time": solve.get("mean_time", ""),
        "median_solve_time": solve.get("median_time", ""),
        "solved_count": solve.get("solved_count", ""),
        "mean_nodes": branch.get("mean_nodes", ""),
        "median_nodes": branch.get("median_nodes", ""),
        "valid_node_count": branch.get("valid_instances", ""),
        "rootgap_original_mean": stat_from_comparison_csv(rootgap, "mean", 1),
        "rootgap_generated_mean": stat_from_comparison_csv(rootgap, "mean", 2),
        "rootgap_w1": stat_from_comparison_csv(rootgap, "Wasserstein_Distance", 1),
        "cutplane_pc1_w1": distance_from_cutplane(cutplane, "PC1"),
        "cutplane_pc2_w1": distance_from_cutplane(cutplane, "PC2"),
        "cutplane_pc3_w1": distance_from_cutplane(cutplane, "PC3"),
        "heur_original_mean": stat_from_comparison_csv(heuristic, "mean", 1),
        "heur_generated_mean": stat_from_comparison_csv(heuristic, "mean", 2),
        "heur_w1": stat_from_comparison_csv(heuristic, "Wasserstein_Distance", 1),
        "solver_info_original": str(original_solver_info or ""),
        "solver_info_generated": str(generated_solver_info or ""),
        "evidence_similarity": str(similarity_evidence or ""),
        "evidence_feasibility": feas.get("result_json", ""),
        "evidence_solving_time": str(solve_path or ""),
        "evidence_branching_nodes": str(branch_evidence or ""),
        "evidence_rootgap": str(rootgap or ""),
        "evidence_cutplane": str(cutplane or ""),
        "evidence_heuristic": str(heuristic or ""),
        "notes": notes,
    }


def numeric_mean(rows: Sequence[dict], key: str) -> Optional[float]:
    values = [to_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def coverage(rows: Sequence[dict], key: str) -> int:
    return sum(1 for row in rows if to_float(row.get(key)) is not None)


def evidence_coverage(rows: Sequence[dict], key: str) -> int:
    return sum(1 for row in rows if row.get(key))


def method_problem_rows(summary_rows: Sequence[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for row in summary_rows:
        groups.setdefault((row["method"], row["problem"]), []).append(row)
    out = []
    for (method, problem), rows in sorted(groups.items()):
        out.append(
            {
                "method": method,
                "problem": problem,
                "rows": len(rows),
                "similarity": numeric_mean(rows, "similarity"),
                "feasible_percent": numeric_mean(rows, "feasible_percent"),
                "mean_solve_time": numeric_mean(rows, "mean_solve_time"),
                "mean_nodes": numeric_mean(rows, "mean_nodes"),
                "rootgap_w1": numeric_mean(rows, "rootgap_w1"),
                "cutplane_pc1_w1": numeric_mean(rows, "cutplane_pc1_w1"),
                "heur_w1": numeric_mean(rows, "heur_w1"),
            }
        )
    return out


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


ETA_ORDER = ("eta0.01", "eta0.05", "eta0.10", "eta0.20")
METHOD_ORDER = ("acm-milp", "dig-milp", "g2milp", "milp-evolve", "milp-studio", "milp-fbgen")
PROBLEM_ORDER = ("ca", "cfl", "mik", "mis", "setcover")
METHOD_LABELS = {
    "acm-milp": "ACM-MILP",
    "dig-milp": "DIG-MILP",
    "g2milp": "G2MILP",
    "milp-evolve": "MILP-Evolve",
    "milp-studio": "MILP-StuDio",
    "milp-fbgen": "MILP-FBGen",
}
PROBLEM_LABELS = {
    "ca": "CA",
    "cfl": "CFL",
    "mik": "MIK",
    "mis": "MIS",
    "setcover": "SetCover",
}


def sort_key_method_problem(item: tuple) -> tuple:
    method, problem = item
    method_rank = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    problem_rank = PROBLEM_ORDER.index(problem) if problem in PROBLEM_ORDER else len(PROBLEM_ORDER)
    return method_rank, problem_rank, method, problem


def method_problem_metric_table(
    summary_rows: Sequence[dict],
    metric_key: str,
    caption: str,
    label: str,
    digits: int = 3,
    percent: bool = False,
) -> List[str]:
    groups: Dict[tuple, Dict[str, dict]] = {}
    for row in summary_rows:
        groups.setdefault((row["method"], row["problem"]), {})[row["eta"]] = row

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begingroup")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular*}{0.78\textwidth}{@{\extracolsep{\fill}}lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Problem} & \multicolumn{4}{c}{\textbf{Target $\eta$}} \\")
    lines.append(r"\cmidrule(lr){2-5}")
    lines.append(r" & \textbf{0.01} & \textbf{0.05} & \textbf{0.10} & \textbf{0.20} \\")
    lines.append(r"\midrule")
    last_method = None
    for method, problem in sorted(groups, key=sort_key_method_problem):
        if method != last_method:
            if last_method is not None:
                lines.append(r"\addlinespace[0.35em]")
            method_label = latex_escape(METHOD_LABELS.get(method, method))
            lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{method_label}}}}} \\")
            lines.append(r"\addlinespace[0.05em]")
            last_method = method
        eta_rows = groups[(method, problem)]
        values = []
        for eta in ETA_ORDER:
            value = to_float(eta_rows.get(eta, {}).get(metric_key))
            values.append(fmt_pct(value, 1) if percent else fmt_num(value, digits))
        problem_label = latex_escape(PROBLEM_LABELS.get(problem, problem))
        lines.append(" & ".join([problem_label] + values) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\endgroup")
    lines.append(r"\end{table*}")
    lines.append("")
    return lines


def make_latex(summary_rows: Sequence[dict], agg_rows: Sequence[dict], audit: dict) -> str:
    lines: List[str] = []
    lines.append("% Auto-generated by scripts/summarize_genbench_results.py.")
    lines.append(r"\section{Full 120-Group Results}")
    lines.append(r"\label{sec:full_120_results}")
    lines.append("")
    lines.append(
        "We compare 120 generated datasets against their corresponding original distributions: "
        r"6 methods $\times$ 5 problem families $\times$ 4 target $\eta$ values. "
        "CVS and IIS are excluded from this final grid. "
        "All Gurobi-based runs use one solver thread per instance; parallel runs were capped at 64 worker processes."
    )
    lines.append("")
    lines.append(
        "The input audit covered "
        f"{audit.get('datasets_audited', '--')} unique directories with "
        f"{audit.get('missing_dirs', '--')} missing directories, "
        f"{audit.get('dirs_with_small_files', '--')} directories containing small files, and "
        f"{audit.get('main_readability_failures', audit.get('readability_failures', '--'))} main-grid readability failures. "
        f"The {audit.get('audit_only_readability_failures', 0)} audit-only failures are retained raw DIG-MILP files "
        "that are intentionally excluded from the score tables."
    )
    lines.append("")
    lines.append(
        r"Tables~\ref{tab:genbench_similarity}--\ref{tab:genbench_heuristic_w1} report one metric per table. "
        r"Each table keeps the four $\eta$ settings as separate columns; missing entries indicate unavailable finite values, not omitted comparisons."
    )
    lines.append("")
    metric_tables = [
        (
            "similarity",
            r"Structural similarity by method, problem, and $\eta$; higher is better.",
            "tab:genbench_similarity",
            3,
            False,
        ),
        (
            "feasible_percent",
            r"Solver-certified feasible-and-bounded ratio within 300 seconds (\%) by method, problem, and $\eta$; higher is better. Time-limit cases remain uncertified rather than being labeled infeasible.",
            "tab:genbench_feasibility",
            1,
            True,
        ),
        (
            "mean_solve_time",
            r"Mean Gurobi solve time in seconds by method, problem, and $\eta$.",
            "tab:genbench_solve_time",
            3,
            False,
        ),
        (
            "mean_nodes",
            r"Mean Gurobi branch-and-bound node count by method, problem, and $\eta$.",
            "tab:genbench_nodes",
            1,
            False,
        ),
        (
            "rootgap_w1",
            r"Root-gap 1-Wasserstein distance by method, problem, and $\eta$; lower is closer to the original distribution.",
            "tab:genbench_rootgap_w1",
            3,
            False,
        ),
        (
            "cutplane_pc1_w1",
            r"Cut-plane usage PC1 1-Wasserstein distance by method, problem, and $\eta$; lower is closer.",
            "tab:genbench_cutplane_pc1_w1",
            3,
            False,
        ),
        (
            "cutplane_pc2_w1",
            r"Cut-plane usage PC2 1-Wasserstein distance by method, problem, and $\eta$; lower is closer.",
            "tab:genbench_cutplane_pc2_w1",
            3,
            False,
        ),
        (
            "cutplane_pc3_w1",
            r"Cut-plane usage PC3 1-Wasserstein distance by method, problem, and $\eta$; lower is closer.",
            "tab:genbench_cutplane_pc3_w1",
            3,
            False,
        ),
        (
            "heur_w1",
            r"Heuristic-count 1-Wasserstein distance by method, problem, and $\eta$; lower is closer.",
            "tab:genbench_heuristic_w1",
            3,
            False,
        ),
    ]
    for metric_key, caption, label, digits, percent in metric_tables:
        lines.extend(method_problem_metric_table(summary_rows, metric_key, caption, label, digits, percent))
    lines.append(
        r"\paragraph{Caveats.} MIS and independent set denote the same problem family; "
        r"the MILP-Evolve directory uses the historical name \texttt{is}. "
        r"All DIG-MILP main results use canonical \texttt{gurobi\_fixed\_eta*} LP files whose variable, constraint, nonzero, and objective-sense signatures match between PySCIPOpt and Gurobi; raw DIG files are audit-only. "
        r"DIG-MILP CA at $\eta=0.20$ is canonicalized from the fixed-sense companion source. "
        r"A dash in root-gap or higher cut-PCA tables denotes an undefined comparison caused by absent finite observations or insufficient effective feature dimension. "
        r"MIK originals are MPS files while generated MIK instances are LP files. "
        r"MILP-StuDio and MILP-FBGen are local reproductions because the original code was not public."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", type=Path, default=PROJECT_ROOT / "outputs/genbench_milp_comparison/20260505")
    parser.add_argument(
        "--fallback-output-base",
        action="append",
        type=Path,
        default=[],
        help="Additional evidence tree, searched in order after --output-base.",
    )
    parser.add_argument("--tex", type=Path, default=PROJECT_ROOT / "ICML-GenBench-MILP/results.tex")
    parser.add_argument(
        "--problems",
        nargs="+",
        default=[],
        help="Optional problem-family filter, for example: --problems cfl.",
    )
    parser.add_argument(
        "--summary-prefix",
        default="final_120",
        help="Prefix for CSV/JSON summary artifacts (default: final_120).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_base = args.output_base.resolve()
    output_bases = [output_base]
    for fallback in args.fallback_output_base:
        resolved = fallback.resolve()
        if resolved not in output_bases:
            output_bases.append(resolved)
    manifest = read_csv(output_base / "manifest_main.csv")
    if args.problems:
        requested_problems = set(args.problems)
        manifest = [row for row in manifest if row.get("problem") in requested_problems]
    summary_prefix = sanitize_id(args.summary_prefix)
    if not summary_prefix:
        raise ValueError("--summary-prefix must contain at least one alphanumeric character")
    similarity_by_id: Dict[str, dict] = {}
    feasibility_by_id: Dict[str, dict] = {}
    for evidence_base in reversed(output_bases):
        similarity_by_id.update(
            {row["comparison_id"]: row for row in read_csv(evidence_base / "similarity_summary.csv")}
        )
        feasibility_by_id.update(
            {row["dataset_id"]: row for row in read_csv(evidence_base / "feasibility_summary.csv")}
        )
    audit = {}
    for evidence_base in output_bases:
        candidate = load_json(evidence_base / "input_audit_summary.json")
        if candidate:
            audit = candidate
            break

    summary_rows = [collect_row(output_bases, row, similarity_by_id, feasibility_by_id) for row in manifest]
    fields = list(summary_rows[0].keys()) if summary_rows else []
    write_csv(output_base / f"{summary_prefix}_summary.csv", summary_rows, fields)

    agg_rows = method_problem_rows(summary_rows)
    write_csv(
        output_base / f"{summary_prefix}_method_problem_summary.csv",
        agg_rows,
        [
            "method",
            "problem",
            "rows",
            "similarity",
            "feasible_percent",
            "mean_solve_time",
            "mean_nodes",
            "rootgap_w1",
            "cutplane_pc1_w1",
            "heur_w1",
        ],
    )

    metric_keys = [
        "similarity",
        "feasible_percent",
        "mean_solve_time",
        "mean_nodes",
        "rootgap_w1",
        "cutplane_pc1_w1",
        "heur_w1",
    ]
    evidence_keys = [
        "evidence_similarity",
        "evidence_feasibility",
        "evidence_solving_time",
        "evidence_branching_nodes",
        "evidence_rootgap",
        "evidence_cutplane",
        "evidence_heuristic",
    ]
    coverage_report = {
        "manifest_rows": len(manifest),
        "summary_rows": len(summary_rows),
        "problems": sorted({row.get("problem", "") for row in manifest}),
        "evidence_coverage": {key: evidence_coverage(summary_rows, key) for key in evidence_keys},
        "finite_value_coverage": {key: coverage(summary_rows, key) for key in metric_keys},
        "input_audit": audit,
        "tex": str(args.tex),
    }
    (output_base / f"{summary_prefix}_coverage.json").write_text(
        json.dumps(coverage_report, indent=2), encoding="utf-8"
    )

    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text(make_latex(summary_rows, agg_rows, audit), encoding="utf-8")
    print(json.dumps(coverage_report, indent=2))


if __name__ == "__main__":
    main()
