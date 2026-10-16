#!/usr/bin/env python
"""Verify generated-instance coverage for the 4 x 4 x 5 benchmark grid."""

import argparse
import csv
import json
from pathlib import Path


METHODS = ("acm-milp", "dig-milp", "g2milp", "milp-evolve")
PROBLEMS = ("mis", "setcover", "ca", "cfl", "mik")
ETAS = ("eta0.01", "eta0.05", "eta0.10", "eta0.20")
INSTANCE_EXTS = (".lp", ".mps", ".lp.gz", ".mps.gz")

ALIASES = {
    "acm-milp": {
        "eta0.10": ("eta0.10", "eta0.1"),
        "eta0.20": ("eta0.20", "eta0.2"),
    },
    "dig-milp": {
        "eta0.10": ("eta0.10", "eta0.1"),
        "eta0.20": ("fixed_eta0.20", "eta0.20", "eta0.2"),
    },
    "g2milp": {
        "eta0.10": ("eta0.10", "eta0.1"),
        "eta0.20": ("eta0.20", "eta0.2"),
    },
    "milp-evolve": {
        "eta0.10": ("eta0.10", "eta0.1"),
        "eta0.20": ("eta0.20", "eta0.2"),
    },
}

PROBLEM_ALIASES = {
    "mis": ("mis", "is"),
    "setcover": ("setcover", "set_cover", "sc"),
    "ca": ("ca", "cauctions"),
    "cfl": ("cfl", "capacitated_facility_location"),
    "mik": ("mik",),
}


def eta_aliases(method: str, problem: str, eta: str):
    aliases = ALIASES.get(method, {}).get(eta, (eta,))
    if method == "dig-milp" and problem != "ca":
        aliases = tuple(alias for alias in aliases if not alias.startswith("fixed_"))
    return aliases


def resolve_dir(
    generated_root: Path,
    method: str,
    problem: str,
    eta: str,
    require_dig_gurobi_fixed: bool = False,
):
    for problem_alias in PROBLEM_ALIASES[problem]:
        base = generated_root / method / problem_alias
        if method == "dig-milp" and require_dig_gurobi_fixed:
            candidate = base / f"gurobi_fixed_{eta}"
            if candidate.exists():
                return candidate
            continue
        for eta_alias in eta_aliases(method, problem, eta):
            candidate = base / eta_alias
            if candidate.exists():
                return candidate
    return generated_root / method / PROBLEM_ALIASES[problem][0] / eta_aliases(method, problem, eta)[0]


def instance_files(directory: Path):
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and any(path.name.endswith(ext) for ext in INSTANCE_EXTS)
    )


def verify(data_root: Path, expected_count: int, require_dig_gurobi_fixed: bool = False):
    generated_root = data_root / "generated"
    rows = []
    for method in METHODS:
        for problem in PROBLEMS:
            for eta in ETAS:
                directory = resolve_dir(
                    generated_root,
                    method,
                    problem,
                    eta,
                    require_dig_gurobi_fixed=require_dig_gurobi_fixed,
                )
                files = instance_files(directory)
                rows.append(
                    {
                        "method": method,
                        "problem": problem,
                        "eta": eta,
                        "generated_dir": str(directory),
                        "exists": directory.exists(),
                        "n_instances": len(files),
                        "status": "ok" if directory.exists() and len(files) >= expected_count else "missing",
                    }
                )
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data1/luoyidong/benchmark_data/data"))
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("outputs/generation_coverage/coverage_80.csv"))
    parser.add_argument(
        "--require-dig-gurobi-fixed",
        action="store_true",
        help="Resolve DIG rows from gurobi_fixed_eta* directories used by the main evaluation.",
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    rows = verify(
        args.data_root,
        args.expected_count,
        require_dig_gurobi_fixed=args.require_dig_gurobi_fixed,
    )
    write_csv(args.output, rows)
    summary = {
        "expected_groups": len(rows),
        "complete_groups": sum(row["status"] == "ok" for row in rows),
        "missing_groups": sum(row["status"] != "ok" for row in rows),
        "expected_count": args.expected_count,
        "require_dig_gurobi_fixed": args.require_dig_gurobi_fixed,
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.fail_on_missing and summary["missing_groups"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
