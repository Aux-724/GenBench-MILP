#!/usr/bin/env python
"""Prepare train/test input directories for ACM-MILP and G2MILP full coverage."""

import argparse
import shutil
from pathlib import Path


DATASETS = {
    "mis": ("independent_set", "*.lp", 900),
    "setcover": ("set_cover", "*.lp", 900),
    "ca": ("combinatorial_auction", "*.lp", 900),
    "cfl": ("capacitated_facility_location", "*.lp", 900),
    "mik": ("mik", "*.mps", 80),
}


def reset_dir(path: Path, force: bool):
    if path.exists() or path.is_symlink():
        if path.is_dir() and any(path.iterdir()) and not force:
            return False
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return True


def link_or_copy(src: Path, dst: Path, copy: bool):
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def prepare_project(project_root: Path, original_root: Path, copy: bool, force: bool, datasets):
    for dataset in datasets:
        source_name, pattern, train_count = DATASETS[dataset]
        files = sorted((original_root / source_name).glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {original_root / source_name}")
        train_files = files[: min(train_count, len(files))]
        test_files = files[min(train_count, len(files)) :]
        if not test_files:
            test_files = files[-min(100, len(files)) :]

        train_dir = project_root / "data" / dataset / "train"
        test_dir = project_root / "data" / dataset / "test"
        if not reset_dir(train_dir, force) or not reset_dir(test_dir, force):
            print(f"{project_root.name}/{dataset}: existing data found, skipping; use --force to rebuild")
            continue

        for src in train_files:
            link_or_copy(src.resolve(), train_dir / src.name, copy)
        for src in test_files:
            link_or_copy(src.resolve(), test_dir / src.name, copy)
        print(f"{project_root.name}/{dataset}: train={len(train_files)} test={len(test_files)}")


def convert_to_lp(src: Path, dst: Path):
    from pyscipopt import Model

    model = Model()
    model.hideOutput()
    model.readProblem(str(src))
    model.writeProblem(str(dst))


def prepare_dig_inputs(dig_root: Path, original_root: Path, copy: bool, force: bool, datasets):
    for dataset in datasets:
        source_name, pattern, train_count = DATASETS[dataset]
        if dataset in {"setcover", "ca"}:
            continue
        files = sorted((original_root / source_name).glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {original_root / source_name}")
        files = files[: min(train_count, len(files))]
        base = dig_root / "data" / "full_coverage" / dataset
        primal_dir = base / "primal_format"
        if not reset_dir(primal_dir, force):
            print(f"{dig_root}/full_coverage/{dataset}: existing primal data found, skipping; use --force to rebuild")
            continue
        for subdir in (
            "primal_solution",
            "primal_slack",
            "dual_solution",
            "dual_slack",
            "normalize_primal_solution",
            "normalize_primal_slack",
            "normalize_dual_solution",
            "normalize_dual_slack",
            "normalize_statistics",
        ):
            reset_dir(base / subdir, force)
        for idx, src in enumerate(files):
            dst = primal_dir / f"{idx}.lp"
            if src.suffix.lower() == ".mps":
                convert_to_lp(src, dst)
            elif copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())
        print(f"{dig_root}/full_coverage/{dataset}: primal={len(files)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, default=Path("/mnt/data1/luoyidong/benchmark_data/data/original"))
    parser.add_argument("--projects", nargs="+", default=["ACM-MILP", "L2O-G2MILP"])
    parser.add_argument("--prepare-dig", action="store_true")
    parser.add_argument("--dig-root", type=Path, default=Path("DIG_MILP/iis"))
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    parser.add_argument("--force", action="store_true", help="Rebuild existing non-empty prepared directories.")
    args = parser.parse_args()

    for project in args.projects:
        prepare_project(Path(project), args.original_root, args.copy, args.force, args.datasets)
    if args.prepare_dig:
        prepare_dig_inputs(args.dig_root, args.original_root, args.copy, args.force, args.datasets)


if __name__ == "__main__":
    main()
