#!/usr/bin/env python
"""Create fixed-sense DIG-MILP CA companion LP files from generated LP files."""

import argparse
import re
from pathlib import Path


TERM_RE = re.compile(r"([+-])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(x\d+)")
VAR_RE = re.compile(r"\bx\d+\b")
CON_RE = re.compile(r"^C\d+:")


def flip_objective(line):
    prefix, rest = line.split(":", 1)

    def repl(match):
        leading_sign, value, var_name = match.groups()
        coefficient = float(value)
        if leading_sign == "-":
            coefficient = -coefficient
        coefficient = -coefficient
        sign = "+" if coefficient >= 0 else "-"
        return f" {sign} {abs(coefficient)} {var_name}"

    return prefix + ":" + TERM_RE.sub(repl, rest)


def convert_file(src, dst):
    lines = src.read_text().splitlines()
    if not lines or lines[0].strip().lower() != "maximize":
        raise ValueError(f"Unexpected first line in {src}")
    if len(lines) < 2 or not lines[1].startswith("OBJ:"):
        raise ValueError(f"Missing objective in {src}")

    binary_pos = next(
        (idx for idx, line in enumerate(lines) if line.strip().lower() == "binary"),
        None,
    )
    if binary_pos is None or binary_pos + 1 >= len(lines):
        raise ValueError(f"Missing binary section in {src}")

    variables = sorted(set(VAR_RE.findall(lines[binary_pos + 1])), key=lambda name: int(name[1:]))
    num_constraints = sum(1 for line in lines if CON_RE.match(line.strip()))
    out_lines = [
        r"\ SCIP STATISTICS",
        r"\   Problem name     : model",
        rf"\   Variables        : {len(variables)} ({len(variables)} binary, 0 integer, 0 implicit integer, 0 continuous)",
        rf"\   Constraints      : {num_constraints}",
        "Minimize",
        flip_objective(lines[1]),
    ]
    out_lines.extend(lines[2:])
    dst.write_text("\n".join(out_lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(args.num_samples):
        src = args.source_dir / f"{idx}.lp"
        dst = args.output_dir / f"instance_{idx + 1}.lp"
        convert_file(src, dst)
    print(f"wrote {len(list(args.output_dir.glob('*.lp')))} files to {args.output_dir}")


if __name__ == "__main__":
    main()
