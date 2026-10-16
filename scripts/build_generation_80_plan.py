#!/usr/bin/env python
"""Build a command plan for producing the full 80 generated-instance groups."""

import argparse
import csv
import shlex
from pathlib import Path

from verify_generation_coverage import ETAS, PROBLEMS, verify


RATIO_BY_ETA = {
    "eta0.01": "0.01",
    "eta0.05": "0.05",
    "eta0.10": "0.1",
    "eta0.20": "0.2",
}

GEN_NAME = {
    "acm-milp": "acm-milp",
    "dig-milp": "dig-milp",
    "g2milp": "g2milp",
    "milp-evolve": "milp-evolve",
}


def q(value):
    return shlex.quote(str(value))


def eta_dir(method: str, eta: str):
    if method == "milp-evolve":
        return eta
    if eta == "eta0.10":
        return "eta0.10"
    if eta == "eta0.20":
        return "eta0.20"
    return eta


def generated_dir(data_root: Path, method: str, problem: str, eta: str):
    problem_name = "is" if method == "milp-evolve" and problem == "mis" else problem
    return data_root / "generated" / GEN_NAME[method] / problem_name / eta_dir(method, eta)


def acm_or_g2_commands(method: str, problem: str, etas, args):
    project = "ACM-MILP" if method == "acm-milp" else "L2O-G2MILP"
    work_dir = Path(project).resolve()
    model_dir = work_dir / "outputs" / "full80" / problem / "model"
    train_dir = work_dir / "outputs" / "full80" / problem
    hydra_train_dir = work_dir / "outputs" / "full80" / problem / "hydra_train"
    train_overrides = []
    if method == "acm-milp" and args.acm_steps:
        train_overrides.extend(
            [
                f"trainer.steps={args.acm_steps}",
                f"trainer.save_step={args.acm_save_step}",
                f"trainer.save_start_stage_1={args.acm_save_start}",
                f"trainer.save_start_stage_2={args.acm_save_start}",
            ]
        )
    if method == "g2milp" and args.g2_steps:
        train_overrides.extend(
            [
                f"trainer.steps={args.g2_steps}",
                f"trainer.save_step={args.g2_save_step}",
                f"trainer.save_start={args.g2_save_start}",
            ]
        )
    train_override_text = " " + " ".join(train_overrides) if train_overrides else ""
    commands = [
        {
            "stage": "preprocess",
            "method": method,
            "problem": problem,
            "eta": "",
            "command": (
                f"cd {q(work_dir)} && conda run -n {q(args.env)} python preprocess.py "
                f"dataset={q(problem)} num_workers={args.num_workers} solve_instances=false"
            ),
        },
        {
            "stage": "train",
            "method": method,
            "problem": problem,
            "eta": "",
            "command": (
                f"cd {q(work_dir)} && conda run -n {q(args.env)} python train.py "
                f"dataset={q(problem)} job_name={q('full80_' + problem)} cuda={args.cuda} "
                f"num_workers={args.num_workers} skip_final_eval=true "
                f"hydra.run.dir={q(hydra_train_dir)} "
                f"paths.train_dir={q(train_dir)} paths.model_dir={q(model_dir)} "
                f"paths.samples_dir={q(train_dir / 'samples')} "
                f"paths.benchmark_dir={q(train_dir / 'benchmark')} "
                f"paths.tensorboard_dir={q(train_dir / 'tensorboard')}"
                f"{train_override_text}"
            ),
        },
    ]
    for eta in etas:
        out_dir = generated_dir(args.data_root, method, problem, eta)
        commands.append(
            {
                "stage": "generate",
                "method": method,
                "problem": problem,
                "eta": eta,
                "command": (
                    f"mkdir -p {q(out_dir)} && cd {q(work_dir)} && "
                    f"conda run -n {q(args.env)} python generate.py "
                    f"dataset={q(problem)} cuda={args.cuda} num_workers={args.num_workers} "
                    f"dir={q(train_dir)} generator.mask_ratio={RATIO_BY_ETA[eta]} "
                    f"generator.num_samples={args.num_instances} skip_benchmark=true "
                    f"paths.samples_dir={q(out_dir)} "
                    f"paths.benchmark_dir={q(train_dir / ('benchmark_' + eta))}"
                ),
            }
        )
    return commands


def dig_commands(problem: str, etas, args):
    base = Path("DIG_MILP/iis").resolve()
    data_base = base / "data" / "full_coverage" / problem
    primal = data_base / "primal_format"
    graph_dataset = base / "graph_dataset" / "full_coverage" / problem
    train_folder = base / "train_files" / f"full80_{problem}"
    common = (
        f"--primal_format {q(primal)}/ --num_instance {args.dig_num_instances} "
        f"--save_folder {q(graph_dataset)}/ "
        f"--primal_solution {q(data_base / 'normalize_primal_solution')}/ "
        f"--primal_slack {q(data_base / 'normalize_primal_slack')}/ "
        f"--dual_solution {q(data_base / 'normalize_dual_solution')}/ "
        f"--dual_slack {q(data_base / 'normalize_dual_slack')}/ "
        f"--normalize_statistics {q(data_base / 'normalize_statistics')}/"
    )
    commands = [
        {
            "stage": "dig_primal",
            "method": "dig-milp",
            "problem": problem,
            "eta": "",
            "command": (
                f"cd {q(base)} && conda run -n {q(args.env)} python _2_infer_scip_primal.py "
                f"--primal_folder {q(primal)}/ --solution_folder {q(data_base / 'primal_solution')}/ "
                f"--slack_folder {q(data_base / 'primal_slack')}/ --n_jobs {args.dig_solver_jobs}"
            ),
        },
        {
            "stage": "dig_dual",
            "method": "dig-milp",
            "problem": problem,
            "eta": "",
            "command": (
                f"cd {q(base)} && conda run -n {q(args.env)} python _4_1_infer_scip_dual_solution.py "
                f"--primal_folder {q(primal)}/ --solution_folder {q(data_base / 'dual_solution')}/ "
                f"--num_instance {args.dig_num_instances} --n_jobs {args.dig_solver_jobs}"
            ),
        },
        {
            "stage": "dig_dual_slack",
            "method": "dig-milp",
            "problem": problem,
            "eta": "",
            "command": (
                f"cd {q(base)} && conda run -n {q(args.env)} python _4_2_infer_scip_dual_slack.py "
                f"--primal_folder {q(primal)}/ --solution_folder {q(data_base / 'dual_solution')}/ "
                f"--slack_folder {q(data_base / 'dual_slack')}/ --num_instance {args.dig_num_instances} "
                f"--n_jobs {args.dig_solver_jobs}"
            ),
        },
        {
            "stage": "dig_normalize",
            "method": "dig-milp",
            "problem": problem,
            "eta": "",
            "command": (
                f"cd {q(base)} && conda run -n {q(args.env)} python _5_normalize_nodes.py "
                f"--primal_folder {q(primal)}/ --num_instance {args.dig_num_instances} "
                f"--primal_solution {q(data_base / 'primal_solution')}/ "
                f"--primal_slack {q(data_base / 'primal_slack')}/ "
                f"--dual_solution {q(data_base / 'dual_solution')}/ "
                f"--dual_slack {q(data_base / 'dual_slack')}/ "
                f"--normalize_primal_solution {q(data_base / 'normalize_primal_solution')}/ "
                f"--normalize_primal_slack {q(data_base / 'normalize_primal_slack')}/ "
                f"--normalize_dual_solution {q(data_base / 'normalize_dual_solution')}/ "
                f"--normalize_dual_slack {q(data_base / 'normalize_dual_slack')}/ "
                f"--normalize_statistics {q(data_base / 'normalize_statistics')}/"
            ),
        },
        {
            "stage": "dig_weight",
            "method": "dig-milp",
            "problem": problem,
            "eta": "",
            "command": (
                f"cd {q(base)} && conda run -n {q(args.env)} python _6_normalize_weight.py "
                f"--primal_folder {q(primal)}/ --normalize_statistics {q(data_base / 'normalize_statistics')}/ "
                f"--num_instance {args.dig_num_instances}"
            ),
        },
        {
            "stage": "dig_graph",
            "method": "dig-milp",
            "problem": problem,
            "eta": "",
            "command": f"cd {q(base)} && conda run -n {q(args.env)} python _7_generate_vcgraph.py {common}",
        },
        {
            "stage": "dig_degree",
            "method": "dig-milp",
            "problem": problem,
            "eta": "",
            "command": (
                f"cd {q(base)} && conda run -n {q(args.env)} python _8_normalize_degree.py {common}"
            ),
        },
        {
            "stage": "dig_train",
            "method": "dig-milp",
            "problem": problem,
            "eta": "",
            "command": (
                f"mkdir -p {q(train_folder)} && cd {q(base)} && conda run -n {q(args.env)} python _9_train.py "
                f"{common} --train_folder {q(train_folder)}/ --gpu {args.cuda} --epoch {args.dig_epochs} "
                f"--batch {args.dig_batch}"
            ),
        },
    ]
    for eta in etas:
        out_dir = generated_dir(args.data_root, "dig-milp", problem, eta)
        sol_dir = args.data_root / "generated" / "dig-milp" / f"{problem}_solutions" / eta_dir("dig-milp", eta)
        commands.append(
            {
                "stage": "dig_generate",
                "method": "dig-milp",
                "problem": problem,
                "eta": eta,
                "command": (
                    f"mkdir -p {q(out_dir)} {q(sol_dir)} && cd {q(base)} && "
                    f"conda run -n {q(args.env)} python _10_generate.py {common} "
                    f"--train_folder {q(train_folder)}/ --generate_folder {q(out_dir)}/ "
                    f"--solution_folder {q(sol_dir)}/ --eta {RATIO_BY_ETA[eta]} "
                    f"--device {q(args.dig_device)} --num_samples {args.num_instances} --skip_existing 1"
                ),
            }
        )
    return commands


def evolve_commands(problem: str, etas, args):
    eta_values = " ".join(eta.replace("eta", "") for eta in etas)
    return [
        {
            "stage": "generate",
            "method": "milp-evolve",
            "problem": problem,
            "eta": ",".join(etas),
            "command": (
                f"conda run -n {q(args.env)} python scripts/generate_milp_evolve_instances.py "
                f"--problems {q('is' if problem == 'mis' else problem)} "
                f"--etas {eta_values} "
                f"--num-instances {args.num_instances}"
            ),
        }
    ]


def build_plan(args):
    rows = verify(args.data_root, args.num_instances)
    missing = [row for row in rows if row["status"] != "ok"]
    by_method_problem = {}
    for row in missing:
        by_method_problem.setdefault((row["method"], row["problem"]), []).append(row["eta"])

    commands = [
        {
            "stage": "prepare_inputs",
            "method": "all",
            "problem": "",
            "eta": "",
            "command": (
                f"conda run -n {q(args.env)} python scripts/prepare_generation_inputs.py "
                f"--prepare-dig"
            ),
        }
    ]
    for (method, problem), etas in sorted(by_method_problem.items()):
        if method in {"acm-milp", "g2milp"}:
            commands.extend(acm_or_g2_commands(method, problem, etas, args))
        elif method == "dig-milp":
            commands.extend(dig_commands(problem, etas, args))
        elif method == "milp-evolve":
            commands.extend(evolve_commands(problem, etas, args))
    return commands


def write_plan(commands, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "generation_80_commands.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "method", "problem", "eta", "command"])
        writer.writeheader()
        writer.writerows(commands)
    shell_path = output_dir / "generation_80_commands.sh"
    with shell_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for command in commands:
            handle.write(f"# {command['stage']} {command['method']} {command['problem']} {command['eta']}\n")
            handle.write(command["command"] + "\n\n")
    return csv_path, shell_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data1/luoyidong/benchmark_data/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generation_coverage"))
    parser.add_argument("--env", default="b4milp")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--num-instances", type=int, default=1000)
    parser.add_argument("--dig-num-instances", type=int, default=1000)
    parser.add_argument("--dig-solver-jobs", type=int, default=32)
    parser.add_argument("--dig-epochs", type=int, default=200)
    parser.add_argument("--dig-batch", type=int, default=128)
    parser.add_argument("--dig-device", default="cpu")
    parser.add_argument("--acm-steps", type=int, default=None)
    parser.add_argument("--acm-save-step", type=int, default=100000)
    parser.add_argument("--acm-save-start", type=int, default=100000)
    parser.add_argument("--g2-steps", type=int, default=None)
    parser.add_argument("--g2-save-step", type=int, default=100000)
    parser.add_argument("--g2-save-start", type=int, default=100000)
    args = parser.parse_args()

    if not 1 <= args.dig_solver_jobs <= 150:
        parser.error("--dig-solver-jobs must be between 1 and 150")

    commands = build_plan(args)
    csv_path, shell_path = write_plan(commands, args.output_dir)
    print(f"wrote {len(commands)} commands to {csv_path} and {shell_path}")


if __name__ == "__main__":
    main()
