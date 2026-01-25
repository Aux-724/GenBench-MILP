# GenBench-MILP

A comprehensive benchmarking framework for evaluating Mixed Integer Linear Programming (MILP) instances and generation methods, with a focus on Learning to Optimize (L2O) applications.

## Overview

This repository provides tools for preprocessing, evaluating, and analyzing MILP instances. It supports various metrics to assess the quality of MILP instances, including feasibility ratio and solving characteristics. The framework now includes GNN-based models for predicting MILP instance properties.

## Requirements

This project depends on Python 3.8+ and Gurobi Optimizer 10.0+. All Python dependencies are listed in the `requirements.txt` file.

You can install all dependencies using the following command:

```bash
pip install -r requirements.txt
```

For some special dependencies:
-   **Gurobi Installation**: Please visit the [Gurobi Official Website](https://www.gurobi.com/downloads/) for a license and installation guide.
-   **PyTorch Geometric (PyG) related dependencies**: Some dependencies may need to be installed separately. For details, see the [PyG Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
-   **Ecole Library**: May need to be built from source. For details, see the [Ecole Installation Guide](https://github.com/ds4dm/ecole).

## Modules

### Instance Synthesis System

Provides flexible MILP instance generation via the `data/synthesize.py` script and YAML configuration files.

**Usage:**

```bash
# Generate an instance set with the default configuration
python -m data.synthesize

# Specify a custom configuration file
python -m data.synthesize --config configs/synthesize.yaml
```

-   **Support for Multiple Problem Types** (More aspects of instances can be found in links):
    -   Set Cover Problem
    -   Capacitated Facility Location Problem
    -   Combinatorial Auction Problem
    -   Independent Set Problem

#### Output Structure

Generated instances are organized by type in the `data/raw/` directory:
```
data/raw/
├── set_cover/
│   ├── instance_0001.mps
│   └── instance_0002.mps
└── facility_location/
    ├── instance_0001.lp
    └── instance_0002.lp
```

---

### Branching Nodes Analysis Tool

Calculates branching node information for MILP instances within a single directory and generates detailed statistical analysis using the `scripts/calculate_branching_nodes.py` script.

**Usage:**

```bash
# Basic usage
python scripts/calculate_branching_nodes.py --instances_dir /path/to/instances --output_dir /path/to/output

# Usage with other parameters
python scripts/calculate_branching_nodes.py --instances_dir /path/to/instances --output_dir /path/to/output --time_limit 60 --parallel True --max_threads 8 --max_instances 100
```

---

### Solver Behavior Analysis System

#### Solver Information Extraction and Analysis

Extracts and analyzes key behavioral metrics of the Gurobi solver when processing MILP instances, including root node Gap, cutting plane statistics, and heuristic usage, via the `scripts/evaluate_solver.py` script.

**Usage:**

```bash
# Basic usage
python scripts/evaluate_solver.py hydra.run.dir=outputs/solver_info/my_dataset

# Specify a different instance directory
python scripts/evaluate_solver.py dataset.instances_dir=/path/to/instances hydra.run.dir=outputs/solver_info/custom_dataset
```

#### Distribution Comparison Tools

##### Root Node Gap Distribution Comparison

Compares the root node Gap distributions of two datasets using the `scripts/compare_solver_rootgap.py` script.

**Usage:**

```bash
# Compare root node Gap distributions of two datasets using default configurations
python scripts/compare_solver_rootgap.py

# Compare using custom CSV file paths
python scripts/compare_solver_rootgap.py paths.csv_file1=/path/to/first.csv paths.csv_file2=/path/to/second.csv
```

##### Heuristic Method Usage Distribution Comparison

Compares the heuristic method usage of two datasets using the `scripts/compare_solver_heur.py` script.

**Usage:**

```bash
# Compare heuristic method usage distributions of two datasets using default configurations
python scripts/compare_solver_heur.py

# Compare using custom CSV file paths and heuristic method column
python scripts/compare_solver_heur.py paths.csv_file1=/path/to/first.csv paths.csv_file2=/path/to/second.csv comparison.heur_column=heur_FoundHeuristic
```

---

### Feasible and Bounded Rate Calculation Tool

Calculates the feasible and bounded rate of MILP instances within a single directory using the `scripts/calculate_feasibility.py` script.

**Usage:**

```bash
# Calculate the feasible and bounded rate for instances in the specified directory using default configurations
python scripts/calculate_feasibility.py

# Specify a different instance directory
python scripts/calculate_feasibility.py single.dir=/path/to/instances
```

---

### Instance Solving Time Analysis Tool

Analyzes the solving time characteristics of MILP instances in a single directory and generates a detailed report using the `scripts/calculate_instance_solving_time.py` script.

**Usage:**

```bash
# Analyze instance solving time in the specified directory using default configurations
python scripts/calculate_instance_solving_time.py paths.instances_dir=data/raw/independent_set

# Limit the number of instances for analysis
python scripts/calculate_instance_solving_time.py paths.instances_dir=data/raw/independent_set calculate_instance.max_instances=20

# Other common parameters
python scripts/calculate_instance_solving_time.py paths.instances_dir=data/raw/independent_set solve.time_limit=60 solve.threads=8
```

---

### Instance Similarity Comparison Tool

Compares the structural similarity of MILP instances between two directories using the `scripts/compare_similarity.py` script.

**Usage:**

```bash
# Compare instance similarity between two directories using default configurations
python scripts/compare_similarity.py

# Modify the directories for comparison
python scripts/compare_similarity.py paths.set1_dir=/path/to/original paths.set2_dir=/path/to/generated

# Modify the output directory for results
python scripts/compare_similarity.py paths.output_dir=/path/to/output_directory
```

---

### Gurobi Hyperparameter Tuning

Evaluates the performance of **Default** (Gurobi default parameters) and tuned parameter configurations on an instance set, comparing solving times and performance improvements.

**Running:**

```bash
# Use default paths from configs/gurobi_param_tuning.yaml
python scripts/run_gurobi_tuning.py

# Override paths for training and generated instances via command line
python scripts/run_gurobi_tuning.py paths.training_instances_dir=/path/to/your/training/set paths.generated_instances_dir=/path/to/your/generated/set

# (Other configuration items can also be overridden via command line, e.g., tuning.n_trials=100)
```

---

### MILP Initial Basis Prediction

**Overview:**

This module implements a GNN (Graph Neural Network)-based model for predicting the initial basis of MILPs. It can predict the optimal initial basis for variables and slack variables in Linear Programs (LPs). Setting a high-quality initial basis can significantly speed up the solving process for MILP problems.

**Model Training:**

Use the `scripts/train_basis_prediction.py` script to train the initial basis prediction GNN model.

**Usage:**

```bash
# Train the model using default configurations
python scripts/train_basis_prediction.py

# Specify a custom dataset directory
python scripts/train_basis_prediction.py datasets.train_dir=/path/to/training/instances

# Customize model and training parameters
python scripts/train_basis_prediction.py model.hidden_dim=256 model.num_layers=4 training.learning_rate=0.0005
```

**Model Evaluation:**

After training, use the `scripts/evaluate_basis_prediction.py` script to evaluate model performance.

**Usage:**

```bash
# Evaluate the default model
python scripts/evaluate_basis_prediction.py

# Specify a custom model and test set
python scripts/evaluate_basis_prediction.py custom_models.model_path=/path/to/model.pth datasets.test_dir=/path/to/test/instances
```

## License

MIT
