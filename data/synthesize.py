
"""
A script for synthesizing MILP instances using the ecole library.
Supports generating four types of MILP instances:
- Set Cover (SC)
- Capacitated Facility Location (CFL)
- Combinatorial Auction (CA)
- Independent Set (IS)

The generated instances will be saved in .lp file format for further processing.
"""

import os
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from omegaconf import DictConfig, OmegaConf

import hydra
import numpy as np
import ecole
from tqdm import tqdm

# Set logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseProblemGenerator:
    """Base problem generator providing common functionalities"""
    
    def __init__(
        self,
        problem_type: str,
        output_dir: str,
        seed: int = 42,
    ):
        """
        Initialize the generator
        
        Args:
            problem_type: Problem type identifier
            output_dir: Output directory
            seed: Random seed
        """
        self.problem_type = problem_type
        self.output_dir = Path(output_dir) / problem_type
        self.seed = seed
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_instance(self, **kwargs) -> Tuple[ecole.scip.Model, Dict[str, Any]]:
        """Generate an instance, must be implemented in subclass"""
        raise NotImplementedError
    
    def save_instance(self, model: ecole.scip.Model, filename: str) -> str:
        """Save instance as LP file"""
        filepath = os.path.join(self.output_dir, filename)
        model.write_problem(filepath)
        return filepath
    
    def generate_and_save(
        self,
        config: DictConfig,
        **kwargs
    ) -> List[str]:
        """Generate and save multiple instances"""
        filepaths = []
        
        # Get problem-specific config or use general config
        problem_config = config.problems.get(self.problem_type, {})
        
        # Determine parameter ranges, prefer problem-specific config, then global config
        min_constraints = problem_config.get('min_constraints', config.size_ranges.min_constraints)
        max_constraints = problem_config.get('max_constraints', config.size_ranges.max_constraints)
        min_variables = problem_config.get('min_variables', config.size_ranges.min_variables)
        max_variables = problem_config.get('max_variables', config.size_ranges.max_variables)
        min_density = problem_config.get('min_density', config.size_ranges.min_density)
        max_density = problem_config.get('max_density', config.size_ranges.max_density)
        
        n_constraints_range = (min_constraints, max_constraints)
        n_variables_range = (min_variables, max_variables) 
        density_range = (min_density, max_density)
        
        # Get number of instances
        num_instances = config.general.num_instances
        
        for i in tqdm(range(num_instances), desc=f"Generating {self.problem_type} instances"):
            # Randomly select parameters
            n_constraints = random.randint(*n_constraints_range)
            n_variables = random.randint(*n_variables_range)
            density = random.uniform(*density_range)
            
            # Generate instance
            try:
                instance_seed = self.seed + i  # Use a different seed for each instance
                model, params = self.generate_instance(
                    n_constraints=n_constraints,
                    n_variables=n_variables,
                    density=density,
                    seed=instance_seed,
                    **kwargs
                )
                
                # Build filename
                param_str = "_".join([f"{k}-{v}" for k, v in params.items()])
                filename = f"{self.problem_type}_{param_str}_{i}.lp"
                
                # Save instance
                filepath = self.save_instance(model, filename)
                filepaths.append(filepath)
                
            except Exception as e:
                logger.error(f"Failed to generate instance {i}: {e}")
                continue
                
        return filepaths


class SetCoverGenerator(BaseProblemGenerator):
    """Set Cover problem generator"""
    
    def __init__(self, output_dir: str, seed: int = 42):
        super().__init__("set_cover", output_dir, seed)
    
    def generate_instance(
        self,
        n_constraints: int,
        n_variables: int,
        density: float,
        seed: int,
        **kwargs
    ) -> Tuple[ecole.scip.Model, Dict[str, Any]]:
        """
        Generate Set Cover instance
        
        Args:
            n_constraints: Number of constraints (elements to cover)
            n_variables: Number of variables (available sets)
            density: Density (probability each set covers an element)
            seed: Random seed
        Returns:
            Generated model and instance parameters
        """
        # Set Ecole global seed
        ecole.seed(seed)
        
        # Create generator
        generator = ecole.instance.SetCoverGenerator(
            n_rows=n_constraints,
            n_cols=n_variables,
            density=density,
        )
        
        # Synthesize instances
        model = next(generator)
        
        params = {
            "n_cons": n_constraints,
            "n_vars": n_variables,
            "dens": round(density, 2)
        }
        
        return model, params


class CapacitatedFacilityLocationGenerator(BaseProblemGenerator):
    """Capacitated Facility Location problem generator"""
    
    def __init__(self, output_dir: str, seed: int = 42):
        super().__init__("capacitated_facility_location", output_dir, seed)
    
    def generate_instance(
        self,
        n_constraints: int,
        n_variables: int,
        density: float,
        seed: int,
        ratio: Optional[float] = None,
        **kwargs
    ) -> Tuple[ecole.scip.Model, Dict[str, Any]]:
        """
        Generate Capacitated Facility Location instance
        
        Args:
            n_constraints: Affects total number of constraints (actual number may be larger)
            n_variables: Affects total number of variables (actual number may be larger)
            density: Not directly used, kept for consistency
            seed: Random seed
            ratio: The ratio of facilities to customers
            
        Returns:
            The parameters
        """
        # Set Ecole global seed
        ecole.seed(seed)
        
        # Convert n_constraints and n_variables to facility and customer counts
        n_customers = max(int(n_constraints * 0.8), 5)  # Ensure at least 5 customers
        
        # If ratio is None, use default value 0.5
        ratio = 0.5 if ratio is None else ratio
        n_facilities = max(int(n_customers * ratio), 2)  # Ensure at least 2 facilities
        
        # Create generator
        generator = ecole.instance.CapacitatedFacilityLocationGenerator(
            n_customers=n_customers,
            n_facilities=n_facilities,
            ratio=ratio,
        )
        
        # Synthesize instances
        model = next(generator)
        
        # Use known parameters to estimate constraint and variable counts
        n_cons_actual = n_customers + n_facilities  # Constraint count estimate
        n_vars_actual = n_customers * n_facilities  # Variable count estimate
        
        params = {
            "n_custs": n_customers,
            "n_facs": n_facilities,
            "n_cons": n_cons_actual,
            "n_vars": n_vars_actual,
            "ratio": round(ratio, 2)
        }
        
        return model, params


class CombinatorialAuctionGenerator(BaseProblemGenerator):
    """Combinatorial Auction problem generator"""
    
    def __init__(self, output_dir: str, seed: int = 42):
        super().__init__("combinatorial_auction", output_dir, seed)
    
    def generate_instance(
        self,
        n_constraints: int,
        n_variables: int,
        density: float,
        seed: int,
        n_items: Optional[int] = None,
        **kwargs
    ) -> Tuple[ecole.scip.Model, Dict[str, Any]]:
        """
        Generate Combinatorial Auction instance
        
        Args:
            n_constraints: Number of constraints (affects generated model)
            n_variables: Number of variables (affects generated model)
            density: Density (affects bidders' interest in items)
            seed: Random seed
            n_items: Number of items, if None will be auto-calculated
        Returns:
            Generated model and instance parameters
        """
        # Set Ecole global seed
        ecole.seed(seed)
        
        # If n_items is not specified, calculate a reasonable number of items based on constraint and variable counts
        if n_items is None:
            n_items = max(int(min(n_constraints, n_variables) * 0.3), 5)
        
        # Calculate the number of bidders (bids)
        n_bids = max(int(n_variables * 0.8), 10)
        
        # Create generator
        generator = ecole.instance.CombinatorialAuctionGenerator(
            n_items=n_items,
            n_bids=n_bids,
            min_value=1,
            max_value=100,
        )
        
        # Synthesize instances
        model = next(generator)
        
        # Use known parameters to estimate constraint and variable counts
        n_cons_actual = n_items + 1  # Constraint count estimate (one constraint per item plus one overall constraint)
        n_vars_actual = n_bids  # Variable count estimate (one variable per bid)
        
        params = {
            "n_items": n_items,
            "n_bids": n_bids,
            "n_cons": n_cons_actual,
            "n_vars": n_vars_actual
        }
        
        return model, params


class IndependentSetGenerator(BaseProblemGenerator):
    """Independent Set problem generator"""
    
    def __init__(self, output_dir: str, seed: int = 42):
        super().__init__("independent_set", output_dir, seed)
    
    def generate_instance(
        self,
        n_constraints: int,
        n_variables: int,
        density: float,
        seed: int,
        **kwargs
    ) -> Tuple[ecole.scip.Model, Dict[str, Any]]:
        """
        Generate Independent Set instance
        
        Args:
            n_constraints: Not directly used, kept for consistency
            n_variables: Number of nodes
            density: Edge density of the graph
            seed: Random seed
        Returns:
            Generated model and instance parameters
        """
        # Set Ecole global seed
        ecole.seed(seed)
        
        # Use n_variables as the number of nodes
        n_nodes = n_variables
        
        # Create generator
        generator = ecole.instance.IndependentSetGenerator(
            n_nodes=n_nodes,
            edge_probability=density,
        )
        
        # Synthesize instances
        model = next(generator)
        
        # Use known parameters to estimate constraint and variable counts
        expected_edges = int(n_nodes * (n_nodes - 1) / 2 * density)
        n_cons_actual = expected_edges  # One constraint per edge
        n_vars_actual = n_nodes  # One variable per node
        
        params = {
            "n_nodes": n_nodes,
            "edge_prob": round(density, 2),
            "n_cons": n_cons_actual,
            "n_vars": n_vars_actual
        }
        
        return model, params


@hydra.main(version_base=None, config_path="../configs", config_name="synthesize")
def synthesize(config: DictConfig) -> None:
    """Generate all types of problem instances"""
    logger.info("Configuration info:\n" + OmegaConf.to_yaml(config))
    
    output_dir = config.output_dir
    
    # Set random seed
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Problem generator mapping
    problem_map = {
        "set_cover": SetCoverGenerator,
        "capacitated_facility_location": CapacitatedFacilityLocationGenerator,
        "combinatorial_auction": CombinatorialAuctionGenerator,
        "independent_set": IndependentSetGenerator
    }
    
    # Generate enabled problem instances
    for prob_type, prob_class in problem_map.items():
        prob_config = config.problems.get(prob_type, {})
        
        # Check if this problem type is enabled
        if not prob_config.get("enabled", True):
            logger.info(f"Skip generating {prob_type} instances (disabled)")
            continue
        
        logger.info(f"Start generating {prob_type} instances...")
        
        # Create generator instance
        generator = prob_class(output_dir, config.seed)
        
        # Prepare problem-specific arguments
        specific_args = {}
        
        # Add problem-specific arguments
        if prob_type == "capacitated_facility_location" and "ratio" in prob_config:
            specific_args["ratio"] = prob_config.ratio
        elif prob_type == "combinatorial_auction":
            if "n_items" in prob_config:
                specific_args["n_items"] = prob_config.n_items
            if "min_value" in prob_config and "max_value" in prob_config:
                specific_args["min_value"] = prob_config.min_value
                specific_args["max_value"] = prob_config.max_value
        
        # Synthesize instances
        generator.generate_and_save(config, **specific_args)
    
    logger.info("All instances generated!")


# The parse_args() function is no longer needed after using Hydra


if __name__ == "__main__":
    synthesize()
