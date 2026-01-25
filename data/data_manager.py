"""
Data management module for MILP instance benchmarking.
"""

import os
import shutil
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path

from utils.logger import setup_logger
from utils.common import get_instance_files, count_variables


class DataManager:
    """
    Manager for handling MILP instance datasets.
    
    This class provides functionality for loading, preprocessing, and splitting
    MILP instance datasets.
    """
    
    def __init__(self, 
                data_dir: str,
                cache_dir: Optional[str] = None,
                **kwargs):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Directory containing MILP instances.
            cache_dir: Directory for caching processed data. If None, a default
                      directory will be created within data_dir.
            **kwargs: Additional parameters.
        """
        self.data_dir = os.path.abspath(data_dir)
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(self.data_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        
        # Set up logger
        self.logger = setup_logger("data_manager", log_dir=cache_dir)
        
        self.params = kwargs
    
    def get_problem_types(self) -> List[str]:
        """
        Get a list of available problem types in the data directory.
        
        Returns:
            List of problem type names.
        """
        problem_types = []
        
        for item in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, item)):
                if item != ".cache" and not item.startswith("."):
                    problem_types.append(item)
        
        return problem_types
    
    def load_instances(self, 
                      problem_type: Optional[str] = None,
                      split: Optional[str] = None,
                      size_category: Optional[str] = None) -> List[str]:
        """
        Load MILP instances from the data directory.
        
        Args:
            problem_type: Type of problem (e.g., 'mik', 'mis', 'setcover').
                         If None, instances of all types will be loaded.
            split: Data split to load (e.g., 'train', 'validation', 'test').
                  If None, instances from all splits will be loaded.
            size_category: Size category to load (e.g., 'small', 'medium', 'large').
                          If None, instances of all sizes will be loaded.
            
        Returns:
            List of paths to the loaded instances.
        """
        instance_files = []
        
        # Determine directories to search
        search_dirs = []
        
        if problem_type is None:
            problem_types = self.get_problem_types()
        else:
            problem_types = [problem_type]
        
        for prob_type in problem_types:
            prob_dir = os.path.join(self.data_dir, prob_type)
            
            if not os.path.isdir(prob_dir):
                self.logger.warning(f"Problem type directory not found: {prob_dir}")
                continue
            
            if split is None:
                # Check for split directories
                if any(s in os.listdir(prob_dir) for s in ['train', 'validation', 'test']):
                    for s in ['train', 'validation', 'test']:
                        split_dir = os.path.join(prob_dir, s)
                        if os.path.isdir(split_dir):
                            search_dirs.append(split_dir)
                else:
                    # No split directories
                    search_dirs.append(prob_dir)
            else:
                split_dir = os.path.join(prob_dir, split)
                if os.path.isdir(split_dir):
                    search_dirs.append(split_dir)
                else:
                    self.logger.warning(f"Split directory not found: {split_dir}")
        
        # Get instance files from all search directories
        for search_dir in search_dirs:
            if size_category is None:
                # Check for size category directories
                if any(s in os.listdir(search_dir) for s in ['small', 'medium', 'large']):
                    for s in ['small', 'medium', 'large']:
                        size_dir = os.path.join(search_dir, s)
                        if os.path.isdir(size_dir):
                            instance_files.extend(get_instance_files(size_dir))
                else:
                    # No size category directories
                    instance_files.extend(get_instance_files(search_dir))
            else:
                size_dir = os.path.join(search_dir, size_category)
                if os.path.isdir(size_dir):
                    instance_files.extend(get_instance_files(size_dir))
                else:
                    self.logger.warning(f"Size category directory not found: {size_dir}")
        
        return instance_files
    
    def split_dataset(self, 
                     problem_type: str,
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     seed: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Split a problem type dataset into train, validation, and test sets.
        
        Args:
            problem_type: Type of problem (e.g., 'mik', 'mis', 'setcover').
            train_ratio: Ratio of instances to use for training.
            val_ratio: Ratio of instances to use for validation.
            test_ratio: Ratio of instances to use for testing.
            seed: Random seed for reproducibility.
            
        Returns:
            Dictionary mapping split names to lists of instance paths.
        """
        if train_ratio + val_ratio + test_ratio != 1.0:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Get all instances for the problem type
        all_instances = self.load_instances(problem_type=problem_type)
        
        if not all_instances:
            self.logger.warning(f"No instances found for problem type: {problem_type}")
            return {"train": [], "validation": [], "test": []}
        
        # Shuffle instances
        random.shuffle(all_instances)
        
        # Calculate split indices
        n = len(all_instances)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split instances
        train_instances = all_instances[:train_end]
        val_instances = all_instances[train_end:val_end]
        test_instances = all_instances[val_end:]
        
        return {
            "train": train_instances,
            "validation": val_instances,
            "test": test_instances
        }
    
    def organize_by_size(self, 
                        problem_type: str,
                        output_dir: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Organize instances by size category.
        
        Args:
            problem_type: Type of problem (e.g., 'mik', 'mis', 'setcover').
            output_dir: Directory where organized instances will be saved.
                       If None, instances will be organized in-place.
            
        Returns:
            Dictionary mapping size categories to lists of instance paths.
        """
        # Get all instances for the problem type
        all_instances = self.load_instances(problem_type=problem_type)
        
        if not all_instances:
            self.logger.warning(f"No instances found for problem type: {problem_type}")
            return {"small": [], "medium": [], "large": []}
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, problem_type, "classified")
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, "small"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "medium"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "large"), exist_ok=True)
        
        # Classify instances
        small_instances = []
        medium_instances = []
        large_instances = []
        
        for instance_path in all_instances:
            # Count variables
            num_vars = count_variables(instance_path)
            
            # Classify instance
            size_category = self._classify_instance(num_vars, problem_type)
            
            # Copy to appropriate directory
            filename = os.path.basename(instance_path)
            dest_path = os.path.join(output_dir, size_category, filename)
            
            shutil.copy2(instance_path, dest_path)
            
            # Add to results
            if size_category == "small":
                small_instances.append(dest_path)
            elif size_category == "medium":
                medium_instances.append(dest_path)
            else:  # large
                large_instances.append(dest_path)
        
        return {
            "small": small_instances,
            "medium": medium_instances,
            "large": large_instances
        }
    
    def _classify_instance(self, num_vars: int, problem_type: str) -> str:
        """
        Classify an instance based on its size.
        
        Args:
            num_vars: Number of variables in the instance.
            problem_type: Type of problem.
            
        Returns:
            Size category ('small', 'medium', or 'large').
        """
        if problem_type == 'mis':
            if num_vars < 200:
                return 'small'
            elif num_vars <= 1000:
                return 'medium'
            else:
                return 'large'
        elif problem_type == 'mik':  
            if num_vars < 500:
                return 'small'
            elif num_vars <= 2000:
                return 'medium'
            else:
                return 'large'
        else:  # setcover
            if num_vars < 500:
                return 'small'
            elif num_vars <= 2000:
                return 'medium'
            else:
                return 'large'
    
    def save_dataset_info(self, dataset_path: str, info: Dict[str, Any]) -> None:
        """
        Save dataset information to a JSON file.
        
        Args:
            dataset_path: Path to the dataset.
            info: Dictionary containing dataset information.
        """
        # Create info directory if it doesn't exist
        info_dir = os.path.join(os.path.dirname(dataset_path), "info")
        os.makedirs(info_dir, exist_ok=True)
        
        # Create info file path
        dataset_name = os.path.basename(dataset_path)
        info_path = os.path.join(info_dir, f"{dataset_name}_info.json")
        
        # Save info as JSON
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def load_dataset_info(self, dataset_path: str) -> Optional[Dict[str, Any]]:
        """
        Load dataset information from a JSON file.
        
        Args:
            dataset_path: Path to the dataset.
            
        Returns:
            Dictionary containing dataset information, or None if not found.
        """
        # Create info file path
        dataset_name = os.path.basename(dataset_path)
        info_dir = os.path.join(os.path.dirname(dataset_path), "info")
        info_path = os.path.join(info_dir, f"{dataset_name}_info.json")
        
        # Check if info file exists
        if not os.path.exists(info_path):
            return None
        
        # Load info from JSON
        with open(info_path, 'r') as f:
            return json.load(f)
