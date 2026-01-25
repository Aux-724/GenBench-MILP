
"""
MILP initial basis prediction GNN model training script

This script is used to train a GNN model to predict the effective initial basis in the LP relaxation process of MILP problems
"""
import os
import sys
import logging
import time
import json
import pickle
import random
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Add the project root directory to the system path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.Initial_basis_prediction.data_utils import MILPDatasetProcessor, split_dataset
from metrics.Initial_basis_prediction.model import InitialBasisGNN
from metrics.Initial_basis_prediction.trainer import GNNTrainer

# Set the log format
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """Set the random seed to ensure reproducibility
    
    Args:
        seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If CUDA is available, set the CUDA-related random seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    os.environ["PYTHONHASHSEED"] = str(seed)

class MILPDataLoader:
    """
    MILP data loader, used to load data in batches
    
    A simplified data loader that supports batch loading and random shuffling
    """
    
    def __init__(self, data_list: List[Dict], batch_size: int = 1, shuffle: bool = False):
        """
        Initialize the data loader
        
        Args:
            data_list: MILP instance data list
            batch_size: batch size
            shuffle: whether to shuffle the data
        """
        self.data_list = data_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(data_list)
        self.indices = list(range(self.n))
        
    def __iter__(self):
        """Return the data batch iterator"""
        # If shuffling is required, reorder the indices
        if self.shuffle:
            random.shuffle(self.indices)
        
        # Iterate over the data in batches
        for start_idx in range(0, self.n, self.batch_size):
            # Ensure the last batch does not exceed the boundary
            end_idx = min(start_idx + self.batch_size, self.n)
            batch_indices = self.indices[start_idx:end_idx]
            
            # Generate the current batch data
            yield [self.data_list[idx] for idx in batch_indices]
            
    def __len__(self):
        """Return the total number of batches. Use ceiling rounding to ensure all data is processed"""
        return (self.n + self.batch_size - 1) // self.batch_size

@hydra.main(config_path="../configs", config_name="initial_basis_prediction")
def main(cfg: DictConfig) -> None:
    """
    Main function: train the MILP initial basis prediction GNN model
    
    Args:
        cfg: Hydra configuration object
    """
    # ==================== Initialization ====================
    # Output configuration information
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    set_seed(cfg.training.seed)
    
    # Create the output directory
    os.makedirs(cfg.training.save_dir, exist_ok=True)
    
    # Set the calculation device
    device = torch.device(cfg.model.device if torch.cuda.is_available() and "cuda" in cfg.model.device else "cpu")
    logger.info(f"Using device: {device}")
    
    # ==================== Data preparation ====================
    # Dataset and cache path
    dataset_dir = cfg.datasets.train_dir
    model_name = "basis_prediction_model"
    logger.info(f"Training dataset: {dataset_dir}")
    
    # Create the cache directory
    cache_dir = Path(cfg.training.save_dir) / "cache"
    cache_dir.mkdir(exist_ok=True)
    processed_data_path = cache_dir / "processed_data.pkl"
    
    # Try to load cached data
    processed_data = None
    if processed_data_path.exists():
        try:
            with open(processed_data_path, "rb") as f:
                processed_data = pickle.load(f)
                if processed_data and len(processed_data) > 0:
                    logger.info(f"Loaded data from cache: {len(processed_data)} instances")
                else:
                    logger.warning(f"Cache file exists but data is empty: {processed_data_path}")
                    processed_data = None
        except Exception as e:
            logger.warning(f"Failed to load cache file: {str(e)}")
            processed_data = None
    
    # If there is no valid cache, process the dataset
    if processed_data is None:
        logger.info(f"Processing dataset: {dataset_dir}")
        
        # Check if the dataset directory exists
        if not Path(dataset_dir).exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
        
        # Process the dataset
        processor = MILPDatasetProcessor(dataset_dir, cache_dir=str(cache_dir))
        processed_data = processor.process_dataset(max_instances=cfg.datasets.max_instances)
        
        # Verify the processed results
        if not processed_data or len(processed_data) == 0:
            raise ValueError(f"The dataset after processing is empty: {dataset_dir}")
        
        # Cache the processed results
        with open(processed_data_path, "wb") as f:
            pickle.dump(processed_data, f)
        logger.info(f"Cached {len(processed_data)} processed instances")
    
    # ==================== Data splitting ====================
    # Split the training set and validation set
    train_data, val_data = split_dataset(
        processed_data, 
        train_ratio=cfg.datasets.train_ratio, 
        val_ratio=cfg.datasets.val_ratio
    )
    
    logger.info(f"Data splitting result: {len(train_data)} training instances, {len(val_data)} validation instances")
    
    # Create data loaders
    train_loader = MILPDataLoader(train_data, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = MILPDataLoader(val_data, batch_size=cfg.training.batch_size)
    
    # ==================== Model creation and training ====================
    # Create the model
    model = InitialBasisGNN(
        var_feat_dim=8,       # Variable node feature dimension
        constr_feat_dim=8,    # Constraint node feature dimension
        hidden_dim=cfg.model.hidden_dim,  # Hidden layer dimension
        num_layers=cfg.model.num_layers,  # Number of GNN layers
        dropout=cfg.model.dropout         # Dropout rate
    )
    
    logger.info(f"Created GNN model: hidden dimension={cfg.model.hidden_dim}, number of layers={cfg.model.num_layers}, feature dimension=8")
    
    # Configure the trainer
    trainer_config = {
        "learning_rate": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "epochs": cfg.training.epochs,
        "early_stopping": cfg.training.early_stopping,
        "save_dir": str(Path(cfg.training.save_dir) / "checkpoints")
    }
    
    # Create the trainer
    trainer = GNNTrainer(model, trainer_config, device=device)
    
    # Start training
    logger.info(f"Training the model (up to {cfg.training.epochs} epochs)...")
    training_results = trainer.train(train_loader, val_loader)
    
    # ==================== Model saving ====================
    # Save the final model
    final_model_path = Path(cfg.training.save_dir) / f"{model_name}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "training_results": training_results,
        "config": OmegaConf.to_container(cfg)
    }, final_model_path)
    
    # Generate and save the model information file
    model_info = {
        "model_path": str(final_model_path),
        "train_dataset": dataset_dir,
        "training_config": OmegaConf.to_container(cfg.training),
        "model_config": OmegaConf.to_container(cfg.model),
        "train_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "train_size": len(train_data),
        "validation_size": len(val_data),
        "best_epoch": training_results["best_epoch"],
        "best_val_loss": float(training_results["best_val_loss"])
    }
    
    model_info_path = Path(cfg.training.save_dir) / "model_info.json"
    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)
    
    # ==================== Result output ====================
    # Training completion information
    best_epoch = training_results["best_epoch"]
    best_val_loss = training_results["best_val_loss"]
    logger.info(f"Model training completed, saved to: {final_model_path}")
    logger.info(f"Best model (epoch {best_epoch}): validation loss = {best_val_loss:.4f}")
    logger.info(f"Model information saved to: {model_info_path}")
    
    return training_results
    
    return training_results

if __name__ == "__main__":
    main()
