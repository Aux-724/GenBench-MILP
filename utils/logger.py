"""
Logging utilities for the benchmark framework.
"""

import os
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any


def setup_logger(name: str, 
                log_dir: Optional[str] = None, 
                level: int = logging.INFO,
                console_output: bool = True) -> logging.Logger:
    """
    Set up a logger with the given name and configuration.
    
    Args:
        name: Logger name.
        log_dir: Directory where log files will be stored. If None, logs will
                only be output to console.
        level: Logging level.
        console_output: Whether to output logs to console.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Define log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


class ExperimentLogger:
    """
    Logger for tracking experiment progress and performance.
    """
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        """
        Initialize the experiment logger.
        
        Args:
            experiment_name: Name of the experiment.
            log_dir: Directory where logs will be stored.
        """
        self.experiment_name = experiment_name
        self.start_time = time.time()
        self.log_dir = os.path.join(log_dir, experiment_name)
        
        # Create experiment log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger(
            f"experiment_{experiment_name}",
            log_dir=self.log_dir
        )
        
        self.step_times: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}
    
    def log_start(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Log the start of an experiment.
        
        Args:
            config: Experiment configuration to log.
        """
        self.logger.info(f"Started experiment: {self.experiment_name}")
        if config:
            self.logger.info(f"Configuration: {config}")
    
    def log_step(self, step_name: str, message: str = "") -> None:
        """
        Log the start of a step in the experiment.
        
        Args:
            step_name: Name of the step.
            message: Additional message to log.
        """
        self.step_times[step_name] = time.time()
        log_msg = f"Starting step: {step_name}"
        if message:
            log_msg += f" - {message}"
        self.logger.info(log_msg)
    
    def log_step_end(self, step_name: str, message: str = "") -> float:
        """
        Log the end of a step and return the elapsed time.
        
        Args:
            step_name: Name of the step.
            message: Additional message to log.
            
        Returns:
            Elapsed time for the step in seconds.
        """
        if step_name not in self.step_times:
            self.logger.warning(f"Step '{step_name}' was not started")
            return 0.0
        
        elapsed = time.time() - self.step_times[step_name]
        log_msg = f"Completed step: {step_name} (Elapsed: {elapsed:.2f}s)"
        if message:
            log_msg += f" - {message}"
        self.logger.info(log_msg)
        return elapsed
    
    def log_metric(self, metric_name: str, value: Any) -> None:
        """
        Log a metric value.
        
        Args:
            metric_name: Name of the metric.
            value: Metric value.
        """
        self.metrics[metric_name] = value
        self.logger.info(f"Metric - {metric_name}: {value}")
    
    def log_exception(self, exception: Exception) -> None:
        """
        Log an exception.
        
        Args:
            exception: Exception to log.
        """
        self.logger.exception(f"Exception occurred: {str(exception)}")
    
    def log_end(self) -> float:
        """
        Log the end of the experiment and return the total elapsed time.
        
        Returns:
            Total elapsed time for the experiment in seconds.
        """
        elapsed = time.time() - self.start_time
        self.logger.info(f"Experiment completed in {elapsed:.2f}s")
        if self.metrics:
            self.logger.info(f"Final metrics: {self.metrics}")
        return elapsed
