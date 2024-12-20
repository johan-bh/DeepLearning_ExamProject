import logging
from pathlib import Path
from datasets import load_from_disk
import os

logger = logging.getLogger(__name__)

def load_dataset(cfg):
    """Load preprocessed dataset from disk"""
    # Look for dataset in the small_subset directory
    dataset_path = Path("big_subset/data")
    
    logger.info(f"Looking for dataset at {dataset_path}...")
    
    if dataset_path.exists():
        logger.info(f"Loading dataset from {dataset_path}")
        return load_from_disk(str(dataset_path))
    else:
        # Try absolute path as fallback
        absolute_path = Path(os.getcwd()) / dataset_path
        if absolute_path.exists():
            logger.info(f"Loading dataset from {absolute_path}")
            return load_from_disk(str(absolute_path))
        
        raise FileNotFoundError(
            f"No dataset found at either:\n"
            f"  {dataset_path}\n"
            f"  {absolute_path}\n"
            f"Please run download_dataset.py first to create the dataset."
        )
