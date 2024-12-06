import logging
from pathlib import Path
from datasets import load_from_disk
import os

logger = logging.getLogger(__name__)

def load_dataset(cfg):
    """Load preprocessed dataset from disk"""
    # Look for preprocessed dataset in root directory
    root_dir = Path("/home/shadeform")  # Root directory where the dataset is located
    preprocessed_path = root_dir / f"preprocessed_coral_{cfg.dataset.train_size}_{cfg.dataset.val_size}"
    
    logger.info(f"Looking for preprocessed dataset at {preprocessed_path}...")
    
    if preprocessed_path.exists():
        logger.info(f"Loading preprocessed dataset from {preprocessed_path}")
        return load_from_disk(str(preprocessed_path))
    else:
        # Try current directory as fallback
        current_dir_path = Path(f"preprocessed_coral_{cfg.dataset.train_size}_{cfg.dataset.val_size}")
        if current_dir_path.exists():
            logger.info(f"Loading preprocessed dataset from {current_dir_path}")
            return load_from_disk(str(current_dir_path))
        
        raise FileNotFoundError(
            f"No preprocessed dataset found at either:\n"
            f"  {preprocessed_path}\n"
            f"  {current_dir_path}\n"
            f"Please ensure the preprocessed dataset is in the correct location."
        )
