import logging
from pathlib import Path
from datasets import load_from_disk
import os

logger = logging.getLogger(__name__)

def load_dataset(cfg):
    """Load preprocessed dataset from disk"""
    # Set up cache directory path
    cache_dir = Path("/ephemeral/huggingface_cache")
    
    # Look for preprocessed dataset in cache directory
    preprocessed_path = cache_dir / f"preprocessed_coral_{cfg.dataset.train_size}_{cfg.dataset.val_size}"
    
    # Also check the old location as fallback
    old_path = Path("processed_datasets") / f"coral_{cfg.dataset.train_size}_{cfg.dataset.val_size}"
    
    logger.info(f"Looking for preprocessed dataset...")
    
    if preprocessed_path.exists():
        logger.info(f"Loading preprocessed dataset from {preprocessed_path}")
        return load_from_disk(str(preprocessed_path))
    elif old_path.exists():
        logger.info(f"Loading preprocessed dataset from {old_path}")
        return load_from_disk(str(old_path))
    else:
        raise FileNotFoundError(
            f"No preprocessed dataset found at either:\n"
            f"  {preprocessed_path}\n"
            f"  {old_path}\n"
            f"Please run preprocess_dataset.py first."
        )
