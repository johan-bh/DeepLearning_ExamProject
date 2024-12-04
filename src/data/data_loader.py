import logging
from pathlib import Path
from datasets import load_from_disk
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def load_dataset(cfg: DictConfig):
    """Load preprocessed dataset from disk"""
    processed_dir = Path("processed_datasets")
    dataset_path = processed_dir / f"coral_{cfg.dataset.train_size}_{cfg.dataset.val_size}"
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {dataset_path}. "
            "Please run preprocess_dataset.py first."
        )
    
    logger.info(f"Loading preprocessed dataset from {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    
    logger.info(f"Train size: {len(dataset['train'])}")
    logger.info(f"Validation size: {len(dataset['validation'])}")
    
    return dataset
