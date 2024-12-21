import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def download_dataset(cfg: DictConfig):
    """
    Download and prepare dataset subset
    Args:
        cfg: Configuration from train_config.yaml
    """
    # Get parameters from config
    train_size = cfg.dataset.download.train_size
    val_size = cfg.dataset.download.val_size
    output_dir = Path(cfg.dataset.download.output_dir)
    
    logger.info(f"Downloading dataset with:")
    logger.info(f"- Train size: {train_size}")
    logger.info(f"- Validation size: {val_size}")
    logger.info(f"- Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset
    dataset = load_dataset(cfg.dataset.name)

    # Create subsets
    train_subset = dataset["train"].select(range(min(train_size, len(dataset["train"]))))
    val_subset = dataset["validation"].select(range(min(val_size, len(dataset["validation"]))))

    # Save subsets
    logger.info("Saving train subset...")
    train_subset.save_to_disk(output_dir / "data" / "train")
    
    logger.info("Saving validation subset...")
    val_subset.save_to_disk(output_dir / "data" / "val")

    logger.info(f"Dataset downloaded and saved to {output_dir}")
    logger.info(f"Train size: {len(train_subset)}")
    logger.info(f"Validation size: {len(val_subset)}")

if __name__ == "__main__":
    download_dataset()
