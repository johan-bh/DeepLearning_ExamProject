import hydra
from omegaconf import DictConfig
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import logging
from pathlib import Path
from tqdm.auto import tqdm
from transformers import WhisperProcessor
import numpy as np

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def preprocess_dataset(cfg: DictConfig):
    """Preprocess and save dataset for later use"""
    setup_logging()
    logger.info("=== Starting Dataset Preprocessing ===")
    
    # Define paths
    processed_dir = Path("processed_datasets")
    processed_dir.mkdir(exist_ok=True)
    
    # Load processor for audio settings
    logger.info("\nLoading processor...")
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language="da",
        task="transcribe"
    )
    
    # Load streaming datasets
    logger.info("\nLoading datasets...")
    train_dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.config,
        split=cfg.dataset.train_split,
        streaming=True
    ).shuffle(seed=cfg.dataset.seed)
    
    eval_dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.config,
        split=cfg.dataset.eval_split,
        streaming=True
    ).shuffle(seed=cfg.dataset.seed)
    
    # Convert streaming datasets to lists with progress bars
    logger.info(f"\nTaking {cfg.dataset.train_size} training samples...")
    train_data = []
    with tqdm(total=cfg.dataset.train_size, desc="Loading train data") as pbar:
        for item in train_dataset:
            train_data.append(item)
            pbar.update(1)
            if len(train_data) >= cfg.dataset.train_size:
                break
    
    logger.info(f"\nTaking {cfg.dataset.val_size} validation samples...")
    val_data = []
    with tqdm(total=cfg.dataset.val_size, desc="Loading validation data") as pbar:
        for item in eval_dataset:
            val_data.append(item)
            pbar.update(1)
            if len(val_data) >= cfg.dataset.val_size:
                break
    
    # Process audio format
    def prepare_audio(example):
        """Process single audio example"""
        audio = example["audio"]
        
        if isinstance(audio, dict) and "array" in audio:
            # Convert array to numpy if it's a list
            if isinstance(audio["array"], list):
                audio["array"] = np.array(audio["array"], dtype=np.float32)
            return example
            
        return {
            **example,
            "audio": {
                "array": np.array(audio["array"], dtype=np.float32),
                "sampling_rate": 16000
            }
        }
    
    # Create dataset dictionary
    logger.info("\nCreating dataset dictionary...")
    dataset = DatasetDict({
        'train': Dataset.from_list([
            prepare_audio(example) for example in train_data
        ]),
        'validation': Dataset.from_list([
            prepare_audio(example) for example in val_data
        ])
    })
    
    logger.info(f"Train size: {len(dataset['train'])}")
    logger.info(f"Validation size: {len(dataset['validation'])}")
    
    # Add verification step
    logger.info("\nVerifying processed samples...")
    train_sample = dataset['train'][0]
    
    # Convert audio array to numpy if it's a list
    if isinstance(train_sample['audio']['array'], list):
        import numpy as np
        train_sample['audio']['array'] = np.array(train_sample['audio']['array'], dtype=np.float32)
    
    logger.info(f"Sample audio shape: {train_sample['audio']['array'].shape}")
    logger.info(f"Sample audio type: {train_sample['audio']['array'].dtype}")
    logger.info(f"Sample text: {train_sample['text']}")
    
    # Save processed dataset with size info in filename
    save_path = processed_dir / f"coral_{cfg.dataset.train_size}_{cfg.dataset.val_size}"
    logger.info(f"\nSaving processed dataset to {save_path}")
    dataset.save_to_disk(str(save_path))
    logger.info("✓ Dataset saved successfully")

if __name__ == "__main__":
    preprocess_dataset() 