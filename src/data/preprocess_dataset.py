import os 
import hydra
from omegaconf import DictConfig
from datasets import load_dataset, DatasetDict
import logging
from pathlib import Path
from tqdm.auto import tqdm
from transformers import WhisperProcessor
import numpy as np
import datasets
import time
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import HfFileSystem
from datasets.download.download_manager import DownloadMode
from datasets import load_dataset, Audio, Dataset, DatasetDict

# Set cache directories before importing HuggingFace libraries
cache_dir = Path("huggingface_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Force HuggingFace to use our cache directory
os.environ["HF_HOME"] = str(cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Add retry decorator for handling rate limits (for HuggingFace API, not used atm)
# @retry(
#     stop=stop_after_attempt(5),  # Try 5 times
#     wait=wait_exponential(multiplier=1, min=4, max=60),  # Wait between 4 and 60 seconds
#     retry=lambda retry_state: isinstance(retry_state.outcome.exception(), HfHubHTTPError)
#     and retry_state.outcome.exception().response.status_code == 429
# )

def fetch_item(dataset_iter):
    try:
        return next(dataset_iter)
    except StopIteration:
        return None

@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def preprocess_dataset(cfg: DictConfig):
    """Preprocess and save dataset for later use"""
    setup_logging()
    logger.info("=== Starting Dataset Preprocessing ===")
    
    # Define output path in root directory
    output_path = Path(f"preprocessed_coral_{cfg.dataset.train_size}_{cfg.dataset.val_size}")
    
    # Load processor for audio settings
    logger.info("\nLoading processor...")
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language="da",
        task="transcribe",
        cache_dir=cache_dir / "transformers"
    )
    
    # Added for debugging
    logger.info(f"Cache directory contents:")
    for path in cache_dir.glob("*"):
        logger.info(f"  {path}")

    # Load dataset with correct cache path
    try:
        # We load the cached dataset I downloaded in the finetuning script. It's in subfolders of the /ephemeral volume
        train_dataset = load_dataset(
            cfg.dataset.name,
            "read_aloud",
            split=cfg.dataset.train_split,
            streaming=False,
            cache_dir=cache_dir
        )
        
        eval_dataset = load_dataset(
            cfg.dataset.name,
            "read_aloud",
            split=cfg.dataset.eval_split,
            streaming=False,
            cache_dir=cache_dir
        )
        
        # Take random samples
        train_dataset = train_dataset.shuffle(seed=cfg.dataset.seed).select(range(cfg.dataset.train_size))
        eval_dataset = eval_dataset.shuffle(seed=cfg.dataset.seed).select(range(cfg.dataset.val_size))
        
        # Cast column audio (vital for Whisper to work)
        # Nb: This doesnt work when loading the dataset with streaming=True...
        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
        eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        logger.info("Successfully loaded cached dataset!")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.info("Full error:", exc_info=True)
        raise RuntimeError("Failed to load dataset from cache") from e

    # Process audio format
    def prepare_audio(example):
        """Process single audio example"""
        audio = example["audio"]
        
        # Extract features using the processor
        if isinstance(audio, dict):
            if 'array' in audio:
                # Already decoded audio
                audio_array = audio['array']
            elif 'path' in audio:
                # Need to decode from path
                audio_data = datasets.Audio(sampling_rate=16000).decode_example({"path": audio["path"], "bytes": None})
                audio_array = audio_data['array']
            else:
                raise ValueError(f"Unexpected audio format: {audio.keys()}")
        else:
            # Direct array
            audio_array = audio
            
        # Extract features
        features = processor.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="np"
        ).input_features[0]
        
        # Process text, preserving Danish characters
        text = example["text"].lower()
        labels = processor.tokenizer(
            text,
            return_tensors="np",
            padding=False,
            add_special_tokens=True
        ).input_ids[0]

        # Convert numpy arrays to lists for compatibility
        features = features.tolist()
        labels = labels.tolist()

        return {
            "input_features": features,
            "labels": labels,
            "text": text
        }
    
    # Process train dataset
    logger.info("\nProcessing training data...")
    train_dataset = train_dataset.map(
        prepare_audio, 
        remove_columns=train_dataset.column_names, 
        desc="Processing train data"
    )

    # Process validation dataset
    logger.info("\nProcessing validation data...")
    eval_dataset = eval_dataset.map(
        prepare_audio, 
        remove_columns=eval_dataset.column_names, 
        desc="Processing validation data"
    )

    # Create dataset dictionary
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': eval_dataset
    })
    
    # Verify dataset structure before saving
    logger.info("\nVerifying dataset structure...")
    train_sample = dataset['train'][0]
    logger.info("Dataset keys:")
    logger.info(train_sample.keys())

    logger.info("\nSample shapes:")
    for key, value in train_sample.items():
        if isinstance(value, list):
            logger.info(f"{key}: List with length {len(value)}")
        elif isinstance(value, np.ndarray):
            logger.info(f"{key}: numpy array with shape {value.shape}")
        else:
            logger.info(f"{key}: {type(value)}")
    
    # Save processed dataset in the root directory
    logger.info(f"\nSaving processed dataset to {output_path}")
    dataset.save_to_disk(str(output_path))
    logger.info("✓ Dataset saved successfully")
    
    # List saved files
    logger.info("\nVerifying saved files:")
    if output_path.exists():
        logger.info(f"Dataset saved at: {output_path}")
        for file in output_path.glob("**/*"):
            logger.info(f"  {file.relative_to(output_path)}")
    else:
        logger.error(f"Failed to find saved dataset at {output_path}")

if __name__ == "__main__":
    preprocess_dataset()