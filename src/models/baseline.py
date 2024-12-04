import torch
from datasets import load_dataset, config
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from jiwer import wer, cer
from tqdm import tqdm
import numpy as np
import os
import yaml
from pathlib import Path
import hydra
import logging
from omegaconf import DictConfig
from itertools import islice
from datasets import Dataset
from datasets.features import Audio

# Set custom cache directory
config.HF_DATASETS_CACHE = "data/cached_datasets"  # relative to project root

# clear cuda cache
torch.cuda.empty_cache()

def load_asr_pipeline(model_name):
    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )
    return asr_pipeline

def process_audio_batch(batch, asr_pipeline):
    """Process a batch of audio files ensuring consistent format"""
    # Ensure all audio arrays are the same format
    processed_batch = []
    for audio in batch:
        # Convert to numpy array if needed
        if isinstance(audio["array"], list):
            audio["array"] = np.array(audio["array"])
        processed_batch.append({
            "array": audio["array"],
            "sampling_rate": audio["sampling_rate"]
        })
    
    # Generate transcriptions
    outputs = asr_pipeline(processed_batch, generate_kwargs={"language": "da"})
    return outputs

def save_results(predictions, references, word_error_rate, character_error_rate, model_name, dataset_name, dataset_config, split):
    """Save results to file with detailed naming"""
    # Create evaluation directory if it doesn't exist
    os.makedirs("evaluation/baseline", exist_ok=True)
    
    # Clean up names for file path (replace slashes and special characters)
    model_name_clean = model_name.replace('/', '_')
    dataset_name_clean = dataset_name.replace('/', '_')
    
    # Create filename with model, dataset and split information
    if dataset_config:
        results_file = f"evaluation/baseline/{model_name_clean}_{dataset_name_clean}_{dataset_config}_{split}.txt"
        dataset_display = f"{dataset_name} ({dataset_config}, {split} split)"
    else:
        results_file = f"evaluation/baseline/{model_name_clean}_{dataset_name_clean}_{split}.txt"
        dataset_display = f"{dataset_name} ({split} split)"
    
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"{model_name} Baseline Results on {dataset_display}\n")
        f.write(f"Word Error Rate (WER): {word_error_rate:.4f}\n")
        f.write(f"Character Error Rate (CER): {character_error_rate:.4f}\n")

        # Save some example predictions
        f.write("\nExample Predictions:\n")
        for i in range(min(5, len(predictions))):
            f.write(f"\nReference: {references[i]}")
            f.write(f"\nPrediction: {predictions[i]}\n")

def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def batch_iterator(dataset, batch_size):
    """Create batches from dataset"""
    dataset_iter = iter(dataset)
    while True:
        batch = list(islice(dataset_iter, batch_size))
        if not batch:
            break
        yield batch

def load_model_and_processor(cfg):
    """Load model and processor from config"""
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language="da",
        task="transcribe"
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.name,
        device_map=f"cuda:{cfg.model.device}",
        torch_dtype=torch.float16 if cfg.model.fp16 else torch.float32
    )
    
    # Set forced decoder IDs for Danish
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="da", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    
    return model, processor

@hydra.main(config_path="../../configs", config_name="baseline_config", version_base=None)
def evaluate_baseline(cfg: DictConfig):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=== Starting Baseline Model Evaluation ===")

    # Create output directory
    output_dir = Path("evaluation/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset in streaming mode
    logger.info(f"\nStep 1/4: Loading dataset '{cfg.dataset.name}' in streaming mode...")
    dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.config,
        split=cfg.dataset.split,
        streaming=True
    )
    logger.info(f"✓ Dataset loaded successfully in streaming mode")

    # Convert streaming dataset to regular dataset with seed for reproducibility
    logger.info(f"Taking {cfg.dataset.max_samples} samples...")
    data = []
    pbar = tqdm(
        desc="Loading data",
        total=cfg.dataset.max_samples,
        unit=" samples",
        ncols=100,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    # Use seed for reproducible sampling
    dataset = dataset.shuffle(seed=cfg.dataset.seed)
    
    for item in dataset:
        data.append(item)
        pbar.update(1)
        if len(data) >= cfg.dataset.max_samples:
            break
    pbar.close()

    # Create dataset
    logger.info("Creating dataset...")
    dataset = Dataset.from_list(data)
    
    # Cast audio column
    logger.info("Casting audio column...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def transform_fn(example):
        # Handle both single examples and batches
        if isinstance(example["audio"], list):
            # Batch processing
            input_features = processor.feature_extractor(
                [audio["array"] for audio in example["audio"]],
                sampling_rate=16000,
                return_tensors="np"
            ).input_features

            return {
                "input_features": input_features,
                "text": [text.lower() for text in example["text"]],
                "dialect": example["dialect"]
            }
        else:
            # Single example processing
            audio_array = example["audio"]["array"]
            input_features = processor.feature_extractor(
                audio_array,
                sampling_rate=16000,
                return_tensors="np"
            ).input_features[0]

            return {
                "input_features": input_features,
                "text": example["text"].lower(),
                "dialect": example["dialect"]
            }

    # Apply the transformation lazily
    logger.info("Applying lazy transformations...")
    dataset.set_transform(transform_fn)

    # Load model and processor
    logger.info(f"\nStep 2/4: Loading model from {cfg.model.name}...")
    model, processor = load_model_and_processor(cfg)
    logger.info(f"✓ Model loaded successfully on {next(model.parameters()).device}")

    logger.info("\nStep 3/4: Processing audio samples...")
    # Initialize lists
    predictions = []
    references = []
    dialects = []

    # Create batch iterator
    batch_size = cfg.dataset.batch_size
    batch_iter = batch_iterator(dataset, batch_size)

    # Calculate total number of samples to process
    total_samples = len(dataset)
    
    # Create progress bar
    pbar = tqdm(
        total=cfg.dataset.max_samples,
        desc="Processing audio samples",
        unit=" samples",
        ncols=100,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    # Process batches
    sample_count = 0
    for i in range(0, len(dataset), cfg.dataset.batch_size):
        try:
            # Get batch
            batch = dataset[i:i + cfg.dataset.batch_size]
            
            # Stack input features
            input_features = torch.from_numpy(
                np.stack(batch["input_features"])
            ).to(model.device)

            # Generate transcriptions
            with torch.no_grad():
                outputs = model.generate(
                    input_features,
                    language="da",
                    task="transcribe"
                )

            # Decode predictions
            batch_predictions = processor.batch_decode(outputs, skip_special_tokens=True)
            batch_references = batch["text"]

            # Store results
            predictions.extend(batch_predictions)
            references.extend(batch_references)
            if "dialect" in batch:  # Check if dialect exists
                dialects.extend(batch["dialect"])
            else:
                dialects.extend(["Unknown"] * len(batch_predictions))
            
            

            # Update progress
            processed_samples = len(batch_predictions)
            pbar.update(processed_samples)
            
            sample_count += processed_samples

        except Exception as e:
            logger.error(f"\nError processing batch: {str(e)}")
            continue

    pbar.close()

    # Calculate metrics
    logger.info("\nStep 4/4: Calculating metrics...")
    word_error_rate = wer(references, predictions)
    character_error_rate = cer(references, predictions)

    logger.info(f"\nResults:")
    logger.info(f"Word Error Rate (WER): {word_error_rate:.4f}")
    logger.info(f"Character Error Rate (CER): {character_error_rate:.4f}")

    # Save results
    save_results(
        predictions=predictions,
        references=references,
        word_error_rate=word_error_rate,
        character_error_rate=character_error_rate,
        model_name=cfg.model.name,
        dataset_name=cfg.dataset.name,
        dataset_config=cfg.dataset.config,
        split=cfg.dataset.split
    )

    logger.info("\n=== Evaluation complete! ===")

if __name__ == "__main__":
    evaluate_baseline()