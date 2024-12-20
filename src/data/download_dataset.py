import os
import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import datasets
from datasets import load_dataset, Audio
from transformers import WhisperProcessor
import soundfile as sf  # for saving audio arrays as .wav
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def create_small_subset(cfg: DictConfig):
    """
    1. Stream a large dataset and take only first N samples from train and val.
    2. Decode audio and save locally as WAV files.
    3. Create a Dataset from these local files.
    4. cast_column("audio", Audio(...)) on the local dataset.
    """

    # Desired subset sizes
<<<<<<< HEAD
    train_size = 50000
    val_size = 500

    output_dir = Path("huge_subset/data")
=======
    train_size = 10000
    val_size = 250

    output_dir = Path("big_subset/data")
>>>>>>> b791ce11ff8eaa2164e4a1e8d9ed53d2b6600e45
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = output_dir / "audio_files"
    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language="da",
        task="transcribe"
    )

    logger.info("Streaming the train split...")
    train_stream = load_dataset(
        cfg.dataset.name,
        cfg.dataset.config,
        split=cfg.dataset.train_split,
        streaming=True
    )

    logger.info("Streaming the validation split...")
    val_stream = load_dataset(
        cfg.dataset.name,
        cfg.dataset.config,
        split=cfg.dataset.eval_split,
        streaming=True
    )

    def decode_and_save_audio(example, idx, split):
        # 'audio' should have 'path' or 'array' accessible via streaming
        # If streaming provides a dict with {'path': ...}:
        # Decode using datasets.Audio
        audio_info = example["audio"]
        if "array" in audio_info:
            # If the dataset already returns decoded arrays
            audio_array = audio_info["array"]
            sr = audio_info.get("sampling_rate", 16000)
        else:
            # Manually decode from path
            audio_data = datasets.Audio(sampling_rate=16000).decode_example(
                {"path": audio_info["path"], "bytes": audio_info.get("bytes")}
            )
            audio_array = audio_data["array"]
            sr = 16000

        # Process text
        text = example["text"]
        
        # Save audio as a local WAV file
        filename = f"{split}_{idx}.wav"
        filepath = audio_dir / filename
        sf.write(filepath, audio_array, sr)

        # Return dictionary with local file path and text
        return {"audio": str(filepath), "text": text}

    def collect_samples(stream, count, split):
        samples = []
        for i, ex in enumerate(stream):
            if i >= count:
                break
            sample = decode_and_save_audio(ex, i, split)
            samples.append(sample)
        return samples

    logger.info(f"Collecting {train_size} train samples via streaming...")
    train_samples = collect_samples(train_stream, train_size, "train")

    logger.info(f"Collecting {val_size} validation samples via streaming...")
    val_samples = collect_samples(val_stream, val_size, "validation")

    # Create datasets from lists
    logger.info("Creating arrow-based datasets from lists...")
    train_ds = datasets.Dataset.from_list(train_samples)
    val_ds = datasets.Dataset.from_list(val_samples)

    # Combine into a DatasetDict
    subset = datasets.DatasetDict({
        "train": train_ds,
        "validation": val_ds
    })

    # Now we have a local dataset with references to local WAV files (audio column is a string path)
    # We can cast the audio column to use the Audio feature
    logger.info("Casting the audio column to Audio feature...")
    subset = subset.cast_column("audio", Audio(sampling_rate=16000))

    # Now subset is a random-access dataset with a proper Audio column, without downloading entire data.
    subset_path = output_dir / "data"
    subset.save_to_disk(str(subset_path))
    logger.info(f"Saved final subset with cast audio column at {subset_path}")

if __name__ == "__main__":
    create_small_subset()
