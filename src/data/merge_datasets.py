import os
import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import datasets
from datasets import load_dataset, Audio
from transformers import WhisperProcessor
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def create_small_subset(cfg: DictConfig):
    # Path to the previously saved Coral subset dataset
    coral_subset_path = "/home/shadeform/DeepLearning_ExamProject/large_subset/data/data"

    if os.path.exists(coral_subset_path):
        logger.info("Loading the pre-downloaded Coral dataset from disk...")
        coral_subset = datasets.DatasetDict.load_from_disk(coral_subset_path)
    else:
        raise ValueError(f"Coral subset not found at {coral_subset_path}. Please create/download it first.")
    
    # coral_subset['train'] and coral_subset['validation'] are available here.
    # coral_subset['train'] ~ 50k samples and coral_subset['validation'] ~ 500 samples.

    # Load Mozilla Common Voice
    logger.info("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language="da",
        task="transcribe"
    )

    logger.info("Streaming Mozilla Common Voice (train and validation) for Danish...")
    mozilla_dataset_name = "mozilla-foundation/common_voice_17_0"
    mozilla_lang = "da"

    # Load each split separately in streaming mode
    mozilla_train_stream = load_dataset(
        mozilla_dataset_name,
        mozilla_lang,
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    mozilla_val_stream = load_dataset(
        mozilla_dataset_name,
        mozilla_lang,
        split="validation",
        streaming=True,
        trust_remote_code=True
    )

    # Directory for audio
    output_dir = Path("/home/shadeform/DeepLearning_ExamProject/huge_combined")
    audio_dir = output_dir / "audio_files"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def decode_and_save_audio(example, idx, split, text_column="sentence"):
        audio_info = example["audio"]
        if "array" in audio_info:
            audio_array = audio_info["array"]
            sr = audio_info.get("sampling_rate", 16000)
        else:
            audio_data = datasets.Audio(sampling_rate=16000).decode_example(
                {"path": audio_info["path"], "bytes": audio_info.get("bytes")}
            )
            audio_array = audio_data["array"]
            sr = 16000

        text = example[text_column]
        filename = f"{split}_{idx}.wav"
        filepath = audio_dir / filename
        sf.write(filepath, audio_array, sr)
        return {"audio": str(filepath), "text": text}

    def collect_all_samples(stream, split, text_column="sentence"):
        samples = []
        for i, ex in enumerate(stream):
            sample = decode_and_save_audio(ex, i, split, text_column=text_column)
            samples.append(sample)
        return samples

    logger.info("Collecting all Mozilla Common Voice train samples...")
    mozilla_train_samples = collect_all_samples(mozilla_train_stream, "mozilla_train", text_column="sentence")

    logger.info("Collecting all Mozilla Common Voice validation samples...")
    mozilla_val_samples = collect_all_samples(mozilla_val_stream, "mozilla_val", text_column="sentence")

    # Create datasets from lists
    mozilla_train_ds = datasets.Dataset.from_list(mozilla_train_samples)
    mozilla_val_ds = datasets.Dataset.from_list(mozilla_val_samples)

    # Combine Mozilla train + validation into one dataset
    mozilla_ds = datasets.concatenate_datasets([mozilla_train_ds, mozilla_val_ds])

    # IMPORTANT: Cast mozilla_ds audio column to Audio to match coral_subset's schema
    mozilla_ds = mozilla_ds.cast_column("audio", Audio(sampling_rate=16000))

    # Combine Coral train set with Mozilla
    combined_train_ds = datasets.concatenate_datasets([coral_subset["train"], mozilla_ds]).shuffle(seed=42)

    # Create a new DatasetDict with combined train and original Coral validation
    final_subset = datasets.DatasetDict({
        "train": combined_train_ds,
        "validation": coral_subset["validation"]
    })

    # final_subset already has `Audio` features for all splits now.

    # Save back to disk (this will be a new dataset that includes both Coral and Mozilla)
    final_subset.save_to_disk(str(output_dir / "data"))
    logger.info(f"Saved final subset with cast audio column at {output_dir / 'data'}")

# if __name__ == "__main__":
#     create_small_subset()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def create_3_combined_datasets(cfg: DictConfig):
    # Paths to pre-downloaded Coral dataset
    coral_subset_path = "/home/shadeform/DeepLearning_ExamProject/huge_subset/data/data"

    # Load Coral subset from disk
    if not os.path.exists(coral_subset_path):
        raise ValueError(f"Coral subset not found at {coral_subset_path}. Please create/download it first.")

    logger.info("Loading the pre-downloaded Coral dataset from disk...")
    coral_subset = datasets.DatasetDict.load_from_disk(coral_subset_path)
    # coral_subset: "train" ~ 50k, "validation" ~ 500

    # Load processor (Whisper)
    logger.info("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language="da",
        task="transcribe"
    )

    # ============ Mozilla Common Voice =============
    logger.info("Streaming Mozilla Common Voice (train and validation) for Danish...")
    mozilla_dataset_name = "mozilla-foundation/common_voice_17_0"
    mozilla_lang = "da"

    mozilla_train_stream = load_dataset(
        mozilla_dataset_name,
        mozilla_lang,
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    mozilla_val_stream = load_dataset(
        mozilla_dataset_name,
        mozilla_lang,
        split="validation",
        streaming=True,
        trust_remote_code=True
    )

    # Directory for all audio
    output_dir = Path("/home/shadeform/DeepLearning_ExamProject/triple_combined_datasets")
    audio_dir = output_dir / "audio_files"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def decode_and_save_audio(example, idx, split, text_column="text"):
        audio_info = example["audio"]
        if "array" in audio_info:
            audio_array = audio_info["array"]
            sr = audio_info.get("sampling_rate", 16000)
        else:
            audio_data = datasets.Audio(sampling_rate=16000).decode_example(
                {"path": audio_info["path"], "bytes": audio_info.get("bytes")}
            )
            audio_array = audio_data["array"]
            sr = 16000

        text = example[text_column]
        filename = f"{split}_{idx}.wav"
        filepath = audio_dir / filename
        sf.write(filepath, audio_array, sr)
        return {"audio": str(filepath), "text": text}

    def collect_all_samples(stream, split, text_column="text"):
        samples = []
        for i, ex in enumerate(stream):
            sample = decode_and_save_audio(ex, i, split, text_column=text_column)
            samples.append(sample)
        return samples

    def collect_samples(stream, count, split, text_column="text"):
        samples = []
        for i, ex in enumerate(stream):
            if i >= count:
                break
            sample = decode_and_save_audio(ex, i, split, text_column=text_column)
            samples.append(sample)
        return samples

    # Collect Mozilla train
    logger.info("Collecting all Mozilla Common Voice train samples...")
    mozilla_train_samples = collect_all_samples(mozilla_train_stream, "mozilla_train", text_column="sentence")

    # Collect Mozilla validation
    logger.info("Collecting all Mozilla Common Voice validation samples...")
    mozilla_val_samples = collect_all_samples(mozilla_val_stream, "mozilla_val", text_column="sentence")

    mozilla_train_ds = datasets.Dataset.from_list(mozilla_train_samples)
    mozilla_val_ds = datasets.Dataset.from_list(mozilla_val_samples)

    # Cast the columns to Audio to match Coral
    mozilla_train_ds = mozilla_train_ds.cast_column("audio", Audio(sampling_rate=16000))
    mozilla_val_ds = mozilla_val_ds.cast_column("audio", Audio(sampling_rate=16000))

    # Combine Coral + Mozilla
    # Training: coral_subset["train"] + mozilla_train_ds
    combined_train_ds = datasets.concatenate_datasets([coral_subset["train"], mozilla_train_ds])

    # Validation: coral_subset["validation"] + mozilla_val_ds
    combined_val_ds = datasets.concatenate_datasets([coral_subset["validation"], mozilla_val_ds])

    # ============ NST-DA Dataset =============
    # "alexandrainst/nst-da" with "train" and "test"
    # Take 5k for train and 500 for validation from the train split
    logger.info("Streaming NST (alexandrainst/nst-da) train split...")
    nst_train_stream = load_dataset(
        "alexandrainst/nst-da",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    nst_count = 5500
    logger.info(f"Collecting {nst_count} NST train samples (5k for train, 500 for validation)...")

    nst_samples = []
    for i, ex in enumerate(nst_train_stream):
        if i >= nst_count:
            break
        sample = decode_and_save_audio(ex, i, "nst", text_column="text")
        nst_samples.append(sample)

    nst_train_samples = nst_samples[:5000]
    nst_val_samples = nst_samples[5000:5500]

    nst_train_ds = datasets.Dataset.from_list(nst_train_samples)
    nst_val_ds = datasets.Dataset.from_list(nst_val_samples)

    # Cast NST to Audio
    nst_train_ds = nst_train_ds.cast_column("audio", Audio(sampling_rate=16000))
    nst_val_ds = nst_val_ds.cast_column("audio", Audio(sampling_rate=16000))

    # Add NST to combined datasets
    combined_train_ds = datasets.concatenate_datasets([combined_train_ds, nst_train_ds])
    combined_val_ds = datasets.concatenate_datasets([combined_val_ds, nst_val_ds])

    # Shuffle final train set
    combined_train_ds = combined_train_ds.shuffle(seed=42)

    # Create the final DatasetDict
    final_subset = datasets.DatasetDict({
        "train": combined_train_ds,
        "validation": combined_val_ds
    })

    # Save back to disk
    final_subset.save_to_disk(str(output_dir / "data"))
    logger.info(f"Saved final merged subset with cast audio column at {output_dir / 'data'}")

if __name__ == "__main__":
    create_3_combined_datasets()