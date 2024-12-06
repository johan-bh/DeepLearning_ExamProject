import os
from pathlib import Path
from datasets import load_dataset

# Configure cache directories
cache_dir = Path("huggingface_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Force HuggingFace to use our cache directory
os.environ["HF_HOME"] = str(cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")

# Dataset parameters (as provided)
DATASET_NAME = "alexandrainst/coral"
TRAIN_SPLIT = "train"
EVAL_SPLIT = "val"
TRAIN_SIZE = 1000
VAL_SIZE = 100
SEED = 42

if __name__ == "__main__":
    # Download the train split
    print("Downloading train split...")
    train_dataset = load_dataset(
        DATASET_NAME,
        "read_aloud",
        split=TRAIN_SPLIT,
        cache_dir=cache_dir
    )

    # Download the val split
    print("Downloading validation split...")
    eval_dataset = load_dataset(
        DATASET_NAME,
        "read_aloud",
        split=EVAL_SPLIT,
        cache_dir=cache_dir
    )
    
    # Optionally, select a subset for caching (this will ensure data is actually accessed and fully downloaded)
    # If you want full dataset downloads, omit the `select(range())` calls below.
    # By accessing elements, we force dataset download from the Hub to cache.
    train_dataset = train_dataset.shuffle(seed=SEED).select(range(min(TRAIN_SIZE, len(train_dataset))))
    eval_dataset = eval_dataset.shuffle(seed=SEED).select(range(min(VAL_SIZE, len(eval_dataset))))

    # Iterate over the entire subsets to ensure all audio and metadata are downloaded into the cache
    print("Ensuring all train samples are cached...")
    for _ in train_dataset:
        pass

    print("Ensuring all eval samples are cached...")
    for _ in eval_dataset:
        pass

    print("All requested samples have been downloaded and cached.")
