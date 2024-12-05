import os
from pathlib import Path

# Set cache directories before importing HuggingFace libraries
cache_dir = Path("/ephemeral/huggingface_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Force HuggingFace to use our cache directory
os.environ["HF_HOME"] = str(cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")

# Now import HuggingFace libraries
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
import evaluate
from functools import partial
import logging
import json
import hydra
from omegaconf import DictConfig
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.data_loader import load_dataset

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features):
        # Extract inputs and labels
        input_features = [feature["input_features"] for feature in features]
        labels = [feature["labels"] for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(
            {"input_features": input_features}, padding=True, return_tensors="pt"
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
            return_attention_mask=False
        )

        # Replace padding token id's of the labels by -100
        labels_batch["input_ids"] = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        batch["labels"] = labels_batch["input_ids"]

        return batch

def compute_metrics(pred, processor):
    metric = evaluate.load("cer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 in the labels as we can't decode them
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def train_whisper(cfg: DictConfig):
    # Initialize logging first
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=== Starting Whisper Fine-tuning ===")

    # Create output directory
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and processor
    logger.info("\nLoading tokenizer and processor...")
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language="da",
        task="transcribe",
        cache_dir=cache_dir / "transformers"
    )
    logger.info("✓ Processor loaded")

    # Load model
    logger.info("\nLoading model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.name,
        cache_dir=cache_dir / "transformers"
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    logger.info("✓ Model loaded")

    # Load preprocessed dataset
    logger.info("\nLoading preprocessed dataset...")
    dataset = load_dataset(cfg)
    logger.info(f"Train size: {len(dataset['train'])}")
    logger.info(f"Validation size: {len(dataset['validation'])}")

    def transform_fn(example):
        """Transform function for preprocessed data"""
        return {
            "input_features": example["input_features"],
            "labels": example["labels"]
        }

    # Apply the transformation
    logger.info("Applying transformations...")
    dataset['train'].set_transform(transform_fn)
    dataset['validation'].set_transform(transform_fn)

    # Training setup
    total_train_batch_size = (
        cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    )
    logger.info(f"Total training samples: {len(dataset['train'])}")
    logger.info(f"Effective batch size: {total_train_batch_size}")

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        fp16=cfg.training.fp16,
        save_strategy="epoch",
        logging_steps=cfg.training.logging_steps,
        evaluation_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=225,
        generation_num_beams=5,
        push_to_hub=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, processor=processor),
    )

    # Train the model
    logger.info("\n=== Starting training ===")
    trainer.train()

    # Save model
    logger.info("\nSaving model and processor...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    # Evaluate the model
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f)

    logger.info("\n=== Training complete! ===")

if __name__ == "__main__":
    train_whisper()
