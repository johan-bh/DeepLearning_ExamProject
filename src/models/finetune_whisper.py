import os
from pathlib import Path

# Set cache directories before importing HuggingFace libraries
cache_dir = Path("huggingface_cache")
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
    Seq2SeqTrainer,
    TrainerCallback
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
import numpy as np

# ADDED: Imports for plotting and W&B logging
import matplotlib.pyplot as plt

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features):
        input_features = [feature["input_features"].clone().detach().to(torch.bfloat16) 
                         for feature in features]
        labels = [feature["labels"] for feature in features]

        # Pad input features
        batch = {
            "input_features": torch.nn.utils.rnn.pad_sequence(
                input_features, 
                batch_first=True
            )
        }

        # Pad labels and attention masks
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )

        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, 
            -100
        )
        batch["attention_mask"] = labels_batch["attention_mask"]

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

# ADDED: A callback to plot loss and metrics at the end of training
class PlotLossCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        log_history = state.log_history

        train_steps = []
        train_loss = []
        eval_steps = []
        eval_loss = []
        eval_cer = []

        for log in log_history:
            if "loss" in log and "learning_rate" in log:
                # Training logs
                train_steps.append(log["step"])
                train_loss.append(log["loss"])
            if "eval_loss" in log:
                eval_steps.append(log["step"])
                eval_loss.append(log["eval_loss"])
            if "eval_cer" in log:
                eval_cer.append(log["eval_cer"])

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10,6))

        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss", color="tab:red")
        ax1.plot(train_steps, train_loss, label="Training Loss", color="tab:red")
        if eval_loss:
            ax1.plot(eval_steps, eval_loss, label="Eval Loss", color="tab:purple")

        ax1.tick_params(axis='y', labelcolor="tab:red")
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines, labels, loc="upper left")

        if eval_cer:
            ax2 = ax1.twinx()
            ax2.set_ylabel("CER", color="tab:blue")
            ax2.plot(eval_steps, eval_cer, label="Eval CER", color="tab:blue")
            ax2.tick_params(axis='y', labelcolor="tab:blue")

            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper center")

        fig.tight_layout()
        fig_path = Path(args.output_dir) / "training_progress.png"
        plt.savefig(fig_path)

        plt.close(fig)

@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def train_whisper(cfg: DictConfig):
    # Initialize logging first
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=== Starting Whisper Fine-tuning ===")

    # Add device checking and setup
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        logger.info(f"Found {n_gpu} GPU(s) available")
        device = torch.device("cuda:0")  # Default to first GPU
    else:
        logger.info("No GPU available, using CPU")
        device = torch.device("cpu")

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

    # Load model with bfloat16
    logger.info("\nLoading model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.name,
        cache_dir=cache_dir / "transformers",
        torch_dtype=torch.bfloat16
    ).to(device)

    # After loading the model
    from transformers.cache_utils import Cache
    model.use_cache = True  # Enable caching

    # ADDED: Set forced decoder IDs & generation config as in distillation
    model.config.forced_decoder_ids = None  # Remove forced decoder IDs
    model.config.suppress_tokens = []
    # ADDED: Also set generation_config to ensure correct decoding
    model.generation_config.language = "da"
    model.generation_config.task = "transcribe"

    logger.info("✓ Model loaded")

    # Load preprocessed dataset
    logger.info("\nLoading preprocessed dataset...")
    dataset = load_dataset(cfg)

    # column cast audio 16khz, for extra safety
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    logger.info(f"Train size: {len(dataset['train'])}")
    logger.info(f"Validation size: {len(dataset['validation'])}")

    def transform_fn(example):
        """Transform function for preprocessed data"""
        # Process audio to input features
        input_features = processor(
            [audio_item["array"] for audio_item in example["audio"]],
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        ).input_features.squeeze(0)

        # Process text to labels
        label_tokens = processor.tokenizer(
            example["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=448,
            return_attention_mask=True
        )

        return {
            "input_features": input_features,
            "labels": label_tokens.input_ids.squeeze(0),
            "attention_mask": label_tokens.attention_mask.squeeze(0)
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
        per_device_eval_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        num_train_epochs=cfg.training.num_train_epochs,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        warmup_steps=500,
        warmup_ratio=0.1,    # Add warmup ratio
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=500,
        eval_strategy="steps",
        eval_steps=1000,      # More frequent evaluation
        predict_with_generate=True,
        generation_max_length=448,
        generation_num_beams=1,
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
        metric_for_best_model="cer",
        greater_is_better=False,
        optim="adamw_hf",
        lr_scheduler_type="linear",
        max_grad_norm=1,
        logging_dir=str(output_dir / "logs"),
        report_to=[],
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, processor=processor),
        callbacks=[PlotLossCallback()]  # ADDED: callback for plotting at the end
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
    with open(output_dir / "OOB_results.json", "w") as f:
        json.dump(eval_results, f)

    logger.info("\n=== Training complete! ===")

if __name__ == "__main__":
    train_whisper()
