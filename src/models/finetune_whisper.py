import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from functools import partial
import os
import hydra
from omegaconf import DictConfig
from pathlib import Path

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for preparing batches for training"""
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

def prepare_dataset(batch, processor):
    """Prepare a batch of audio data for training"""
    # Load and resample audio
    audio = batch["audio"]
    
    # Compute input features
    batch["input_features"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]
    
    # Compute labels
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

def compute_metrics(pred, processor):
    """Compute WER metric during training"""
    metric = evaluate.load("wer")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def train_whisper(cfg: DictConfig):
    """
    Finetune Whisper model on Danish NST dataset using Hydra configuration
    
    Args:
        cfg: Hydra configuration object
    """
    # Create output directory if it doesn't exist
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset(cfg.dataset.name)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=cfg.dataset.sampling_rate))

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(cfg.model.name)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model.name)

    # Freeze encoder if specified
    if cfg.model.freeze_encoder:
        for param in model.get_encoder().parameters():
            param.requires_grad = False

    # Prepare dataset
    prepare_dataset_partial = partial(prepare_dataset, processor=processor)
    tokenized_dataset = dataset.map(
        prepare_dataset_partial,
        remove_columns=dataset["train"].column_names,
        num_proc=cfg.dataset.num_proc
    )

    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        max_steps=cfg.training.max_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        predict_with_generate=True,
        fp16=cfg.training.fp16,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        logging_steps=cfg.training.logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, processor=processor),
        tokenizer=processor.feature_extractor,
    )

    # Train model
    trainer.train()

    # Save final model
    trainer.save_model()
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    train_whisper()
