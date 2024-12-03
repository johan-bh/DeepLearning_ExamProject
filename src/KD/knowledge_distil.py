import torch
from datasets import load_dataset, Audio, Dataset, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from transformers.models.whisper.modeling_whisper import shift_tokens_right
from torch.nn import functional as F
from dataclasses import dataclass
import evaluate
from functools import partial
import hydra
from omegaconf import DictConfig
from pathlib import Path

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor  # Use a single processor

    def __call__(self, features):
        # Extract labels
        labels = [feature["labels"] for feature in features]

        # Prepare input features
        input_features = [feature["input_features"] for feature in features]
        batch = self.processor.feature_extractor.pad(
            {"input_features": input_features},
            padding=True,
            return_tensors="pt",
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        # Combine into a single batch
        batch["labels"] = labels

        return batch

class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, teacher_model=None, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        # Move teacher model to the same device as student
        if self.teacher_model is not None:
            self.teacher_model.to(self.model.device)
            self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass for student
        outputs = model(**inputs)
        student_loss = outputs.loss

        # Only compute distillation loss during training
        if self.model.training:
            # Prepare decoder_input_ids for the teacher model
            labels_for_teacher = inputs["labels"].clone()
            labels_for_teacher[labels_for_teacher == -100] = self.teacher_model.config.pad_token_id
            decoder_input_ids = shift_tokens_right(
                labels_for_teacher,
                self.teacher_model.config.pad_token_id,
                self.teacher_model.config.decoder_start_token_id
            )

            # Prepare inputs for teacher
            teacher_inputs = {
                "input_features": inputs["input_features"],
                "decoder_input_ids": decoder_input_ids
            }

            # Forward pass for teacher
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**teacher_inputs)
                teacher_logits = teacher_outputs.logits

            # Compute distillation loss
            student_logits = outputs.logits

            # Ensure logits have the same shape
            if student_logits.shape != teacher_logits.shape:
                # Truncate or pad logits to match shapes
                min_length = min(student_logits.size(1), teacher_logits.size(1))
                student_logits = student_logits[:, :min_length, :]
                teacher_logits = teacher_logits[:, :min_length, :]

            distill_loss = F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)

            # Combine losses
            loss = (self.alpha * distill_loss) + ((1 - self.alpha) * student_loss)
        else:
            # During evaluation, only use student loss
            loss = student_loss

        return (loss, outputs) if return_outputs else loss

def prepare_dataset(example, processor):
    # Load audio
    audio_array = example["audio"]["array"]

    # Process input features
    input_features = processor.feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors=None
    ).input_features[0]

    # Tokenize labels
    labels = processor.tokenizer(
        example["text"].lower(),
        return_tensors=None
    ).input_ids

    return {
        "input_features": input_features,
        "labels": labels
    }

def compute_metrics(pred, processor):
    """Compute CER metric during training"""
    metric = evaluate.load("cer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute CER
    cer = metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

@hydra.main(config_path="../../configs", config_name="distill_config", version_base=None)
def distill_whisper(cfg: DictConfig):
    print("\n=== Starting Whisper Knowledge Distillation ===")

    # Create output directory
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load processor and models
    print("\nLoading processor and models...")
    processor = WhisperProcessor.from_pretrained(
        cfg.teacher_model.name,
        language="da",
        task="transcribe"
    )

    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.teacher_model.name,
        device_map=f"cuda:{cfg.teacher_model.device}",
        torch_dtype=torch.float16 if cfg.teacher_model.fp16 else torch.float32
    )

    student_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.student_model.name,
        device_map=f"cuda:{cfg.student_model.device}",
        torch_dtype=torch.float16 if cfg.student_model.fp16 else torch.float32
    )

    # Ensure student model has the same tokenizer and vocab size
    student_model.config.vocab_size = teacher_model.config.vocab_size
    student_model.config.pad_token_id = teacher_model.config.pad_token_id
    student_model.config.decoder_start_token_id = teacher_model.config.decoder_start_token_id
    student_model.config.eos_token_id = teacher_model.config.eos_token_id
    student_model.resize_token_embeddings(teacher_model.config.vocab_size)

    # Load dataset
    print("\nLoading dataset...")
    train_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.train_split,
        streaming=True
    )

    eval_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.eval_split,
        streaming=True
    )

    # Convert streaming datasets to regular datasets with limited samples
    train_data = list(train_dataset.take(cfg.dataset.train_samples))
    val_data = list(eval_dataset.take(cfg.dataset.eval_samples))

    dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data)
    })

    # Cast audio column
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")

    # Preprocess datasets
    print("\nPreprocessing datasets...")
    dataset = dataset.map(
        partial(prepare_dataset, processor=processor),
        remove_columns=dataset["train"].column_names,
        num_proc=cfg.dataset.num_proc,
        desc="Processing datasets",
        batched=False
    )

    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        fp16=cfg.training.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=cfg.training.logging_steps,
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=1,
        include_for_metrics=["input_features", "labels"],
        max_grad_norm=1.0,
        dataloader_num_workers=0
    )

    # Create trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=cfg.training.temperature,
        alpha=cfg.training.alpha,
        model=student_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, processor=processor),
    )

    # Train model
    print("\n=== Starting training ===")
    trainer.train()

    # Save model and processor
    print("\nSaving model and processor...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    print("\n=== Training complete! ===")

if __name__ == "__main__":
    distill_whisper()
