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
    student_processor: WhisperProcessor
    teacher_processor: WhisperProcessor

    def __call__(self, features):
        # Extract labels
        labels = [feature["labels"] for feature in features]

        # Prepare student input features
        student_input_features = [feature["student_input_features"] for feature in features]
        student_batch = self.student_processor.feature_extractor.pad(
            {"input_features": student_input_features},
            padding=True,
            return_tensors="pt",
        )

        # Prepare teacher input features if available
        if "teacher_input_features" in features[0]:
            teacher_input_features = [feature["teacher_input_features"] for feature in features]
            teacher_batch = self.teacher_processor.feature_extractor.pad(
                {"input_features": teacher_input_features},
                padding=True,
                return_tensors="pt",
            )
            teacher_input_features = teacher_batch["input_features"]
        else:
            teacher_input_features = None

        # Pad labels
        labels_batch = self.student_processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.student_processor.tokenizer.pad_token_id, -100
        )

        # Combine into a single batch
        batch = {
            "student_input_features": student_batch["input_features"],
            "labels": labels
        }

        if teacher_input_features is not None:
            batch["teacher_input_features"] = teacher_input_features

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
        # Extract inputs for student
        student_inputs = {"input_features": inputs["student_input_features"], "labels": inputs["labels"]}
        # Forward pass for student
        outputs = model(**student_inputs)
        student_loss = outputs.loss

        # Only compute distillation loss during training
        if self.model.training and "teacher_input_features" in inputs:
            teacher_inputs = {"input_features": inputs["teacher_input_features"]}

            # Prepare decoder_input_ids for the teacher model
            labels_for_teacher = inputs["labels"].clone()
            labels_for_teacher[labels_for_teacher == -100] = self.teacher_model.config.pad_token_id
            decoder_input_ids = shift_tokens_right(
                labels_for_teacher,
                self.teacher_model.config.pad_token_id,
                self.teacher_model.config.decoder_start_token_id
            )
            teacher_inputs["decoder_input_ids"] = decoder_input_ids

            # Forward pass for teacher
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**teacher_inputs)
                teacher_logits = teacher_outputs.logits

            # Get the minimum sequence length between teacher and student
            min_length = min(teacher_logits.size(1), outputs.logits.size(1))
            
            # Truncate both logits to the same length
            teacher_logits = teacher_logits[:, :min_length, :]
            student_logits = outputs.logits[:, :min_length, :]

            # Compute distillation loss
            distill_loss = F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
                reduction='batchmean',
                log_target=False
            ) * (self.temperature ** 2)

            # Combine losses
            loss = (self.alpha * distill_loss) + ((1 - self.alpha) * student_loss)
        else:
            # During evaluation, only use student loss
            loss = student_loss

        return (loss, outputs) if return_outputs else loss

    def _prepare_inputs(self, inputs):
        # Move inputs to the correct device
        inputs = {k: v.to(self.args.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # Move teacher inputs to the teacher model's device if available
        if self.model.training and 'teacher_input_features' in inputs:
            inputs["teacher_input_features"] = inputs["teacher_input_features"].to(self.teacher_model.device)
        return inputs

def prepare_dataset(example, student_processor, teacher_processor, is_train=True):
    # Load audio
    audio_array = example["audio"]["array"]

    # Process student input features
    student_input_features = student_processor.feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors=None
    ).input_features[0]

    # Tokenize labels
    labels = student_processor.tokenizer(
        example["text"].lower(),
        return_tensors=None
    ).input_ids

    result = {
        "student_input_features": student_input_features,
        "labels": labels
    }

    # Include teacher input features during training
    if is_train:
        teacher_input_features = teacher_processor.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors=None
        ).input_features[0]
        result["teacher_input_features"] = teacher_input_features

    return result

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
    
    # Load teacher model and processor
    print("\nLoading teacher model and processor...")
    teacher_processor = WhisperProcessor.from_pretrained(
        cfg.teacher_model.name,
        language="da",
        task="transcribe"
    )
    
    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.teacher_model.name,
        device_map=f"cuda:{cfg.teacher_model.device}",
        torch_dtype=torch.float16 if cfg.teacher_model.fp16 else torch.float32
    )
    
    # Load student model and processor
    print("\nLoading student model and processor...")
    student_processor = WhisperProcessor.from_pretrained(
        cfg.student_model.name,
        language="da",
        task="transcribe"
    )
    
    student_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.student_model.name,
        device_map=f"cuda:{cfg.student_model.device}",
        torch_dtype=torch.float16 if cfg.student_model.fp16 else torch.float32
    )
    
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
    
    # Preprocess training dataset
    print("\nPreprocessing training dataset...")
    dataset["train"] = dataset["train"].map(
        partial(prepare_dataset, student_processor=student_processor, teacher_processor=teacher_processor, is_train=True),
        remove_columns=dataset["train"].column_names,
        num_proc=cfg.dataset.num_proc,
        desc="Processing training dataset",
        batched=False
    )

    # Preprocess validation dataset
    print("\nPreprocessing validation dataset...")
    dataset["validation"] = dataset["validation"].map(
        partial(prepare_dataset, student_processor=student_processor, teacher_processor=teacher_processor, is_train=False),
        remove_columns=dataset["validation"].column_names,
        num_proc=cfg.dataset.num_proc,
        desc="Processing validation dataset",
        batched=False
    )

    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        student_processor=student_processor,
        teacher_processor=teacher_processor
    )

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
        compute_metrics=partial(compute_metrics, processor=student_processor),
    )
    
    # Train model
    print("\n=== Starting training ===")
    trainer.train()
    
    # Save model and processor
    print("\nSaving model and processor...")
    trainer.save_model()
    student_processor.save_pretrained(output_dir)
    
    print("\n=== Training complete! ===")

if __name__ == "__main__":
    distill_whisper()
