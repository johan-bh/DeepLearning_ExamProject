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
from tqdm.auto import tqdm
import logging
import transformers

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

            # Ensure input features are in the correct dtype
            input_features = inputs["input_features"]
            if self.teacher_model.dtype == torch.float16:
                input_features = input_features.to(torch.float16)

            # Prepare inputs for teacher
            teacher_inputs = {
                "input_features": input_features,
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

def compute_metrics(pred, processor):
    """Compute cer metric during training"""
    metric = evaluate.load("cer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute cer
    cer = metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

@hydra.main(config_path="../../configs", config_name="distill_config", version_base=None)
def distill_whisper(cfg: DictConfig):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=== Starting Whisper Knowledge Distillation ===")

    # Create output directory
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load processor and models
    logger.info("\nLoading processor and models...")
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
    logger.info("Loading dataset...")
    train_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.train_split,
        streaming=True
    ).shuffle(seed=cfg.dataset.seed)

    eval_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.eval_split,
        streaming=True
    ).shuffle(seed=cfg.dataset.seed)

    # Convert streaming datasets to regular datasets with limited samples
    logger.info(f"Taking {cfg.dataset.train_samples} training samples...")
    train_data = []
    pbar = tqdm(
        desc="Loading train data",
        total=cfg.dataset.train_samples,
        unit=" samples",
        ncols=100,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for item in train_dataset:
        train_data.append(item)
        pbar.update(1)
        if len(train_data) >= cfg.dataset.train_samples:
            break
    pbar.close()

    logger.info(f"Taking {cfg.dataset.eval_samples} validation samples...")
    val_data = []
    pbar = tqdm(
        desc="Loading validation data",
        total=cfg.dataset.eval_samples,
        unit=" samples",
        ncols=100,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for item in eval_dataset:
        val_data.append(item)
        pbar.update(1)
        if len(val_data) >= cfg.dataset.eval_samples:
            break
    pbar.close()

    logger.info("Creating dataset dictionary...")
    logger.info("Converting training data to Dataset...")
    
    def create_dataset_with_progress(data_list, desc):
        chunk_size = 500  # Process 500 items at a time
        chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
        
        all_features = []
        with tqdm(total=len(data_list), desc=desc) as pbar:
            for chunk in chunks:
                chunk_dataset = Dataset.from_list(chunk)
                all_features.extend([{k: example[k] for k in example} for example in chunk_dataset])
                pbar.update(len(chunk))
        
        return Dataset.from_list(all_features)

    train_dataset = create_dataset_with_progress(train_data, "Creating training dataset")
    validation_dataset = create_dataset_with_progress(val_data, "Creating validation dataset")

    logger.info("Combining into DatasetDict...")
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })

    logger.info(f"Train size: {len(dataset['train'])}")
    logger.info(f"Validation size: {len(dataset['validation'])}")

    # Cast audio column without loading data into memory
    logger.info("Casting audio column using cast_column...")
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

            labels = processor.tokenizer(
                [text.lower() for text in example["text"]],
                return_tensors="np"
            ).input_ids

            return {
                "input_features": input_features,
                "labels": labels
            }
        else:
            # Single example processing
            audio_array = example["audio"]["array"]
            input_features = processor.feature_extractor(
                audio_array,
                sampling_rate=16000,
                return_tensors="np"
            ).input_features[0]

            labels = processor.tokenizer(
                example["text"].lower(),
                return_tensors="np"
            ).input_ids[0]

            return {
                "input_features": input_features,
                "labels": labels
            }

    # Apply the transformation lazily
    logger.info("Applying lazy transformations with set_transform...")
    dataset['train'].set_transform(transform_fn)
    dataset['validation'].set_transform(transform_fn)

    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        eval_accumulation_steps=cfg.training.eval_accumulation_steps,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=-1,
        warmup_steps=cfg.training.warmup_steps,
        warmup_ratio=cfg.training.warmup_ratio,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=cfg.training.logging_steps,
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
        load_best_model_at_end=False,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=1,
        include_for_metrics=["input_features", "labels"],
        max_grad_norm=1.0,
        dataloader_num_workers=cfg.training.dataloader_workers
    )

    # Before creating the trainer
    total_train_batch_size = (
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps
    )
    
    total_steps = (
        (len(dataset['train']) // total_train_batch_size)
        * cfg.training.num_train_epochs
    )
    
    logger.info(f"Total training samples: {len(dataset['train'])}")
    logger.info(f"Effective batch size: {total_train_batch_size}")
    logger.info(f"Expected training steps: {total_steps}")

    # After training arguments, before creating trainer
    class MetricCallback(transformers.TrainerCallback):
        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)
            self.results = {
                'train_loss': [],
                'eval_loss': [],
                'eval_cer': [],
                'learning_rate': [],
                'epoch': []
            }
            
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            # Store metrics
            self.results['eval_loss'].append(metrics.get('eval_loss', None))
            self.results['eval_cer'].append(metrics.get('eval_cer', None))
            self.results['epoch'].append(state.epoch)
            
        def on_log(self, args, state, control, logs, **kwargs):
            # Store training metrics
            if 'loss' in logs:
                self.results['train_loss'].append(logs['loss'])
            if 'learning_rate' in logs:
                self.results['learning_rate'].append(logs['learning_rate'])
                
        def on_train_end(self, args, state, control, **kwargs):
            # Save results to file
            results_file = self.output_dir / 'training_results.txt'
            
            with open(results_file, 'w') as f:
                f.write("=== Knowledge Distillation Results ===\n\n")
                
                # Write training configuration
                f.write("Training Configuration:\n")
                f.write(f"Teacher Model: {cfg.teacher_model.name}\n")
                f.write(f"Student Model: {cfg.student_model.name}\n")
                f.write(f"Temperature: {cfg.training.temperature}\n")
                f.write(f"Alpha: {cfg.training.alpha}\n")
                f.write(f"Training Samples: {len(dataset['train'])}\n")
                f.write(f"Validation Samples: {len(dataset['validation'])}\n")
                f.write(f"Batch Size: {cfg.training.per_device_train_batch_size}\n")
                f.write(f"Learning Rate: {cfg.training.learning_rate}\n")
                f.write(f"Epochs: {cfg.training.num_train_epochs}\n\n")
                
                # Write final metrics
                f.write("Final Results:\n")
                if self.results['eval_cer']:
                    f.write(f"Best CER: {min(self.results['eval_cer'])}\n")
                    f.write(f"Final CER: {self.results['eval_cer'][-1]}\n")
                if self.results['eval_loss']:
                    f.write(f"Best Loss: {min(self.results['eval_loss'])}\n")
                    f.write(f"Final Loss: {self.results['eval_loss'][-1]}\n\n")
                
                # Write detailed metrics per epoch
                f.write("Detailed Results per Epoch:\n")
                for i in range(len(self.results['epoch'])):
                    f.write(f"\nEpoch {self.results['epoch'][i]}:\n")
                    if i < len(self.results['eval_cer']):
                        f.write(f"  CER: {self.results['eval_cer'][i]}\n")
                    if i < len(self.results['eval_loss']):
                        f.write(f"  Eval Loss: {self.results['eval_loss'][i]}\n")

    # Create the callback
    metric_callback = MetricCallback(output_dir)

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
        callbacks=[metric_callback]  # Add the callback here
    )

    # Train model
    logger.info("\n=== Starting training ===")
    trainer.train()

    # Save model and processor
    logger.info("\nSaving model and processor...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    logger.info("\n=== Training complete! ===")

if __name__ == "__main__":
    distill_whisper()
