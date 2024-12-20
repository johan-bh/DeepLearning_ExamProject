import os
from pathlib import Path
import sys
import logging
import torch
import numpy as np
import evaluate
from dataclasses import dataclass
from functools import partial
from omegaconf import DictConfig
import hydra
import transformers
from peft import LoraConfig, get_peft_model, TaskType

from datasets import Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from transformers.models.whisper.modeling_whisper import shift_tokens_right
from torch.nn import functional as F

# Ensure the root directory is on the path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.data_loader import load_dataset

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features):
        # Convert input features to float32 numpy arrays (bf16 not supported in numpy)
        for f in features:
            if isinstance(f["input_features"], list):
                f["input_features"] = np.array(f["input_features"], dtype=np.float32)
            else:
                f["input_features"] = f["input_features"].astype(np.float32)

        # Determine max length for padding audio features
        max_length = max(f["input_features"].shape[0] for f in features)

        # Limit maximum sequence length
        max_length = min(max_length, 30000)  # Adjust this value based on your needs

        # Pad all input_features to the same length
        padded_inputs = []
        for f in features:
            inp = f["input_features"]
            padding_length = max_length - inp.shape[0]
            if padding_length > 0:
                inp = np.pad(inp, ((0, padding_length), (0, 0)), mode='constant', constant_values=0)
            padded_inputs.append(inp)

        # Stack into a single NumPy array, then convert to tensor with bf16
        padded_inputs = np.stack(padded_inputs, axis=0)
        batch = {
            "input_features": torch.tensor(padded_inputs, dtype=torch.bfloat16),
        }

        # Pad labels
        labels = [f["labels"] for f in features]
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt"
        )
        
        # Convert padding token ids to -100 for PyTorch CE loss
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, 
            -100
        )

        return batch

class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, teacher_model=None, temperature=4.0, alpha=0.1, processor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.processor = processor

        if self.teacher_model is not None:
            self.teacher_model.to(self.model.device)
            self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Convert input features to bfloat16 to match model parameters
        inputs["input_features"] = inputs["input_features"].to(dtype=torch.bfloat16)
        
        # For LoRA-wrapped model, we need to handle the base model's forward pass
        if hasattr(model, "base_model"):
            outputs = model.base_model(
                input_features=inputs["input_features"],
                labels=inputs["labels"] if "labels" in inputs else None
            )
        else:
            outputs = model(
                input_features=inputs["input_features"],
                labels=inputs["labels"] if "labels" in inputs else None
            )
            
        student_loss = outputs.loss

        # Only compute distillation loss during training
        if self.model.training:
            with torch.no_grad():
                # Get teacher predictions
                teacher_pred = self.teacher_model.generate(
                    inputs["input_features"],
                    language="da",
                    task="transcribe",
                    forced_decoder_ids=self.teacher_model.generation_config.forced_decoder_ids
                )
                
                # Get student predictions
                if hasattr(model, "base_model"):
                    student_pred = model.base_model.generate(
                        inputs["input_features"],
                        language="da",
                        task="transcribe",
                        forced_decoder_ids=model.generation_config.forced_decoder_ids
                    )
                else:
                    student_pred = model.generate(
                        inputs["input_features"],
                        language="da",
                        task="transcribe",
                        forced_decoder_ids=model.generation_config.forced_decoder_ids
                    )

                # Prepare teacher inputs
                labels_for_teacher = inputs["labels"].clone()
                labels_for_teacher[labels_for_teacher == -100] = self.teacher_model.config.pad_token_id
                decoder_input_ids = shift_tokens_right(
                    labels_for_teacher,
                    self.teacher_model.config.pad_token_id,
                    self.teacher_model.config.decoder_start_token_id
                )

                # Teacher forward pass
                with torch.set_grad_enabled(True):  # Enable gradients for teacher outputs
                    teacher_outputs = self.teacher_model(
                        input_features=inputs["input_features"],
                        decoder_input_ids=decoder_input_ids,
                        output_hidden_states=True
                    )
                    teacher_logits = teacher_outputs.logits.detach()  # Detach teacher logits

                    # Compute distillation loss with gradients enabled
                    student_logits = outputs.logits
                    if student_logits.shape != teacher_logits.shape:
                        min_length = min(student_logits.size(1), teacher_logits.size(1))
                        student_logits = student_logits[:, :min_length, :]
                        teacher_logits = teacher_logits[:, :min_length, :]

                    # Ensure student logits require gradients
                    student_logits = student_logits.requires_grad_(True)

                    distill_loss = (
                        F.kl_div(
                            F.log_softmax(student_logits / self.temperature, dim=-1),
                            F.softmax(teacher_logits / self.temperature, dim=-1),
                            reduction='batchmean',
                            log_target=False
                        ) * (self.temperature ** 2)
                    )

                    loss = (self.alpha * distill_loss) + ((1 - self.alpha) * student_loss)
                    loss = loss.requires_grad_(True)  # Ensure final loss requires gradients
        else:
            loss = student_loss

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Ensure we're using the device the model is on 
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        
        # If we're generating predictions
        if not prediction_loss_only:
            # Handle LoRA model's generate method differently
            if hasattr(model, "base_model"):
                generated_ids = model.base_model.generate(
                    inputs["input_features"],
                    language="da",
                    task="transcribe",
                    forced_decoder_ids=model.generation_config.forced_decoder_ids
                )
            else:
                generated_ids = model.generate(
                    inputs["input_features"],
                    language="da",
                    task="transcribe",
                    forced_decoder_ids=model.generation_config.forced_decoder_ids
                )

            # Compute loss if labels are provided
            with torch.no_grad():
                if hasattr(model, "base_model"):
                    outputs = model.base_model(
                        input_features=inputs["input_features"], 
                        labels=inputs["labels"]
                    )
                else:
                    outputs = model(
                        input_features=inputs["input_features"], 
                        labels=inputs["labels"]
                    )
            loss = outputs.loss
            return (loss.detach(), generated_ids.detach(), inputs["labels"].detach())

        # If prediction_loss_only=True
        with torch.no_grad():
            if hasattr(model, "base_model"):
                outputs = model.base_model(
                    input_features=inputs["input_features"], 
                    labels=inputs["labels"]
                )
            else:
                outputs = model(
                    input_features=inputs["input_features"], 
                    labels=inputs["labels"]
                )
        return (outputs.loss.detach(), None, inputs["labels"].detach())


def compute_metrics(pred, processor):
    metric = evaluate.load("cer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

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
    
    # Set memory efficient settings
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Check CUDA availability and select device
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        logger.info(f"Found {n_gpu} GPU(s)")
        # Use the first available GPU
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU available, using CPU")
    
    processor = WhisperProcessor.from_pretrained(
        cfg.teacher_model.name,
        language="da",
        task="transcribe"
    )

    # Load models with memory optimization
    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.teacher_model.name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Enable automatic device mapping
        low_cpu_mem_usage=True
    ).to(device)

    student_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.student_model.name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    ).to(device)

    # Add LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg.lora.rank,  # LoRA attention dimension
        lora_alpha=cfg.lora.alpha,  # Alpha scaling
        lora_dropout=cfg.lora.dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        bias="none",
        inference_mode=False,
    )

    # First freeze all parameters of the student model
    for param in student_model.parameters():
        param.requires_grad = False

    # Wrap student model with LoRA
    student_model = get_peft_model(student_model, lora_config)
    
    # Only enable training for LoRA parameters
    for name, param in student_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Remove the previous parameter handling code that was enabling too many parameters
    student_model.print_trainable_parameters()

    # Freeze teacher model
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Print parameter status
    logger.info("\nParameter status:")
    total_params = 0
    trainable_params = 0
    lora_params = 0
    
    for name, param in student_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if "lora_" in name:
                lora_params += param.numel()
            logger.info(f"Trainable: {name}")
    
    # Calculate percentages
    trainable_pct = (trainable_params / total_params) * 100
    frozen_pct = 100 - trainable_pct
    lora_pct = (lora_params / total_params) * 100
    
    # Print summary
    logger.info("\nParameter Summary:")
    logger.info(f"Total parameters:      {total_params:,}")
    logger.info(f"Trainable parameters:  {trainable_params:,} ({trainable_pct:.2f}%)")
    logger.info(f"Frozen parameters:     {total_params - trainable_params:,} ({frozen_pct:.2f}%)")
    logger.info(f"LoRA parameters:       {lora_params:,} ({lora_pct:.2f}%)")

    # Ensure student matches teacher's vocab config
    student_model.config.vocab_size = teacher_model.config.vocab_size
    student_model.config.pad_token_id = teacher_model.config.pad_token_id
    student_model.config.decoder_start_token_id = teacher_model.config.decoder_start_token_id
    student_model.config.eos_token_id = teacher_model.config.eos_token_id
    student_model.resize_token_embeddings(teacher_model.config.vocab_size)

    # Set forced decoder IDs & generation configs
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="da", task="transcribe")
    teacher_model.config.forced_decoder_ids = forced_decoder_ids
    teacher_model.generation_config.forced_decoder_ids = forced_decoder_ids
    student_model.config.forced_decoder_ids = forced_decoder_ids
    student_model.generation_config.forced_decoder_ids = forced_decoder_ids

    teacher_model.generation_config.language = "da"
    teacher_model.generation_config.task = "transcribe"
    student_model.generation_config.language = "da"
    student_model.generation_config.task = "transcribe"

    # Load preprocessed dataset
    logger.info("\nLoading preprocessed dataset...")
    dataset = load_dataset(cfg)

    # Limit validation dataset size
    if len(dataset['validation']) > 100:
        dataset['validation'] = dataset['validation'].select(range(100))

    logger.info(f"Train size: {len(dataset['train'])}")
    logger.info(f"Validation size: {len(dataset['validation'])}")

    def transform_fn(example):
        """Transform dataset examples into the format expected by the model."""
        # Process audio to input features
        audio = example["audio"]["array"]
        input_features = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features[0]
        
        # Process text to labels
        labels = processor(
            text=example["text"],
            return_tensors="pt"
        ).input_ids[0]
        
        # Move tensors to the correct device
        input_features = input_features.to(device)
        labels = labels.to(device)
        
        return {
            "input_features": input_features,
            "labels": labels
        }

    logger.info("Transforming datasets...")
    dataset['train'] = dataset['train'].map(transform_fn)
    dataset['validation'] = dataset['validation'].map(transform_fn)

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,   # Reduce eval batch size further
        eval_accumulation_steps=2,       # Reduce eval accumulation steps
        gradient_accumulation_steps=2,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=-1,
        warmup_steps=cfg.training.warmup_steps,
        warmup_ratio=cfg.training.warmup_ratio,
        fp16=False,
        bf16=True,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
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
        max_grad_norm=0.5,
        dataloader_pin_memory=False,
        dataloader_num_workers=1,
        optim="adafactor",
        eval_steps=100,              # Evaluate more frequently
    )

    # No need to manually set device properties
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

    class MetricCallback(transformers.TrainerCallback):
        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.results = {
                'train_loss': [],
                'eval_loss': [],
                'eval_cer': [],
                'learning_rate': [],
                'epoch': []
            }
            
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            self.results['eval_loss'].append(metrics.get('eval_loss', None))
            self.results['eval_cer'].append(metrics.get('eval_cer', None))
            self.results['epoch'].append(state.epoch)
            self.save_results()
                
        def on_log(self, args, state, control, logs, **kwargs):
            if 'loss' in logs:
                self.results['train_loss'].append(logs['loss'])
            if 'learning_rate' in logs:
                self.results['learning_rate'].append(logs['learning_rate'])
            self.save_results()
                
        def save_results(self):
            results_file = self.output_dir / 'training_results.txt'
            
            with open(results_file, 'w') as f:
                f.write("=== Knowledge Distillation Results ===\n\n")
                
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
                
                if self.results['eval_cer']:
                    f.write(f"Best CER: {min(self.results['eval_cer'])}\n")
                    f.write(f"Latest CER: {self.results['eval_cer'][-1]}\n")
                if self.results['eval_loss']:
                    f.write(f"Best Loss: {min(self.results['eval_loss'])}\n")
                    f.write(f"Latest Loss: {self.results['eval_loss'][-1]}\n\n")
                
                f.write("Detailed Results per Epoch:\n")
                for i in range(len(self.results['epoch'])):
                    f.write(f"\nEpoch {self.results['epoch'][i]}:\n")
                    if i < len(self.results['eval_cer']):
                        f.write(f"  CER: {self.results['eval_cer'][i]}\n")
                    if i < len(self.results['eval_loss']):
                        f.write(f"  Eval Loss: {self.results['eval_loss'][i]}\n")
            
        def on_train_end(self, args, state, control, **kwargs):
            self.save_results()

    metric_callback = MetricCallback(output_dir)

    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=cfg.training.temperature,
        alpha=cfg.training.alpha,
        processor=processor,
        model=student_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, processor=processor),
        callbacks=[metric_callback]
    )

    logger.info("\n=== Starting training ===")
    trainer.train()

    logger.info("\nSaving model and processor...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    logger.info("\n=== Training complete! ===")

if __name__ == "__main__":
    distill_whisper()
