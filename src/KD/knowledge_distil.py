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
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.data_loader import load_dataset
import numpy as np
import os

# Set cache directories before importing HuggingFace libraries
cache_dir = Path("huggingface_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Force HuggingFace to use our cache directory
os.environ["HF_HOME"] = str(cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features):
        # Convert input features to numpy arrays if they're lists
        features = [{
            "input_features": feature["input_features"] if isinstance(feature["input_features"], np.ndarray) 
                            else np.array(feature["input_features"]),
            "labels": feature["labels"] if isinstance(feature["labels"], np.ndarray)
                     else np.array(feature["labels"])
        } for feature in features]
        
        # Get max length for padding
        max_length = max(feature["input_features"].shape[1] for feature in features)
        
        # Pad input features
        padded_inputs = []
        for feature in features:
            input_feature = feature["input_features"]
            padding_length = max_length - input_feature.shape[1]
            
            if padding_length > 0:
                padded_feature = np.pad(
                    input_feature,
                    ((0, 0), (0, padding_length), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
            else:
                padded_feature = input_feature
                
            padded_inputs.append(padded_feature)
        
        # Convert to tensor with bfloat16 dtype
        batch = {
            "input_features": torch.tensor(np.array(padded_inputs), dtype=torch.bfloat16),
        }

        # Pad labels
        labels = [feature["labels"] for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt"
        )
        
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id,
            -100
        )

        return batch

class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, teacher_model=None, temperature=2.0, alpha=0.5, processor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.processor = processor

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
            # Get teacher predictions first to verify they're correct
            with torch.no_grad():
                teacher_pred = self.teacher_model.generate(
                    inputs["input_features"],
                    language="da",
                    task="transcribe",
                    forced_decoder_ids=self.teacher_model.generation_config.forced_decoder_ids
                )
                
                # Log teacher predictions at the start of training
                if self.state.global_step == 0:
                    teacher_text = self.processor.batch_decode(teacher_pred, skip_special_tokens=True)
                    print("\nTeacher predictions:")
                    print(teacher_text[:2])
                    print("\nGround truth:")
                    ground_truth = self.processor.batch_decode(
                        inputs["labels"].masked_fill(inputs["labels"] == -100, self.processor.tokenizer.pad_token_id),
                        skip_special_tokens=True
                    )
                    print(ground_truth[:2])

            # Get teacher logits with proper input preparation
            labels_for_teacher = inputs["labels"].clone()
            labels_for_teacher[labels_for_teacher == -100] = self.teacher_model.config.pad_token_id
            decoder_input_ids = shift_tokens_right(
                labels_for_teacher,
                self.teacher_model.config.pad_token_id,
                self.teacher_model.config.decoder_start_token_id
            )

            # Teacher forward pass
            teacher_outputs = self.teacher_model(
                input_features=inputs["input_features"],
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True
            )
            teacher_logits = teacher_outputs.logits

            # Compute distillation loss with temperature scaling
            student_logits = outputs.logits
            if student_logits.shape != teacher_logits.shape:
                min_length = min(student_logits.size(1), teacher_logits.size(1))
                student_logits = student_logits[:, :min_length, :]
                teacher_logits = teacher_logits[:, :min_length, :]

            # KL divergence loss with temperature scaling
            distill_loss = (
                F.kl_div(
                    F.log_softmax(student_logits / self.temperature, dim=-1),
                    F.softmax(teacher_logits / self.temperature, dim=-1),
                    reduction='batchmean',
                    log_target=False
                ) * (self.temperature ** 2)
            )

            # Combine losses with alpha weighting
            loss = (self.alpha * distill_loss) + ((1 - self.alpha) * student_loss)

            # Log losses periodically
            if self.state.global_step % 100 == 0:
                print(f"\nStep {self.state.global_step}:")
                print(f"Student Loss: {student_loss.item():.4f}")
                print(f"Distill Loss: {distill_loss.item():.4f}")
                print(f"Combined Loss: {loss.item():.4f}")
        else:
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

    # Load processor and models with safe device allocation
    logger.info("\nLoading processor and models...")
    processor = WhisperProcessor.from_pretrained(
        cfg.teacher_model.name,
        language="da",
        task="transcribe"
    )

    # Modify model loading to use safe device allocation
    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.teacher_model.name,
        torch_dtype=torch.bfloat16
    ).to(device)

    student_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.student_model.name,
        torch_dtype=torch.bfloat16
    ).to(device)

    # Ensure student model has the same tokenizer and vocab size
    student_model.config.vocab_size = teacher_model.config.vocab_size
    student_model.config.pad_token_id = teacher_model.config.pad_token_id
    student_model.config.decoder_start_token_id = teacher_model.config.decoder_start_token_id
    student_model.config.eos_token_id = teacher_model.config.eos_token_id
    student_model.resize_token_embeddings(teacher_model.config.vocab_size)

    # After loading models and before loading dataset
    # Set forced decoder IDs for Danish
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="da", task="transcribe")
    teacher_model.config.forced_decoder_ids = forced_decoder_ids
    teacher_model.generation_config.forced_decoder_ids = forced_decoder_ids
    student_model.config.forced_decoder_ids = forced_decoder_ids
    student_model.generation_config.forced_decoder_ids = forced_decoder_ids

    # Set generation config
    teacher_model.generation_config.language = "da"
    teacher_model.generation_config.task = "transcribe"
    student_model.generation_config.language = "da"
    student_model.generation_config.task = "transcribe"

    # Verify teacher model outputs during training
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass for student
        outputs = model(**inputs)
        student_loss = outputs.loss

        # Only compute distillation loss during training
        if self.model.training:
            # Get teacher predictions first to verify they're correct
            with torch.no_grad():
                teacher_pred = self.teacher_model.generate(
                    inputs["input_features"],
                    language="da",
                    task="transcribe"
                )
                if self.state.global_step == 0:
                    teacher_text = self.processor.batch_decode(teacher_pred, skip_special_tokens=True)
                    print("\nTeacher predictions:")
                    print(teacher_text[:2])  # Print first two predictions

    # Load preprocessed dataset
    dataset = load_dataset(cfg)

    logger.info(f"Train size: {len(dataset['train'])}")
    logger.info(f"Validation size: {len(dataset['validation'])}")

    #  only take 500 samples from val
    dataset['validation'] = dataset['validation'].select(range(500))

    # Remove or comment out the audio casting part since data is already processed
    # logger.info("Casting audio column using cast_column...")
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def transform_fn(example):
        """Transform function for preprocessed data"""
        # First, let's log the structure to see what we're working with
        if not hasattr(transform_fn, 'structure_logged'):
            print("Example type:", type(example))
            print("Example structure:", example)
            transform_fn.structure_logged = True
        
        # Handle the specific dataset structure
        if isinstance(example, dict):
            # Process audio features - take only the first audio sample for now
            if 'audio' in example and isinstance(example['audio'], list):
                # Extract features using the processor
                input_features = processor(
                    [audio_item['array'] for audio_item in example['audio']],
                    sampling_rate=16000,
                    return_tensors="np"
                ).input_features
            else:
                raise KeyError("Expected 'audio' key with list of audio samples")
            
            # Process text labels
            if 'text' in example and isinstance(example['text'], list):
                # Convert text to labels using the processor
                labels = processor(
                    text=example['text'],
                    return_tensors="np",
                    padding=True
                ).input_ids
            else:
                raise KeyError("Expected 'text' key with list of transcriptions")
            
            return {
                "input_features": input_features,
                "labels": labels
            }
        else:
            raise TypeError(f"Expected dict, got {type(example)}")

    # Before applying the transform, let's inspect the dataset structure
    logger.info("Dataset structure before transform:")
    try:
        sample = dataset['train'][0]
        logger.info(f"Sample type: {type(sample)}")
        logger.info(f"Sample content: {sample}")
    except Exception as e:
        logger.error(f"Error inspecting dataset: {str(e)}")

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
        fp16=False,
        bf16=True,
        save_strategy="epoch",           # Changed back to "epoch"
        save_total_limit=2,             # Keep only the last 2 checkpoints
        evaluation_strategy="epoch",     # Changed back to "epoch"
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
        report_to=[],
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
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
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
            
            # Save results after each evaluation
            self.save_results()
                
        def on_log(self, args, state, control, logs, **kwargs):
            # Store training metrics
            if 'loss' in logs:
                self.results['train_loss'].append(logs['loss'])
            if 'learning_rate' in logs:
                self.results['learning_rate'].append(logs['learning_rate'])
            
            # Save results after each log
            self.save_results()
                
        def save_results(self):
            """Save current results to file"""
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
                
                # Write current metrics
                f.write("Current Results:\n")
                if self.results['eval_cer']:
                    f.write(f"Best CER: {min(self.results['eval_cer'])}\n")
                    f.write(f"Latest CER: {self.results['eval_cer'][-1]}\n")
                if self.results['eval_loss']:
                    f.write(f"Best Loss: {min(self.results['eval_loss'])}\n")
                    f.write(f"Latest Loss: {self.results['eval_loss'][-1]}\n\n")
                
                # Write detailed metrics per epoch
                f.write("Detailed Results per Epoch:\n")
                for i in range(len(self.results['epoch'])):
                    f.write(f"\nEpoch {self.results['epoch'][i]}:\n")
                    if i < len(self.results['eval_cer']):
                        f.write(f"  CER: {self.results['eval_cer'][i]}\n")
                    if i < len(self.results['eval_loss']):
                        f.write(f"  Eval Loss: {self.results['eval_loss'][i]}\n")
            
        def on_train_end(self, args, state, control, **kwargs):
            # Final save of results
            self.save_results()

    # Create the callback
    metric_callback = MetricCallback(output_dir)

    # Create trainer
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