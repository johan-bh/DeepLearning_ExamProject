import torch
from datasets import load_dataset, Audio, Dataset, DatasetDict
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
import hydra
from omegaconf import DictConfig
from pathlib import Path

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features):
        # Extract input features and labels
        input_features = [feature["input_features"] for feature in features]
        labels = [feature["labels"] for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(
            {"input_features": input_features},
            padding=True,
            return_tensors="pt",
        )

        # Pad labels without returning attention_mask
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
            return_attention_mask=False,
        )

        # Replace padding tokens with -100
        labels_padded = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        batch["labels"] = labels_padded

        # Ensure 'input_ids' is not in the batch
        if 'input_ids' in batch:
            del batch['input_ids']

        return batch

def prepare_dataset(example, processor):
    """Prepare a single example for training"""
    # Load and resample audio
    audio_array = example["audio"]["array"]
    
    # Process audio features
    input_features = processor.feature_extractor(
        audio_array, 
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features[0]

    # Tokenize text
    labels = processor.tokenizer(
        example["transcription"],
        return_tensors="pt"
    ).input_ids[0]

    return {
        "input_features": input_features,
        "labels": labels
    }

def compute_metrics(pred, processor):
    """Compute CER metric during training"""
    metric = evaluate.load("cer")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute CER
    cer = metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def _prepare_inputs(self, inputs):
        # Remove 'input_ids' from inputs if present
        if 'input_ids' in inputs:
            del inputs['input_ids']
        return super()._prepare_inputs(inputs)

@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def train_whisper(cfg: DictConfig):
    print("\n=== Starting Whisper Fine-tuning ===")
    
    # Create output directory
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and processor
    print("\nLoading tokenizer and processor...")
    tokenizer = WhisperTokenizer.from_pretrained(
        cfg.model.name, 
        language="Danish", 
        task="transcribe"
    )
    
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language="Danish",
        task="transcribe"
    )
    processor.tokenizer = tokenizer  # Update tokenizer in processor
    
    # Load model

    print("\nLoading model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.name,
        device_map="auto",
        # Remove torch_dtype to load model in FP32
        # torch_dtype=torch.float16 if cfg.training.fp16 else torch.float32
    )
    model.config.use_cache = False  # Important for gradient checkpointing
        
    
    # Load dataset
    print("\nLoading dataset...")
    print(f"Using {cfg.dataset.train_split} split for training")
    print(f"Using {cfg.dataset.eval_split} split for evaluation")
    
    train_dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.config,
        split=cfg.dataset.train_split,
        streaming=True
    )
    
    eval_dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.config,
        split=cfg.dataset.eval_split,
        streaming=True
    )
    
    # Convert streaming datasets to regular datasets
    train_data = list(train_dataset.take(2000))  # Adjust number as needed
    val_data = list(eval_dataset.take(500))  # Adjust number as needed
    
    # Create datasets
    dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data)
    })
    
    # Cast audio column
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    
    # Preprocess dataset
    print("\nPreprocessing dataset...")
    dataset = dataset.map(
        partial(prepare_dataset, processor=processor),
        remove_columns=dataset["train"].column_names,
        num_proc=cfg.dataset.num_proc,
        desc="Processing dataset",
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=cfg.training.logging_steps,
        remove_unused_columns=True,
        label_names=["labels"],
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        eval_accumulation_steps=None,  # Disable eval accumulation
        predict_with_generate=True,    # Important for seq2seq evaluation
        generation_max_length=256,     # Set max generation length
        generation_num_beams=1,        # Use greedy decoding for faster eval
        include_inputs_for_metrics=True,
        gradient_checkpointing=False,
        optim="adafactor"
    )
    
    # Create trainer
    trainer = CustomSeq2SeqTrainer(
        args=training_args,
        model=model,
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
    train_whisper()
 