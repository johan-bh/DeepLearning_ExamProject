import torch
import torch.nn as nn
from torch.nn.functional import softmax
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, Trainer, TrainingArguments

# Load dataset
dataset = load_dataset("your_danish_dataset")

# Load teacher and student models
teacher_model = WhisperForConditionalGeneration.from_pretrained("path_to_finetuned_large_turbo")
student_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")  # Or smaller model

# Tokenizer (assumes teacher and student share the same tokenizer)
tokenizer = teacher_model.tokenizer

# Preprocess dataset for input IDs
def preprocess_data(batch):
    inputs = tokenizer(batch["audio_transcription"], return_tensors="pt", padding="longest", truncation=True)
    batch["input_ids"] = inputs["input_ids"]
    batch["labels"] = inputs["input_ids"]  # Labels for CrossEntropy loss
    return batch

processed_dataset = dataset.map(preprocess_data, batched=True)

# Generate teacher logits for soft targets
def generate_teacher_logits(batch):
    with torch.no_grad():
        outputs = teacher_model(input_ids=batch["input_ids"])
        # Use logits and apply softmax to get probabilities (soft targets)
        soft_targets = softmax(outputs.logits, dim=-1)
    batch["soft_targets"] = soft_targets
    return batch

soft_dataset = processed_dataset.map(generate_teacher_logits, batched=True)

# Define Knowledge Distillation loss
alpha = 0.5  # Balance between soft loss and hard loss
temperature = 3.0  # Higher temperature smooths teacher probabilities

def kd_loss(student_logits, teacher_probs, true_labels):
    # Soft loss (KL divergence with temperature scaling)
    soft_loss = nn.KLDivLoss()(nn.functional.log_softmax(student_logits / temperature, dim=-1),
                               teacher_probs / temperature)
    # Hard loss (CrossEntropy loss on true labels)
    hard_loss = nn.CrossEntropyLoss()(student_logits.view(-1, student_logits.size(-1)), true_labels.view(-1))
    return alpha * soft_loss + (1 - alpha) * hard_loss

# Custom Trainer for KD loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass for student model
        student_outputs = model(input_ids=inputs["input_ids"], labels=inputs["labels"])
        student_logits = student_outputs.logits

        # Get teacher's soft targets
        teacher_probs = inputs["soft_targets"]

        # Calculate KD loss
        loss = kd_loss(student_logits, teacher_probs, inputs["labels"])
        return (loss, student_outputs) if return_outputs else loss

# Training arguments
training_args = TrainingArguments(
    output_dir="path_to_student_model",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    save_strategy="epoch",
    logging_dir="logs",  # Directory for logs
    load_best_model_at_end=True,
    fp16=True,  # Mixed precision for faster training (optional)
)

# Initialize CustomTrainer
trainer = CustomTrainer(
    model=student_model,
    args=training_args,
    train_dataset=soft_dataset["train"],
    eval_dataset=soft_dataset["validation"],
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Evaluate the student model
def evaluate_student(student_model, dataset):
    predictions = []
    references = []
    for batch in dataset:
        with torch.no_grad():
            outputs = student_model.generate(input_ids=batch["input_ids"])
        predicted_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(predicted_texts)
        references.extend(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))
    return predictions, references

# Example evaluation
from jiwer import wer

predictions, references = evaluate_student(student_model, soft_dataset["test"])
test_wer = wer(references, predictions)
print(f"Student Model WER: {test_wer}")
