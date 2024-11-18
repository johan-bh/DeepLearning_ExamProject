import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
from tqdm import tqdm

def load_model_and_processor():
    # Load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return model, processor, device

def evaluate_whisper():
    # Load Danish subset of Fleurs dataset
    dataset = load_dataset("google/fleurs", "da_dk", split="test")
    
    # Load model and processor
    model, processor, device = load_model_and_processor()
    
    # Initialize lists to store predictions and references
    predictions = []
    references = []
    
    # Process each audio file
    for item in tqdm(dataset, desc="Processing audio files"):
        # Process audio
        input_features = processor(
            item["audio"]["array"],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # Decode prediction
        predicted_text = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        # Store prediction and reference
        predictions.append(predicted_text.lower())
        references.append(item["transcription"].lower())
    
    # Calculate WER and CER
    word_error_rate = wer(references, predictions)
    character_error_rate = cer(references, predictions)
    
    print(f"\nResults for Whisper-large-v3 on Danish Fleurs:")
    print(f"Word Error Rate (WER): {word_error_rate:.4f}")
    print(f"Character Error Rate (CER): {character_error_rate:.4f}")
    
    # Save results to file
    with open("baseline_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Whisper-large-v3 Baseline Results on Danish Fleurs\n")
        f.write(f"Word Error Rate (WER): {word_error_rate:.4f}\n")
        f.write(f"Character Error Rate (CER): {character_error_rate:.4f}\n")
        
        # Save some example predictions
        f.write("\nExample Predictions:\n")
        for i in range(min(5, len(predictions))):
            f.write(f"\nReference: {references[i]}")
            f.write(f"\nPrediction: {predictions[i]}\n")

if __name__ == "__main__":
    evaluate_whisper()
