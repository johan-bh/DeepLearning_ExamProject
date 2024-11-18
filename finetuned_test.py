import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
from tqdm import tqdm
import os
import json

def load_model_and_processor(model_path):
    """
    Load a finetuned model and processor from the specified path
    """
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return model, processor, device

def evaluate_finetuned_model(model_path, output_dir="evaluation_results"):
    """
    Evaluate a finetuned Whisper model on the Danish Fleurs test set
    
    Args:
        model_path: Path to the finetuned model
        output_dir: Directory to save evaluation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Danish subset of Fleurs dataset
    dataset = load_dataset("google/fleurs", "da_dk", split="test")
    
    # Load model and processor
    model, processor, device = load_model_and_processor(model_path)
    
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
    
    # Prepare results
    results = {
        "model_path": model_path,
        "metrics": {
            "wer": float(word_error_rate),
            "cer": float(character_error_rate)
        },
        "examples": []
    }
    
    # Add example predictions
    for i in range(min(5, len(predictions))):
        results["examples"].append({
            "reference": references[i],
            "prediction": predictions[i]
        })
    
    # Save results
    model_name = os.path.basename(model_path)
    output_file = os.path.join(output_dir, f"{model_name}_results.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults for {model_name}:")
    print(f"Word Error Rate (WER): {word_error_rate:.4f}")
    print(f"Character Error Rate (CER): {character_error_rate:.4f}")
    print(f"Full results saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a finetuned Whisper model")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the finetuned model directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    evaluate_finetuned_model(args.model_path, args.output_dir) 