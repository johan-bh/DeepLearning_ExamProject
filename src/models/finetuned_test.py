import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
from tqdm import tqdm
import os
import json
import yaml

def load_config(config_path="configs/test_finetuning.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_and_processor(config):
    """Load a finetuned model and processor from the config"""
    model_path = config['model']['path']
    print(f"Loading model from: {model_path}")
    
    # Load processor with language and task
    processor = WhisperProcessor.from_pretrained(
        model_path,
        language="Danish",
        task="transcribe",
        legacy_format=False
    )
    
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if config['model']['fp16'] else torch.float32,
        device_map=None,
        low_cpu_mem_usage=True
    )
    
    # Set forced decoder IDs
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="Danish", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    
    # Set other generation config parameters if needed
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.num_beams = 1
    model.generation_config.max_length = 256
    
    # Force using specific GPU
    torch.cuda.set_device(config['model']['device'])
    model = model.to(f"cuda:{config['model']['device']}")
    
    return model, processor

def evaluate_finetuned_model(config_path="configs/test_finetuning.yaml"):
    """Evaluate a finetuned Whisper model using config file"""
    # Load config
    config = load_config(config_path)
    print("\n=== Starting Finetuned Model Evaluation ===")
    
    # Create output directory
    output_dir = "evaluation/finetuned"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset: {config['dataset']['name']}")
    dataset = load_dataset(
        config['dataset']['name'],
        config['dataset']['config'],
        split=config['dataset']['split']
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}")
    
    # Load model and processor
    model, processor = load_model_and_processor(config)
    print(f"Model loaded successfully on {next(model.parameters()).device}")
    
    # Initialize lists
    predictions = []
    references = []
    
    # Process in batches
    batch_size = config['dataset']['batch_size']
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing audio"):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        
        try:
            # Get audio arrays and ensure proper length
            audio_inputs = processor(
                [dataset[j]["audio"]["array"] for j in batch_indices],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                max_length=processor.feature_extractor.n_samples,  # Use processor's expected length
                truncation=True,
                return_attention_mask=True
            )
            
            # Convert to FP16 if needed
            if config['model']['fp16']:
                input_features = audio_inputs.input_features.to(torch.float16)
            else:
                input_features = audio_inputs.input_features
            
            # Move to device
            input_features = input_features.to(model.device)
            attention_mask = audio_inputs.attention_mask.to(model.device)
            
            # Print shape for debugging
            print(f"Input features shape: {input_features.shape}")
            
            # Generate transcriptions
            with torch.no_grad():
                outputs = model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    language="da",
                    task="transcribe",
                    return_dict_in_generate=True,
                    max_length=256
                )
            
            # Decode predictions
            batch_predictions = processor.batch_decode(
                outputs.sequences,
                skip_special_tokens=True
            )
            
            # Store valid predictions and references
            for idx, pred in enumerate(batch_predictions):
                if pred.strip():  # Only store non-empty predictions
                    predictions.append(pred.lower())
                    references.append(dataset[batch_indices[idx]]["transcription"].lower())
                    
                    # Print for debugging
                    print(f"\nPrediction {len(predictions)}:")
                    print(f"Reference: {references[-1]}")
                    print(f"Predicted: {predictions[-1]}")
            
        except Exception as e:
            print(f"Error processing batch at index {i}: {str(e)}")
            continue
    
    # Validate predictions and references
    if not predictions or not references:
        raise ValueError("No valid predictions or references generated")
    
    print(f"\nProcessed {len(predictions)} valid samples out of {len(dataset)} total")
    
    # Ensure all entries are non-empty
    valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) 
                   if ref and pred]
    
    if not valid_pairs:
        raise ValueError("No valid prediction-reference pairs found")
    
    references, predictions = zip(*valid_pairs)
    references = list(references)
    predictions = list(predictions)
    
    # Print some examples
    print("\nFirst few predictions and references:")
    for i in range(min(5, len(predictions))):
        print(f"\nExample {i+1}:")
        print(f"Reference: '{references[i]}'")
        print(f"Prediction: '{predictions[i]}'")
        print("-" * 50)
    
    # Calculate metrics
    try:
        word_error_rate = wer(references, predictions)
        character_error_rate = cer(references, predictions)
        
        print(f"\nResults:")
        print(f"Word Error Rate (WER): {word_error_rate:.4f}")
        print(f"Character Error Rate (CER): {character_error_rate:.4f}")
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        raise
    
    # Prepare results
    results = {
        "model_path": config['model']['path'],
        "metrics": {
            "wer": float(word_error_rate),
            "cer": float(character_error_rate)
        },
        "examples": []
    }
    
    # Add examples
    for i in range(min(5, len(predictions))):
        results["examples"].append({
            "reference": references[i],
            "prediction": predictions[i]
        })
    
    # Save results
    model_name = os.path.basename(config['model']['path'])
    output_file = os.path.join(output_dir, f"{model_name}_results.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nFull results saved to: {output_file}")

if __name__ == "__main__":
    evaluate_finetuned_model()
