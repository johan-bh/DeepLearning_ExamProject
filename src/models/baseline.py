import torch
from datasets import load_dataset, config
from transformers import pipeline
from jiwer import wer, cer
from tqdm import tqdm
import numpy as np
import os
import yaml
from pathlib import Path

# Set custom cache directory
config.HF_DATASETS_CACHE = "data/cached_datasets"  # relative to project root

# clear cuda cache
torch.cuda.empty_cache()

def load_asr_pipeline(model_name):
    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )
    return asr_pipeline

def process_audio_batch(batch, asr_pipeline):
    """Process a batch of audio files ensuring consistent format"""
    # Ensure all audio arrays are the same format
    processed_batch = []
    for audio in batch:
        # Convert to numpy array if needed
        if isinstance(audio["array"], list):
            audio["array"] = np.array(audio["array"])
        processed_batch.append({
            "array": audio["array"],
            "sampling_rate": audio["sampling_rate"]
        })
    
    # Generate transcriptions
    outputs = asr_pipeline(processed_batch, generate_kwargs={"language": "da"})
    return outputs

def save_results(predictions, references, word_error_rate, character_error_rate, model_name, dataset_name, dataset_config, split):
    """Save results to file with detailed naming"""
    # Create evaluation directory if it doesn't exist
    os.makedirs("evaluation/baseline", exist_ok=True)
    
    # Clean up names for file path (replace slashes and special characters)
    model_name_clean = model_name.replace('/', '_')
    dataset_name_clean = dataset_name.replace('/', '_')
    
    # Create filename with model, dataset and split information
    if dataset_config:
        results_file = f"evaluation/baseline/{model_name_clean}_{dataset_name_clean}_{dataset_config}_{split}.txt"
        dataset_display = f"{dataset_name} ({dataset_config}, {split} split)"
    else:
        results_file = f"evaluation/baseline/{model_name_clean}_{dataset_name_clean}_{split}.txt"
        dataset_display = f"{dataset_name} ({split} split)"
    
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"{model_name} Baseline Results on {dataset_display}\n")
        f.write(f"Word Error Rate (WER): {word_error_rate:.4f}\n")
        f.write(f"Character Error Rate (CER): {character_error_rate:.4f}\n")

        # Save some example predictions
        f.write("\nExample Predictions:\n")
        for i in range(min(5, len(predictions))):
            f.write(f"\nReference: {references[i]}")
            f.write(f"\nPrediction: {predictions[i]}\n")

def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

def evaluate_whisper(config_path="configs/model_config.yaml"):
    # Load configurations
    config = load_config(config_path)
    model_config = config['baseline_model']
    
    # Extract parameters from config
    model_name = model_config['name']
    dataset_name = model_config['dataset_name']
    dataset_config = model_config.get('dataset_config')
    label_column = model_config.get('label_column', 'text')
    batch_size = model_config['batch_size']
    device = model_config['device']
    max_samples = model_config.get('max_test_samples', None)
    
    # Use configured splits or fallback to defaults
    split = model_config.get('eval_split', 'validation')  # Use validation as test set
    
    try:
        # Load dataset with streaming enabled
        if dataset_config:
            dataset = load_dataset(
                dataset_name, 
                dataset_config, 
                split=split,  # Use the configured split
                streaming=True
            )
        else:
            dataset = load_dataset(
                dataset_name, 
                split=split,  # Use the configured split
                streaming=True
            )
        
        # Convert to iterator and take only what we need
        dataset = dataset.take(max_samples) if max_samples else dataset
        dataset = list(dataset)
        print(f"Dataset size: {len(dataset)}")
        
        # Print available columns for debugging
        # print(f"Available columns: {dataset[0].keys()}")
        
        # Create the pipeline
        asr_pipeline = pipeline(
            "automatic-speech-recognition", 
            model=model_name,
            torch_dtype=torch.float16,
            device=device
        )
        
        # Initialize lists for predictions and references
        predictions = []
        references = []  # Initialize references here instead of upfront

        # Process audio files in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing audio"):
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            
            # Get audio arrays and ensure consistent format
            audio_inputs = []
            batch_references = []  # Collect references corresponding to valid audio inputs
            for j in batch_indices:
                audio = dataset[j]["audio"]
                # Ensure audio is a dict with array and sampling_rate
                if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
                    audio_inputs.append({
                        "array": audio["array"],
                        "sampling_rate": audio["sampling_rate"]
                    })
                    # Append the corresponding reference
                    batch_references.append(dataset[j][label_column].lower())
                else:
                    print(f"Warning: Skipping malformed audio at index {j}")
                    continue
            
            if not audio_inputs:
                continue  # Skip the batch if no valid audio inputs
            
            # Get transcriptions
            try:
                outputs = asr_pipeline(
                    audio_inputs,
                    batch_size=len(audio_inputs),  # Use actual batch size
                    generate_kwargs={
                        "language": "da",
                        "task": "transcribe",
                        "num_beams": 1,
                        "max_length": 256
                    }
                )
                
                # Store predictions
                if isinstance(outputs, list):
                    batch_predictions = [output["text"].lower() for output in outputs]
                elif isinstance(outputs, dict):
                    batch_predictions = [outputs["text"].lower()]
                else:
                    print(f"Unexpected output type: {type(outputs)}")
                    continue
                    
                predictions.extend(batch_predictions)
                references.extend(batch_references)
                
            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                continue

        
        # Add this before the WER calculation
        if len(references) != len(predictions):
            print("\nDetailed length analysis:")
            print(f"References length: {len(references)}")
            print(f"Predictions length: {len(predictions)}")
            print("\nFirst few mismatched items:")
            for i in range(min(len(references), len(predictions), 5)):
                if references[i] != predictions[i]:
                    print(f"\nIndex {i}:")
                    print(f"Reference: {references[i]}")
                    print(f"Prediction: {predictions[i]}")
        
        # Calculate WER and CER
        word_error_rate = wer(references, predictions)
        character_error_rate = cer(references, predictions)
        
        print(f"\nResults for {model_name} on {dataset_name}:")
        print(f"Word Error Rate (WER): {word_error_rate:.4f}")
        print(f"Character Error Rate (CER): {character_error_rate:.4f}")
        
        # Save results
        save_results(
            predictions=predictions,
            references=references,
            word_error_rate=word_error_rate,
            character_error_rate=character_error_rate,
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split
        )
        
    except KeyError as e:
        print(f"Error: Column '{label_column}' not found in dataset.")
        print(f"Available columns: {dataset[0].keys()}")
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_whisper()