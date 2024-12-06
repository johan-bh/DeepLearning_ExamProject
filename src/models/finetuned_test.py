import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
from tqdm import tqdm
import os
import json
import yaml
from itertools import islice
import librosa  # For audio resampling
from datasets import load_dataset, Audio, Dataset, DatasetDict

def load_config(config_path="configs/test_finetuning.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def batch_iterator(dataset, batch_size):
    dataset_iter = iter(dataset)
    while True:
        batch = list(islice(dataset_iter, batch_size))
        if not batch:
            break
        yield batch

def resample_audio(audio_array, original_sampling_rate, target_sampling_rate):
    if original_sampling_rate != target_sampling_rate:
        audio_array = librosa.resample(
            audio_array,
            orig_sr=original_sampling_rate,
            target_sr=target_sampling_rate
        )
    return audio_array

def load_model_and_processor(config):
    """Load a finetuned model and processor from the config"""
    model_path = config['model']['path']
    print(f"Loading processor...")
    
    # Load processor
    processor = WhisperProcessor.from_pretrained(model_path)
    print("✓ Processor loaded")
    
    print(f"Loading model weights (this might take a few minutes)...")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if config['model']['fp16'] else torch.float32,
        device_map=None,
        low_cpu_mem_usage=True
    )
    print("✓ Model weights loaded")
    
    print("Setting up model configuration...")
    try:
        # Get forced decoder IDs for Danish
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="da", task="transcribe")
        model.config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.forced_decoder_ids = forced_decoder_ids
        print("✓ Forced decoder IDs set for Danish")
    except Exception as e:
        print(f"Warning: Could not set forced decoder IDs: {str(e)}")
        print("The model may still work, but language forcing might not be optimal")

    # Move model to specified GPU
    available_devices = torch.cuda.device_count()
    if config['model']['device'] >= available_devices:
        raise ValueError(f"Invalid device ID {config['model']['device']}. Available devices: {available_devices}")
    print(f"Moving model to GPU {config['model']['device']}...")
    model = model.to(f"cuda:{config['model']['device']}")
    print("✓ Model ready on GPU")
    
    return model, processor

def evaluate_finetuned_model(config_path="configs/test_finetuning.yaml"):
    """Evaluate a finetuned Whisper model using the config file"""
    # Load config
    config = load_config(config_path)
    print("\n=== Starting Finetuned Model Evaluation ===")

    # Create output directory
    output_dir = "evaluation/finetuned"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset in streaming mode
    print(f"\nStep 1/4: Loading dataset '{config['dataset']['name']}' in streaming mode...")
    dataset = load_dataset(
        config['dataset']['name'],
        config['dataset']['config'],
        split=config['dataset']['split'],
        streaming=True
    )
    print(f"✓ Dataset loaded successfully in streaming mode")

    # cast_column
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))  

    # Load model and processor
    print(f"\nStep 2/4: Loading model from {config['model']['path']}...")
    model, processor = load_model_and_processor(config)
    print(f"✓ Model loaded successfully on {next(model.parameters()).device}")

    print("\nStep 3/4: Processing audio samples...")
    # Initialize lists
    predictions = []
    references = []
    dialects = []
    max_samples = 100

    # Create batch iterator
    batch_size = config['dataset']['batch_size']
    batch_iter = batch_iterator(dataset, batch_size)

    # Calculate total number of samples to process
    total_samples = min(max_samples, 100)  # Using max_samples or 100 as default
    
    # Create progress bar
    pbar = tqdm(
        total=total_samples,
        desc="Processing audio samples",
        unit=" samples",
        ncols=100,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    sample_count = 0
    max_length_samples = 30 * processor.feature_extractor.sampling_rate
    target_sampling_rate = processor.feature_extractor.sampling_rate

    for batch in batch_iter:
        try:
            # Get audio arrays, transcriptions, and dialects from the batch
            audio_arrays = []
            transcriptions = []
            batch_dialects = []  # Temporary list for the batch
            for sample in batch:
                if sample.get("audio") and sample["audio"].get("array") is not None:
                    audio_array = sample["audio"]["array"]
                    original_sampling_rate = sample["audio"]["sampling_rate"]
                    # Resample audio if necessary
                    audio_array = resample_audio(audio_array, original_sampling_rate, target_sampling_rate)
                    audio_arrays.append(audio_array)
                    transcriptions.append(sample["text"])
                    # Extract dialect, handle if missing
                    dialect = sample.get("dialect", "Unknown")
                    batch_dialects.append(dialect)
                else:
                    print("Skipping invalid sample")

            if not audio_arrays:
                continue  # Skip if no valid audio arrays

            # Preprocess audio inputs with correct padding
            audio_inputs = processor(
                audio_arrays,
                sampling_rate=target_sampling_rate,
                return_tensors="pt",
                padding="max_length",
                max_length_samples=max_length_samples,  # Pad/truncate to 480,000 samples
                truncation=True,
                return_attention_mask=False  # Whisper does not use attention masks
            )

            # Convert to desired dtype and move to device
            input_features = audio_inputs.input_features.to(
                torch.float16 if config['model']['fp16'] else torch.float32
            ).to(model.device)

            # Generate transcriptions
            with torch.no_grad():
                outputs = model.generate(
                    input_features,
                    return_dict_in_generate=True,
                    max_length=256,
                    temperature=0.0,  # Make output more deterministic
                    do_sample=False,  # Don't use sampling
                    num_beams=1  # Use greedy decoding
                )

            # Decode predictions
            batch_predictions = processor.batch_decode(
                outputs.sequences,
                skip_special_tokens=True
            )

            # Store predictions, references, and dialects
            for ref, pred, dialect in zip(transcriptions, batch_predictions, batch_dialects):
                if pred.strip():
                    predictions.append(pred.lower())
                    references.append(ref.lower())
                    dialects.append(dialect)

            # Update progress bar with actual processed samples
            processed_samples = len(batch_predictions)
            pbar.update(processed_samples)
            
            sample_count += processed_samples
            if sample_count >= max_samples:
                print(f"\nReached maximum sample limit of {max_samples}. Stopping.")
                break

        except Exception as e:
            print(f"\nError processing batch: {str(e)}")
            continue

    pbar.close()

    print("\nStep 4/4: Calculating final metrics...")

    # Validate predictions and references
    if not predictions or not references:
        raise ValueError("No valid predictions or references generated")

    # Report the number of processed samples
    print(f"\nProcessed {len(predictions)} valid samples")

    # Ensure all entries are non-empty
    valid_pairs = [(ref, pred, dialect) for ref, pred, dialect in zip(references, predictions, dialects)
                   if ref and pred]

    if not valid_pairs:
        raise ValueError("No valid prediction-reference pairs found")

    references, predictions, dialects = zip(*valid_pairs)
    references = list(references)
    predictions = list(predictions)
    dialects = list(dialects)

    # Print some examples including dialect
    print("\nFirst few predictions, references, and dialects:")
    for i in range(min(5, len(predictions))):
        print(f"\nExample {i+1}:")
        print(f"Dialect: '{dialects[i]}'")
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

    # Add examples including dialect
    for ref, pred, dialect in zip(references[:5], predictions[:5], dialects[:5]):
        results["examples"].append({
            "dialect": dialect,
            "reference": ref,
            "prediction": pred
        })

    # Save results
    model_name = os.path.basename(config['model']['path'])
    output_file = os.path.join(output_dir, f"{model_name}_results_test.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nFull results saved to: {output_file}")

if __name__ == "__main__":
    evaluate_finetuned_model()
