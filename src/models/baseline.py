import torch
from datasets import load_dataset
from transformers import pipeline
from jiwer import wer, cer
from tqdm import tqdm
import numpy as np

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

def evaluate_whisper(model_name):
    # Load Danish subset of the Fleurs dataset
    dataset = load_dataset("google/fleurs", "da_dk", split="test", trust_remote_code=True)
    
    # Initialize the ASR pipeline
    asr_pipeline = load_asr_pipeline(model_name)
    
    # Initialize lists to store predictions and references
    predictions = []
    references = [item["transcription"].lower() for item in dataset]
    
    # Process audio files in smaller batches
    batch_size = 8  # Reduced batch size for stability
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing audio batches"):
        batch = [dataset[j]["audio"] for j in range(i, min(i + batch_size, len(dataset)))]
        try:
            outputs = process_audio_batch(batch, asr_pipeline)
            
            # Handle outputs
            if isinstance(outputs, dict):
                predictions.append(outputs["text"].lower())
            else:
                predictions.extend([output["text"].lower() for output in outputs])
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {str(e)}")
            continue
    
    # Calculate WER and CER
    word_error_rate = wer(references, predictions)
    character_error_rate = cer(references, predictions)
    
    print(f"\nResults for {model_name} on Danish Fleurs:")
    print(f"Word Error Rate (WER): {word_error_rate:.4f}")
    print(f"Character Error Rate (CER): {character_error_rate:.4f}")
    
    # Save results to file
    with open(f"evaluation/baseline/baseline_{model_name.replace('/', '_')}.txt", "w", encoding="utf-8") as f:
        f.write(f"{model_name} Baseline Results on Danish Fleurs\n")
        f.write(f"Word Error Rate (WER): {word_error_rate:.4f}\n")
        f.write(f"Character Error Rate (CER): {character_error_rate:.4f}\n")
        
        # Save some example predictions
        f.write("\nExample Predictions:\n")
        for i in range(min(5, len(predictions))):
            f.write(f"\nReference: {references[i]}")
            f.write(f"\nPrediction: {predictions[i]}\n")

if __name__ == "__main__":
    evaluate_whisper("openai/whisper-large-v3") 