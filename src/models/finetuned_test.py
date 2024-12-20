import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
from tqdm import tqdm
import os
import json
import yaml
from itertools import islice
import librosa
import random
import pandas as pd
import matplotlib.pyplot as plt

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
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="da", task="transcribe")
        model.config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.forced_decoder_ids = forced_decoder_ids
        print("✓ Forced decoder IDs set for Danish")
    except Exception as e:
        print(f"Warning: Could not set forced decoder IDs: {str(e)}")

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

    # Create top-level output directory
    top_level_output_dir = "evaluation/finetuned"
    os.makedirs(top_level_output_dir, exist_ok=True)

    # Extract model name and create a subfolder for this model
    model_name = os.path.basename(config['model']['path'])
    output_dir = os.path.join(top_level_output_dir, model_name)
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

    # cast_column to Audio with a fixed sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))  

    # Load model and processor
    print(f"\nStep 2/4: Loading model from {config['model']['path']}...")
    model, processor = load_model_and_processor(config)
    print(f"✓ Model loaded successfully on {next(model.parameters()).device}")

    print("\nStep 3/4: Processing audio samples...")
    predictions = []
    references = []
    dialects = []
    ages = []
    genders = []

    # Set max samples
    max_samples = 2500
    batch_size = config['dataset']['batch_size']
    batch_iter = batch_iterator(dataset, batch_size)

    pbar = tqdm(
        total=max_samples,
        desc="Processing audio samples",
        unit=" samples",
        ncols=100
    )

    sample_count = 0
    max_length_samples = 30 * processor.feature_extractor.sampling_rate
    target_sampling_rate = processor.feature_extractor.sampling_rate

    for batch in batch_iter:
        try:
            audio_arrays = []
            transcriptions = []
            batch_dialects = []     
            batch_ages = []
            batch_genders = []

            for sample in batch:
                if sample.get("audio") and sample["audio"].get("array") is not None:
                    audio_array = sample["audio"]["array"]
                    original_sampling_rate = sample["audio"]["sampling_rate"]
                    # Resample audio if necessary
                    audio_array = resample_audio(audio_array, original_sampling_rate, target_sampling_rate)
                    audio_arrays.append(audio_array)
                    transcriptions.append(sample["text"])
                    # Extract dialect, age, gender if present, else label "Unknown"
                    dia = sample.get("dialect", "Unknown")
                    ag = sample.get("age", "Unknown")
                    gen = sample.get("gender", "Unknown")

                    # If age is numeric, map to bins
                    if isinstance(ag, (int, float)):
                        if 0 <= ag <= 25:
                            ag = "0-25"
                        elif 25 < ag <= 50:
                            ag = "25-50"
                        else:
                            ag = "50+"

                    batch_dialects.append(dia)
                    batch_ages.append(ag)
                    batch_genders.append(gen)
                else:
                    print("Skipping invalid sample")

            if not audio_arrays:
                continue

            audio_inputs = processor(
                audio_arrays,
                sampling_rate=target_sampling_rate,
                return_tensors="pt",
                padding="max_length",
                max_length_samples=max_length_samples,
                truncation=True,
                return_attention_mask=False
            )

            input_features = audio_inputs.input_features.to(
                torch.float16 if config['model']['fp16'] else torch.float32
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_features,
                    return_dict_in_generate=True,
                    max_length=256,
                    temperature=1,
                    do_sample=False,
                    num_beams=5
                )

            batch_predictions = processor.batch_decode(
                outputs.sequences,
                skip_special_tokens=True
            )

            for ref, pred, dia, ag, gen in zip(transcriptions, batch_predictions, batch_dialects, batch_ages, batch_genders):
                pred = pred.strip().lower()
                ref = ref.strip().lower()
                if pred:
                    predictions.append(pred)
                    references.append(ref)
                    dialects.append(dia)
                    ages.append(ag)
                    genders.append(gen)

            processed_samples = len(batch_predictions)
            sample_count += processed_samples
            pbar.update(processed_samples)

            if sample_count >= max_samples:
                print(f"\nReached maximum sample limit of {max_samples}. Stopping.")
                break

        except Exception as e:
            print(f"\nError processing batch: {str(e)}")
            continue

    pbar.close()

    print("\nStep 4/4: Calculating final metrics...")

    if not predictions or not references:
        raise ValueError("No valid predictions or references generated")

    print(f"\nProcessed {len(predictions)} valid samples")

    # Prepare DataFrame for grouping
    data = pd.DataFrame({
        "reference": references,
        "prediction": predictions,
        "dialect": dialects,
        "age": ages,
        "gender": genders
    })

    # Filter out empty references or predictions if any
    data = data[(data["reference"].str.strip() != "") & (data["prediction"].str.strip() != "")]

    # Compute overall metrics
    overall_wer = wer(data["reference"].tolist(), data["prediction"].tolist())
    overall_cer = cer(data["reference"].tolist(), data["prediction"].tolist())

    # Function to compute metrics
    def compute_metrics(group):
        w = wer(group["reference"].tolist(), group["prediction"].tolist())
        c = cer(group["reference"].tolist(), group["prediction"].tolist())
        return pd.Series({"wer": w, "cer": c})

    # Compute metrics by groups
    dialect_stats = data.groupby("dialect").apply(compute_metrics).reset_index()
    age_stats = data.groupby("age").apply(compute_metrics).reset_index()
    gender_stats = data.groupby("gender").apply(compute_metrics).reset_index()

    # Create a combined set of categories:
    categories = []

    # Add gender categories (if present)
    for gcat in ["female", "male"]:
        if gcat in gender_stats["gender"].values:
            categories.append((gcat, gender_stats.loc[gender_stats["gender"] == gcat, "cer"].iloc[0],
                                     gender_stats.loc[gender_stats["gender"] == gcat, "wer"].iloc[0]))
        else:
            categories.append((gcat, float('nan'), float('nan')))

    # Add age categories
    for acat in ["0-25", "25-50", "50+"]:
        if acat in age_stats["age"].values:
            categories.append((acat, age_stats.loc[age_stats["age"] == acat, "cer"].iloc[0],
                                     age_stats.loc[age_stats["age"] == acat, "wer"].iloc[0]))
        else:
            categories.append((acat, float('nan'), float('nan')))

    # Add dialect categories
    for d in dialect_stats["dialect"].unique():
        cer_val = dialect_stats.loc[dialect_stats["dialect"] == d, "cer"].iloc[0]
        wer_val = dialect_stats.loc[dialect_stats["dialect"] == d, "wer"].iloc[0]
        categories.append((d, cer_val, wer_val))

    # Add overall at the end
    categories.append(("overall", overall_cer, overall_wer))

    # Convert categories to DataFrame
    cat_df = pd.DataFrame(categories, columns=["category", "cer", "wer"])

    # Rename long dialect names if needed
    rename_dict = {
        "syd for rigsgrænsen: mellemslesvisk, angelmål, fjoldemål": "syd for rigsgrænsen",
        "vendsysselsk (m. hanherred og læsø)": "vendsysselsk"
    }

    cat_df["category"] = cat_df["category"].replace(rename_dict)

    # Plot CER
    plt.figure(figsize=(12, 6))
    plt.bar(cat_df["category"], cat_df["cer"], color="steelblue")
    plt.title("Character Error Rate by Group (Lower is Better)")
    plt.xlabel("Group")
    plt.ylabel("Character Error Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cer_plot_path = os.path.join(output_dir, "cer_by_all_categories.png")
    plt.savefig(cer_plot_path)
    plt.close()

    # Plot WER
    plt.figure(figsize=(12, 6))
    plt.bar(cat_df["category"], cat_df["wer"], color="steelblue")
    plt.title("Word Error Rate by Group (Lower is Better)")
    plt.xlabel("Group")
    plt.ylabel("Word Error Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    wer_plot_path = os.path.join(output_dir, "wer_by_all_categories.png")
    plt.savefig(wer_plot_path)
    plt.close()

    # Prepare results
    results = {
        "model_path": config['model']['path'],
        "num_entries_processed": len(data),
        "metrics": {
            "overall_wer": float(overall_wer),
            "overall_cer": float(overall_cer)
        },
        "group_metrics": cat_df.to_dict(orient="records"),
        "examples": []
    }

    # Add 10 random samples
    sampled_data = data.sample(min(10, len(data)), random_state=42)
    for _, row in sampled_data.iterrows():
        results["examples"].append({
            "dialect": row["dialect"],
            "age": row["age"],
            "gender": row["gender"],
            "reference": row["reference"],
            "prediction": row["prediction"]
        })

    output_file = os.path.join(output_dir, f"{model_name}_results_extended.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nFull results saved to: {output_file}")
    print("Plots saved in:", output_dir)

if __name__ == "__main__":
    evaluate_finetuned_model()