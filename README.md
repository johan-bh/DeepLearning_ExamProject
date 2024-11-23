# Danish ASR Optimization

This project focuses on fine-tuning and pruning ASR models for Danish language processing.

## Project Structure

- `configs/`: Configuration files for models and training
- `data/`: Data storage and processing scripts
- `models/`: Model definitions and saved models
- `notebooks/`: Jupyter notebooks for exploration
- `src/`: Source code
- `tests/`: Unit tests

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

# Training and Evaluation

## Fine-tuning Whisper

The project uses Hydra for configuration management. The training script can be run in several ways:

1. Train with default configuration:
```bash
python src/models/finetune_whisper.py
```

2. Override specific parameters:
```bash
python src/models/finetune_whisper.py training.batch_size=16 training.learning_rate=2e-5
```

3. Use a different configuration file:
```bash
python src/models/finetune_whisper.py --config-name=experiment1_config
```

The training configuration can be modified in `configs/train_config.yaml`.

### Configuration Options

Key configuration parameters in `train_config.yaml`:

```yaml
training:
  output_dir: Directory to save model checkpoints
  num_train_epochs: Number of training epochs
  batch_size: Training batch size
  learning_rate: Learning rate for optimization
  fp16: Whether to use mixed precision training

model:
  name: Base model to fine-tune ("openai/whisper-large-v3-turbo")
  freeze_encoder: Whether to freeze encoder parameters

dataset:
  name: Dataset to use for training ("alexandrainst/nst-da")
  sampling_rate: Audio sampling rate
```

## Evaluating Models

To evaluate a fine-tuned model on the Danish Fleurs test set:
```bash
python src/models/finetuned_test.py path/to/finetuned/model --output_dir evaluation_results
```

To evaluate the baseline Whisper model:
```bash
python src/models/baseline.py
```

### Evaluation Metrics

The evaluation produces the following metrics:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Example predictions for manual inspection

Results will be saved in the `evaluation/` directory:
- `evaluation/baseline/`: Results for baseline models
- `evaluation/finetuned/`: Results for fine-tuned models

## Notes

- The training script automatically handles device placement (GPU/CPU)
- Gradient accumulation is used to handle larger effective batch sizes
- The encoder is frozen by default to speed up training
- Checkpoints are saved based on the best WER score
- Mixed precision training (fp16) is enabled by default for efficiency
```

