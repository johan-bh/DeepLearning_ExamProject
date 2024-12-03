#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J whisper_finetune
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=A100_80GB"
### -- set walltime limit: minutes --
#BSUB -W 240
### -- request 40GB of system-memory and ensure single host --
#BSUB -R "rusage[mem=40GB] span[hosts=1]"
### -- set the email address --
#BSUB -u s194495@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o whisper_train_%J.out
#BSUB -e whisper_train_%J.err
# -- end of LSF options --

# Load necessary modules
module load cuda/11.6
module load python3/3.10.14

# Create and activate a virtual environment
python3 -m venv env
source env/bin/activate

# Install required packages
pip install --upgrade pip
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
pip install transformers==4.30.2
pip install datasets
pip install evaluate
pip install jiwer
pip install hydra-core
pip install omegaconf
pip install pyyaml

# Print GPU information
nvidia-smi

# Run the training script
python3 src/models/finetune_whisper.py 