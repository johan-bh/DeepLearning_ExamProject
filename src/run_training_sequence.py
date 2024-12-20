import subprocess
import sys
from pathlib import Path
import logging
import time
import os
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_sequence.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def clear_cuda_memory():
    """Clear CUDA memory and cache"""
    if torch.cuda.is_available():
        # Empty the cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Try to reset GPU without killing our process
        try:
            import subprocess
            import os
            current_pid = os.getpid()
            # Kill other Python processes but not our own
            subprocess.run(f'for pid in $(pgrep python); do if [ $pid -ne {current_pid} ]; then kill -9 $pid; fi; done', 
                         shell=True, check=False)
            # Reset GPU
            subprocess.run(['nvidia-smi', '--gpu-reset'], check=False)
        except:
            pass
            
        logger.info("Cleared CUDA memory")

def check_gpu_memory():
    """Check and log GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # Convert to GB
            cached = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"GPU {i} Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

def run_script(script_path, description):
    """Run a Python script and handle its completion status"""
    logger.info(f"\n=== Starting {description} ===")
    start_time = time.time()
    
    try:
        # Check memory before clearing
        logger.info("GPU memory before clearing:")
        check_gpu_memory()
        
        # Clear CUDA memory before running script
        clear_cuda_memory()
        
        # Check memory after clearing
        logger.info("GPU memory after clearing:")
        check_gpu_memory()
        
        # Set environment variables to disable wandb and configure PyTorch memory
        env = os.environ.copy()
        env["WANDB_DISABLED"] = "true"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Run the script and stream output in real-time
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                logger.info(output.strip())
        
        # Wait for the process to complete and get return code
        return_code = process.poll()
        
        if return_code == 0:
            duration = time.time() - start_time
            logger.info(f"✓ {description} completed successfully! (Duration: {duration:.2f}s)")
            return True
        else:
            logger.error(f"✗ {description} failed with exit code {return_code}")
            return False
        
    except Exception as e:
        logger.error(f"✗ Unexpected error running {description}: {str(e)}")
        return False

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Define script paths
    distil_script = project_root / "src" / "KD" / "knowledge_distil.py"
    finetune_script = project_root / "src" / "models" / "finetune_whisper.py"
    
    # Verify scripts exist
    if not distil_script.exists():
        logger.error(f"Knowledge distillation script not found at {distil_script}")
        return
    if not finetune_script.exists():
        logger.error(f"Fine-tuning script not found at {finetune_script}")
        return
    
    # Run knowledge distillation first
    if run_script(distil_script, "Knowledge Distillation"):
        logger.info("\nKnowledge distillation completed successfully. Starting fine-tuning...")
        
        # Clear CUDA memory between runs
        clear_cuda_memory()
        
        # Run fine-tuning
        if run_script(finetune_script, "Fine-tuning"):
            logger.info("\n=== All training completed successfully! ===")
        else:
            logger.error("\n=== Fine-tuning failed! ===")
    else:
        logger.error("\n=== Knowledge distillation failed! Fine-tuning will not be started. ===")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining sequence interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main sequence: {str(e)}", exc_info=True) 