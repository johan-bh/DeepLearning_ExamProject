from datasets import load_dataset
from typing import Tuple, List
import torch

class DanishASRDataLoader:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def load_fleurs_dataset(self, split: str = "test") -> dataset:
        """Load Danish Fleurs dataset."""
        return load_dataset("google/fleurs", "da_dk", split=split, trust_remote_code=True)

    def prepare_batch(self, batch: List) -> Tuple[torch.Tensor, List[str]]:
        """Prepare a batch of data."""
        audio = [item["audio"] for item in batch]
        transcriptions = [item["transcription"].lower() for item in batch]
        return audio, transcriptions 