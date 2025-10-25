from pathlib import Path
import torchaudio, torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, data_dir: Path, max_len: int = 48000*4):
        self.files = [p for p in Path(data_dir).glob("*.wav")]
        self.max_len = max_len

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])
        x = wav.mean(0)  # mono
        if x.numel() >= self.max_len:
            x = x[:self.max_len]
        else:
            x = torch.nn.functional.pad(x, (0, self.max_len - x.numel()))
        return x
