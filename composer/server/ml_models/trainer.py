from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from .models import build_model
from .dataset import AudioDataset

def train_model(data_path: Path, output_path: Path, epochs: int = 5, batch_size: int = 4, lr: float = 3e-4,
                scale: str = "small", mixed_precision: bool = True, distributed: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(scale=scale).to(device)
    ds = AudioDataset(data_path)
    if len(ds) == 0:
        raise RuntimeError("No WAV files found in uploads directory")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = GradScaler(enabled=mixed_precision)

    model.train()
    for ep in range(epochs):
        pbar = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}")
        for xb in pbar:
            xb = xb.to(device)
            with autocast(enabled=mixed_precision):
                out = model(xb)
                loss = loss_fn(out, xb)  # self-reconstruction objective (placeholder)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return output_path
