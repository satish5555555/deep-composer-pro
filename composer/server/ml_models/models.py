import torch, torch.nn as nn

class SimpleAudioModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=1024, num_layers=6, nhead=8):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B,T] -> [B,T,1]
        x = x.unsqueeze(-1)
        x = self.proj(x)
        x = self.enc(x)
        return self.head(x).squeeze(-1)

def build_model(scale="small"):
    if scale == "small":
        return SimpleAudioModel(hidden_dim=512, num_layers=4, nhead=8)
    if scale == "medium":
        return SimpleAudioModel(hidden_dim=1024, num_layers=8, nhead=16)
    if scale == "large":
        return SimpleAudioModel(hidden_dim=2048, num_layers=16, nhead=32)
    raise ValueError("unknown scale")
