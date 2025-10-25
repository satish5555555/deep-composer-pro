import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            ResBlock1D(out_ch),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.net = nn.Sequential(
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            ResBlock1D(out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            if diff > 0:
                x = F.pad(x, (0, diff))
            else:
                x = x[:, :, :skip.size(-1)]
        x = x + skip
        return self.net(x)


class MultiScaleUNet(nn.Module):
    """1D Multi-scale U-Net with residual connections"""
    def __init__(self, in_ch=2, base_ch=64, num_scales=5):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=15, padding=7),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True)
        )

        enc_chs = [base_ch * (2**i) for i in range(num_scales)]
        self.downs = nn.ModuleList()
        prev_ch = base_ch
        for ch in enc_chs[1:]:
            self.downs.append(DownBlock(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = nn.Sequential(
            ResBlock1D(prev_ch),
            nn.Conv1d(prev_ch, prev_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        dec_chs = list(reversed(enc_chs[:-1]))
        self.ups = nn.ModuleList()
        for ch in dec_chs:
            self.ups.append(UpBlock(prev_ch, ch))
            prev_ch = ch

        self.out_conv = nn.Sequential(
            nn.Conv1d(base_ch, in_ch, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        x0 = self.in_conv(x)
        skips = [x0]
        cur = x0
        for d in self.downs:
            cur = d(cur)
            skips.append(cur)
        cur = self.bottleneck(cur)
        for u, skip in zip(self.ups, reversed(skips[:-1])):
            cur = u(cur, skip)
        out = self.out_conv(cur)
        return out

