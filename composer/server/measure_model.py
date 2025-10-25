# -*- coding: utf-8 -*-
"""
measure_model.py — Robust model measurement script for MultiScaleUNet.

This version:
- Works from anywhere (inside 'server/' or from project root)
- Automatically finds and imports model_service
- Reports parameter count, model size, memory, inference speed, FLOPs (optional), and GPU VRAM usage
"""

import os
import sys
import time
import torch

# ---------------------------------------------------------------------
# Safe import setup
# ---------------------------------------------------------------------
# Ensure project root is in sys.path (handles both 'python measure_model.py' and 'python -m server.measure_model')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # Try absolute import (when run as module)
    from server.model_service import MultiScaleUNet
except ImportError:
    # Fallback for direct execution
    from server.ml_models.unet_model import MultiScaleUNet


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def human_format(num):
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            return f"{num:.2f}{unit}"
        num /= 1000.0
    return f"{num:.2f}T"


# ---------------------------------------------------------------------
# Core measurement logic
# ---------------------------------------------------------------------
def measure_model(device=None, input_seconds=4.0, sr=48000, save_tmp=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Measuring MultiScaleUNet on device: {device} ===")

    model = MultiScaleUNet(in_ch=2, base_ch=64, num_scales=5).to(device)
    model.eval()

    # Parameter summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    print(f"\n[PARAMETERS]")
    print(f"  Total Parameters     : {total_params:,} ({human_format(total_params)})")
    print(f"  Trainable Parameters : {trainable_params:,} ({human_format(trainable_params)})")
    print(f"  Model Memory (float32): {param_mem_mb:.2f} MB")

    # Save model size to disk
    if save_tmp:
        tmp_path = os.path.join(CURRENT_DIR, "tmp_model.pt")
        torch.save(model.state_dict(), tmp_path)
        file_size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
        print(f"  File Size on Disk     : {file_size_mb:.2f} MB")
        os.remove(tmp_path)

    # Dummy input
    seconds = input_seconds
    num_samples = int(sr * seconds)
    x = torch.randn(1, 2, num_samples).to(device)

    # Warm-up runs
    with torch.no_grad():
        for _ in range(2):
            _ = model(x)

    # GPU memory tracking
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Inference timing
    start = time.time()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    end = time.time()

    inference_time = end - start
    rtf = inference_time / seconds
    print(f"\n[INFERENCE PERFORMANCE]")
    print(f"  Input duration        : {seconds:.2f}s ({num_samples:,} samples)")
    print(f"  Inference time        : {inference_time:.4f}s")
    print(f"  Real-time factor (RTF): {rtf:.4f}x (<1.0 means faster than real time)")

    # GPU memory usage
    if device.startswith("cuda"):
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  Peak GPU VRAM usage   : {peak_mem_mb:.2f} MB")

    # FLOPs estimate
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(x,), verbose=False)
        print(f"\n[COMPUTATION ESTIMATE]")
        print(f"  Approx. FLOPs         : {flops / 1e9:.2f} GFLOPs")
    except ImportError:
        print("\n[COMPUTATION ESTIMATE]")
        print("  (Install 'thop' to estimate FLOPs: pip install thop)")

    print("\n✅ Measurement complete.\n")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    measure_model()

