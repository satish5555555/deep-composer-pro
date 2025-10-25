from pathlib import Path
import numpy as np
import soundfile as sf

def generate_audio(prompt: str, out_path: Path, seconds: int = 8, sr: int = 48000):
    # Lightweight synthesis so generation works without external weights.
    base = 220.0 + (hash(prompt) % 7) * 20.0
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    sig = (
        0.6*np.sin(2*np.pi*base*t) +
        0.4*np.sin(2*np.pi*(base*5/4)*t) +
        0.3*np.sin(2*np.pi*(base*3/2)*t)
    )
    a = np.clip(t/0.3, 0, 1)
    r = np.clip((seconds - t)/0.8, 0, 1)
    env = np.minimum(a, r)
    sig = sig * env
    sig = (sig / (np.max(np.abs(sig))+1e-9) * 0.9).astype(np.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), sig, sr)
    return str(out_path)
