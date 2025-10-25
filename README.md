# 🎼 Deep Composer Pro (Open, GPU-Ready, Docker)

A production-ready, **open-source** music generation stack inspired by AWS DeepComposer — built with **FastAPI (Python)**, **PyTorch**, and a **React UI (CDN)**. Train on your **own WAV songs**, generate new tracks, and enhance with DSP — all in Docker with GPU support.

## 🚀 Quick start

```bash
unzip deep-composer-pro-full.zip
cd deep-composer-pro
./bootstrap.sh   # downloads MIT-licensed vocoder weights, CC0 samples, soundfont (optional)
docker compose build
docker compose up
```

- UI → http://localhost:8000  
- CLI → `python -m cli --help`

## 🧭 Architecture

```mermaid
graph TD
  A[React UI (CDN)] -->|REST| B(FastAPI Backend)
  B --> C[Trainer (PyTorch, AMP)]
  B --> D[Generator]
  D --> E[HiFi-GAN Vocoder*]
  E --> F[Sox/FFmpeg DSP]
  F --> A
```
\* Vocoder weights fetched by `bootstrap.sh` (MIT). Baseline generator works without them so the endpoint always returns audio.

## 📡 REST API
- `POST /api/upload` (multipart, multiple files via `files`)
- `POST /api/train`  body: `{epochs, batch_size, lr, scale}`
- `POST /api/generate` body: `{prompt, duration}`
- `POST /api/improve` form: `filename`
- `GET  /api/status/{job}`
- `GET  /api/health`

## 🐍 CLI Examples
```bash
python -m cli upload ./data/samples
python -m cli train --epochs 10
python -m cli generate --prompt "uplifting cinematic pad" --duration 16
python -m cli health
```

## 🧠 Training
- Upload multiple WAVs via UI (or place in `data/uploads/` once container is running).
- Training objective is a placeholder (self-reconstruction). Swap in your own loss/objective as needed in `trainer.py`.
- AMP enabled; will use GPU if available.

## 🎼 Generation
- The provided generator returns audio without external weights (baseline synthesis).
- For higher fidelity, integrate a spectrogram generator and enable the MIT-licensed **HiFi-GAN** vocoder weights downloaded by `bootstrap.sh`.

## 🪄 Improve (DSP)
- Provide a filename you uploaded; pipeline applies sample-rate convert to 48k, stereo, silence removal, high/low-pass, and gain normalization. Output: `*-improved.wav`.

## 🐳 Docker
- Base: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
- Installs PyTorch CUDA wheels via `requirements.txt`
- Requires NVIDIA Container Toolkit on the host for GPU access.

## 🪙 Licensing
- Code: MIT
- No proprietary models included; you own trained weights and outputs.
- Assets pulled by `bootstrap.sh` are MIT/CC0/open-licensed.
