#!/usr/bin/env bash
set -e
echo "🎧 Bootstrap: fetching optional assets (weights, samples, soundfonts)"
mkdir -p composer/server/ml_models/weights data/samples assets/soundfonts

# MIT-licensed HiFi-GAN universal generator (public link; if it moves, replace with your mirror)
echo "📥 (Optional) Downloading HiFi-GAN vocoder weights..."
curl -L -o composer/server/ml_models/weights/hifigan_universal.pth "https://huggingface.co/jik876/hifi-gan/resolve/main/generator_universal.pth" || true

# CC0 samples (tiny) — you can delete after testing
echo "📥 Downloading sample WAV files..."
curl -L -o data/samples/sample1.wav "https://cdn.pixabay.com/download/audio/2022/03/15/audio_b9dfe763ff.wav" || true
curl -L -o data/samples/sample2.wav "https://cdn.pixabay.com/download/audio/2022/03/15/audio_3e05d75a9b.wav" || true

# Open soundfont
echo "📥 Downloading FluidR3 GM soundfont..."
curl -L -o assets/soundfonts/FluidR3_GM.sf2 "https://github.com/urish/cybr/releases/download/v0.1.0/FluidR3_GM.sf2" || true

echo "✅ Done. Now run:  docker compose up --build"
