from pathlib import Path
import torchaudio

def improve_audio_file(src: Path, dst: Path):
    effects = [
        ["rate", "48000"],
        ["channels", "2"],
        ["silenceremove", "1", "0.1", "0.01%", "-1", "0.5", "0.01%"],
        ["highpass", "40"],
        ["lowpass", "18000"],
        ["gain", "-n", "-1"],
    ]
    wav, sr = torchaudio.sox_effects.apply_effects_file(str(src), effects=effects)
    torchaudio.save(str(dst), wav, sr)
    return dst
