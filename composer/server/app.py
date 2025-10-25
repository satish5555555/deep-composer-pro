import uuid, threading, subprocess
from pathlib import Path
from typing import Dict, Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import torch, torchaudio

from .model_service import improve_audio_file
from .ml_models.trainer import train_model
from .generator import generate_audio

# ======================================================
# Paths
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
CKPT_DIR = DATA_DIR / "checkpoints"

for p in (UPLOADS_DIR, OUTPUTS_DIR, CKPT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ======================================================
# App
# ======================================================
app = FastAPI(title="Deep Composer Pro", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# Static & UI Mount
# ======================================================
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR), html=False), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR), html=False), name="outputs")
app.mount("/checkpoints", StaticFiles(directory=str(CKPT_DIR), html=False), name="checkpoints")
app.mount("/ui", StaticFiles(directory=str(BASE_DIR / "server" / "static"), html=True), name="ui")

@app.get("/")
def root_redirect():
    """Redirect root URL to the UI"""
    return RedirectResponse(url="/ui/")

# ======================================================
# Jobs
# ======================================================
class JobStatus(BaseModel):
    id: str
    kind: str
    status: str
    message: Optional[str] = None
    artifact_path: Optional[str] = None

JOBS: Dict[str, JobStatus] = {}

def _run_job(job_id: str, fn, *args, **kwargs):
    JOBS[job_id].status = "running"
    try:
        artifact = fn(*args, **kwargs)
        JOBS[job_id].status = "done"
        JOBS[job_id].artifact_path = str(artifact) if artifact is not None else None
    except Exception as e:
        JOBS[job_id].status = "error"
        JOBS[job_id].message = repr(e)

def _start(kind: str, fn, *args, **kwargs) -> JobStatus:
    jid = uuid.uuid4().hex[:12]
    st = JobStatus(id=jid, kind=kind, status="pending")
    JOBS[jid] = st
    threading.Thread(target=_run_job, args=(jid, fn, *args), kwargs=kwargs, daemon=True).start()
    return st

# ======================================================
# API Endpoints
# ======================================================
@app.get("/api/health")
def health():
    cuda = torch.cuda.is_available()
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else []
    return {
        "status": "ok",
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "cuda": cuda,
        "gpus": gpus,
    }

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for file in files:
        dst = UPLOADS_DIR / file.filename
        with open(dst, "wb") as f:
            f.write(await file.read())
        # Auto convert MP3 -> WAV for training compatibility
        if dst.suffix.lower() == ".mp3":
            wav_path = dst.with_suffix(".wav")
            subprocess.run(["ffmpeg", "-y", "-i", str(dst), str(wav_path)], check=True)
            dst.unlink()
            dst = wav_path
        saved.append(dst.name)
    return {"ok": True, "files": saved}

@app.post("/api/improve")
def improve(filename: str = Form(...)):
    src = UPLOADS_DIR / filename
    if not src.exists():
        raise HTTPException(404, f"File not found: {filename}")
    def task():
        out = OUTPUTS_DIR / f"{src.stem}-improved.wav"
        improve_audio_file(src, out)
        return str(out)
    job = _start("improve", task)
    return {"job": job.id}

class TrainRequest(BaseModel):
    epochs: int = 5
    batch_size: int = 4
    lr: float = 3e-4
    scale: str = "small"

@app.post("/api/train")
def train(req: TrainRequest):
    def task():
        ckpt = CKPT_DIR / "music_model.pt"
        train_model(
            data_path=UPLOADS_DIR,
            output_path=ckpt,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            scale=req.scale,
            mixed_precision=True,
            distributed=False,
        )
        return str(ckpt)
    job = _start("train", task)
    return {"job": job.id}

class GenRequest(BaseModel):
    prompt: str = "Calm ambient pad"
    duration: int = 12

@app.post("/api/generate")
def generate(req: GenRequest):
    def task():
        out = OUTPUTS_DIR / "generated.wav"
        generate_audio(req.prompt, out, seconds=req.duration)
        return str(out)
    job = _start("generate", task)
    return {"job": job.id}

@app.get("/api/status/{job_id}")
def status(job_id: str):
    st = JOBS.get(job_id)
    if not st:
        raise HTTPException(404, "Unknown job")
    return st

