import typer, requests, os, glob

BASE = "http://localhost:8000"
app = typer.Typer(help="Deep Composer Pro CLI")

@app.command()
def upload(path: str = typer.Argument(..., help="Glob or folder of wav files")):
    files = []
    if os.path.isdir(path):
        files = [os.path.join(path,f) for f in os.listdir(path) if f.lower().endswith('.wav')]
    else:
        files = glob.glob(path)
    if not files:
        typer.echo("No files matched"); raise typer.Exit(1)
    m = []
    for f in files:
        m.append(('files', (os.path.basename(f), open(f,'rb'), 'audio/wav')))
    r = requests.post(f"{BASE}/api/upload", files=m)
    typer.echo(r.text)

@app.command()
def train(epochs: int = 5, batch_size: int = 4, lr: float = 3e-4, scale: str = "small"):
    r = requests.post(f"{BASE}/api/train", json={"epochs":epochs,"batch_size":batch_size,"lr":lr,"scale":scale})
    typer.echo(r.text)

@app.command()
def generate(prompt: str = "Calm ambient pad", duration: int = 12):
    r = requests.post(f"{BASE}/api/generate", json={"prompt":prompt,"duration":duration})
    typer.echo(r.text)

@app.command()
def status(job_id: str):
    r = requests.get(f"{BASE}/api/status/{job_id}")
    typer.echo(r.text)

@app.command()
def health():
    r = requests.get(f"{BASE}/api/health")
    typer.echo(r.text)

if __name__ == "__main__":
    app()
