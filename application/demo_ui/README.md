# CSLR Demo UI (Separate Showcase UI)

This is a separate **demo-only interface** for presenting the full CSLR pipeline.

## Stack
- Backend: FastAPI + WebSocket (Python)
- Frontend: Vanilla JS + Tailwind CSS + GSAP
- Camera: WebRTC (`getUserMedia`)
- Audio status: simulated output state for showcase mode

## What it shows
- Real-time module-by-module pipeline cards
- Each module has: Input, Process, Output, short technical note
- Live gloss stream and corrected sentence
- Confidence animation and latency/FPS indicators
- Camera preview with pulse effect when active

## Run

```bash
cd demo_ui
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Open `http://localhost:8080`.

## Note
This UI is intentionally separate from the main application UI and focuses on premium demo presentation.
