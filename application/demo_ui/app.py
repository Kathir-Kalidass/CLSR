from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_FILE = BASE_DIR / "templates" / "index.html"
REPORT_DIAGRAMS_DIR = BASE_DIR.parent / "report_pages" / "architecture_diagram"

app = FastAPI(title="CSLR Pipeline Demo UI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if REPORT_DIAGRAMS_DIR.exists():
    app.mount("/report-diagrams", StaticFiles(directory=str(REPORT_DIAGRAMS_DIR)), name="report-diagrams")


class DemoEngine:
    def __init__(self) -> None:
        self.gloss_tokens = [
            "HELLO",
            "HOW",
            "YOU",
            "ME",
            "GO",
            "SCHOOL",
            "TOMORROW",
            "BOOK",
            "WHERE",
            "NAME",
            "THANK-YOU",
            "FINE",
        ]
        self.sentences = [
            "Hello, how are you?",
            "I will go to school tomorrow.",
            "Where is the red book?",
            "What is your name?",
            "I am fine, thank you.",
        ]
        self.pipeline_order = ["module1", "module2", "module3", "module4", "module5"]
        self.evidence_notes = [
            "Dual-stream RGB+Pose follows 01_project_overview_merged.md and architecture mapping.",
            "Attention fusion and temporal modeling align with Module2 flow in 03_complete_modular_flow.md.",
            "CTC decoding and sliding-window inference follow CSLR_Project_Report.md real-time constraints.",
            "Gloss-to-text refinement reflects language correction design in 02_architecture_overview.md.",
            "Output text+speech delivery maps to sign_archi-module4 and deployment workflow.",
        ]

    def _module_payload(self, tick: int, client_state: dict[str, Any] | None = None) -> dict[str, Any]:
        client_state = client_state or {}
        camera_active = bool(client_state.get("camera_active", False))
        frame_hint = int(client_state.get("frame_hint", 0))
        client_resolution = str(client_state.get("resolution", "unknown"))

        fps = random.randint(24, 30)
        frame_count = 64
        confidence = round(random.uniform(0.74, 0.96), 2)
        latency_ms = random.randint(255, 455)

        sample_gloss = random.sample(self.gloss_tokens, k=3)
        sentence = self.sentences[tick % len(self.sentences)]
        partial = " ".join(sample_gloss)

        window_start = tick * 32
        window_end = window_start + frame_count - 1
        pose_detected = random.randint(66, 75)
        hand_visibility = round(random.uniform(0.78, 0.99), 2)
        attn_rgb = round(random.uniform(0.42, 0.71), 2)
        attn_pose = round(1.0 - attn_rgb, 2)
        beam_width = random.choice([5, 6, 8])

        module_data = {
            "module1": {
                "title": "Module 1: Video Acquisition + Preprocessing",
                "input": f"Live camera stream ({fps} FPS), raw RGB frames, client={client_resolution}",
                "process": "Decode -> sample -> resize 224x224 -> normalize -> pose/landmark extraction",
                "output": "RGB tensor 64x3x224x224 | Pose tensor 64x75x3",
                "note": "Implements preprocessing and temporal standardization before fusion.",
                "parse": [
                    f"window=frames[{window_start}:{window_end}] stride=32",
                    f"client_camera_active={camera_active} | client_frame_hint={frame_hint}",
                    f"pose_landmarks={pose_detected}/75, hand_visibility={hand_visibility}",
                    "normalization=ImageNet(mean/std), corrupt_frame_check=passed",
                ],
            },
            "module2": {
                "title": "Module 2: Feature Extraction + Attention Fusion",
                "input": "RGB tensor + Pose tensor",
                "process": "ResNet RGB stream + Pose encoder + cross-modal attention fusion",
                "output": "Fused embedding sequence 64x512",
                "note": "Follows efficient multi-feature attention design from referenced architecture.",
                "parse": [
                    "rgb_feat=64x512, pose_feat=64x256",
                    f"attention_weights: alpha_rgb={attn_rgb}, beta_pose={attn_pose}",
                    "fusion_out=64x512, fp16_path=enabled",
                ],
            },
            "module3": {
                "title": "Module 3: Temporal Modeling + CTC Decoding",
                "input": "Fused sequence embeddings",
                "process": "BiLSTM/Transformer temporal encoder + CTC logits + beam decoding",
                "output": f"Partial gloss stream: {partial}",
                "note": "Sliding-window decoding keeps latency in real-time budget.",
                "parse": [
                    f"temporal_in=64x512 -> ctc_logits=64x{len(self.gloss_tokens) + 1}",
                    f"beam_width={beam_width}, blank_pruned={random.randint(11, 24)}",
                    f"gloss_partial='{partial}'",
                ],
            },
            "module4": {
                "title": "Module 4: Translation + Grammar Correction",
                "input": f"Gloss tokens: {partial}",
                "process": "Gloss-to-text translation + grammar correction + fluency adjustment",
                "output": sentence,
                "note": "Converts gloss ordering to fluent English sentence structure.",
                "parse": [
                    f"translator_in=[{', '.join(sample_gloss)}]",
                    "decoder=t5/bart-style seq2seq, grammar_fix=on",
                    f"sentence_out='{sentence}'",
                ],
            },
            "module5": {
                "title": "Module 5: Output Layer (Caption + TTS)",
                "input": sentence,
                "process": "Subtitle render + confidence tagging + TTS queue/playback",
                "output": f"Caption published, audio_state={random.choice(['playing', 'ready'])}",
                "note": "Final multimodal output pipeline for text and speech feedback.",
                "parse": [
                    f"subtitle_commit_ts={int(time.time() * 1000)}",
                    f"tts_voice={random.choice(['female_neural', 'male_neural'])}, rate={random.choice(['0.95x', '1.0x', '1.05x'])}",
                    f"end_to_end_latency={latency_ms}ms",
                ],
            },
        }

        parser_console = []
        for module_key in self.pipeline_order:
            parser_console.extend([f"[{module_key}] {line}" for line in module_data[module_key]["parse"]])

        return {
            "timestamp": time.time(),
            "tick": tick,
            "status": "active",
            "latency_ms": latency_ms,
            "fps": fps,
            "confidence": confidence,
            "audio_state": random.choice(["speaking", "queued", "idle"]),
            "partial_gloss": partial,
            "final_sentence": sentence,
            "active_stage": self.pipeline_order[tick % len(self.pipeline_order)],
            "pipeline_order": self.pipeline_order,
            "evidence_note": self.evidence_notes[tick % len(self.evidence_notes)],
            "metrics": {
                "wer_proxy": round(random.uniform(0.11, 0.24), 3),
                "bleu_proxy": round(random.uniform(0.29, 0.41), 3),
                "window_frames": frame_count,
                "stride": 32,
            },
            "module1": module_data["module1"],
            "module2": module_data["module2"],
            "module3": module_data["module3"],
            "module4": module_data["module4"],
            "module5": module_data["module5"],
            "parser_console": parser_console,
        }

    async def stream(self, websocket: WebSocket, client_state: dict[str, Any]) -> None:
        tick = 0
        while True:
            payload = self._module_payload(tick, client_state=client_state)
            await websocket.send_text(json.dumps(payload))
            tick += 1
            await asyncio.sleep(1.05)


engine = DemoEngine()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return TEMPLATE_FILE.read_text(encoding="utf-8")


@app.websocket("/ws/demo")
async def websocket_demo(websocket: WebSocket) -> None:
    await websocket.accept()
    client_state: dict[str, Any] = {
        "camera_active": False,
        "frame_hint": 0,
        "resolution": "unknown",
    }

    async def recv_client() -> None:
        while True:
            text = await websocket.receive_text()
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                continue

            if message.get("type") == "client_video_stats":
                client_state["camera_active"] = bool(message.get("camera_active", False))
                client_state["frame_hint"] = int(message.get("frame_hint", 0))
                client_state["resolution"] = str(message.get("resolution", "unknown"))

    try:
        sender = asyncio.create_task(engine.stream(websocket, client_state=client_state))
        receiver = asyncio.create_task(recv_client())
        done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc:
                raise exc
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
