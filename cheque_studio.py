"""
Cheque Verification Studio
===========================
FastAPI + SSE backend with inline HTML/CSS/JS frontend.

Two-phase pipeline:
  Tab 1 - Signature Verification
    Phase 1 Detection:  Falcon Perception locates the signature region on the cheque,
                        then the Line Sweep algorithm tightly crops it.
    Phase 2 Verification: Signature SVM classifies the cropped signature as
                          GENUINE or FORGED.

  Tab 2 - Data Extraction
    Gemma 4 E2B (mlx_vlm) reads 11 structured cheque fields.

  Tab 3 - Visual Reasoning
    Gemma 4 E2B answers free-form questions about the cheque image.

Entry point:
    python cheque_studio.py
    → http://localhost:7860

Pre-requisites:
    1. Signature SVM assets in signature_svm/data and signature_svm/model.pkl.
    2. Gemma 4 E2B auto-downloaded from HuggingFace on first run via mlx_vlm.

Step outputs (detection crops, signatures) are saved to step_outputs/<timestamp>/.
"""

import sys
import io
import time
import json
import base64
import traceback
from pathlib import Path

from PIL import Image
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.responses import Response
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from detection.ocr_extractor    import extract_cheque_fields
from detection.Line_Sweep.lineSweepDetect import line_sweep_with_bounds
from signature_svm.verifier import is_trained, verify_signature_pil

# Lazy Gemma 4 import — only loaded when reasoning tab is used
_gemma_load_fn = None
_gemma_vlm_fn  = None

def _try_load_gemma_fns():
    global _gemma_load_fn, _gemma_vlm_fn
    if _gemma_load_fn is not None:
        return True
    try:
        from agent_studio import _load_gemma, _vlm
        _gemma_load_fn = _load_gemma
        _gemma_vlm_fn  = _vlm
        return True
    except Exception:
        return False

STEP_OUTPUTS_DIR = BASE_DIR / "step_outputs"
STEP_OUTPUTS_DIR.mkdir(exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def decode_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def img_to_b64(img: Image.Image, max_w: int = 1200) -> str:
    if img.width > max_w:
        r   = max_w / img.width
        img = img.resize((max_w, int(img.height * r)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


_SSE_HEADERS = {
    "Cache-Control":     "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection":        "keep-alive",
}


# ── Tab 1: Signature Verification — SSE event generator ──────────────────────

def execute_signature_events(img: Image.Image):
    """
    Yields SSE event dicts for the Signature Verification pipeline:
      Phase 1: Falcon Perception → detect signature region (instance segmentation)
               Fallback: heuristic bottom-right crop if Falcon not installed
      Phase 1: Line Sweep → tight crop
      Phase 2: Signature SVM -> classify GENUINE / FORGED

    Each run saves intermediate outputs to step_outputs/<timestamp>/.
    """
    run_id   = int(time.time() * 1000)
    step_dir = STEP_OUTPUTS_DIR / str(run_id)
    step_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1a: Falcon Perception — signature detection ─────────────────
    yield {
        "type":       "detect_start",
        "label":      "Detecting signature — Falcon Perception",
        "model":      "Falcon Perception",
        "model_size": "0.6B",
        "task":       "Instance Segmentation",
        "phase":      "Phase 1 — Detection",
        "color":      "#6366f1",
    }

    t0             = time.time()
    bbox           = None
    method         = "heuristic"
    annotated      = img.copy()
    falcon_used    = False
    falcon_ran     = False   # True = _detect() completed without exception

    try:
        from agent_studio import _load_falcon, _detect, _render_detections
        _load_falcon()
        dets = _detect(img, "signature", task="segmentation")
        falcon_ran = True

        if dets:
            det = max(dets, key=lambda d: (
                (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
                if "bbox" in d else 0
            ))
            if "bbox" in det:
                bbox        = det["bbox"]
                method      = "falcon-perception"
                falcon_used = True
                annotated   = _render_detections(img, [det], "signature")

    except Exception as e:
        print(f"[Detection] Falcon error ({e}), using heuristic.")

    dt_detect = round(time.time() - t0, 2)

    # ── Heuristic fallback when Falcon produced no usable bbox ────────────
    # This handles: short signatures, disconnected strokes, Falcon misses.
    # Never return UNSIGNED here — always run SVM on the best available crop.
    if bbox is None:
        w, h  = img.size
        bbox  = [int(w * 0.50), int(h * 0.58), w, h]
        method = "heuristic"
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(annotated)
        draw.rectangle(bbox, outline=(99, 102, 241), width=3)
        draw.rectangle([bbox[0], max(0, bbox[1] - 22), bbox[2], bbox[1]],
                       fill=(99, 102, 241))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        except Exception:
            font = None
        draw.text((bbox[0] + 4, max(0, bbox[1] - 20)),
                  "Signature Region (heuristic)", fill=(255, 255, 255), font=font)
        if falcon_ran:
            # Falcon ran but found nothing — short / disconnected signature possible
            yield {
                "type":    "detect_notice",
                "message": (
                    "Falcon Perception did not detect a signature region. "
                    "Using heuristic crop (bottom-right area) — "
                    "short or disconnected signatures may still be verified by the Signature SVM."
                ),
            }

    try:
        annotated.save(step_dir / "1_detected_region.jpg", quality=90)
    except Exception:
        pass

    x1, y1, x2, y2 = bbox
    yield {
        "type":       "detect_complete",
        "count":      1,
        "image_b64":  img_to_b64(annotated),
        "duration_s": dt_detect,
        "bboxes":     [{"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "w": x2 - x1, "h": y2 - y1}],
        "method":     method,
        "model":      "Falcon Perception 0.6B" if falcon_used else "Heuristic",
        "color":      "#6366f1",
    }

    # ── Phase 1b: Crop + Line Sweep refinement ────────────────────────────
    yield {"type": "crop_start", "label": "Line Sweep — tight crop",
           "phase": "Phase 1 — Detection", "color": "#f59e0b"}

    w_img, h_img = img.size
    pad    = 20
    region = img.crop((max(0, x1 - pad), max(0, y1 - pad),
                       min(w_img, x2 + pad), min(h_img, y2 + pad)))

    sweep_result = line_sweep_with_bounds(region)

    # Guard: if Line Sweep returns a degenerate crop (< 30 px tall) use the
    # padded Falcon region directly — avoids the 10-px strip issue.
    if sweep_result["success"] and sweep_result["image"].size[1] >= 30:
        cropped  = sweep_result["image"]
        sweep_ok = True
    else:
        cropped  = region
        sweep_ok = False

    dt_sweep = round(time.time() - t0 - dt_detect, 4)

    try:
        cropped.save(step_dir / "2_cropped_signature.jpg", quality=90)
    except Exception:
        pass

    crop_info = {"w": cropped.width, "h": cropped.height, "sweep_success": sweep_ok}

    yield {
        "type":       "crop_complete",
        "image_b64":  img_to_b64(cropped),
        "duration_s": dt_sweep,
        "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "crop_size":  crop_info,
        "color":      "#f59e0b",
    }

    # ── Phase 2: Signature SVM Verification ───────────────────────────────
    if not is_trained():
        yield {
            "type":    "verify_stub",
            "message": (
                "Signature SVM assets are not ready. "
                "Check signature_svm/data/genuine, signature_svm/data/forged, and signature_svm/model.pkl."
            ),
        }
    else:
        yield {
            "type":  "verify_start",
            "label": "Signature SVM - Genuine vs Forged",
            "phase": "Phase 2 - Verification",
            "model": "Signature SVM",
            "color": "#10b981",
        }

        try:
            svm_label, conf = verify_signature_pil(cropped)
            verdict = "GENUINE" if svm_label == "REAL" else "FORGED" if svm_label == "FORGED" else "INCONCLUSIVE"
            dt_svm  = round(time.time() - t0 - dt_detect - dt_sweep, 4)

            yield {
                "type":       "verify_complete",
                "verdict":    verdict,
                "confidence": conf,
                "duration_s": dt_svm,
                "model":      "Signature SVM",
                "color":      "#10b981" if verdict == "GENUINE" else "#f43f5e" if verdict == "FORGED" else "#f59e0b",
            }
            try:
                (step_dir / "3_verdict.txt").write_text(
                    f"verdict={verdict}\n"
                    f"confidence={conf}\n"
                    f"detection_method={method}\n"
                )
            except Exception:
                pass

        except Exception as e:
            from signature_svm.verifier import ForgeryServerUnavailable
            if isinstance(e, ForgeryServerUnavailable):
                yield {
                    "type":    "verify_stub",
                    "message": (
                        "Signature SVM unavailable. Check the local signature_svm "
                        "dataset folders and model.pkl marker file."
                    ),
                }
            else:
                yield {"type": "error", "message": f"Signature verification failed: {e}",
                       "detail": traceback.format_exc()}
            return

    yield {
        "type":        "done",
        "json_output": {
            "image_size":        {"width": img.width, "height": img.height},
            "detected_bbox":     {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "crop_size":         crop_info,
            "detection_method":  method,
            "detection_time_s":  dt_detect,
        },
    }


# ── Tab 2: Data Extraction — SSE event generator ──────────────────────────────

def execute_extraction_events(img: Image.Image):
    """Yields SSE dicts for the Data Extraction tab (Gemma 4 E2B via mlx_vlm)."""

    yield {"type": "loading_models"}
    yield {"type": "models_ready"}

    yield {
        "type":  "extract_start",
        "label": "Reading cheque fields",
        "model": "Gemma 4 E2B",
        "color": "#8b5cf6",
    }

    t0 = time.time()
    try:
        fields = extract_cheque_fields(img)
        dt     = fields.pop("_duration_s", round(time.time() - t0, 2))

        if "error" in fields:
            yield {
                "type":    "extract_unavailable",
                "message": fields["error"],
            }
            yield {"type": "done", "json_output": {}}
            return

        yield {
            "type":       "extract_complete",
            "fields":     fields,
            "duration_s": dt,
            "model":      "Gemma 4 E2B",
            "color":      "#8b5cf6",
        }

    except Exception as e:
        yield {
            "type":    "error",
            "message": f"Extraction failed: {e}",
            "detail":  traceback.format_exc(),
        }
        return

    yield {"type": "done", "json_output": fields}


# ── Tab 3: Visual Reasoning — SSE event generator ────────────────────────────

def execute_reasoning_events(img: Image.Image, question: str):
    """Yields SSE dicts for the Visual Reasoning tab (Gemma 4 E2B via mlx_vlm)."""

    yield {"type": "loading_models", "model": "Gemma 4 E2B", "color": "#8b5cf6"}

    if not _try_load_gemma_fns():
        yield {
            "type":    "reason_unavailable",
            "message": (
                "Gemma 4 (mlx_vlm) is not installed or could not be loaded. "
                "Install with:  pip install mlx-vlm  (Apple Silicon required)."
            ),
        }
        yield {"type": "done"}
        return

    try:
        _gemma_load_fn()
    except Exception as e:
        yield {
            "type":    "error",
            "message": f"Failed to load Gemma 4 E2B: {e}",
            "detail":  traceback.format_exc(),
        }
        return

    yield {"type": "models_ready", "model": "Gemma 4 E2B", "color": "#8b5cf6"}

    yield {
        "type":     "reason_start",
        "question": question,
        "model":    "Gemma 4 E2B (2B)",
        "task":     "Visual Reasoning",
        "color":    "#8b5cf6",
    }

    t0 = time.time()
    try:
        answer = _gemma_vlm_fn(img, question)
        dt     = round(time.time() - t0, 1)
        yield {
            "type":       "reason_complete",
            "answer":     answer,
            "duration_s": dt,
            "model":      "Gemma 4 E2B",
            "color":      "#8b5cf6",
        }
    except Exception as e:
        yield {
            "type":    "error",
            "message": f"Gemma 4 inference failed: {e}",
            "detail":  traceback.format_exc(),
        }
        return

    yield {"type": "done"}


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Cheque Verification Studio")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.post("/api/verify/stream")
async def verify_stream(request: Request):
    try:
        data = await request.json()
        img  = decode_image(data["image_b64"])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    def generate():
        try:
            for event in execute_signature_events(img):
                yield sse(event)
        except Exception as e:
            traceback.print_exc()
            yield sse({"type": "error", "message": str(e),
                       "detail": traceback.format_exc()[-6000:]})

    return StreamingResponse(generate(),
                             media_type="text/event-stream; charset=utf-8",
                             headers=dict(_SSE_HEADERS))


@app.post("/api/extract/stream")
async def extract_stream(request: Request):
    try:
        data = await request.json()
        img  = decode_image(data["image_b64"])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    def generate():
        try:
            for event in execute_extraction_events(img):
                yield sse(event)
        except Exception as e:
            traceback.print_exc()
            yield sse({"type": "error", "message": str(e),
                       "detail": traceback.format_exc()[-6000:]})

    return StreamingResponse(generate(),
                             media_type="text/event-stream; charset=utf-8",
                             headers=dict(_SSE_HEADERS))


@app.post("/api/reason/stream")
async def reason_stream(request: Request):
    try:
        data     = await request.json()
        img      = decode_image(data["image_b64"])
        question = data.get("question", "").strip()
        if not question:
            return JSONResponse({"error": "question is required"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    def generate():
        try:
            for event in execute_reasoning_events(img, question):
                yield sse(event)
        except Exception as e:
            traceback.print_exc()
            yield sse({"type": "error", "message": str(e),
                       "detail": traceback.format_exc()[-6000:]})

    return StreamingResponse(generate(),
                             media_type="text/event-stream; charset=utf-8",
                             headers=dict(_SSE_HEADERS))


@app.get("/api/model/status")
async def model_status():
    """Return whether the local signature verifier assets are ready."""
    ready = is_trained()
    return JSONResponse({
        "trained": ready,
        "ready": ready,
        "model": "Signature SVM",
    })


def _b64_png(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def _read_upload(request: Request):
    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return None
    img_bytes = await upload.read()
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _verdict_payload(ver: dict) -> dict:
    """Map agent verification output to the frontend response shape."""
    conf = float(ver.get("confidence", 0.5) or 0.5)
    model_name = ver.get("model", "Signature SVM")

    if ver.get("verdict") == "GENUINE":
        return {
            "message": "The signature is genuine",
            "prediction": [[round(conf * 100, 2), round((1.0 - conf) * 100, 2)]],
            "confidence": conf,
            "model": model_name,
        }
    if ver.get("verdict") == "FORGED":
        return {
            "message": "The signature appears forged",
            "prediction": [[round((1.0 - conf) * 100, 2), round(conf * 100, 2)]],
            "confidence": conf,
            "model": model_name,
        }
    if ver.get("verdict") == "INCONCLUSIVE":
        return {
            "message": "The signature result is inconclusive",
            "prediction": [[None, None]],
            "confidence": conf,
            "model": model_name,
            "note": ver.get(
                "note",
                "The SVM margin is too close to the decision boundary. Use enrolled reference signatures for a reliable identity decision.",
            ),
        }
    return {
        "error": ver.get("error", "Verification unavailable. Check signature_svm assets."),
        "confidence": conf,
        "model": model_name,
    }


@app.post("/api/signature/verify-crop")
async def signature_verify_crop(request: Request):
    """
    Fast step-2 endpoint. Verifies an already-cropped signature image without
    rerunning Falcon detection or Line Sweep.
    """
    sig_img = await _read_upload(request)
    if sig_img is None:
        return JSONResponse({"error": "no signature crop uploaded"}, status_code=400)

    t0 = time.time()
    try:
        label, conf = verify_signature_pil(sig_img)
        verdict = "GENUINE" if label == "REAL" else "FORGED" if label == "FORGED" else "INCONCLUSIVE"
        ver = {
            "verdict": verdict,
            "confidence": conf,
            "model": "Signature SVM",
            "note": "Low SVM margin; enrolment/reference signatures are needed for a reliable identity decision." if verdict == "INCONCLUSIVE" else "",
            "duration_s": round(time.time() - t0, 3),
        }
    except Exception as e:
        ver = {
            "verdict": "ERROR",
            "confidence": 0.0,
            "model": "Signature SVM",
            "error": str(e),
            "duration_s": round(time.time() - t0, 3),
        }

    return JSONResponse({
        "verification": ver,
        "verdict": _verdict_payload(ver),
    })


@app.post("/api/cheque/verify")
async def cheque_verify(request: Request):
    """
    Detect + crop the signature, then classify with the Signature SVM.
    Returns crop images + verdict in the format expected by the JS frontend.
    """
    img = await _read_upload(request)
    if img is None:
        return JSONResponse({"error": "no file uploaded"}, status_code=400)

    from agent import ChequeVerificationAgent
    agent = ChequeVerificationAgent()

    det = agent.detect_signature(img)
    ls  = agent.line_sweep_crop(img, det["bbox"])
    sig_img = ls.get("image") or img.crop(det["bbox"])

    ver = agent.verify_signature(sig_img)

    return JSONResponse({
        "detection":          {"bbox": det["bbox"], "method": det["method"],
                               "duration_s": det["duration_s"]},
        "line_sweep":         {"success": ls.get("success", False)},
        "verification":       ver,
        "verdict":            _verdict_payload(ver),
        "annotated_b64":      _b64_png(det.get("annotated", img)),
        "signature_crop_b64": _b64_png(sig_img),
    })


@app.post("/api/cheque/crop")
async def cheque_crop(request: Request):
    """
    Step 1 — detect + line-sweep only. Returns annotated cheque and cropped
    signature as base64. Does not run signature verification.
    """
    img = await _read_upload(request)
    if img is None:
        return JSONResponse({"error": "no file uploaded"}, status_code=400)

    from agent import ChequeVerificationAgent
    agent = ChequeVerificationAgent()

    det = agent.detect_signature(img)
    ls  = agent.line_sweep_crop(img, det["bbox"])
    sig_img = ls.get("image") or img.crop(det["bbox"])

    return JSONResponse({
        "detection": {
            "bbox":       det["bbox"],
            "method":     det["method"],
            "duration_s": det["duration_s"],
        },
        "line_sweep":        {"success": ls.get("success", False)},
        "annotated_b64":     _b64_png(det.get("annotated", img)),
        "signature_crop_b64": _b64_png(sig_img),
    })


@app.post("/api/cheque/extract")
async def cheque_extract(request: Request):
    """
    "Extract Cheque Data" button - data extraction only.
    Pipeline: Qwen OCR hint + Gemma 4 + regex cleanup -> 11 cheque fields.
    Safe to run with just python app.py.
    """
    img = await _read_upload(request)
    if img is None:
        return JSONResponse({"error": "no file uploaded"}, status_code=400)

    from agent import ChequeVerificationAgent
    agent = ChequeVerificationAgent()

    try:
        fields = agent.extract_fields(img)
        fields.pop("_duration_s", None)
    except Exception as e:
        fields = {"error": str(e)}

    return JSONResponse({"extraction": fields})


@app.post("/api/cheque/forgery")
async def cheque_forgery(request: Request):
    """
    Compatibility endpoint: detection + Line Sweep + Signature SVM forgery check.
    """
    img = await _read_upload(request)
    if img is None:
        return JSONResponse({"error": "no file uploaded"}, status_code=400)

    from agent import ChequeVerificationAgent
    agent = ChequeVerificationAgent()

    det = agent.detect_signature(img)
    ls  = agent.line_sweep_crop(img, det["bbox"])
    sig_img = ls.get("image") or img.crop(det["bbox"])

    try:
        ver = agent.verify_signature(sig_img)
    except Exception as e:
        ver = {
            "verdict": "ERROR",
            "confidence": 0.0,
            "model": "none",
            "error": str(e),
        }

    return JSONResponse({
        "detection": {
            "bbox":   det["bbox"],
            "method": det["method"],
            "duration_s": det["duration_s"],
        },
        "line_sweep": {"success": ls.get("success", False)},
        "verification": ver,
        "signature_crop_b64": _b64_png(sig_img),
        "annotated_b64":      _b64_png(det.get("annotated", img)),
    })


# ── HTML frontend ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cheque Verification System</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b1020;--surface:#111827;--elevated:#1f2937;--soft:#172033;--border:#334155;
  --t1:#f8fafc;--t2:#cbd5e1;--t3:#94a3b8;
  --indigo:#4f46e5;--amber:#f59e0b;--emerald:#059669;--rose:#e11d48;--violet:#7c3aed;--cyan:#0891b2;
  --font:-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",Roboto,sans-serif;
  --mono:"SF Mono","Fira Code",monospace;
}
html{background:var(--bg);color:var(--t1);font-family:var(--font);font-size:14px;line-height:1.5}
body{min-height:100vh;display:flex;flex-direction:column;align-items:center;padding-bottom:48px;
  background:
    radial-gradient(circle at 15% -10%, rgba(8,145,178,.18), transparent 34%),
    radial-gradient(circle at 88% 4%, rgba(5,150,105,.14), transparent 28%),
    linear-gradient(180deg,#0b1020 0%,#0f172a 55%,#111827 100%)}

/* ── Header ── */
.header{text-align:center;padding:32px 20px 18px;width:100%;max-width:1320px}
.header h1{font-size:28px;font-weight:750;letter-spacing:0}
.header p{font-size:13px;color:var(--t2);margin-top:6px}
.badges{display:flex;gap:8px;justify-content:center;margin-top:10px;flex-wrap:wrap}
.badge{font-size:11px;padding:5px 10px;border-radius:999px;font-weight:650;border:1px solid}
.badge.glm   {color:var(--cyan);  background:rgba(6,182,212,.1); border-color:rgba(6,182,212,.3)}
.badge.sweep {color:var(--amber); background:rgba(245,158,11,.1);border-color:rgba(245,158,11,.3)}
.badge.verifier{color:var(--emerald);background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.3)}
.badge.phase {color:var(--indigo);background:rgba(99,102,241,.1);border-color:rgba(99,102,241,.3)}

/* ── App wrapper ── */
.app{width:100%;max-width:1320px;padding:0 20px}

/* ── Phase banner ── */
.phase-banner{display:flex;gap:0;margin-bottom:18px;border-radius:12px;overflow:hidden;
  border:1px solid rgba(148,163,184,.18);box-shadow:0 18px 40px rgba(0,0,0,.18)}
.phase-seg{flex:1;padding:12px 16px;background:rgba(17,24,39,.88);text-align:center}
.phase-seg:not(:last-child){border-right:1px solid rgba(148,163,184,.14)}
.phase-seg .ps-num{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:.08em;color:var(--t3);margin-bottom:2px}
.phase-seg .ps-name{font-size:12px;font-weight:600;color:var(--t2)}
.phase-seg .ps-model{font-size:11px;color:var(--t3);margin-top:1px}
.phase-seg.active .ps-name{color:var(--t1)}
.phase-seg.p1{border-left:3px solid var(--indigo)}
.phase-seg.p2{border-left:3px solid var(--emerald)}

/* ── Tab nav ── */
.tab-nav{display:flex;gap:4px;margin:0 auto 20px;background:rgba(17,24,39,.9);
  border:1px solid rgba(148,163,184,.18);border-radius:12px;padding:5px;width:fit-content;
  box-shadow:0 14px 34px rgba(0,0,0,.2)}
.tab-btn{padding:9px 18px;border-radius:9px;border:none;background:transparent;
  color:var(--t2);font-size:13px;font-weight:500;cursor:pointer;
  font-family:var(--font);transition:all .15s;display:flex;align-items:center;gap:6px}
.tab-btn.active{background:var(--elevated);color:var(--t1);box-shadow:inset 0 0 0 1px rgba(255,255,255,.04)}
.tab-btn:hover:not(.active){color:var(--t1)}
.tab-pane{display:none}.tab-pane.active{display:block}

/* ── Grids ── */
.grid-3{display:grid;grid-template-columns:320px 1fr 1fr;gap:20px;align-items:start}
.grid-2{display:grid;grid-template-columns:360px 1fr;gap:20px;align-items:start}
@media(max-width:960px){.grid-3,.grid-2{grid-template-columns:1fr}}

/* ── Panel ── */
.panel{background:rgba(17,24,39,.92);border:1px solid rgba(148,163,184,.16);
  border-radius:12px;overflow:hidden;margin-bottom:16px;box-shadow:0 18px 44px rgba(0,0,0,.18)}
.panel-hd{padding:14px 18px 11px;border-bottom:1px solid rgba(148,163,184,.12);
  display:flex;align-items:center;gap:8px}
.panel-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.panel-title{font-size:13px;font-weight:600}
.panel-sub{font-size:11px;color:var(--t3);margin-top:1px}
.panel-dur{font-size:11px;color:var(--t3);font-family:var(--mono);margin-left:auto}
.panel-bd{padding:16px}

/* ── Upload zone ── */
.upload-zone{border:2px dashed rgba(148,163,184,.34);border-radius:12px;padding:32px 16px;
  text-align:center;cursor:pointer;transition:border-color .2s;min-height:180px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  background:linear-gradient(180deg,rgba(31,41,55,.8),rgba(23,32,51,.85));overflow:hidden;position:relative}
.upload-zone:hover{border-color:var(--t3)}
.upload-zone.dragover{border-color:var(--indigo);background:rgba(99,102,241,.05)}
.upload-zone.has-image{padding:0;border-style:solid;border-color:var(--border)}
.upload-zone.has-image .upload-ph{display:none}
.upload-preview{width:100%;height:auto;max-height:280px;object-fit:contain;
  border-radius:8px;display:none}
.upload-zone.has-image .upload-preview{display:block}
.upload-clear{position:absolute;top:8px;right:8px;width:26px;height:26px;
  border-radius:50%;background:rgba(0,0,0,.75);border:none;color:#fff;cursor:pointer;
  display:none;align-items:center;justify-content:center;font-size:16px;z-index:10;line-height:1}
.upload-zone.has-image .upload-clear{display:flex}
.upload-icon{width:28px;height:28px;color:var(--t3);margin-bottom:8px}
.upload-text{font-size:13px;color:var(--t3)}
.upload-hint{font-size:11px;color:var(--t3);margin-top:4px}

/* Extract tab preview zone */
.preview-zone{border:1px solid rgba(148,163,184,.18);border-radius:12px;overflow:hidden;
  min-height:160px;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(180deg,rgba(31,41,55,.8),rgba(23,32,51,.85));cursor:pointer;transition:border-color .2s;position:relative}
.preview-zone:hover{border-color:var(--t3)}
.preview-zone.has-image{min-height:unset}
.preview-zone .upload-ph{padding:32px 16px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center}
.preview-zone.has-image .upload-ph{display:none}
.preview-img-ext{width:100%;height:auto;max-height:280px;object-fit:contain;display:none}
.preview-zone.has-image .preview-img-ext{display:block}

/* ── Buttons ── */
.btn{width:100%;margin-top:12px;padding:11px 14px;border:none;border-radius:9px;
  background:var(--indigo);color:#fff;font-size:14px;font-weight:600;cursor:pointer;
  font-family:var(--font);transition:all .2s;box-shadow:0 12px 24px rgba(79,70,229,.2)}
.btn:hover:not(:disabled){filter:brightness(1.08);transform:translateY(-1px)}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn.busy{background:var(--elevated);color:var(--t2)}
.btn.violet{background:var(--violet)}
.btn.violet:hover:not(:disabled){box-shadow:0 0 20px rgba(139,92,246,.3)}
.btn.secondary{background:var(--elevated);border:1px solid var(--border);color:var(--t2);
  font-size:12px;margin-top:8px}
.btn.secondary:hover:not(:disabled){color:var(--t1);border-color:var(--t2)}

/* ── Empty ── */
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:220px;color:var(--t3);font-size:13px;text-align:center;padding:20px;gap:9px}
.empty-icon{font-size:28px;opacity:.4}

/* ── Loading ── */
.loading-row{display:flex;align-items:center;gap:10px;padding:14px 0;
  color:var(--t2);font-size:13px}
.spinner{width:15px;height:15px;border:2px solid var(--border);
  border-top-color:var(--indigo);border-radius:50%;
  animation:spin .6s linear infinite;flex-shrink:0}
.spinner.amber{border-top-color:var(--amber)}
.spinner.emerald{border-top-color:var(--emerald)}
.spinner.violet{border-top-color:var(--violet)}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Results ── */
.result-img{width:100%;border-radius:10px;overflow:hidden;border:1px solid rgba(148,163,184,.16);margin-top:12px;background:#0f172a}
.result-img img{width:100%;display:block}
.bbox-list{margin-top:12px;display:flex;flex-direction:column;gap:6px}
.bbox-row{background:rgba(31,41,55,.72);border-radius:8px;padding:7px 12px;
  font-size:11px;font-family:var(--mono);color:var(--t2);display:flex;align-items:center;gap:8px}
.bbox-dot{width:7px;height:7px;border-radius:50%;background:var(--indigo);flex-shrink:0}
.section-lbl{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--t3)}
.method-pill{display:inline-block;font-size:10px;font-weight:600;padding:2px 8px;
  border-radius:10px;margin-left:6px;background:rgba(6,182,212,.1);color:var(--cyan);
  border:1px solid rgba(6,182,212,.3)}

/* ── Verdict ── */
.verdict{margin-top:12px;border-radius:12px;padding:18px 20px;text-align:center}
.verdict.genuine {background:rgba(16,185,129,.08); border:1px solid rgba(16,185,129,.25)}
.verdict.forged  {background:rgba(244,63,94,.08);  border:1px solid rgba(244,63,94,.25)}
.verdict.unsigned{background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.25)}
.verdict.stub    {background:var(--elevated);      border:1px solid var(--border)}
.verdict-lbl{font-size:20px;font-weight:700;letter-spacing:.05em}
.verdict-lbl.genuine {color:var(--emerald)}
.verdict-lbl.forged  {color:var(--rose)}
.verdict-lbl.unsigned{color:var(--amber)}
.verdict-lbl.stub    {color:var(--t3)}
.detect-notice{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.25);
  border-radius:8px;padding:10px 14px;font-size:12px;color:var(--amber);margin-bottom:10px;
  display:flex;align-items:flex-start;gap:8px;line-height:1.5}
.verdict-meta{font-size:12px;color:var(--t2);margin-top:6px;line-height:1.6}

/* ── Extract table ── */
.extract-summary{font-size:11px;color:var(--t3);margin-bottom:10px;display:flex;align-items:center;gap:6px}
.extract-summary .pill{background:var(--elevated);border:1px solid var(--border);
  border-radius:20px;padding:2px 9px;font-size:10px;font-weight:600}
.pill.ok{color:var(--emerald);border-color:rgba(16,185,129,.3)}
.pill.partial{color:var(--amber);border-color:rgba(245,158,11,.3)}
.extract-table{width:100%;border-collapse:separate;border-spacing:0}
.extract-table th{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;
  color:var(--t3);padding:7px 12px;text-align:left;border-bottom:1px solid var(--elevated)}
.extract-table td{padding:10px 12px;font-size:13px;border-bottom:1px solid rgba(148,163,184,.11);vertical-align:middle}
.extract-table tr:last-child td{border-bottom:none}
.extract-table tr.row-hl{background:rgba(139,92,246,.04)}
.extract-table tr:hover td{background:rgba(255,255,255,.02)}
.td-icon{width:30px;padding-right:0;font-size:16px;text-align:center}
.td-field{width:170px;color:var(--t2);font-size:12px}
.td-val{color:var(--t1);font-weight:500;word-break:break-all}
.td-val.null{color:var(--t3);font-style:italic;font-weight:400}
.td-status{width:26px;text-align:center;font-size:13px}
.raw-resp{background:var(--surface);border:1px solid var(--elevated);border-radius:8px;
  padding:14px;margin-top:12px;font-family:var(--mono);font-size:11px;color:var(--t2);
  line-height:1.6;overflow:auto;max-height:320px;white-space:pre-wrap;word-break:break-word}

/* ── Export bar ── */
.export-bar{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
.export-btn{flex:1;min-width:110px;padding:8px 14px;border-radius:8px;
  border:1px solid var(--border);background:var(--elevated);color:var(--t2);
  font-size:12px;font-weight:500;cursor:pointer;font-family:var(--font);
  transition:all .2s;display:flex;align-items:center;justify-content:center;gap:5px}
.export-btn:hover:not(:disabled){border-color:var(--t2);color:var(--t1)}
.export-btn:disabled{opacity:.35;cursor:not-allowed}
.export-btn.copied{color:var(--emerald);border-color:rgba(16,185,129,.4)}

/* ── JSON ── */
.json-btn{width:100%;background:var(--surface);border:1px solid var(--elevated);
  color:var(--t2);padding:8px 14px;border-radius:8px;font-size:12px;cursor:pointer;
  font-family:var(--font);transition:all .2s;margin-top:8px}
.json-btn:hover{border-color:var(--t3);color:var(--t1)}
.json-pre{background:var(--surface);border:1px solid var(--elevated);border-radius:8px;
  padding:14px;margin-top:6px;font-family:var(--mono);font-size:11px;color:var(--t2);
  line-height:1.6;overflow:auto;max-height:280px;white-space:pre-wrap;word-break:break-word}

/* ── Train panel ── */
.train-box{background:var(--surface);border:1px solid var(--elevated);border-radius:10px;
  padding:14px 16px;margin-top:12px}
.train-box p{font-size:12px;color:var(--t2);margin-bottom:8px;line-height:1.6}

/* ── Visual Reasoning (Tab 3) ── */
.question-area{width:100%;background:var(--elevated);border:1px solid var(--border);
  border-radius:8px;color:var(--t1);padding:10px 14px;font-size:13px;
  font-family:var(--font);resize:vertical;min-height:80px;outline:none;
  transition:border .2s;line-height:1.6}
.question-area:focus{border-color:var(--violet)}
.question-area::placeholder{color:var(--t3)}
.hint-pills{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.hint-pill{font-size:11px;padding:4px 10px;border-radius:20px;cursor:pointer;
  background:rgba(139,92,246,.08);border:1px solid rgba(139,92,246,.25);
  color:var(--violet);transition:all .15s;white-space:nowrap}
.hint-pill:hover{background:rgba(139,92,246,.18);border-color:var(--violet)}
.reason-answer{background:var(--elevated);border:1px solid var(--border);
  border-radius:10px;padding:18px 20px;margin-top:12px;font-size:13px;
  color:var(--t1);line-height:1.8;white-space:pre-wrap;word-break:break-word}
.reason-meta{display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap}
.reason-badge{font-size:10px;font-weight:600;padding:3px 9px;border-radius:10px;
  border:1px solid rgba(139,92,246,.35);background:rgba(139,92,246,.08);color:var(--violet)}
.reason-time{font-size:11px;color:var(--t3);font-family:var(--mono);margin-left:auto}
</style>
</head>
<body>

<input type="file" id="file-input" accept="image/*" style="display:none">

<!-- ── Header ── -->
<div class="header">
  <h1>Cheque Verification System</h1>
  <p>Signature detection, forgery verification, cheque field extraction, and visual reasoning</p>
  <div class="badges">
    <span class="badge phase">Phase 1 — Detection</span>
    <span class="badge" style="color:var(--indigo);background:rgba(99,102,241,.1);border-color:rgba(99,102,241,.3)">Falcon Perception 0.6B</span>
    <span class="badge sweep">Line Sweep</span>
    <span class="badge phase">Phase 2 — Verification</span>
    <span class="badge verifier">Signature SVM</span>
    <span class="badge" style="color:var(--violet);background:rgba(139,92,246,.1);border-color:rgba(139,92,246,.3)">Gemma 4 E2B</span>
  </div>
</div>

<div class="app">

  <!-- ── Tab navigation ── -->
  <div class="tab-nav">
    <button class="tab-btn active" data-tab="verify">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
      </svg>
      Signature Verification
    </button>
    <button class="tab-btn" data-tab="extract">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
      </svg>
      Data Extraction
    </button>
    <button class="tab-btn" data-tab="reason">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>
        <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>
      </svg>
      Visual Reasoning
    </button>
  </div>

  <!-- ══════════════════════════════════════════════════
       TAB 1 — Signature Verification
  ═══════════════════════════════════════════════════════ -->
  <div id="tab-verify" class="tab-pane active">

    <!-- Phase indicator -->
    <div class="phase-banner">
      <div class="phase-seg p1 active" id="ph1-seg">
        <div class="ps-num">Phase 1</div>
        <div class="ps-name">Detection</div>
        <div class="ps-model">Falcon Perception → Line Sweep</div>
      </div>
      <div class="phase-seg p2" id="ph2-seg">
        <div class="ps-num">Phase 2</div>
        <div class="ps-name">Verification</div>
        <div class="ps-model">Signature SVM</div>
      </div>
    </div>

    <div class="grid-3">

      <!-- Col 1: Upload + train -->
      <div>
        <div class="panel">
          <div class="panel-hd">
            <span class="panel-dot" style="background:var(--indigo)"></span>
            <div>
              <div class="panel-title">Cheque Image</div>
              <div class="panel-sub">Upload or drop a cheque scan</div>
            </div>
          </div>
          <div class="panel-bd">
            <div class="upload-zone" id="upload-zone">
              <div class="upload-ph">
                <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                  <path d="M12 16V4m0 0L7 9m5-5l5 5M4 20h16"/>
                </svg>
                <div class="upload-text">Drop cheque image or click to upload</div>
                <div class="upload-hint">PNG &middot; JPG &middot; any scan quality</div>
              </div>
              <img class="upload-preview" id="upload-preview" src="" alt="">
              <button class="upload-clear" id="clear-btn" title="Remove">&times;</button>
            </div>
            <button class="btn" id="detect-btn">Detect Signature</button>
            <button class="btn" id="verify-btn" disabled style="margin-top:8px;background:var(--emerald);opacity:.45">Verify Signature</button>
            <div style="font-size:11px;color:var(--t3);margin-top:6px;text-align:center">
              Step 1: Detect and crop. Step 2: Verify the saved crop.
            </div>

            <div class="train-box" id="train-box">
              <p>Signature verifier assets were not found.<br>
                 Check <strong>signature_svm/data</strong> and <strong>signature_svm/model.pkl</strong>.
              </p>
            </div>
          </div>
        </div>
      </div>

      <!-- Col 2: Detected region -->
      <div>
        <div class="panel">
          <div class="panel-hd">
            <span class="panel-dot" style="background:var(--indigo)"></span>
            <div>
              <div class="panel-title">Phase 1a — Falcon Perception</div>
              <div class="panel-sub">Signature region detection · 0.6B</div>
            </div>
            <span class="panel-dur" id="detect-time"></span>
          </div>
          <div class="panel-bd">
            <div id="detect-panel">
              <div class="empty"><span class="empty-icon">🔍</span><span>Signature region will appear here</span></div>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-hd">
            <span class="panel-dot" style="background:var(--amber)"></span>
            <div>
              <div class="panel-title">Phase 1b — Line Sweep</div>
              <div class="panel-sub">Tight signature crop</div>
            </div>
            <span class="panel-dur" id="crop-time"></span>
          </div>
          <div class="panel-bd">
            <div id="crop-panel">
              <div class="empty"><span class="empty-icon">✂️</span><span>Cropped signature will appear here</span></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Col 3: Signature SVM result -->
      <div>
        <div class="panel" id="verdict-section">
          <div class="panel-hd">
            <span class="panel-dot" style="background:var(--emerald)"></span>
            <div>
              <div class="panel-title">Phase 2 — Forgery Verdict</div>
              <div class="panel-sub">Signature SVM · SIFT BoVW + LinearSVC</div>
            </div>
            <span class="panel-dur" id="svm-time"></span>
          </div>
          <div class="panel-bd">
            <div id="verdict-panel">
              <div class="empty"><span class="empty-icon">🤖</span><span>Click Verify Signature after detecting the crop</span></div>
            </div>
          </div>
        </div>

        <div id="verify-json-wrap" style="display:none">
          <button class="json-btn" id="verify-json-btn">Show JSON Output</button>
          <pre id="verify-json-pre" class="json-pre" style="display:none"></pre>
        </div>
      </div>

    </div><!-- /grid-3 -->
  </div><!-- /tab-verify -->

  <!-- ══════════════════════════════════════════════════
       TAB 2 — Data Extraction
  ═══════════════════════════════════════════════════════ -->
  <div id="tab-extract" class="tab-pane">
    <div class="grid-2">

      <!-- Col 1: Preview + Extract -->
      <div>
        <div class="panel">
          <div class="panel-hd">
            <span class="panel-dot" style="background:var(--violet)"></span>
            <div>
              <div class="panel-title">Cheque Image</div>
              <div class="panel-sub">Click to upload &bull; shared with Verify tab</div>
            </div>
          </div>
          <div class="panel-bd">
            <div class="preview-zone" id="ext-zone">
              <div class="upload-ph">
                <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                  <path d="M12 16V4m0 0L7 9m5-5l5 5M4 20h16"/>
                </svg>
                <div class="upload-text">Drop cheque image or click to upload</div>
                <div class="upload-hint">PNG &middot; JPG &middot; any scan quality</div>
              </div>
              <img class="preview-img-ext" id="ext-preview" src="" alt="">
            </div>
            <button class="btn violet" id="extract-btn">Extract Cheque Data</button>
          </div>
        </div>
      </div>

      <!-- Col 2: Extracted fields -->
      <div>
        <div class="panel">
          <div class="panel-hd">
            <span class="panel-dot" style="background:var(--violet)"></span>
            <div>
              <div class="panel-title">Extracted Fields</div>
              <div class="panel-sub">Qwen OCR hint &bull; Gemma structured extraction</div>
            </div>
            <span class="panel-dur" id="extract-time"></span>
          </div>
          <div class="panel-bd">
            <div id="extract-panel">
              <div class="empty"><span class="empty-icon">📄</span><span>Upload a cheque and click<br>Extract Cheque Data</span></div>
            </div>
          </div>
        </div>

        <div id="export-bar" class="export-bar" style="display:none">
          <button class="export-btn" id="exp-csv"  onclick="exportCSV()"    disabled>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 16V4m0 12-4-4m4 4 4-4M4 20h16"/>
            </svg>Export CSV</button>
          <button class="export-btn" id="exp-json" onclick="exportJSON()"   disabled>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
            </svg>Export JSON</button>
          <button class="export-btn" id="exp-copy" onclick="copyFields()"   disabled>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>Copy</button>
        </div>

        <div id="extract-json-wrap" style="display:none">
          <button class="json-btn" id="extract-json-btn">Show Raw JSON</button>
          <pre id="extract-json-pre" class="json-pre" style="display:none"></pre>
        </div>
      </div>

    </div><!-- /grid-2 -->
  </div><!-- /tab-extract -->

  <!-- ══════════════════════════════════════════════════
       TAB 3 — Visual Reasoning (Gemma 4 E2B)
  ═══════════════════════════════════════════════════════ -->
  <div id="tab-reason" class="tab-pane">
    <div class="grid-2">

      <!-- Col 1: Image + question -->
      <div>
        <div class="panel">
          <div class="panel-hd">
            <span class="panel-dot" style="background:var(--violet)"></span>
            <div>
              <div class="panel-title">Cheque Image</div>
              <div class="panel-sub">Shared with other tabs</div>
            </div>
          </div>
          <div class="panel-bd">
            <div class="preview-zone" id="reason-zone">
              <div class="upload-ph">
                <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                  <path d="M12 16V4m0 0L7 9m5-5l5 5M4 20h16"/>
                </svg>
                <div class="upload-text">Drop cheque image or click to upload</div>
                <div class="upload-hint">PNG &middot; JPG &middot; any scan quality</div>
              </div>
              <img class="preview-img-ext" id="reason-preview" src="" alt="">
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-hd">
            <span class="panel-dot" style="background:var(--violet)"></span>
            <div>
              <div class="panel-title">Question</div>
              <div class="panel-sub">Ask anything about the cheque</div>
            </div>
          </div>
          <div class="panel-bd">
            <textarea class="question-area" id="reason-question"
              placeholder="e.g. Is this cheque signature genuine or does it look forged?&#10;What bank issued this cheque?&#10;List all the details visible on this cheque."></textarea>
            <div class="hint-pills">
              <span class="hint-pill" onclick="setQuestion('Describe all the details visible on this cheque.')">All details</span>
              <span class="hint-pill" onclick="setQuestion('Does the signature on this cheque look genuine or suspicious? Explain your reasoning.')">Signature quality</span>
              <span class="hint-pill" onclick="setQuestion('What is the amount written on this cheque in both words and figures?')">Amount</span>
              <span class="hint-pill" onclick="setQuestion('What is the name of the bank and account holder on this cheque?')">Bank &amp; holder</span>
              <span class="hint-pill" onclick="setQuestion('Read the MICR code and cheque number from this cheque.')">MICR &amp; cheque#</span>
              <span class="hint-pill" onclick="setQuestion('What date is written on this cheque? Is it post-dated?')">Date check</span>
              <span class="hint-pill" onclick="setQuestion('Are there any signs of tampering, alterations, or corrections visible on this cheque?')">Tampering check</span>
            </div>
            <button class="btn violet" id="reason-btn">Ask Gemma 4</button>
          </div>
        </div>
      </div>

      <!-- Col 2: Gemma 4 response -->
      <div>
        <div class="panel">
          <div class="panel-hd">
            <span class="panel-dot" style="background:var(--violet)"></span>
            <div>
              <div class="panel-title">Gemma 4 E2B — Visual Reasoning</div>
              <div class="panel-sub">2B parameter VLM &bull; Local MLX inference</div>
            </div>
            <span class="panel-dur" id="reason-time"></span>
          </div>
          <div class="panel-bd">
            <div id="reason-panel">
              <div class="empty">
                <span class="empty-icon">🧠</span>
                <span>Upload a cheque, type your question, and click Ask Gemma 4</span>
              </div>
            </div>
          </div>
        </div>

        <div id="reason-json-wrap" style="display:none">
          <button class="json-btn" id="reason-json-btn">Show Raw Answer</button>
          <pre id="reason-json-pre" class="json-pre" style="display:none"></pre>
        </div>
      </div>

    </div><!-- /grid-2 -->
  </div><!-- /tab-reason -->

</div><!-- /app -->

<script>
// ═══════════════════════════════════════════════════════
// Shared upload state
// ═══════════════════════════════════════════════════════
const fileInput  = document.getElementById('file-input');
const uploadZone = document.getElementById('upload-zone');
const previewImg = document.getElementById('upload-preview');
const clearBtn   = document.getElementById('clear-btn');
const extZone    = document.getElementById('ext-zone');
const extPreview = document.getElementById('ext-preview');
let uploadedImage = null;
let signatureCropDataUrl = null;

function syncPreviews() {
  if (uploadedImage) {
    previewImg.src = uploadedImage; uploadZone.classList.add('has-image');
    extPreview.src = uploadedImage; extZone.classList.add('has-image');
    const rp = document.getElementById('reason-preview');
    const rz = document.getElementById('reason-zone');
    if (rp) rp.src = uploadedImage;
    if (rz) rz.classList.add('has-image');
  } else {
    previewImg.src = ''; uploadZone.classList.remove('has-image');
    extPreview.src = ''; extZone.classList.remove('has-image');
    const rp = document.getElementById('reason-preview');
    const rz = document.getElementById('reason-zone');
    if (rp) rp.src = '';
    if (rz) rz.classList.remove('has-image');
    fileInput.value = '';
  }
}
function loadFile(file) {
  const r = new FileReader();
  r.onload = e => {
    uploadedImage = e.target.result;
    syncPreviews();
    resetVerifyPanels();
    resetExtractPanels();
    if (typeof resetReasonPanels === 'function') resetReasonPanels();
  };
  r.readAsDataURL(file);
}
function clearUpload() {
  uploadedImage = null;
  syncPreviews();
  resetVerifyPanels();
  resetExtractPanels();
  if (typeof resetReasonPanels === 'function') resetReasonPanels();
}

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => { e.preventDefault(); uploadZone.classList.remove('dragover'); if(e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); });
clearBtn.addEventListener('click', e => { e.stopPropagation(); clearUpload(); });

extZone.addEventListener('click', () => fileInput.click());
extZone.addEventListener('dragover',  e => { e.preventDefault(); extZone.classList.add('dragover'); });
extZone.addEventListener('dragleave', () => extZone.classList.remove('dragover'));
extZone.addEventListener('drop', e => { e.preventDefault(); extZone.classList.remove('dragover'); if(e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => { if(fileInput.files[0]) loadFile(fileInput.files[0]); });

// ═══════════════════════════════════════════════════════
// Tab switching
// ═══════════════════════════════════════════════════════
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ═══════════════════════════════════════════════════════
// SSE reader
// ═══════════════════════════════════════════════════════
async function readSSE(response, handler) {
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      if (line.startsWith('data: ')) { try { handler(JSON.parse(line.slice(6))); } catch(_){} }
    }
  }
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ═══════════════════════════════════════════════════════
// Signature verifier status check
// ═══════════════════════════════════════════════════════
const trainBox = document.getElementById('train-box');

// Hide the verifier notice when the local assets exist
fetch('/api/model/status')
  .then(r => r.json())
  .then(d => { if (d.trained) trainBox.style.display = 'none'; })
  .catch(() => {});

// ═══════════════════════════════════════════════════════
// TAB 1 — Signature Verification (two-step)
// ═══════════════════════════════════════════════════════
const detectBtn = document.getElementById('detect-btn');
const verifyBtn = document.getElementById('verify-btn');

function resetVerifyPanels() {
  signatureCropDataUrl = null;
  document.getElementById('detect-panel').innerHTML =
    `<div class="empty"><span class="empty-icon">🔍</span><span>Signature region will appear here</span></div>`;
  document.getElementById('crop-panel').innerHTML =
    `<div class="empty"><span class="empty-icon">✂️</span><span>Cropped signature will appear here</span></div>`;
  document.getElementById('verdict-panel').innerHTML =
    `<div class="empty"><span class="empty-icon">🤖</span><span>Click Verify Signature after detecting the crop</span></div>`;
  document.getElementById('detect-time').textContent = '';
  document.getElementById('crop-time').textContent   = '';
  document.getElementById('svm-time').textContent    = '';
  document.getElementById('verify-json-wrap').style.display = 'none';
  document.getElementById('ph1-seg').classList.add('active');
  document.getElementById('ph2-seg').classList.remove('active');
  verifyBtn.disabled = true;
  verifyBtn.style.opacity = '.45';
}

// ── Step 1: Detect & Crop (REST /api/cheque/crop) ──────────────────
detectBtn.addEventListener('click', async () => {
  if (!uploadedImage) { alert('Please upload a cheque image first.'); return; }

  signatureCropDataUrl = null;
  detectBtn.disabled = true;
  detectBtn.textContent = 'Detecting...';
  detectBtn.classList.add('busy');
  verifyBtn.disabled = true;
  verifyBtn.style.opacity = '.45';

  document.getElementById('detect-panel').innerHTML =
    `<div class="loading-row"><span class="spinner"></span>Falcon Perception is locating the signature...</div>`;
  document.getElementById('crop-panel').innerHTML =
    `<div class="empty"><span class="empty-icon">✂️</span><span>Waiting for detection...</span></div>`;
  document.getElementById('verdict-panel').innerHTML =
    `<div class="empty"><span class="empty-icon">🤖</span><span>Click Verify Signature after detecting the crop</span></div>`;

  try {
    // Convert base64 data URL to blob for multipart upload
    const blob = await (await fetch(uploadedImage)).blob();
    const fd = new FormData();
    fd.append('file', blob, 'cheque.png');

    const res = await fetch('/api/cheque/crop', { method: 'POST', body: fd });
    const d   = await res.json();

    if (d.error) { showVerifyErr(d.error); return; }

    signatureCropDataUrl = d.signature_crop_b64 || null;
    const det = d.detection || {};
    const ls  = d.line_sweep || {};
    const isFalcon = det.method === 'falcon-perception';
    const methodPill = isFalcon
      ? `<span class="method-pill">Falcon Perception</span>`
      : `<span class="method-pill" style="background:rgba(245,158,11,.1);color:var(--amber);border-color:rgba(245,158,11,.3)">heuristic</span>`;

    document.getElementById('detect-time').textContent = (det.duration_s || '') + 's';
    document.getElementById('detect-panel').innerHTML =
      `<div class="section-lbl">${isFalcon ? 'Signature detected' : 'Heuristic region'} ${methodPill}</div>
       <div class="result-img"><img src="${d.annotated_b64}" alt="detected region"></div>
       <div class="bbox-list">
         <div class="bbox-row"><span class="bbox-dot"></span>${JSON.stringify(det.bbox || [])}</div>
       </div>`;

    document.getElementById('crop-time').textContent = '';
    document.getElementById('crop-panel').innerHTML =
      `<div class="section-lbl">Sweep: ${ls.success ? 'refined ✓' : 'coarse (fallback)'}</div>
       <div class="result-img" style="margin-top:8px">
         <img src="${d.signature_crop_b64}" alt="cropped signature">
       </div>`;

    document.getElementById('ph1-seg').classList.add('active');

    // Enable Verify button now that we have a crop
    verifyBtn.disabled = !signatureCropDataUrl;
    verifyBtn.style.opacity = signatureCropDataUrl ? '1' : '.45';

  } catch(e) {
    showVerifyErr('Detection failed: ' + e.message);
  } finally {
    detectBtn.disabled = false;
    detectBtn.textContent = 'Detect Signature';
    detectBtn.classList.remove('busy');
  }
});

// ── Step 2: Verify Forgery (REST /api/signature/verify-crop) ──
verifyBtn.addEventListener('click', async () => {
  if (!signatureCropDataUrl) { alert('Please detect and crop a signature first.'); return; }

  verifyBtn.disabled = true;
  verifyBtn.textContent = 'Verifying...';
  verifyBtn.classList.add('busy');

  document.getElementById('verdict-panel').innerHTML =
    `<div class="loading-row"><span class="spinner emerald"></span>Signature SVM is verifying the saved crop...</div>`;
  document.getElementById('ph2-seg').classList.add('active');

  try {
    const blob = await (await fetch(signatureCropDataUrl)).blob();
    const fd = new FormData();
    fd.append('file', blob, 'signature_crop.png');

    const res = await fetch('/api/signature/verify-crop', { method: 'POST', body: fd });
    const d   = await res.json();

    const v   = d.verdict || {};
    const ver = d.verification || {};
    if (d.error || v.error) { showVerifyErr(d.error || v.error, 'verdict-panel'); return; }

    const msg = v.message || '';
    const msgLower = msg.toLowerCase();
    const isReal   = msgLower.includes('real') || msgLower.includes('genuine');
    const isForged = msgLower.includes('forg');
    const isInconclusive = msgLower.includes('inconclusive') || msgLower.includes('unknown');
    const cls  = isReal ? 'genuine' : isForged ? 'forged' : 'unsigned';
    const icon = isReal ? '✅' : isForged ? '❌' : '⚠️';
    const verdict  = isReal ? 'GENUINE' : isForged ? 'FORGED' : isInconclusive ? 'INCONCLUSIVE' : 'UNKNOWN';
    const modelLbl = v.model || 'Signature SVM';

    document.getElementById('svm-time').textContent = ver.duration_s ? `${ver.duration_s}s` : '';
    document.getElementById('verdict-panel').innerHTML =
      `<div class="verdict ${cls}">
         <div class="verdict-lbl ${cls}">${icon} ${verdict}</div>
         <div class="verdict-meta">${esc(modelLbl)}${v.confidence ? ` · confidence ${(v.confidence * 100).toFixed(1)}%` : ''}${v.note ? `<br>${esc(v.note)}` : ''}</div>
       </div>`;

    document.getElementById('verify-json-wrap').style.display = 'block';
    document.getElementById('verify-json-pre').textContent =
      JSON.stringify({ verification: d.verification, verdict: d.verdict }, null, 2);

  } catch(e) {
    showVerifyErr('Verification failed: ' + e.message, 'verdict-panel');
  } finally {
    verifyBtn.disabled = false;
    verifyBtn.textContent = 'Verify Signature';
    verifyBtn.classList.remove('busy');
  }
});

function showVerifyErr(msg, panelId='detect-panel') {
  document.getElementById(panelId).innerHTML =
    `<div class="empty" style="color:var(--rose)"><span class="empty-icon">❌</span><span>${esc(msg)}</span></div>`;
}

document.getElementById('verify-json-btn').addEventListener('click', () => {
  const pre = document.getElementById('verify-json-pre');
  const btn = document.getElementById('verify-json-btn');
  const vis = pre.style.display === 'none';
  pre.style.display = vis ? 'block' : 'none';
  btn.textContent   = vis ? 'Hide JSON Output' : 'Show JSON Output';
});

// ═══════════════════════════════════════════════════════
// TAB 2 — Data Extraction
// ═══════════════════════════════════════════════════════
const extractBtn = document.getElementById('extract-btn');

const FIELD_META = [
  { key:'account_holder',   label:'Account Holder Name', icon:'👤', highlight:true  },
  { key:'bank_name',        label:'Bank Name',            icon:'🏦', highlight:true  },
  { key:'branch_name',      label:'Branch Name',          icon:'📍', highlight:false },
  { key:'cheque_number',    label:'Cheque Number',        icon:'#️⃣', highlight:false },
  { key:'date',             label:'Date',                 icon:'📅', highlight:true  },
  { key:'payee_name',       label:'Payee Name',           icon:'🧾', highlight:true  },
  { key:'amount_numeric',   label:'Amount (Numeric)',     icon:'💰', highlight:true  },
  { key:'amount_words',     label:'Amount (in Words)',    icon:'✍️', highlight:true  },
  { key:'signature_present',label:'Signature',            icon:'✒️', highlight:false },
  { key:'ifsc_code',        label:'IFSC Code',            icon:'🏛️', highlight:false },
  { key:'account_number',   label:'Account Number',       icon:'🔢', highlight:false },
];

let currentFields = null;

function hasValue(v) { return v !== null && v !== undefined && v !== ''; }

function renderFields(fields, durationS) {
  currentFields = fields;
  if (fields.raw_response) {
    return `<div class="section-lbl">Raw Response (JSON parse failed)</div>
      <div class="raw-resp">${esc(fields.raw_response)}</div>`;
  }
  const detected = FIELD_META.filter(m => hasValue(fields[m.key])).length;
  const total    = FIELD_META.length;
  const pct      = Math.round(detected / total * 100);
  const pillCls  = pct >= 75 ? 'ok' : 'partial';
  const summary  = `<div class="extract-summary">
    <span>Extracted in ${durationS}s &nbsp;&middot;&nbsp; Gemma 4 E2B</span>
    <span class="pill ${pillCls}">${detected}/${total} fields &nbsp;(${pct}%)</span>
  </div>`;
  const rows = FIELD_META.map(m => {
    const val = fields[m.key]; const ok = hasValue(val);
    let display;
    if (m.key === 'signature_present') {
      const sig = ok ? String(val).toLowerCase() : '';
      if (sig === 'yes')       display = '<span style="color:var(--emerald)">&#10003; Present</span>';
      else if (sig === 'no')   display = '<span style="color:var(--rose)">&#10007; Not present</span>';
      else if (ok)             display = esc(String(val));
      else                     display = '<span class="null">Not detected</span>';
    } else {
      display = ok ? esc(String(val)) : 'Not detected';
    }
    return `<tr class="${m.highlight && ok ? 'row-hl' : ''}">
      <td class="td-icon">${m.icon}</td>
      <td class="td-field">${m.label}</td>
      <td class="td-val${ok ? '' : ' null'}">${display}</td>
      <td class="td-status">${ok ? '&#10003;' : '—'}</td>
    </tr>`;
  }).join('');
  return summary + `<table class="extract-table">
    <thead><tr><th></th><th>Field</th><th>Value</th><th></th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function download(filename, content, mime) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type: mime }));
  a.download = filename; document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(a.href);
}
function exportCSV() {
  if (!currentFields) return;
  const rows = [['Field','Value']];
  FIELD_META.forEach(m => { const v = currentFields[m.key]; rows.push([m.label, hasValue(v) ? String(v) : '']); });
  download('cheque_data.csv', rows.map(r => r.map(c => '"'+String(c).replace(/"/g,'""')+'"').join(',')).join('\r\n'), 'text/csv;charset=utf-8;');
}
function exportJSON() {
  if (!currentFields) return;
  download('cheque_data.json', JSON.stringify(currentFields, null, 2), 'application/json');
}
async function copyFields() {
  if (!currentFields) return;
  const text = FIELD_META.map(m => `${m.label}: ${hasValue(currentFields[m.key]) ? currentFields[m.key] : 'Not detected'}`).join('\n');
  try {
    await navigator.clipboard.writeText(text);
    const btn = document.getElementById('exp-copy');
    btn.classList.add('copied'); btn.textContent = '✓ Copied!';
    setTimeout(() => { btn.classList.remove('copied'); btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`; }, 2000);
  } catch(_) { alert('Copy failed.'); }
}

function resetExtractPanels() {
  currentFields = null;
  document.getElementById('extract-panel').innerHTML =
    `<div class="empty"><span class="empty-icon">📄</span><span>Upload a cheque and click<br>Extract Cheque Data</span></div>`;
  document.getElementById('extract-time').textContent = '';
  document.getElementById('export-bar').style.display = 'none';
  document.getElementById('extract-json-wrap').style.display = 'none';
  ['exp-csv','exp-json','exp-copy'].forEach(id => { document.getElementById(id).disabled = true; });
}

extractBtn.addEventListener('click', runExtract);

async function runExtract() {
  if (!uploadedImage) { alert('Please upload a cheque image first.'); return; }
  extractBtn.disabled = true; extractBtn.textContent = 'Extracting…'; extractBtn.classList.add('busy');
  document.getElementById('extract-panel').innerHTML =
    `<div class="loading-row"><span class="spinner violet"></span>Preparing OCR and reading cheque fields...</div>`;

  let res;
  try {
    res = await fetch('/api/extract/stream', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ image_b64: uploadedImage }),
    });
  } catch(e) { showExtractErr('Network error: ' + e.message); resetExtractBtn(); return; }
  if (!res.ok) { showExtractErr('HTTP ' + res.status); resetExtractBtn(); return; }

  try {
    await readSSE(res, ev => {
      switch(ev.type) {
        case 'loading_models': case 'models_ready': case 'extract_start':
          document.getElementById('extract-panel').innerHTML =
            `<div class="loading-row"><span class="spinner violet"></span>Qwen OCR and Gemma are analysing cheque fields...</div>`;
          break;
        case 'extract_complete':
          document.getElementById('extract-time').textContent = ev.duration_s + 's';
          document.getElementById('extract-panel').innerHTML = renderFields(ev.fields, ev.duration_s);
          document.getElementById('export-bar').style.display = 'flex';
          ['exp-csv','exp-json','exp-copy'].forEach(id => { document.getElementById(id).disabled = false; });
          break;
        case 'extract_unavailable':
          document.getElementById('extract-panel').innerHTML =
            `<div class="empty" style="color:var(--amber)"><span class="empty-icon">⚠️</span>
               <span style="text-align:left;max-width:420px">${esc(ev.message)}</span></div>`;
          break;
        case 'error':
          showExtractErr(ev.message || 'Unknown error'); resetExtractBtn(); break;
        case 'done':
          if (ev.json_output && Object.keys(ev.json_output).length > 0) {
            document.getElementById('extract-json-wrap').style.display = 'block';
            document.getElementById('extract-json-pre').textContent = JSON.stringify(ev.json_output, null, 2);
            if (!currentFields) currentFields = ev.json_output;
          }
          resetExtractBtn(); break;
      }
    });
  } catch(e) { showExtractErr(e.message || String(e)); }
  resetExtractBtn();
}

function resetExtractBtn() {
  extractBtn.disabled = false; extractBtn.textContent = 'Extract Cheque Data'; extractBtn.classList.remove('busy');
}
function showExtractErr(msg) {
  document.getElementById('extract-panel').innerHTML =
    `<div class="empty" style="color:var(--rose)"><span class="empty-icon">❌</span><span>${esc(msg)}</span></div>`;
}

document.getElementById('extract-json-btn').addEventListener('click', () => {
  const pre = document.getElementById('extract-json-pre');
  const btn = document.getElementById('extract-json-btn');
  const vis = pre.style.display === 'none';
  pre.style.display = vis ? 'block' : 'none';
  btn.textContent   = vis ? 'Hide Raw JSON' : 'Show Raw JSON';
});

// ═══════════════════════════════════════════════════════
// TAB 3 — Visual Reasoning (Gemma 4)
// ═══════════════════════════════════════════════════════
const reasonBtn = document.getElementById('reason-btn');
const reasonZone    = document.getElementById('reason-zone');
const reasonPreview = document.getElementById('reason-preview');

reasonZone.addEventListener('click', () => fileInput.click());
reasonZone.addEventListener('dragover',  e => { e.preventDefault(); reasonZone.classList.add('dragover'); });
reasonZone.addEventListener('dragleave', () => reasonZone.classList.remove('dragover'));
reasonZone.addEventListener('drop', e => { e.preventDefault(); reasonZone.classList.remove('dragover'); if(e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); });

function setQuestion(q) {
  document.getElementById('reason-question').value = q;
  document.getElementById('reason-question').focus();
}

function resetReasonPanels() {
  document.getElementById('reason-panel').innerHTML =
    `<div class="empty"><span class="empty-icon">🧠</span>
     <span>Upload a cheque, type your question, and click Ask Gemma 4</span></div>`;
  document.getElementById('reason-time').textContent = '';
  document.getElementById('reason-json-wrap').style.display = 'none';
}

reasonBtn.addEventListener('click', runReason);

async function runReason() {
  if (!uploadedImage) { alert('Please upload a cheque image first.'); return; }
  const question = document.getElementById('reason-question').value.trim();
  if (!question) { alert('Please enter a question.'); return; }

  reasonBtn.disabled = true; reasonBtn.textContent = 'Reasoning…'; reasonBtn.classList.add('busy');
  document.getElementById('reason-panel').innerHTML =
    `<div class="loading-row"><span class="spinner violet"></span>Loading Gemma 4 E2B — first run downloads the model…</div>`;

  let res;
  try {
    res = await fetch('/api/reason/stream', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ image_b64: uploadedImage, question }),
    });
  } catch(e) { showReasonErr('Network error: ' + e.message); resetReasonBtn(); return; }
  if (!res.ok) { showReasonErr('HTTP ' + res.status); resetReasonBtn(); return; }

  let finalAnswer = '';
  let finalDt     = 0;

  try {
    await readSSE(res, ev => {
      switch(ev.type) {
        case 'loading_models':
          document.getElementById('reason-panel').innerHTML =
            `<div class="loading-row"><span class="spinner violet"></span>
             Loading Gemma 4 E2B (first run may take a while to download ~4 GB)…</div>`;
          break;

        case 'models_ready':
          document.getElementById('reason-panel').innerHTML =
            `<div class="loading-row"><span class="spinner violet"></span>
             Gemma 4 ready — running visual inference…</div>`;
          break;

        case 'reason_start':
          document.getElementById('reason-panel').innerHTML =
            `<div class="loading-row"><span class="spinner violet"></span>
             Analysing cheque with Gemma 4 E2B…</div>`;
          break;

        case 'reason_complete': {
          finalAnswer = ev.answer || '';
          finalDt     = ev.duration_s || 0;
          document.getElementById('reason-time').textContent = finalDt + 's';
          document.getElementById('reason-panel').innerHTML =
            `<div class="reason-meta">
               <span class="reason-badge">Gemma 4 E2B &bull; 2B params</span>
               <span class="reason-badge" style="background:rgba(6,182,212,.08);border-color:rgba(6,182,212,.3);color:var(--cyan)">Visual Q&amp;A</span>
               <span class="reason-time">${finalDt}s</span>
             </div>
             <div class="reason-answer">${esc(finalAnswer)}</div>`;
          document.getElementById('reason-json-wrap').style.display = 'block';
          document.getElementById('reason-json-pre').textContent = finalAnswer;
          break;
        }

        case 'reason_unavailable':
          document.getElementById('reason-panel').innerHTML =
            `<div class="empty" style="color:var(--amber)">
               <span class="empty-icon">⚠️</span>
               <span style="text-align:left;max-width:440px">${esc(ev.message)}</span>
             </div>`;
          break;

        case 'error':
          showReasonErr(ev.message || 'Unknown error');
          resetReasonBtn(); break;

        case 'done':
          resetReasonBtn(); break;
      }
    });
  } catch(e) { showReasonErr(e.message || String(e)); }
  resetReasonBtn();
}

function resetReasonBtn() {
  reasonBtn.disabled = false; reasonBtn.textContent = 'Ask Gemma 4'; reasonBtn.classList.remove('busy');
}
function showReasonErr(msg) {
  document.getElementById('reason-panel').innerHTML =
    `<div class="empty" style="color:var(--rose)"><span class="empty-icon">❌</span><span>${esc(msg)}</span></div>`;
}

document.getElementById('reason-json-btn').addEventListener('click', () => {
  const pre = document.getElementById('reason-json-pre');
  const btn = document.getElementById('reason-json-btn');
  const vis = pre.style.display === 'none';
  pre.style.display = vis ? 'block' : 'none';
  btn.textContent   = vis ? 'Hide Raw Answer' : 'Show Raw Answer';
});

// ── Auto-check model status on load ─────────────────────────────────────────
// Silently probe whether the model file exists by pinging the API.
// Check if signature verifier assets are ready and hide the notice.
(async () => {
  try {
    const r = await fetch('/api/model/status');
    const d = await r.json();
    if (d.trained) {
      document.getElementById('train-box').style.display = 'none';
    }
  } catch(_) {}
})();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)
