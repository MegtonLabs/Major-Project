# Cheque Verification System

Automated bank cheque processing powered by **Falcon Perception 0.6B** (signature detection), **Gemma 4 E2B** (field extraction), and a **Signature SVM** classifier (SIFT BoVW + LinearSVC) for forgery detection — fully TensorFlow-free on Apple Silicon.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       cheque_studio.py                           │
│               FastAPI + Server-Sent Events (SSE)                 │
│                     http://localhost:7860                         │
│                                                                   │
│  SSE Tabs (built-in HTML UI)                                      │
│    Tab 1 — Signature Verification  POST /api/verify/stream        │
│    Tab 2 — Data Extraction         POST /api/extract/stream       │
│    Tab 3 — Visual Reasoning        POST /api/reason/stream        │
│                                                                   │
│  REST API (used by React frontend)                                │
│    POST /api/cheque/crop     Detect + crop signature              │
│    POST /api/cheque/verify   Crop + SVM forgery verdict           │
│    POST /api/cheque/extract  EasyOCR + Gemma 4 fields             │
│    GET  /api/model/status    Check if SVM model is ready          │
└───────────────────────┬──────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────────┐
        ▼               ▼                   ▼
┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  Detection  │  │  Data Extraction│  │  Visual Reasoning │
│ Falcon 0.6B │  │  EasyOCR        │  │  Gemma 4 E2B      │
│ (MLX)       │  │  + Gemma 4 E2B  │  │  (mlx_vlm)        │
└──────┬──────┘  │  11 fields      │  └──────────────────┘
       │         └─────────────────┘
┌──────▼──────┐
│  Line Sweep │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│  signature_svm/svm.py  (model.pkl)                               │
│  SIFT BoVW (500 clusters) + 12 Geometric Features               │
│  Pre-trained LinearSVC — no TensorFlow required                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Models

| Model | File | Task |
|-------|------|------|
| Falcon Perception 0.6B | auto-downloaded | Signature region detection (MLX) |
| Gemma 4 E2B | auto-downloaded | Field extraction + visual Q&A (mlx_vlm) |
| EasyOCR | auto-downloaded | Raw text hint for Gemma |
| Signature LinearSVC | `signature_svm/model.pkl` | Forgery classification (sklearn, no TF) |

---

## Project Structure

```
cheque-verify-app/
├── cheque_studio.py          ← Main FastAPI app (SSE + REST endpoints)
├── agent_studio.py           ← Falcon + Gemma model wrappers
├── agent.py                  ← Programmatic pipeline (detect→crop→verify→extract)
├── app.py                    ← Entry-point launcher  (python app.py)
├── main.py                   ← Unified CLI launcher
├── demo.py                   ← CLI demo
├── requirements.txt
│
├── detection/
│   ├── ocr_extractor.py      ← Gemma 4 E2B structured field extraction (11 fields)
│   ├── Line_Sweep/           ← Line Sweep tight-crop algorithm
│   │   └── lineSweepDetect.py
│   ├── OCR/                  ← OCR algorithm module
│   │   └── OCR_Algorithm.py
│   └── Connected_Components/ ← Connected component analysis
│
├── signature_svm/            ← Signature forgery classifier
│   ├── svm.py                ← svm_algo() — full SIFT+SVM pipeline
│   ├── model.pkl             ← Pre-trained LinearSVC (500 SIFT BoVW + 12 geometric)
│   ├── verifier.py           ← Adapter: PIL image → svm_algo() → REAL/FORGED
│   ├── preproc.py            ← Otsu threshold preprocessing
│   ├── features.py           ← 12 geometric feature extractors
│   ├── svm_run.py            ← Training evaluation script (29 user groups)
│   ├── svm_test.py           ← Test script using static/LineSweep_Results/
│   ├── data/
│   │   ├── genuine/          ← 145 genuine signature training images
│   │   ├── forged/           ← 145 forged signature training images
│   │   └── origin/           ← Test/origin signatures
│   └── static/
│       └── LineSweep_Results/ ← Drop-zone for test images (svm_test.py)
│
├── Our_Dataset/              ← Custom cheque images dataset
│   └── cheque_images/
│
├── step_outputs/             ← Per-run detection crops (auto-generated)
└── diagrams/                 ← Architecture diagrams
```

---

## Quick Start

### 1. Install dependencies

```bash
conda activate cheque-verify
pip install -r requirements.txt
```

### 2. Start the server

```bash
python app.py
# → http://0.0.0.0:7860
```

Open **http://localhost:7860** for the built-in SSE UI.

### 3. (Optional) React frontend

```bash
# The frontend is no longer bundled — use the built-in SSE UI at port 7860
```

---

## SSE Tab Behaviour

| Tab | Endpoint | Pipeline |
|-----|----------|----------|
| Signature Verification | `POST /api/verify/stream` | Falcon → Line Sweep → Signature SVM |
| Data Extraction | `POST /api/extract/stream` | EasyOCR → Gemma 4 E2B → regex |
| Visual Reasoning | `POST /api/reason/stream` | Gemma 4 E2B free-form Q&A |

---

## Hybrid OCR Extraction

Two-pass pipeline in `detection/ocr_extractor.py`:

1. **EasyOCR** — raw printed text as context hint
2. **Gemma 4 E2B** — receives image + OCR hint, returns structured JSON
3. **Regex fallback** — fills null fields using Indian cheque patterns

Fields: `account_holder` · `bank_name` · `branch_name` · `cheque_number` · `date` · `payee_name` · `amount_numeric` · `amount_words` · `signature_present` · `ifsc_code` · `account_number`

---

## Signature SVM Training

`model.pkl` is pre-trained on 145 genuine + 145 forged signatures using SIFT BoVW (500 clusters) + 12 geometric features.

```bash
cd signature_svm

# Training evaluation (29 user groups)
python svm_run.py

# Test on new images (place images in static/LineSweep_Results/)
python svm_test.py
```

---

## Environment

- **Python** 3.10+ · conda env `cheque-verify`
- **Platform** macOS Apple Silicon (M-series) — MLX required
- **Key packages** `mlx-vlm` · `falcon-perception` · `easyocr` · `fastapi` · `uvicorn` · `scikit-learn` · `opencv-python` · `Pillow` · `scipy` · `imagehash`
