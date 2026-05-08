# Cheque Verify System — Architecture

## 1. High-Level Pipeline

```
                          User Browser
                              │
                    POST /api/verify/stream
                    POST /api/extract/stream
                    POST /api/reason/stream
                              │
                    ┌─────────▼──────────┐
                    │   cheque_studio.py  │
                    │  FastAPI + uvicorn  │
                    │  SSE streaming      │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼──────────────────┐
              ▼               ▼                  ▼
    ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
    │  Tab 1      │  │  Tab 2       │  │  Tab 3           │
    │  Signature  │  │  Data        │  │  Visual          │
    │  Verify     │  │  Extraction  │  │  Reasoning       │
    └──────┬──────┘  └──────┬───────┘  └──────┬───────────┘
           │                │                  │
    ┌──────▼──────┐  ┌──────▼───────┐  ┌──────▼───────────┐
    │  Falcon     │  │ ocr_         │  │  agent_studio.   │
    │  Perception │  │ extractor.py │  │  _vlm()          │
    │  0.6B (MLX) │  │ (Gemma 4 E2B)│  │  (Gemma 4 E2B)   │
    └──────┬──────┘  └──────────────┘  └──────────────────┘
           │
    ┌──────▼──────┐
    │  Line Sweep │  detection/Line_Sweep/lineSweepDetect.py
    └──────┬──────┘
           │
    ┌──────▼───────────────────────────────────────────────┐
    │  signature_svm/svm.py  —  svm_algo()                 │
    │  SIFT BoVW (500 clusters) + 12 Geometric Features    │
    │  Pre-trained LinearSVC  (signature_svm/model.pkl)    │
    │  No TensorFlow — safe on Apple Silicon               │
    └──────────────────────────────────────────────────────┘
```

---

## 2. Component Map

### `cheque_studio.py` — Main Server

| Function | Role |
|----------|------|
| `execute_signature_events(img)` | SSE generator for Tab 1 (detection + verification) |
| `execute_extraction_events(img)` | SSE generator for Tab 2 (Gemma field extraction) |
| `execute_reasoning_events(img, question)` | SSE generator for Tab 3 (Gemma VQA) |
| `POST /api/verify/stream` | Tab 1 SSE endpoint |
| `POST /api/extract/stream` | Tab 2 SSE endpoint |
| `POST /api/reason/stream` | Tab 3 SSE endpoint |
| `POST /api/cheque/crop` | REST: detect + crop signature |
| `POST /api/cheque/verify` | REST: crop + Signature SVM verdict |
| `POST /api/cheque/extract` | REST: EasyOCR + Gemma fields |
| `GET /` | Serves inline HTML/CSS/JS frontend |

### `agent_studio.py` — Model Wrappers

| Function | Model | Purpose |
|----------|-------|---------|
| `_load_falcon()` | Falcon Perception 0.6B | Lazy-load MLX segmentation model |
| `_load_gemma()` | Gemma 4 E2B | Lazy-load mlx_vlm VLM |
| `_detect(img, query, task)` | Falcon | Instance segmentation → bboxes + RLE masks |
| `_vlm(img, prompt)` | Gemma 4 E2B | Visual language inference → text |

### `agent.py` — Programmatic Pipeline

| Method | Role |
|--------|------|
| `detect_signature(img)` | Falcon + heuristic fallback → bbox |
| `line_sweep_crop(img, bbox)` | Tight signature crop |
| `verify_signature(sig_img)` | Signature SVM → GENUINE / FORGED |
| `extract_fields(img)` | Gemma 4 E2B → 11 structured fields |
| `run(image_path)` | Full pipeline, returns combined result dict |

### `signature_svm/` — Signature Forgery Classifier

| File | Role |
|------|------|
| `svm.py` | `svm_algo()` — full pipeline: load training data + SIFT + k-means vocab + predict |
| `model.pkl` | Pre-trained LinearSVC (512-dim: 500 SIFT BoVW + 12 geometric features) |
| `verifier.py` | Adapter: PIL image → saves to LineSweep_Results → `svm_algo()` → REAL/FORGED |
| `preproc.py` | RGB → grayscale → Otsu threshold → tight binary crop |
| `features.py` | 12 geometric feature extractors |
| `svm_run.py` | Training evaluation script (29 user groups, CLI) |
| `svm_test.py` | Test script for images in `static/LineSweep_Results/` |
| `svm_training_testing.ipynb` | Jupyter notebook: training + evaluation + visualizations |
| `data/genuine/` | 145 genuine signature training images |
| `data/forged/` | 145 forged signature training images |
| `data/origin/` | Origin/test reference signatures |

### `detection/ocr_extractor.py` — Field Extraction

| Item | Detail |
|------|--------|
| Backend | Gemma 4 E2B via `agent_studio._vlm()` (lazy import) |
| OCR hint | EasyOCR raw text passed as context |
| Fields | 11: `account_holder`, `bank_name`, `branch_name`, `cheque_number`, `date`, `payee_name`, `amount_numeric`, `amount_words`, `signature_present`, `ifsc_code`, `account_number` |
| Format | Indian cheque — DD/MM/YYYY dates, "Rupees X Only", IFSC 4+0+6 |

### `detection/Line_Sweep/lineSweepDetect.py` — Tight Crop

Line Sweep algorithm: iterates rows/columns to find tightest bounding box around ink pixels.
Returns `{"image": PIL, "bounds": dict, "success": bool}`.

---

## 3. SSE Event Sequences

### Tab 1 — Signature Verification

```
Client → POST /api/verify/stream {image_b64}
Server → detect_start        {model, task}
Server → detect_complete     {bbox, method, duration_s, annotated_b64}
       OR detect_notice      {message}   ← when Falcon finds nothing
Server → crop_complete       {sig_b64, method}
Server → verify_complete     {verdict, confidence, model, duration_s}
       OR no_detection       {message}
Server → done
```

Verdict values: `GENUINE` | `FORGED` | `UNSIGNED`

### Tab 2 — Data Extraction

```
Client → POST /api/extract/stream {image_b64}
Server → loading_models
Server → models_ready
Server → extract_start       {model: "Gemma 4 E2B"}
Server → extract_complete    {fields, duration_s}
       OR extract_unavailable {message}
Server → done                {json_output}
```

### Tab 3 — Visual Reasoning

```
Client → POST /api/reason/stream {image_b64, question}
Server → loading_models      {model: "Gemma 4 E2B"}
Server → models_ready
Server → reason_start        {question, model, task}
Server → reason_complete     {answer, duration_s}
       OR error              {message}
Server → done
```

---

## 4. Detection Fallback Logic

```
Falcon Perception runs
       │
       ├── dets not empty AND bbox found
       │       └── use Falcon bbox  (method = "falcon-perception")
       │
       └── dets empty OR exception
               └── Heuristic fallback:
                       bbox = [w*0.50, h*0.58, w, h]
                       method = "heuristic"
```

---

## 5. Signature SVM Feature Pipeline

```
Input: PIL signature crop
    │
preproc.preproc()
    │  RGB → grayscale → Otsu threshold → tight binary crop
    ▼
features.py  (12 geometric features)
    │  aspect_ratio, hull/bounding, contour/bounding,
    │  ratio, centroid_0, centroid_1,
    │  eccentricity, solidity,
    │  skewness_0, skewness_1, kurtosis_0, kurtosis_1
    ▼
SIFT keypoints + descriptors  (cv2.SIFT_create)
    │
k-means vocabulary (500 clusters, built from training data)
    │
BoVW histogram  (500-dim)
    ▼
Feature vector: [500 SIFT BoVW] + [12 geometric] = 512-dim
    │
model.pkl  (pre-trained LinearSVC)
    ▼
class 1 → FORGED    class 2 → GENUINE
```

---

## 6. Technology Stack

| Layer | Technology |
|-------|-----------|
| Web server | FastAPI + uvicorn |
| Streaming | Server-Sent Events (SSE) |
| Frontend | Vanilla JS + CSS (inline in cheque_studio.py) |
| Detection model | Falcon Perception 0.6B (MLX, Apple Silicon) |
| VLM | Gemma 4 E2B via mlx_vlm |
| Forgery classifier | LinearSVC — SIFT BoVW + geometric features (sklearn) |
| Image processing | Pillow, OpenCV, NumPy, SciPy |
| Mask decoding | pycocotools (COCO RLE format) |
| Platform | macOS Apple Silicon (M-series) |
