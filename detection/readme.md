## Detection Phase

# Detection Phase

Phase 1 of the two-phase signature verification pipeline.  Given a raw cheque
image, this phase locates and tightly crops the signature region so it can be
fed to Phase 2 (Verification).

### Folder Structure

```
detection/
├── ocr_extractor.py          ← VLM-based cheque field extractor
├── line_sweep.py              ← Thin wrapper re-exporting Line_Sweep functions
├── OCR/
│   ├── OCR_Algorithm.py       ← OCR keyword search + GLM-4V fallback
│   └── readme.md
├── Line_Sweep/
│   ├── lineSweepDetect.py     ← Line Sweep algorithm
│   └── readme.md
└── Connected_Components/
    ├── connectedComponent.py  ← Connected Components algorithm
    ├── unionFindArray.py      ← Union-find data structure
    └── readme.md
```

### Detection Pipeline

The three sub-algorithms are run in a fallback chain to maximise robustness:

```
GLM-4V (offline VLM)
    ↓ (if unavailable or no bbox returned)
OCR keyword search ("Please sign above" — pytesseract)
    ↓ (if keywords not found)
Line Sweep on the bottom-right quadrant of the cheque
    ↓ (last resort)
Connected Components on the bottom-right quadrant
```

### Sub-algorithm Comparison

| Algorithm | Strength | Weakness |
|-----------|----------|---------|
| **GLM-4V** | Visual understanding; robust to layout variation | Requires ~18 GB VRAM; slow on CPU |
| **OCR keyword** | Fast; works on standard cheque layouts | Fails if "Please sign above" text is absent/rotated/faded |
| **Line Sweep** | Handles disconnected signatures; fast | Requires a pre-cropped region near the signature |
| **Connected Components** | Tight crop around connected ink | Fails for disconnected / multi-part signatures |

### Importing

```python
# Preferred: use glm_ocr_detect which orchestrates the full fallback chain
from detection.ocr_extractor import locate_signature_region, extract_cheque_fields

# Individual algorithms
from detection.OCR.OCR_Algorithm import detect_signature_region
from detection.Line_Sweep.lineSweepDetect import line_sweep_with_bounds
from detection.Connected_Components.connectedComponent import connected_component_with_bounds
```

### References

- GLM-4V-9B — https://huggingface.co/THUDM/glm-4v-9b
- pytesseract — https://pypi.org/project/pytesseract/
- OpenCV — https://pypi.org/project/opencv-python/
