## OCR-Based Signature Detection

# OCR — Detection Phase

The OCR approach identifies and extracts text from the cheque image to localise
the signature field.

### Enhancement over original approach

| Step | Original approach | This project |
|------|---------------------|--------------|
| Text recognition | pytesseract | **GLM-4V (offline VLM)** → pytesseract fallback |
| Keyword target | "Please sign above" | Visual understanding of full cheque layout |
| Accuracy | Fails if text is rotated / faded | Robust to scan quality variation |

### Original approach (retained as fallback)

**Assumption** — Cheques contain the phrase *"Please sign above"* near the
signature field.

**Steps:**

1. Read the cheque image with OpenCV (`cv2.imread`).
2. Apply an HSV blue-ink mask and Gaussian blur to suppress noise.
3. Run `pytesseract.image_to_data()` to get per-word bounding boxes.
4. Locate the words **"please"** and **"above"** in the data.
5. Derive the signature bounding box using fixed scale factors:
   - `x1 = please.left  − length × 2.5`
   - `y1 = please.top   − length × 2.0`
   - width  = `length × (2.5 + 0.5 + 1)`
   - height = `length × 2.0`
6. Crop the region and save to `OCR_Results/`.

### Public API

```python
from detection.OCR.OCR_Algorithm import detect_signature_region, run_ocr_on_folder

# Single image
result = detect_signature_region(pil_image)
# → {"bbox": [x1, y1, x2, y2], "method": "glm-4v" | "tesseract"}

# Batch folder processing (original script mode)
run_ocr_on_folder(input_dir, output_dir)
```

### References
- PIL — https://pillow.readthedocs.io/en/stable/
- OpenCV — https://pypi.org/project/opencv-python/
- pytesseract — https://pypi.org/project/pytesseract/
- GLM-4V — https://huggingface.co/THUDM/glm-4v-9b
