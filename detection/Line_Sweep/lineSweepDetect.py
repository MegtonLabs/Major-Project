"""
Line Sweep Algorithm — Signature Region Extractor
==================================================
Source: Detection Phase / Line Sweep / lineSweepDetect.py
        (Thapar Institute of Engineering and Technology Capstone Project)

The Line-Sweeping algorithm is run in both directions (horizontal and vertical)
to tightly fit a rectangle around the signature foreground pixels.
It is better performing than the Connected Components algorithm because it does
not require signatures to be connected.

Changes from original:
  - Original was a file-walking script; exposed here as callable functions so
    other modules (cheque_studio, glm_ocr_detect) can import and reuse it.
  - `line_sweep_crop(img)`          → PIL Image → PIL Image
  - `line_sweep_with_bounds(img)`   → PIL Image → dict with bounds + image
  - `main(input_dir, output_dir)`   → original batch-processing behaviour
  - Logic (threshold values, sweep conditions) is identical to the original.
"""

import os
import numpy as np
import cv2
from PIL import Image


# ── Core algorithm ────────────────────────────────────────────────────────────

def line_sweep_crop(img: Image.Image) -> Image.Image:
    """
    Apply the Line Sweep algorithm to tightly crop the signature region.

    Steps:
      1. Convert image to grayscale (PIL .convert("L")).
      2. Inverse binary threshold at 128 (THRESH_BINARY_INV).
      3. Horizontal sweep → find indexStartX, indexEndX (row range of ink).
      4. Vertical sweep   → find indexStartY, indexEndY (col range of ink).
      5. Crop the original colour array to those bounds.

    Returns the original image unchanged if the sweep yields a degenerate crop.
    """
    result = line_sweep_with_bounds(img)
    return result["image"]


def line_sweep_with_bounds(img: Image.Image) -> dict:
    """
    Apply Line Sweep and also return the pixel bounds found.

    Returns:
        {
            "image":   PIL Image  (cropped, or original on failure),
            "bounds":  {"x1": indexStartY, "y1": indexStartX,
                        "x2": indexEndY,   "y2": indexEndX}  or None,
            "success": bool,
        }
    """
    original_np = np.array(img.convert("RGB"))

    # Step 1+2: grayscale → inverse binary threshold
    grayscale = img.convert("L")
    _, thresh = cv2.threshold(
        np.array(grayscale), 128, 255, cv2.THRESH_BINARY_INV
    )

    rows, cols = thresh.shape

    # ── Horizontal sweep: find row range of ink pixels ────────────────────
    flagx = indexStartX = indexEndX = 0

    for i in range(rows):
        line = thresh[i, :]
        if flagx == 0:
            if np.any(line == 255):
                indexStartX = i
                flagx = 1
        elif flagx == 1:
            if np.any(line == 255):
                indexEndX = i
            elif indexStartX + 5 > indexEndX:
                # Too few rows — reset and keep searching
                indexStartX = 0
                flagx = 0
            else:
                break

    # ── Vertical sweep: find col range within the found rows ─────────────
    flagy = indexStartY = indexEndY = 0

    for j in range(cols):
        # Sample a 20-column-wide slice
        line = thresh[indexStartX:indexEndX, j:j + 20]
        if flagy == 0:
            if np.any(line == 255):
                indexStartY = j
                flagy = 1
        elif flagy == 1:
            if np.any(line == 255):
                indexEndY = j
            elif indexStartY + 20 > indexEndY:
                indexStartY = 0
                flagy = 0
            else:
                break

    # ── Crop ──────────────────────────────────────────────────────────────
    cropped = original_np[indexStartX:indexEndX + 1,
                          indexStartY:indexEndY + 1]

    if cropped.size == 0 or cropped.shape[0] < 5 or cropped.shape[1] < 5:
        return {"image": img, "bounds": None, "success": False}

    return {
        "image":   Image.fromarray(cropped),
        "bounds":  {
            "x1": indexStartY, "y1": indexStartX,
            "x2": indexEndY,   "y2": indexEndX,
        },
        "success": True,
    }


# ── Batch script mode ───────────────────────────

def main(input_dir: str = None, output_dir: str = None):
    """
    Process every image in `input_dir` with the Line Sweep algorithm and
    write the cropped results to `output_dir`.

    Default paths:
      input_dir  → ../OCR/OCR_Results
      output_dir → ./LineSweep_Results
    """
    if input_dir is None:
        input_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../OCR/OCR_Results",
        )
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "LineSweep_Results",
        )

    os.makedirs(output_dir, exist_ok=True)
    total = processed = 0

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        total += 1

        if os.stat(file_path).st_size == 0:
            continue

        print(f"Processing {filename}")
        img    = Image.open(file_path)
        result = line_sweep_with_bounds(img)

        if result["success"]:
            out_name = f"LineSweep_Result_{filename}"
            # Save as BGR (cv2)
            cv2.imwrite(
                os.path.join(output_dir, out_name),
                cv2.cvtColor(np.array(result["image"]), cv2.COLOR_RGB2BGR),
            )
            processed += 1

    print(f"{processed}/{total} files processed successfully")
    print("Processing Complete.")
    print("You may check the Result folder in the same directory to see the cropped images.")


if __name__ == "__main__":
    main()
