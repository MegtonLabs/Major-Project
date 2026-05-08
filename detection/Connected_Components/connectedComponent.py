"""
Connected Components Algorithm — Signature Region Extractor
===========================================================
Source: Detection Phase / Connected Components / connectedComponent.py
        (Thapar Institute of Engineering and Technology Capstone Project)

Two-pass connected component labeling via union-find, followed by
bounding-box fitting.  The largest bounding box (by area) is taken as the
signature region.

Changes from original:
  - Original used `from unionFindArray import *` (same-directory import).
    Exposed here as callable functions so other modules can import and reuse.
  - `connected_component_crop(img)`        → PIL Image → PIL Image
  - `connected_component_with_bounds(img)` → PIL Image → dict with bounds + image
  - `main(input_dir, output_dir)`          → original batch-processing behaviour
  - Logic (pixel scanning, union-find calls) is identical to the original.
"""

import operator
import os
import random
import numpy as np
import cv2
from itertools import product
from PIL import Image

try:
    from .unionFindArray import UFarray
except ImportError:
    from unionFindArray import UFarray


# ── Core algorithm ────────────────────────────────────────────────────────────

def _run(img: Image.Image):
    """
    Two-pass connected component labeling.

    Returns (labels dict, colourised output_img) — same as the original.
    """
    data   = img.load()
    width, height = img.size

    uf     = UFarray()
    labels = {}

    for y, x in product(range(height), range(width)):
        #
        # Pixel names were chosen as shown:
        #
        #       x -->
        #     -------------
        #  y  | a | b | c |
        #     -------------
        #     | d | e |   |
        #     -------------
        #     |   |   |   |
        #     -------------
        #
        # The current pixel is e
        # a, b, c, and d are its neighbors of interest
        #
        # 255 is white, 0 is black
        # White pixels are background — ignored.
        # Pixels outside the image bounds default to white.
        #

        if data[x, y] == 255:
            pass

        elif y > 0 and data[x, y - 1] == 0:
            labels[x, y] = labels[(x, y - 1)]

        elif x + 1 < width and y > 0 and data[x + 1, y - 1] == 0:
            c = labels[(x + 1, y - 1)]
            labels[x, y] = c

            if x > 0 and data[x - 1, y - 1] == 0:
                a = labels[(x - 1, y - 1)]
                uf.union(c, a)
            elif x > 0 and data[x - 1, y] == 0:
                d = labels[(x - 1, y)]
                uf.union(c, d)

        elif x > 0 and y > 0 and data[x - 1, y - 1] == 0:
            labels[x, y] = labels[(x - 1, y - 1)]

        elif x > 0 and data[x - 1, y] == 0:
            labels[x, y] = labels[(x - 1, y)]

        else:
            labels[x, y] = uf.makeLabel()

    uf.flatten()

    colors     = {}
    output_img = Image.new("RGB", (width, height))
    outdata    = output_img.load()

    for (x, y) in labels:
        component = uf.find(labels[(x, y)])
        labels[(x, y)] = component

        if component not in colors:
            colors[component] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        outdata[x, y] = colors[component]

    return labels, output_img


def _crop_by_connected_component(points: dict, original_np: np.ndarray):
    """
    Find the largest bounding-box component and return its crop.
    Crops image region by connected component, returns ndarray.
    """
    sig = {}
    for data in points.values():
        data = np.array(data)
        x, y, w, h = cv2.boundingRect(data)
        sig[(x, y, w, h)] = w * h

    if not sig:
        return None, None

    sorted_x   = sorted(sig.items(), key=operator.itemgetter(1))
    (x, y, w, h), _ = sorted_x[-1]
    return original_np[y: y + h, x: x + w], (x, y, x + w, y + h)


# ── Public API ────────────────────────────────────────────────────────────────

def connected_component_with_bounds(img: Image.Image) -> dict:
    """
    Apply Connected Components labeling and crop the largest component.

    Returns:
        {
            "image":   PIL Image  (cropped, or original on failure),
            "bounds":  {"x1": …, "y1": …, "x2": …, "y2": …}  or None,
            "success": bool,
        }
    """
    original_np = np.array(img.convert("RGB"))

    # Threshold to binary
    thresh_img = img.point(lambda p: p > 128 and 255).convert("1")

    labels, _ = _run(thresh_img)

    # Group pixel coords by component label
    component_points: dict = {}
    for (x, y), comp in labels.items():
        component_points.setdefault(comp, []).append((x, y))

    crop_np, bounds = _crop_by_connected_component(component_points, original_np)

    if crop_np is None or crop_np.size == 0 or crop_np.shape[0] < 5 or crop_np.shape[1] < 5:
        return {"image": img, "bounds": None, "success": False}

    x1, y1, x2, y2 = bounds
    return {
        "image":   Image.fromarray(crop_np),
        "bounds":  {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "success": True,
    }


def connected_component_crop(img: Image.Image) -> Image.Image:
    """
    Apply Connected Components and return just the cropped PIL Image.
    Returns the original image unchanged if detection fails.
    """
    return connected_component_with_bounds(img)["image"]


# ── Batch script mode ───────────────────────────

def main(input_dir: str = None, output_dir: str = None):
    """
    Process every image in `input_dir` with Connected Components and write
    the cropped results to `output_dir`.

    Default paths:
      input_dir  → ../OCR/OCR_Results
      output_dir → ./ConnectedComponents_Results
    """
    if input_dir is None:
        input_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../OCR/OCR_Results",
        )
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ConnectedComponents_Results",
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
        result = connected_component_with_bounds(img)

        if result["success"]:
            out_name = f"CC_Result_{filename}"
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
