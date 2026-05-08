## Line Sweep Algorithm — Signature Region Extractor

# Line Sweep — Detection Phase

The Line Sweep algorithm runs two directional passes (horizontal then vertical)
over the thresholded image to tightly fit a rectangle around foreground ink pixels.

### Why Line Sweep over Connected Components?

| Aspect | Connected Components | Line Sweep |
|--------|---------------------|------------|
| Assumption | Signature is a single connected blob | No connectivity required |
| Disconnected signatures | Fails (picks largest component only) | Works correctly |
| Speed | Slower (union-find over all pixels) | Faster (row/column scans) |

### Steps

1. Convert the OCR-cropped image to grayscale.
2. Apply inverse binary threshold at 128 (`cv2.THRESH_BINARY_INV`) — ink pixels become 255.
3. **Horizontal sweep** — scan each row; find the first and last row that contain any ink pixel (`np.any(line == 255)`). Skip a candidate start if fewer than 5 rows of ink follow it.
4. **Vertical sweep** — within the found row range, scan each column using a 20-column-wide window; find the first and last column range containing ink. Skip a candidate start if fewer than 20 columns of ink follow it.
5. Crop the original colour image to `[indexStartX:indexEndX+1, indexStartY:indexEndY+1]`.

### Public API

```python
from detection.Line_Sweep.lineSweepDetect import line_sweep_crop, line_sweep_with_bounds

# Returns a PIL Image (cropped, or original on failure)
cropped = line_sweep_crop(pil_image)

# Returns dict with image + pixel bounds
result = line_sweep_with_bounds(pil_image)
# → {"image": PIL Image, "bounds": {"x1":…, "y1":…, "x2":…, "y2":…}, "success": bool}

# Batch processing
from detection.Line_Sweep.lineSweepDetect import main
main(input_dir="path/to/OCR_Results", output_dir="path/to/LineSweep_Results")
```

### References
- OpenCV — https://pypi.org/project/opencv-python/
- PIL — https://pillow.readthedocs.io/en/stable/
