## Connected Components Algorithm — Signature Region Extractor

# Connected Components — Detection Phase

The Connected Components algorithm uses union-find labeling to identify distinct
connected regions in a binary image and fits a tight bounding box around the
largest region (assumed to be the signature).

### Major Assumption

The signature we want to extract is a **single connected blob** of ink pixels.
Disconnected signatures (e.g. initials with gaps) will yield a suboptimal crop —
use the Line Sweep algorithm for those cases.

### How It Works

1. Threshold the OCR-cropped image to binary (pixels > 128 → white background, rest → black ink).
2. Run two-pass connected component labeling via the `UFarray` union-find structure.
3. Group all labeled pixels by their component ID.
4. Fit a bounding rectangle (`cv2.boundingRect`) around each component.
5. Select the component with the **largest bounding-box area**.
6. Crop the original colour image to that rectangle.

### Component: `UFarray` (unionFindArray.py)

A custom union-find (disjoint-set) data structure used during the sweep:

| Method | Purpose |
|--------|---------|
| `makeLabel()` | Create and return a new unique label |
| `find(i)` | Find root of label `i` with path compression |
| `union(i, j)` | Merge the sets containing `i` and `j` |
| `flatten()` | Compress all paths in one pass |

### Public API

```python
from detection.Connected_Components.connectedComponent import (
    connected_component_crop,
    connected_component_with_bounds,
)

# Returns a PIL Image (cropped, or original on failure)
cropped = connected_component_crop(pil_image)

# Returns dict with image + pixel bounds
result = connected_component_with_bounds(pil_image)
# → {"image": PIL Image, "bounds": {"x1":…, "y1":…, "x2":…, "y2":…}, "success": bool}

# Batch processing
from detection.Connected_Components.connectedComponent import main
main(input_dir="path/to/OCR_Results", output_dir="path/to/CC_Results")
```

### Failure Scenarios

When the signature has disconnected strokes (e.g. dotted letters, separate initials),
the algorithm selects only the largest blob and misses the rest.
In that case the Line Sweep algorithm (`detection/Line_Sweep/`) should be used instead.

### References
- Connected Component Algorithm — https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
- Union Find Array — https://www.geeksforgeeks.org/number-of-connected-components-of-a-graph-using-disjoint-set-union/
- OpenCV — https://pypi.org/project/opencv-python/
- PIL — https://pillow.readthedocs.io/en/stable/
