# Cellular Imagery Vacuole Variance Heatmap Analyzer

A Python pipeline using **OpenCV (`cv2`)**, **NumPy**, and **Matplotlib** to analyze cellular microscopy imagery and produce quantitative heatmaps of within-cell vacuole size variance.

---

## Target Image Characteristics

- **Background**: Light / white field.
- **Cell Walls**: Thin purple polygonal outlines.
- **Vacuoles**: Small, filled gray circles inside individual cell compartments.

---

## Analytical Pipeline

1. **Color Segmentation (`HSV` Color Space)**:
   - Converts the input image to HSV color space.
   - Applies distinct threshold ranges:
     - **Purple cell walls**: Hue `[115, 165]`, Saturation `[35, 255]`, Value `[30, 255]`. Applies morphological closing to bridge 1-pixel line discontinuities.
     - **Gray vacuoles**: Hue `[0, 180]`, Saturation `[0, 55]`, Value `[30, 200]` (neutral saturation, distinctly darker than light background `V > 200`).
2. **Contour Extraction & Noise Filtering**:
   - Extracts cell boundaries using `cv2.findContours` (`cv2.RETR_CCOMP` to reliably extract enclosed cell cavities within thin polygonal outlines or honeycomb tissue meshes).
   - Extracts vacuole contours using `cv2.findContours` (`cv2.RETR_EXTERNAL`).
   - Filters out single-pixel and sub-resolution noise (`area < 3.0 px`).
   - Computes centroids using spatial moments (`m10 / m00`, `m01 / m00`).
3. **Spatial Mapping**:
   - Determines enclosing cell for each vacuole using `cv2.pointPolygonTest`.
   - Employs `measureDist=True` to assign boundary-adjacent vacuoles to the cell with maximum positive internal distance.
4. **Statistical Analysis**:
   - For each cell, calculates the sample standard deviation ($\sigma$, `ddof=1`) of assigned vacuole pixel areas.
   - Enforces the analytical rule: **cells with fewer than 2 vacuoles receive $\sigma = 0.0$**.
5. **Heatmap Generation**:
   - Allocates a blank canvas matching the original image dimensions.
   - Normalizes standard deviation values across cells (`matplotlib.colors.Normalize`).
   - Uses `cv2.fillPoly` to paint each cell with its mapped colormap value (e.g. `'plasma'`, `'viridis'`).
   - Outlines cell perimeters with contrasting borders for morphological clarity.
6. **Output & Publication-Ready Visualization**:
   - Overlays a calibrated colorbar indicating the variance scale ($\sigma$ in $px^2$).
   - Saves the final visualization image to the local directory.
   - Supports an optional `--multi-panel` flag to generate a 3-panel comparative diagnostic figure (Original Image vs. Segmentations Overlay vs. Heatmap).

---

## Installation & Requirements

Ensure you have Python 3.9+ with the required dependencies:

```bash
pip install opencv-python numpy matplotlib
```

Or using `uv`:

```bash
uv run --with opencv-python --with numpy --with matplotlib python vacuole_variance_analyzer.py ...
```

---

## Usage

### Basic Execution

```bash
python vacuole_variance_analyzer.py path/to/cellular_image.png
```

This saves the heatmap visualization to `path/to/cellular_image_variance_heatmap.png`.

### Advanced Options

```bash
python vacuole_variance_analyzer.py sample_cells.png \
    --output custom_heatmap.png \
    --colormap viridis \
    --min-vacuole-area 4.0 \
    --min-cell-area 150.0 \
    --ddof 1 \
    --multi-panel
```

### CLI Arguments Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `image` | Path to the microscopy image file (positional, required). | - |
| `-o`, `--output` | Output path for the generated heatmap visualization. | `<stem>_variance_heatmap.png` |
| `-c`, `--colormap` | Matplotlib colormap (`plasma`, `viridis`, `magma`, `inferno`, etc.). | `plasma` |
| `--min-vacuole-area`| Minimum vacuole area in pixels to filter out single-pixel noise. | `3.0` |
| `--min-cell-area` | Minimum cell boundary area in pixels. | `50.0` |
| `--max-cell-area` | Maximum cell area threshold (filters slide borders/frames). | `None` (auto-detected) |
| `--ddof` | Degrees of freedom for standard deviation (`1` for sample, `0` for population). | `1` |
| `--multi-panel` | Also export a 3-panel diagnostic figure (Original, Segmentations, Heatmap). | `False` |

---

## Synthetic Image Verification

You can generate a test cellular image with known ground-truth variances:

```bash
python generate_sample_image.py
python vacuole_variance_analyzer.py sample_cells.png --multi-panel
```
