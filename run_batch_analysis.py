"""
Batch Cellular Vacuole Size Variance Analyzer for Folder 2-10
==============================================================
Runs the complete cellular analysis pipeline on all microscopy images in 2-10:
  - Adaptive foreground-background segmentation supporting any color palette
  - Contour extraction & spatial mapping via pointPolygonTest
  - Statistical analysis & standard deviation computation
  - Heatmap generation with plasma colormap & colorbar
  - Directional compass gradient analysis (N, NE, E, SE, S, SW, W, NW)
"""

import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vacuole_variance_analyzer import (
    extract_vacuole_contours,
    extract_cell_contours,
    map_vacuoles_to_cells,
    calculate_cell_statistics,
    generate_heatmap,
    save_visualization,
    Cell,
    Vacuole,
)


def segment_adaptive(img_bgr: np.ndarray):
    """
    Universal adaptive segmentation supporting diverse staining palettes
    (e.g., olive green walls + blue vacuoles, magenta walls + gray vacuoles,
    black walls + brown vacuoles, etc.) on light microscopic backgrounds.
    """
    h, w = img_bgr.shape[:2]
    # Sample background color from central 60% of field
    c = img_bgr[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    gray_c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    thresh_val = np.percentile(gray_c, 75)
    bg_color = np.median(c[gray_c >= thresh_val], axis=0)

    # Euclidean color distance from background
    dist = np.linalg.norm(img_bgr.astype(np.float32) - bg_color, axis=2)
    non_bg = dist[dist > 10]
    thresh = max(18.0, min(30.0, float(np.percentile(non_bg, 20)))) if len(non_bg) > 0 else 22.0
    fg_mask = (dist > thresh).astype(np.uint8) * 255

    # Separate walls from vacuoles by geometric morphology
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    vac_mask = np.zeros((h, w), dtype=np.uint8)
    wall_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
        # Vacuoles: small compact circular components
        if 3 <= area <= 140 and aspect < 2.5 and bw <= 25 and bh <= 25:
            vac_mask[labels == i] = 255
        else:
            # Everything else constitutes the cell wall mesh
            wall_mask[labels == i] = 255

    # Heal 1-pixel micro-gaps in wall outlines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    vac_mask = cv2.bitwise_and(vac_mask, cv2.bitwise_not(wall_mask))

    return wall_mask, vac_mask


def compute_compass_direction(cells, img_shape):
    """
    Computes the 2D gradient of within-cell vacuole size variance across the tissue
    and maps it to one of the 8 compass directions: N, NE, E, SE, S, SW, W, NW.
    """
    for c in cells:
        M = cv2.moments(c.contour)
        c.centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] != 0 else (0.0, 0.0)

    stds = [c.std_dev for c in cells if len(c.vacuoles) >= 2]
    if len(stds) < 10:
        return "UNKNOWN", 0.0, 0.0, 0.0

    # Focus on cells above baseline variance
    top_thresh = np.percentile(stds, 70)
    high_cells = [c for c in cells if len(c.vacuoles) >= 2 and c.std_dev >= top_thresh]
    if len(high_cells) < 10:
        high_cells = [c for c in cells if len(c.vacuoles) >= 2]

    xs = np.array([c.centroid[0] for c in high_cells])
    ys = np.array([c.centroid[1] for c in high_cells])
    vals = np.array([c.std_dev for c in high_cells])

    A = np.column_stack([xs, ys, np.ones_like(xs)])
    coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
    gx, gy = float(coeffs[0]), float(-coeffs[1])  # Invert y because image rows point downward

    angle = float(np.degrees(np.arctan2(gy, gx)) % 360)
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    sec = int(np.round(angle / 45.0)) % 8
    return dirs[sec], angle, gx, gy


def process_image(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load {img_path}")

    # Step 1: Adaptive Segmentation
    w_m, v_m = segment_adaptive(img)

    # Step 2: Extraction
    vacs = extract_vacuole_contours(v_m, min_vacuole_area=3.0)
    cells = extract_cell_contours(w_m, min_cell_area=50.0, image_shape=img.shape)

    # Step 3: Spatial Mapping
    map_vacuoles_to_cells(cells, vacs)

    # Step 4: Statistical Analysis
    calculate_cell_statistics(cells, ddof=1)

    # Step 5: Compass Direction Analysis
    direction, angle, gx, gy = compute_compass_direction(cells, img.shape)

    # Step 6: Heatmap Generation & Output
    base = os.path.splitext(img_path)[0]
    out_heatmap = f"{base}_variance_heatmap.png"

    canvas, norm, cmap = generate_heatmap(img.shape, cells, colormap_name="plasma")
    save_visualization(
        canvas_bgr=canvas,
        norm=norm,
        cmap=cmap,
        output_path=out_heatmap,
        original_image_bgr=img,
        cells=cells,
        vacuoles=vacs,
        save_multi_panel=True,
    )

    stds = [c.std_dev for c in cells if len(c.vacuoles) >= 2]
    return {
        "file": os.path.basename(img_path),
        "total_cells": len(cells),
        "analyzed_cells": len(stds),
        "total_vacuoles": len(vacs),
        "mean_std": float(np.mean(stds)) if stds else 0.0,
        "max_std": float(np.max(stds)) if stds else 0.0,
        "compass_direction": direction,
        "angle_deg": angle,
        "heatmap": out_heatmap,
    }


def main():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2-10")
    files = sorted(glob.glob(os.path.join(folder, "*.webp")))
    if not files:
        print(f"No webp files found in {folder}")
        return

    print(f"\nProcessing {len(files)} cellular images in {folder}...\n")
    results = []
    for f in files:
        res = process_image(f)
        results.append(res)

    print("\n" + "=" * 90)
    print("                     BATCH ANALYSIS PIPELINE RESULTS (FOLDER 2-10)                     ")
    print("=" * 90)
    print(f"{'Filename':<20} | {'Cells':<6} | {'Active':<7} | {'Vacuoles':<9} | {'Mean Std':<9} | {'Max Std':<9} | {'Compass Direction'}")
    print("-" * 90)
    for r in results:
        print(f"{r['file']:<20} | {r['total_cells']:<6} | {r['analyzed_cells']:<7} | {r['total_vacuoles']:<9} | {r['mean_std']:<9.2f} | {r['max_std']:<9.2f} | {r['compass_direction']} ({r['angle_deg']:.1f}°)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
