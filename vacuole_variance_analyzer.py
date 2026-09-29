"""
Cellular Vacuole Size Variance Heatmap Analyzer
===============================================
This module analyzes cellular microscopy imagery featuring:
  - Light/white background
  - Thin purple polygonal outlines (cell walls)
  - Small filled gray circles (vacuoles) inside the cells

Pipeline:
  1. Color Segmentation (HSV color space thresholding)
  2. Contour Extraction (Cell boundaries & vacuole area calculation with noise filtering)
  3. Spatial Mapping (Assigning vacuole centroids to enclosing cell contours via pointPolygonTest)
  4. Statistical Analysis (Within-cell vacuole area standard deviation calculation)
  5. Heatmap Generation (cv2.fillPoly on matching canvas mapped to Matplotlib colormap)
  6. Output (Colorbar overlay and saving publication-ready visualization)

Usage:
  python vacuole_variance_analyzer.py path/to/image.png
  python vacuole_variance_analyzer.py path/to/image.png --colormap viridis --output heatmap.png
"""

import os
import sys
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Configure logging for robust analytical diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VacuoleAnalyzer")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Vacuole:
    """Represents a segmented vacuole within the image."""
    vacuole_id: int
    contour: np.ndarray
    area: float
    centroid: Tuple[float, float]
    assigned_cell_id: Optional[int] = None


@dataclass
class Cell:
    """Represents a segmented cell compartment."""
    cell_id: int
    contour: np.ndarray
    area: float
    vacuoles: List[Vacuole] = field(default_factory=list)
    variance: float = 0.0
    std_dev: float = 0.0

    @property
    def vacuole_areas(self) -> List[float]:
        return [v.area for v in self.vacuoles]


# ============================================================================
# Step 1: Color Segmentation
# ============================================================================

def segment_adaptive(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Universal adaptive segmentation supporting diverse histological/synthetic color schemes
    (e.g., olive green walls + blue vacuoles, magenta walls + gray vacuoles,
    black walls + brown vacuoles, etc.) on light microscopic backgrounds.
    """
    h, w = img_bgr.shape[:2]
    c = img_bgr[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    gray_c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    thresh_val = np.percentile(gray_c, 75)
    bg_color = np.median(c[gray_c >= thresh_val], axis=0)

    dist = np.linalg.norm(img_bgr.astype(np.float32) - bg_color, axis=2)
    non_bg = dist[dist > 10]
    thresh = max(18.0, min(30.0, float(np.percentile(non_bg, 20)))) if len(non_bg) > 0 else 22.0
    fg_mask = (dist > thresh).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    vac_mask = np.zeros((h, w), dtype=np.uint8)
    wall_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
        if 3 <= area <= 140 and aspect < 2.5 and bw <= 25 and bh <= 25:
            vac_mask[labels == i] = 255
        else:
            wall_mask[labels == i] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    vac_mask = cv2.bitwise_and(vac_mask, cv2.bitwise_not(wall_mask))
    return wall_mask, vac_mask


def segment_colors(
    image_bgr: np.ndarray,
    mode: str = "auto",
    purple_hsv_lower: Tuple[int, int, int] = (115, 35, 30),
    purple_hsv_upper: Tuple[int, int, int] = (165, 255, 255),
    gray_hsv_lower: Tuple[int, int, int] = (0, 0, 30),
    gray_hsv_upper: Tuple[int, int, int] = (180, 55, 200),
    close_cell_walls: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment cell walls and vacuoles into binary masks.

    Modes:
      - 'auto' (default): Adaptive color distance & morphology, supporting arbitrary
        color schemes (olive green, magenta, black, purple, brown, etc.).
      - 'hsv': Explicit HSV thresholding for purple cell walls and gray vacuoles.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid image: array is None or empty.")

    if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel color image (BGR), got shape {image_bgr.shape}")

    if mode == "auto":
        wall_mask, vacuole_mask = segment_adaptive(image_bgr)
        if cv2.countNonZero(vacuole_mask) > 100:
            logger.info(
                "Adaptive segmentation complete. Wall mask pixels: %d, Vacuole mask pixels: %d",
                cv2.countNonZero(wall_mask),
                cv2.countNonZero(vacuole_mask),
            )
            return wall_mask, vacuole_mask

    # HSV segmentation for purple cell walls and gray vacuoles
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_purple = np.array(purple_hsv_lower, dtype=np.uint8)
    upper_purple = np.array(purple_hsv_upper, dtype=np.uint8)
    cell_wall_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    if close_cell_walls:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cell_wall_mask = cv2.morphologyEx(cell_wall_mask, cv2.MORPH_CLOSE, kernel)

    lower_gray = np.array(gray_hsv_lower, dtype=np.uint8)
    upper_gray = np.array(gray_hsv_upper, dtype=np.uint8)
    vacuole_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    vacuole_mask = cv2.bitwise_and(vacuole_mask, cv2.bitwise_not(cell_wall_mask))

    logger.info(
        "HSV segmentation complete. Wall mask pixels: %d, Vacuole mask pixels: %d",
        cv2.countNonZero(cell_wall_mask),
        cv2.countNonZero(vacuole_mask),
    )

    return cell_wall_mask, vacuole_mask


# ============================================================================
# Step 2: Contour Extraction & Noise Filtering
# ============================================================================

def extract_vacuole_contours(
    vacuole_mask: np.ndarray,
    min_vacuole_area: float = 3.0,
    max_vacuole_area: Optional[float] = None,
) -> List[Vacuole]:
    """
    Extract individual vacuole contours, compute pixel area, filter out single-pixel noise,
    and calculate accurate centroid coordinates via spatial moments.

    Args:
        vacuole_mask: Binary mask of vacuole regions.
        min_vacuole_area: Minimum contour area threshold (removes single-pixel noise).
        max_vacuole_area: Optional upper area threshold to reject large segmentation artifacts.

    Returns:
        List of Vacuole instances.
    """
    contours, _ = cv2.findContours(
        vacuole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    vacuoles: List[Vacuole] = []
    noise_count = 0

    for i, cnt in enumerate(contours):
        # Calculate pixel area using Green's Theorem via cv2.contourArea
        area = float(cv2.contourArea(cnt))

        # Basic area filtering for single-pixel / sub-resolution noise
        if area < min_vacuole_area or (max_vacuole_area and area > max_vacuole_area):
            noise_count += 1
            continue

        # Calculate centroid using spatial moments
        moments = cv2.moments(cnt)
        if moments["m00"] != 0:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            # Fallback to coordinate mean if area moment is zero
            cx = float(np.mean(cnt[:, 0, 0]))
            cy = float(np.mean(cnt[:, 0, 1]))

        vacuoles.append(
            Vacuole(
                vacuole_id=len(vacuoles),
                contour=cnt,
                area=area,
                centroid=(cx, cy),
            )
        )

    logger.info(
        "Vacuole extraction: detected %d valid vacuoles (filtered out %d noise blobs, min_area=%.1f px).",
        len(vacuoles),
        noise_count,
        min_vacuole_area,
    )
    return vacuoles


def extract_cell_contours(
    cell_wall_mask: np.ndarray,
    min_cell_area: float = 50.0,
    max_cell_area: Optional[float] = None,
    image_shape: Optional[Tuple[int, ...]] = None,
) -> List[Cell]:
    """
    Extract cell boundaries from the thin polygonal cell wall outlines.

    Analytical Architecture:
      - Uses cv2.RETR_CCOMP (two-level hierarchy).
      - In a thin outline or honeycomb mesh, the interior cell cavities are enclosed loops
        where hierarchy indicates parent >= 0 (internal holes enclosed by walls).
      - Filters out macroscopic slide/specimen frame boundaries that span the entire field of view.
      - If discrete, filled, or external contours are present without holes, it falls back
        cleanly to RETR_EXTERNAL to ensure 100% boundary detection coverage across all morphology types.

    Args:
        cell_wall_mask: Binary mask of cell walls.
        min_cell_area: Minimum area threshold to ignore small loops or junction artifacts.
        max_cell_area: Optional maximum area threshold to ignore whole-image bounds.
        image_shape: Optional (height, width) of the image for automatic frame detection.

    Returns:
        List of Cell instances with extracted boundary contours.
    """
    contours, hierarchy = cv2.findContours(
        cell_wall_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None or len(contours) == 0:
        logger.warning("No contours found in cell wall mask.")
        return []

    hier = hierarchy[0]
    selected_contours: List[np.ndarray] = []

    # Check for inner cavities (holes enclosed by cell walls, where parent >= 0)
    enclosed_cavities = [
        contours[i]
        for i in range(len(contours))
        if hier[i][3] >= 0 and cv2.contourArea(contours[i]) >= min_cell_area
    ]

    if len(enclosed_cavities) > 0:
        # Mesh / honeycomb / hollow outlines: the inner cavities are the true cell interiors
        selected_contours = enclosed_cavities
    else:
        # Discrete filled shapes or single contours: use external contours
        ext_contours, _ = cv2.findContours(
            cell_wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        selected_contours = [
            cnt for cnt in ext_contours if cv2.contourArea(cnt) >= min_cell_area
        ]

    # Automatic frame boundary threshold if not explicitly provided
    img_h, img_w = (image_shape[0], image_shape[1]) if image_shape else (None, None)
    total_img_area = (img_h * img_w) if (img_h and img_w) else None

    cells: List[Cell] = []
    filtered_frames = 0

    for cnt in selected_contours:
        area = float(cv2.contourArea(cnt))
        bx, by, bw, bh = cv2.boundingRect(cnt)

        # 1. Filter out slide / field-of-view outer frame (spans > 75% of both dimensions)
        if img_h and img_w and bw > 0.75 * img_w and bh > 0.75 * img_h:
            filtered_frames += 1
            continue

        # 2. Filter out oversized macroscopic containers
        if max_cell_area is not None and area > max_cell_area:
            filtered_frames += 1
            continue
        elif total_img_area and len(selected_contours) > 20 and area > 0.05 * total_img_area:
            # If there are many cells, a region > 5% of entire image is an extracellular border/container
            filtered_frames += 1
            continue

        cells.append(Cell(cell_id=len(cells), contour=cnt, area=area))

    if filtered_frames > 0:
        logger.info("Filtered out %d outer frame / macroscopic container regions.", filtered_frames)

    logger.info("Cell extraction: detected %d valid cell boundaries.", len(cells))
    return cells


# ============================================================================
# Step 3: Spatial Mapping
# ============================================================================

def map_vacuoles_to_cells(
    cells: List[Cell],
    vacuoles: List[Vacuole],
) -> Dict[int, List[Vacuole]]:
    """
    Determine which cell contains each vacuole using cv2.pointPolygonTest on vacuole centroids.

    Analytical Accuracy Detail:
      - Uses measureDist=True to find the exact signed Euclidean distance from centroid to contour.
      - If a vacuole centroid is near a shared wall or tests positive for multiple boundaries,
        it is assigned to the cell whose boundary encloses it with the highest positive distance (deepest inside).
      - If a vacuole centroid lies outside all cell contours (distance < 0), it is flagged as unassigned.

    Args:
        cells: List of extracted Cell objects.
        vacuoles: List of extracted Vacuole objects.

    Returns:
        Mapping of cell_id to list of assigned Vacuole objects.
    """
    unassigned_count = 0

    for vacuole in vacuoles:
        cx, cy = vacuole.centroid
        candidates: List[Tuple[Cell, float]] = []

        for cell in cells:
            # pointPolygonTest: >0 inside, 0 on edge, <0 outside
            dist = cv2.pointPolygonTest(cell.contour, (cx, cy), measureDist=True)
            if dist >= 0:
                candidates.append((cell, dist))

        if candidates:
            # Analytical precision:
            # If a point is enclosed by nested boundaries or touching cells,
            # assign it to the TIGHTEST (smallest area) enclosing cell,
            # breaking ties by deepest internal distance (largest dist).
            candidates.sort(key=lambda item: (item[0].area, -item[1]))
            best_cell = candidates[0][0]
            vacuole.assigned_cell_id = best_cell.cell_id
            best_cell.vacuoles.append(vacuole)
        else:
            unassigned_count += 1

    if unassigned_count > 0:
        logger.warning(
            "%d vacuoles could not be assigned to any cell (likely extracellular debris).",
            unassigned_count,
        )

    logger.info("Spatial mapping complete. %d vacuoles mapped across %d cells.",
                len(vacuoles) - unassigned_count, len(cells))

    return {c.cell_id: c.vacuoles for c in cells}


# ============================================================================
# Step 4: Statistical Analysis
# ============================================================================

def calculate_cell_statistics(cells: List[Cell], ddof: int = 1) -> None:
    """
    Calculate within-cell vacuole area standard deviation for each cell.

    Analytical Rule:
      - If a cell has fewer than 2 vacuoles (< 2), standard deviation is assigned as 0.0.
      - If a cell has >= 2 vacuoles, standard deviation is calculated over vacuole areas
        using specified degrees of freedom (default ddof=1 for sample standard deviation).

    Args:
        cells: List of Cell objects with populated vacuoles.
        ddof: Delta Degrees of Freedom (1 for sample std dev, 0 for population std dev).
    """
    for cell in cells:
        areas = cell.vacuole_areas
        n = len(areas)
        if n < 2:
            cell.variance = 0.0
            cell.std_dev = 0.0
        else:
            # Use sample standard deviation (ddof=1)
            cell.std_dev = float(np.std(areas, ddof=ddof))
            cell.variance = float(np.var(areas, ddof=ddof))

    logger.info("Statistical analysis complete:")
    for cell in cells:
        logger.info(
            "  Cell #%02d | Vacuoles: %d | Areas: [%s] | Std Dev: %.2f px^2",
            cell.cell_id,
            len(cell.vacuoles),
            ", ".join(f"{a:.1f}" for a in cell.vacuole_areas[:5]) + ("..." if len(cell.vacuoles) > 5 else ""),
            cell.std_dev,
        )


def compute_compass_direction(cells: List[Cell]) -> Tuple[str, float]:
    """
    Computes the 2D gradient of within-cell vacuole size variance across the tissue
    and maps it to one of the 8 compass directions: N, NE, E, SE, S, SW, W, NW.
    """
    for c in cells:
        M = cv2.moments(c.contour)
        c.centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] != 0 else (0.0, 0.0)

    stds = [c.std_dev for c in cells if len(c.vacuoles) >= 2]
    if len(stds) < 10:
        return "N/A", 0.0

    # Focus on the upper quartile / region of variance increase
    top_thresh = np.percentile(stds, 70)
    high_cells = [c for c in cells if len(c.vacuoles) >= 2 and c.std_dev >= top_thresh]
    if len(high_cells) < 10:
        high_cells = [c for c in cells if len(c.vacuoles) >= 2]

    xs = np.array([c.centroid[0] for c in high_cells])
    ys = np.array([c.centroid[1] for c in high_cells])
    vals = np.array([c.std_dev for c in high_cells])

    A = np.column_stack([xs, ys, np.ones_like(xs)])
    coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
    gx, gy = float(coeffs[0]), float(-coeffs[1])  # Invert y because image rows point downwards

    angle = float(np.degrees(np.arctan2(gy, gx)) % 360)
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    sec = int(np.round(angle / 45.0)) % 8
    return dirs[sec], angle


# ============================================================================
# Step 5: Heatmap Generation
# ============================================================================

def generate_heatmap(
    image_shape: Tuple[int, ...],
    cells: List[Cell],
    colormap_name: str = "plasma",
    background_color: Tuple[int, int, int] = (255, 255, 255),
    draw_cell_borders: bool = True,
    border_color: Tuple[int, int, int] = (40, 40, 40),
    border_thickness: int = 1,
) -> Tuple[np.ndarray, Normalize, matplotlib.colors.Colormap]:
    """
    Create a canvas matching original image dimensions and render each cell colored
    by its calculated standard deviation mapped to a Matplotlib colormap.

    Args:
        image_shape: Shape tuple of the original image (H, W, ...).
        cells: List of Cell objects with computed std_dev values.
        colormap_name: Matplotlib colormap name (e.g. 'plasma', 'viridis').
        background_color: BGR tuple for empty/extracellular canvas background.
        draw_cell_borders: Whether to draw contrasting outlines around cells.
        border_color: BGR tuple for cell outline borders.
        border_thickness: Thickness in pixels for cell outline borders.

    Returns:
        Tuple of (canvas_bgr, norm_object, colormap_object).
    """
    h, w = image_shape[0], image_shape[1]
    canvas = np.full((h, w, 3), background_color, dtype=np.uint8)

    if not cells:
        logger.warning("No cells provided to generate_heatmap; returning blank canvas.")
        norm = Normalize(vmin=0.0, vmax=1.0)
        cmap = matplotlib.colormaps[colormap_name]
        return canvas, norm, cmap

    # Collect all standard deviations to define dynamic colormap range
    std_values = [cell.std_dev for cell in cells]
    vmin = min(std_values)
    vmax = max(std_values)

    # Prevent division by zero if all cells have identical std_dev (e.g. all 0.0)
    if vmax <= vmin:
        vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)

    try:
        cmap = matplotlib.colormaps[colormap_name]
    except KeyError:
        logger.warning("Colormap '%s' not recognized; falling back to 'plasma'.", colormap_name)
        cmap = matplotlib.colormaps["plasma"]

    # Sort cells by area descending so any larger regions are drawn first,
    # ensuring smaller cellular compartments are always cleanly visible on top
    cells_sorted = sorted(cells, key=lambda c: c.area, reverse=True)
    for cell in cells_sorted:
        norm_val = norm(cell.std_dev)
        rgba = cmap(norm_val)  # returns (r, g, b, a) in [0.0, 1.0]

        # Convert RGB floats [0, 1] to BGR integers [0, 255] for OpenCV
        bgr_color = (
            int(rgba[2] * 255),
            int(rgba[1] * 255),
            int(rgba[0] * 255),
        )

        # cv2.fillPoly expects list of contour arrays
        cv2.fillPoly(canvas, [cell.contour], color=bgr_color)

        if draw_cell_borders:
            cv2.polylines(
                canvas,
                [cell.contour],
                isClosed=True,
                color=border_color,
                thickness=border_thickness,
                lineType=cv2.LINE_AA,
            )

    logger.info(
        "Heatmap canvas rendered (Colormap: %s, Range: [%.2f, %.2f] px^2).",
        colormap_name,
        vmin,
        vmax,
    )
    return canvas, norm, cmap


# ============================================================================
# Step 6: Output & Colorbar Visualization
# ============================================================================

def save_visualization(
    canvas_bgr: np.ndarray,
    norm: Normalize,
    cmap: matplotlib.colors.Colormap,
    output_path: str,
    original_image_bgr: Optional[np.ndarray] = None,
    cells: Optional[List[Cell]] = None,
    vacuoles: Optional[List[Vacuole]] = None,
    save_multi_panel: bool = False,
    dpi: int = 300,
) -> str:
    """
    Overlay a colorbar indicating variance scale and save the publication-quality
    heatmap visualization to the local filesystem.

    Args:
        canvas_bgr: Heatmap canvas in BGR format.
        norm: Matplotlib Normalize instance used for color mapping.
        cmap: Matplotlib colormap instance.
        output_path: Target output image file path.
        original_image_bgr: Optional original image for multi-panel comparison.
        cells: Optional list of cells for inspection.
        vacuoles: Optional list of vacuoles for inspection.
        save_multi_panel: If True, also generates a comprehensive multi-panel figure.
        dpi: Resolution (DPI) for saved figures.

    Returns:
        Absolute path to the saved heatmap image.
    """
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    h, w = canvas_bgr.shape[:2]
    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

    # 1. Primary Output: Heatmap with overlaid/adjacent colorbar
    fig_w = max(6.0, w / 100.0 + 1.8)
    fig_h = max(5.0, h / 100.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.imshow(canvas_rgb)
    ax.axis("off")
    ax.set_title("Within-Cell Vacuole Size Variance Heatmap", fontsize=13, fontweight="bold", pad=12)

    # Add colorbar matching standard deviation scale
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Vacuole Size Standard Deviation $\\sigma$ ($px^2$)", fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    logger.info("Saved heatmap visualization to: %s", output_path)

    # 2. Optional Diagnostic Multi-Panel (Original vs Segmented vs Heatmap)
    if save_multi_panel and original_image_bgr is not None:
        panel_path = os.path.splitext(output_path)[0] + "_comprehensive_analysis.png"
        fig_multi, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)

        # Panel 1: Original Image
        orig_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
        axes[0].imshow(orig_rgb)
        axes[0].set_title("Original Cellular Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Detections Overlay
        overlay_bgr = original_image_bgr.copy()
        if cells:
            for cell in cells:
                cv2.polylines(overlay_bgr, [cell.contour], True, (0, 255, 0), 2)
        if vacuoles:
            for v in vacuoles:
                cv2.circle(overlay_bgr, (int(round(v.centroid[0])), int(round(v.centroid[1]))), 3, (0, 0, 255), -1)
        axes[1].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
        axes[1].set_title(
            f"Segmented Cells (Green) & Vacuoles (Red)\nN_cells={len(cells or [])}, N_vac={len(vacuoles or [])}",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].axis("off")

        # Panel 3: Heatmap with Colorbar
        axes[2].imshow(canvas_rgb)
        axes[2].set_title("Variance Heatmap", fontsize=12, fontweight="bold")
        axes[2].axis("off")
        cbar_multi = fig_multi.colorbar(scalar_mappable, ax=axes[2], fraction=0.046, pad=0.04)
        cbar_multi.set_label("Vacuole Size Std Dev $\\sigma$ ($px^2$)", fontsize=10)

        plt.tight_layout()
        plt.savefig(panel_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig_multi)
        logger.info("Saved multi-panel diagnostic figure to: %s", panel_path)

    return os.path.abspath(output_path)


# ============================================================================
# Main Analytical Pipeline Runner
# ============================================================================

def run_analysis_pipeline(
    image_path: str,
    output_path: Optional[str] = None,
    colormap: str = "plasma",
    min_vacuole_area: float = 3.0,
    min_cell_area: float = 50.0,
    max_cell_area: Optional[float] = None,
    save_multi_panel: bool = False,
    ddof: int = 1,
) -> Dict[str, Any]:
    """
    Execute the full end-to-end cellular analysis pipeline.

    Args:
        image_path: Path to the microscopy image file.
        output_path: Target path for the output heatmap image.
        colormap: Matplotlib colormap name.
        min_vacuole_area: Noise threshold for vacuole area in pixels.
        min_cell_area: Minimum area for cell contour extraction.
        max_cell_area: Optional maximum area threshold to ignore macroscopic frames.
        save_multi_panel: Whether to also save the 3-panel comparative diagnostic figure.
        ddof: Degrees of freedom for standard deviation (1 for sample std dev).

    Returns:
        Dictionary containing pipeline results, summary statistics, and output paths.
    """
    # 0. Validate image path and load image
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image file does not exist: {image_path}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(
            f"Failed to decode image at '{image_path}'. Ensure it is a valid format (PNG, JPEG, TIFF)."
        )

    logger.info("Loaded image '%s' with dimensions %dx%d px.", image_path, image_bgr.shape[1], image_bgr.shape[0])

    # Determine default output path if not specified
    if not output_path:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_variance_heatmap.png"

    # Step 1: Color Segmentation
    cell_wall_mask, vacuole_mask = segment_colors(image_bgr)

    # Step 2: Contour Extraction & Noise Filtering
    vacuoles = extract_vacuole_contours(vacuole_mask, min_vacuole_area=min_vacuole_area)
    cells = extract_cell_contours(
        cell_wall_mask,
        min_cell_area=min_cell_area,
        max_cell_area=max_cell_area,
        image_shape=image_bgr.shape,
    )

    if not cells:
        logger.error("No cell boundaries were identified. Review HSV thresholds or image contrast.")
    if not vacuoles:
        logger.warning("No vacuoles were identified. Review vacuole HSV thresholds.")

    # Step 3: Spatial Mapping
    map_vacuoles_to_cells(cells, vacuoles)

    # Step 4: Statistical Analysis
    calculate_cell_statistics(cells, ddof=ddof)

    # Step 5: Heatmap Generation
    canvas_bgr, norm, cmap = generate_heatmap(
        image_shape=image_bgr.shape,
        cells=cells,
        colormap_name=colormap,
    )

    # Step 6: Output & Colorbar Visualization
    saved_path = save_visualization(
        canvas_bgr=canvas_bgr,
        norm=norm,
        cmap=cmap,
        output_path=output_path,
        original_image_bgr=image_bgr,
        cells=cells,
        vacuoles=vacuoles,
        save_multi_panel=save_multi_panel,
    )

    # Compilation of analytical results
    results = {
        "status": "SUCCESS",
        "image_path": image_path,
        "output_path": saved_path,
        "num_cells": len(cells),
        "num_vacuoles": len(vacuoles),
        "cell_metrics": [
            {
                "cell_id": c.cell_id,
                "cell_area": c.area,
                "vacuole_count": len(c.vacuoles),
                "vacuole_areas": c.vacuole_areas,
                "variance": c.variance,
                "std_dev": c.std_dev,
            }
            for c in cells
        ],
    }

    return results


# ============================================================================
# Command-Line Interface
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cellular imagery and generate a within-cell vacuole size variance heatmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the cellular image file to analyze.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for the generated heatmap visualization image.",
    )
    parser.add_argument(
        "-c", "--colormap",
        type=str,
        default="plasma",
        help="Matplotlib colormap for variance visualization (e.g. 'plasma', 'viridis', 'magma', 'inferno').",
    )
    parser.add_argument(
        "--min-vacuole-area",
        type=float,
        default=3.0,
        help="Minimum vacuole area in pixels to filter out single-pixel noise.",
    )
    parser.add_argument(
        "--min-cell-area",
        type=float,
        default=50.0,
        help="Minimum cell boundary area in pixels.",
    )
    parser.add_argument(
        "--max-cell-area",
        type=float,
        default=None,
        help="Maximum cell area threshold (rejects macroscopic tissue borders or slide frames).",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=1,
        choices=[0, 1],
        help="Degrees of freedom for standard deviation (1 for sample std dev, 0 for population std dev).",
    )
    parser.add_argument(
        "--multi-panel",
        action="store_true",
        help="Also export a multi-panel diagnostic figure (Original, Segmentations, Heatmap).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        results = run_analysis_pipeline(
            image_path=args.image,
            output_path=args.output,
            colormap=args.colormap,
            min_vacuole_area=args.min_vacuole_area,
            min_cell_area=args.min_cell_area,
            max_cell_area=args.max_cell_area,
            save_multi_panel=args.multi_panel,
            ddof=args.ddof,
        )

        print("\n=======================================================")
        print("               ANALYSIS PIPELINE SUMMARY               ")
        print("=======================================================")
        print(f"Input Image:    {results['image_path']}")
        print(f"Output Heatmap: {results['output_path']}")
        print(f"Cells Analyzed: {results['num_cells']}")
        print(f"Vacuoles Found: {results['num_vacuoles']}")
        print("-------------------------------------------------------")
        print(f"{'Cell ID':<8} | {'Vacuoles':<9} | {'Mean Area (px^2)':<16} | {'Std Dev (px^2)':<14}")
        print("-------------------------------------------------------")
        for m in results["cell_metrics"]:
            areas = m["vacuole_areas"]
            mean_area = float(np.mean(areas)) if areas else 0.0
            print(f"{m['cell_id']:<8} | {m['vacuole_count']:<9} | {mean_area:<16.2f} | {m['std_dev']:<14.2f}")
        print("=======================================================\n")

    except Exception as exc:
        logger.exception("Analysis pipeline failed with error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
