"""
Sample Cellular Image Generator
===============================
Generates synthetic microscopy imagery strictly conforming to:
  - Light/white background
  - Thin purple polygonal outlines (cell walls)
  - Small filled gray circles (vacuoles) inside the cells
  - Controlled ground-truth vacuole size variances across cells
  - Random single-pixel noise to test filtering
"""

import os
import cv2
import numpy as np

def generate_cellular_image(output_path: str = "sample_cells.png", width: int = 800, height: int = 650):
    # 1. Background: Light/white with subtle microscopic field variation
    np.random.seed(42)
    img = np.full((height, width, 3), 246, dtype=np.uint8)
    noise = np.random.randint(-4, 5, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Cell walls: Thin purple polygonal outlines
    # In BGR: Purple is around B=140, G=30, R=120
    purple_bgr = (140, 25, 125)

    # 6 distinct polygonal cells with diverse shapes
    cell_polygons = [
        # Cell 0: Top-left (will have HIGH variance)
        np.array([[50, 60], [240, 45], [265, 250], [180, 280], [60, 240]], np.int32),
        # Cell 1: Top-middle (MODERATE variance)
        np.array([[290, 50], [490, 40], [510, 240], [420, 280], [280, 230]], np.int32),
        # Cell 2: Top-right (ZERO variance - identical sizes)
        np.array([[540, 55], [740, 60], [755, 250], [660, 285], [530, 240]], np.int32),
        # Cell 3: Bottom-left (SINGLE vacuole -> std = 0)
        np.array([[60, 320], [230, 310], [250, 560], [170, 595], [55, 540]], np.int32),
        # Cell 4: Bottom-middle (ZERO vacuoles -> std = 0)
        np.array([[280, 320], [480, 315], [500, 565], [390, 600], [275, 550]], np.int32),
        # Cell 5: Bottom-right (LOW variance)
        np.array([[530, 325], [730, 330], [750, 570], [650, 605], [520, 560]], np.int32),
    ]

    for poly in cell_polygons:
        cv2.polylines(img, [poly], isClosed=True, color=purple_bgr, thickness=2, lineType=cv2.LINE_AA)

    # 3. Vacuoles: Small filled gray circles
    # Neutral gray in BGR: e.g. (110, 110, 110)
    gray_bgr = (105, 105, 105)

    # Ground-truth configurations:
    # Cell 0: High variance: radii 3, 6, 9, 13, 17
    vacuoles_cell_0 = [
        ((110, 110), 3),
        ((190, 100), 6),
        ((140, 170), 9),
        ((200, 210), 13),
        ((100, 220), 17),
    ]

    # Cell 1: Moderate variance: radii 6, 8, 11, 14
    vacuoles_cell_1 = [
        ((350, 110), 6),
        ((440, 110), 8),
        ((360, 200), 11),
        ((440, 200), 14),
    ]

    # Cell 2: Zero variance: all radius 8
    vacuoles_cell_2 = [
        ((600, 110), 8),
        ((690, 110), 8),
        ((610, 190), 8),
        ((690, 200), 8),
        ((650, 150), 8),
    ]

    # Cell 3: Exactly 1 vacuole
    vacuoles_cell_3 = [
        ((150, 440), 10),
    ]

    # Cell 4: 0 vacuoles (empty cell)
    vacuoles_cell_4 = []

    # Cell 5: Low variance: radii 7, 8, 8, 9
    vacuoles_cell_5 = [
        ((590, 410), 7),
        ((680, 410), 8),
        ((600, 500), 8),
        ((690, 490), 9),
    ]

    all_vacuoles = (
        vacuoles_cell_0
        + vacuoles_cell_1
        + vacuoles_cell_2
        + vacuoles_cell_3
        + vacuoles_cell_4
        + vacuoles_cell_5
    )

    for center, radius in all_vacuoles:
        cv2.circle(img, center, radius, gray_bgr, -1, lineType=cv2.LINE_AA)

    # 4. Add isolated single-pixel noise in extracellular spaces
    noise_points = [(20, 20), (770, 20), (20, 620), (770, 620), (400, 15), (400, 630)]
    for pt in noise_points:
        img[pt[1], pt[0]] = gray_bgr

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(output_path, img)
    print(f"Sample cellular image successfully generated: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    generate_cellular_image("sample_cells.png")
