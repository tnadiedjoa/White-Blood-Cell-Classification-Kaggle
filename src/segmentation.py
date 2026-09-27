"""
Nucleus / cytoplasm segmentation of white blood cell images:
  1. Reinhard colour normalization (LAB space), robust to staining/illumination variations
  2. Marker-controlled watershed for the cell boundary
  3. Morphological geodesic active contours for a refined nucleus boundary
"""
import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient


def compute_reference_stats(train_df, train_dir, n_per_class=40):
    """Compute mean/std of LAB channels from a stratified sample of training images."""
    l_vals, a_vals, b_vals = [], [], []
    # Stratified sampling: up to n_per_class images per class
    sample = train_df.groupby('label').sample(n=n_per_class, random_state=42, replace=True).drop_duplicates().reset_index(drop=True)
    for _, row in sample.iterrows():
        img = cv2.imread(str(train_dir / row['ID']))
        if img is None:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
        l_vals.append(lab[:, :, 0].mean())
        a_vals.append(lab[:, :, 1].mean())
        b_vals.append(lab[:, :, 2].mean())
    return (np.mean(l_vals), np.mean(a_vals), np.mean(b_vals)), \
           (np.std(l_vals), np.std(a_vals), np.std(b_vals))


def reinhard_normalize(img_bgr, ref_mean, ref_std):
    """Reinhard color normalization in LAB space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    for ch in range(3):
        ch_mean = lab[:, :, ch].mean()
        ch_std = lab[:, :, ch].std() + 1e-6
        lab[:, :, ch] = (lab[:, :, ch] - ch_mean) * (ref_std[ch] / ch_std) + ref_mean[ch]
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def segment_cell(img_bgr, ref_mean, ref_std):
    """
    Segmentation in three steps:
    1. Reinhard color normalization
    2. Marker-controlled Watershed for cell boundary
    3. Morphological GAC for refined nucleus boundary
    Returns: mask_cell, mask_nucleus, mask_cytoplasm
    """
    # Step 0: Color normalization
    img_norm = reinhard_normalize(img_bgr, ref_mean, ref_std)
    img_hsv = cv2.cvtColor(img_norm, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_norm, cv2.COLOR_BGR2LAB)

    # =====================================================
    # CELL SEGMENTATION — Watershed
    # =====================================================
    sat = img_hsv[:, :, 1]

    # Initial Otsu threshold
    _, otsu_mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Compute distance transform for Watershed markers
    dist_transform = cv2.distanceTransform(otsu_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Sure background = dilated Otsu mask
    sure_bg = cv2.dilate(otsu_mask, kernel, iterations=3)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers for Watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Background = 1 (not 0)
    markers[unknown == 255] = 0  # Unknown = 0

    # Apply Watershed
    img_ws = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
    img_ws_3ch = img_ws.copy()
    cv2.watershed(img_ws_3ch, markers)

    # Extract cell mask: keep the largest non-background marker
    unique_markers = np.unique(markers)
    unique_markers = unique_markers[(unique_markers > 1) & (unique_markers != -1)]  # Exclude bg and boundaries

    if len(unique_markers) > 0:
        # Find the marker with the largest area
        areas = [(m, np.sum(markers == m)) for m in unique_markers]
        best_marker = max(areas, key=lambda x: x[1])[0]
        mask_cell = np.uint8(markers == best_marker) * 255
    else:
        # Fallback to Otsu if Watershed fails
        mask_cell = otsu_mask.copy()

    # Keep largest connected component + fill holes
    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask_cell, connectivity=8)
    if num_cc > 1:
        largest = 1 + np.argmax(cc_stats[1:, cv2.CC_STAT_AREA])
        mask_cell = np.uint8(cc_labels == largest) * 255

    mask_cell_filled = ndimage.binary_fill_holes(mask_cell > 0).astype(np.uint8) * 255

    # =====================================================
    # NUCLEUS SEGMENTATION — GAC (Geodesic Active Contours)
    # =====================================================
    L_chan = img_lab[:, :, 0]
    L_masked = L_chan.copy()
    L_masked[mask_cell_filled == 0] = 255  # Background = white

    # Initial estimate via Otsu (used as GAC initialization)
    _, mask_nuc_init = cv2.threshold(L_masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_nuc_init = cv2.bitwise_and(mask_nuc_init, mask_cell_filled)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_nuc_init = cv2.morphologyEx(mask_nuc_init, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Check if Otsu found a reasonable nucleus
    nuc_area_init = np.sum(mask_nuc_init > 0)
    cell_area = np.sum(mask_cell_filled > 0)

    if nuc_area_init > 50 and cell_area > 0 and (nuc_area_init / cell_area) < 0.95:
        # Prepare edge map for GAC (inverse gaussian gradient on L channel)
        L_float = L_masked.astype(np.float64) / 255.0
        gimage = inverse_gaussian_gradient(L_float, alpha=100, sigma=3.0)

        # GAC initialization = Otsu result (binary)
        init_ls = (mask_nuc_init > 0).astype(np.int8)

        # Run morphological GAC — refines the boundary
        # Gentle params to preserve lobulated nuclei (neutrophils, etc.)
        gac_result = morphological_geodesic_active_contour(
            gimage, num_iter=40, init_level_set=init_ls,
            smoothing=1, balloon=0  # No balloon pressure → edge-driven only
        )
        mask_nucleus = (gac_result > 0).astype(np.uint8) * 255

        # Ensure nucleus stays within cell
        mask_nucleus = cv2.bitwise_and(mask_nucleus, mask_cell_filled)

        # Cleanup small fragments
        mask_nucleus = cv2.morphologyEx(mask_nucleus, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask_nucleus = cv2.morphologyEx(mask_nucleus, cv2.MORPH_CLOSE, kernel_small, iterations=1)

        # Safety check: if GAC produced garbage, fallback to Otsu
        nuc_area_gac = np.sum(mask_nucleus > 0)
        if nuc_area_gac < 30 or (nuc_area_gac / cell_area) > 0.95:
            mask_nucleus = mask_nuc_init
    else:
        # Fallback to Otsu-only nucleus
        mask_nucleus = mask_nuc_init
        mask_nucleus = cv2.morphologyEx(mask_nucleus, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # =====================================================
    # CYTOPLASM = cell - nucleus
    # =====================================================
    mask_cytoplasm = cv2.bitwise_and(mask_cell_filled, cv2.bitwise_not(mask_nucleus))

    return mask_cell_filled, mask_nucleus, mask_cytoplasm
