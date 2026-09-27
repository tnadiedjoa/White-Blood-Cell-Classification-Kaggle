"""
Handcrafted features of a segmented white blood cell: shape (asymmetry, border, size),
colour moments, texture (GLCM, LBP), HOG and Fourier descriptors of the nucleus contour.
"""
import cv2
import numpy as np
from scipy import stats
from skimage import measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

from .segmentation import segment_cell

N_HOG = 32  # HOG feature count per region (64x64, ppc=32, cpb=2, orient=8)
N_FOURIER = 10  # Fourier descriptor count


def extract_shape_features(mask_nucleus, mask_cell, mask_cytoplasm):
    """Shape features: Asymmetry, Border, Dimension (A, B, D)."""
    feats = {}

    # --- NUCLEUS ---
    nuc_area = np.sum(mask_nucleus > 0)
    cell_area = np.sum(mask_cell > 0)
    cyto_area = np.sum(mask_cytoplasm > 0)

    feats['nuc_area'] = nuc_area
    feats['cell_area'] = cell_area
    feats['cyto_area'] = cyto_area
    feats['nc_ratio'] = nuc_area / max(cell_area, 1)

    # Nucleus region properties
    nuc_labels = measure.label(mask_nucleus > 0)
    nuc_regions = measure.regionprops(nuc_labels)

    if len(nuc_regions) > 0:
        # Take the largest region as the main nucleus
        main_nuc = max(nuc_regions, key=lambda r: r.area)

        # A — Asymmetry
        feats['nuc_eccentricity'] = main_nuc.eccentricity
        feats['nuc_solidity'] = main_nuc.solidity
        feats['nuc_extent'] = main_nuc.extent
        hu = main_nuc.moments_hu
        for k in range(7):
            feats[f'nuc_hu_{k}'] = -np.sign(hu[k]) * np.log10(max(abs(hu[k]), 1e-30))

        # B — Border irregularity
        perimeter = main_nuc.perimeter
        area = main_nuc.area
        feats['nuc_circularity'] = (4 * np.pi * area) / max(perimeter ** 2, 1)
        feats['nuc_perimeter'] = perimeter
        convex_area = main_nuc.convex_area if hasattr(main_nuc, 'convex_area') else main_nuc.area
        feats['nuc_convexity'] = area / max(convex_area, 1)  # same definition as solidity

        # D — Dimension
        feats['nuc_major_axis'] = main_nuc.major_axis_length
        feats['nuc_minor_axis'] = main_nuc.minor_axis_length
        feats['nuc_axis_ratio'] = main_nuc.minor_axis_length / max(main_nuc.major_axis_length, 1)
        feats['nuc_equiv_diameter'] = main_nuc.equivalent_diameter

        # Number of lobes
        feats['nuc_num_lobes'] = len(nuc_regions)
    else:
        for key in ['nuc_eccentricity', 'nuc_solidity', 'nuc_extent',
                     'nuc_circularity', 'nuc_perimeter', 'nuc_convexity',
                     'nuc_major_axis', 'nuc_minor_axis', 'nuc_axis_ratio',
                     'nuc_equiv_diameter', 'nuc_num_lobes']:
            feats[key] = 0
        for k in range(7):
            feats[f'nuc_hu_{k}'] = 0

    # Whole cell properties
    cell_labels = measure.label(mask_cell > 0)
    cell_regions = measure.regionprops(cell_labels)
    if len(cell_regions) > 0:
        main_cell = max(cell_regions, key=lambda r: r.area)
        feats['cell_eccentricity'] = main_cell.eccentricity
        feats['cell_solidity'] = main_cell.solidity
        feats['cell_circularity'] = (4 * np.pi * main_cell.area) / max(main_cell.perimeter ** 2, 1)
        feats['cell_major_axis'] = main_cell.major_axis_length
        feats['cell_minor_axis'] = main_cell.minor_axis_length
    else:
        feats['cell_eccentricity'] = 0
        feats['cell_solidity'] = 0
        feats['cell_circularity'] = 0
        feats['cell_major_axis'] = 0
        feats['cell_minor_axis'] = 0

    return feats


def extract_color_features(img_bgr, mask_nucleus, mask_cytoplasm):
    """Color features (C) in RGB, HSV, LAB for nucleus and cytoplasm."""
    feats = {}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    spaces = {'rgb': img_rgb, 'hsv': img_hsv, 'lab': img_lab}
    channels = {'rgb': ['R', 'G', 'B'], 'hsv': ['H', 'S', 'V'], 'lab': ['L', 'A', 'B_lab']}
    masks = {'nuc': mask_nucleus, 'cyto': mask_cytoplasm}

    for region_name, mask in masks.items():
        mask_bool = mask > 0
        if np.sum(mask_bool) < 10:
            # Not enough pixels — set features to 0
            for space_name in spaces:
                for ch_name in channels[space_name]:
                    for stat_name in ['mean', 'std', 'skew', 'kurt']:
                        feats[f'{region_name}_{ch_name}_{stat_name}'] = 0
            continue

        for space_name, img_space in spaces.items():
            for ch_idx, ch_name in enumerate(channels[space_name]):
                vals = img_space[:, :, ch_idx][mask_bool].astype(np.float64)
                feats[f'{region_name}_{ch_name}_mean'] = np.mean(vals)
                feats[f'{region_name}_{ch_name}_std'] = np.std(vals)
                feats[f'{region_name}_{ch_name}_skew'] = float(stats.skew(vals))
                feats[f'{region_name}_{ch_name}_kurt'] = float(stats.kurtosis(vals))

    # Global color ratios (on nucleus)
    nuc_bool = mask_nucleus > 0
    if np.sum(nuc_bool) > 10:
        r_mean = np.mean(img_rgb[:, :, 0][nuc_bool].astype(np.float64))
        g_mean = np.mean(img_rgb[:, :, 1][nuc_bool].astype(np.float64))
        b_mean = np.mean(img_rgb[:, :, 2][nuc_bool].astype(np.float64))
        feats['nuc_rg_ratio'] = r_mean / max(g_mean, 1)
        feats['nuc_rb_ratio'] = r_mean / max(b_mean, 1)
        feats['nuc_gb_ratio'] = g_mean / max(b_mean, 1)
    else:
        feats['nuc_rg_ratio'] = 0
        feats['nuc_rb_ratio'] = 0
        feats['nuc_gb_ratio'] = 0

    return feats


def extract_texture_features(img_bgr, mask_nucleus, mask_cytoplasm):
    """Texture features: GLCM and LBP on nucleus and cytoplasm."""
    feats = {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    masks = {'nuc': mask_nucleus, 'cyto': mask_cytoplasm}

    for region_name, mask in masks.items():
        mask_bool = mask > 0
        if np.sum(mask_bool) < 100:
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                feats[f'{region_name}_glcm_{prop}'] = 0
            feats[f'{region_name}_lbp_mean'] = 0
            feats[f'{region_name}_lbp_std'] = 0
            for b in range(10):
                feats[f'{region_name}_lbp_hist_{b}'] = 0
            continue

        # Extract region crop for GLCM
        rows = np.where(mask_bool)[0]
        cols = np.where(mask_bool)[1]
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        crop = gray[r_min:r_max+1, c_min:c_max+1]
        mask_crop = mask[r_min:r_max+1, c_min:c_max+1]
        crop_masked = crop.copy()
        crop_masked[mask_crop == 0] = 0

        # GLCM
        if crop_masked.shape[0] > 2 and crop_masked.shape[1] > 2:
            glcm = graycomatrix(crop_masked, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
                                levels=256, symmetric=True, normed=True)
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                vals = graycoprops(glcm, prop)
                feats[f'{region_name}_glcm_{prop}'] = np.mean(vals)
        else:
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                feats[f'{region_name}_glcm_{prop}'] = 0

        # LBP
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_region = lbp[mask_bool]
        feats[f'{region_name}_lbp_mean'] = np.mean(lbp_region)
        feats[f'{region_name}_lbp_std'] = np.std(lbp_region)
        hist, _ = np.histogram(lbp_region, bins=10, range=(0, 10), density=True)
        for b in range(10):
            feats[f'{region_name}_lbp_hist_{b}'] = hist[b]

    return feats


def extract_hog_features(img_bgr, mask_nucleus, mask_cytoplasm):
    """HOG (Histogram of Oriented Gradients) features on nucleus and cytoplasm."""
    feats = {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    masks = {'nuc': mask_nucleus, 'cyto': mask_cytoplasm}

    for region_name, mask in masks.items():
        mask_bool = mask > 0
        if np.sum(mask_bool) < 100:
            for i in range(N_HOG):
                feats[f'{region_name}_hog_{i}'] = 0
            continue

        # Crop to bounding box
        rows = np.where(mask_bool)[0]
        cols = np.where(mask_bool)[1]
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        crop = gray[r_min:r_max+1, c_min:c_max+1]

        # Resize to fixed size for consistent HOG descriptor
        crop_resized = cv2.resize(crop, (64, 64))

        hog_feats = hog(crop_resized, orientations=8, pixels_per_cell=(32, 32),
                        cells_per_block=(2, 2), feature_vector=True)

        for i in range(min(N_HOG, len(hog_feats))):
            feats[f'{region_name}_hog_{i}'] = hog_feats[i]
        for i in range(len(hog_feats), N_HOG):
            feats[f'{region_name}_hog_{i}'] = 0

    return feats


def extract_fourier_descriptors(mask_nucleus):
    """Fourier descriptors of the nucleus contour, normalized by the DC term. The magnitudes are
    rotation-invariant; the DC term depends on the nucleus position, so they are neither
    translation- nor scale-invariant."""
    feats = {}

    contours, _ = cv2.findContours(mask_nucleus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0 or len(max(contours, key=len)) < 5:
        for i in range(N_FOURIER):
            feats[f'nuc_fourier_{i}'] = 0
        return feats

    contour = max(contours, key=len).squeeze()
    if contour.ndim != 2:
        for i in range(N_FOURIER):
            feats[f'nuc_fourier_{i}'] = 0
        return feats

    # Complex representation of contour
    z = contour[:, 0] + 1j * contour[:, 1]

    # FFT
    fourier = np.fft.fft(z)

    # Magnitudes (rotation invariance), divided by the DC term |F[0]|, which depends on position
    dc = np.abs(fourier[0]) + 1e-10
    magnitudes = np.abs(fourier[1:N_FOURIER+1]) / dc

    for i in range(N_FOURIER):
        if i < len(magnitudes):
            feats[f'nuc_fourier_{i}'] = magnitudes[i]
        else:
            feats[f'nuc_fourier_{i}'] = 0

    return feats


def extract_all_features(img_path, ref_mean, ref_std):
    """All features of one image. `ref_mean`/`ref_std` are the Reinhard reference
    statistics from `segmentation.compute_reference_stats`. Returns None if unreadable."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    # Resize to uniform size
    img_bgr = cv2.resize(img_bgr, (256, 256))

    # Segmentation
    mask_cell, mask_nuc, mask_cyto = segment_cell(img_bgr, ref_mean, ref_std)

    # Extract features on ORIGINAL image (not Reinhard-normalized)
    shape_feats = extract_shape_features(mask_nuc, mask_cell, mask_cyto)
    color_feats = extract_color_features(img_bgr, mask_nuc, mask_cyto)
    texture_feats = extract_texture_features(img_bgr, mask_nuc, mask_cyto)
    hog_feats = extract_hog_features(img_bgr, mask_nuc, mask_cyto)
    fourier_feats = extract_fourier_descriptors(mask_nuc)

    all_feats = {**shape_feats, **color_feats, **texture_feats, **hog_feats, **fourier_feats}
    return all_feats
