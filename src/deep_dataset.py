"""
Data pipeline of the final deep-learning model (deep_9 no-morph).

- Offline steps, run once: oversampling of rare classes, HSV-based white-cell cropping.
- Online steps: CLAHE on the HSV value channel, Albumentations augmentation, normalization.
- Leakage-free train/validation split (augmented copies of validation images are discarded).
"""
import shutil
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, get_worker_info

SEED = 42

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TTA_ROUNDS = 8


# ------------------------------------------------------------------
# Offline oversampling of rare classes
# ------------------------------------------------------------------
def build_augmenter(seed=SEED):
    """Light numpy/OpenCV augmenter used to create offline copies of rare-class images
    (flips, rotation + translation, brightness/contrast jitter, Gaussian noise)."""
    rng = np.random.default_rng(seed)

    def augment(img):
        out = img.copy()
        if rng.random() < 0.5:
            out = cv2.flip(out, 1)
        if rng.random() < 0.5:
            out = cv2.flip(out, 0)
        angle = float(rng.uniform(-20, 20))
        h, w = out.shape[:2]
        rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rot[0, 2] += float(rng.uniform(-0.08, 0.08) * w)
        rot[1, 2] += float(rng.uniform(-0.08, 0.08) * h)
        out = cv2.warpAffine(out, rot, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        alpha = float(rng.uniform(0.9, 1.1))
        beta = float(rng.uniform(-10, 10))
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
        if rng.random() < 0.3:
            noise = rng.normal(0, 4, size=out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out

    return augment


def build_oversampled_dataset(train_df, train_dir, oversample_dir, oversample_csv,
                              rare_classes, rare_threshold, seed=SEED):
    """Copy every training image to `oversample_dir` and add augmented copies of each rare
    class until it reaches `rare_threshold` images. Copies are named `<stem>_aug_<k>`.
    Skipped if `oversample_csv` already exists. Returns the oversampled metadata."""
    if not Path(oversample_csv).exists():
        print("Creating oversampled dataset...")
        Path(oversample_dir).mkdir(exist_ok=True)
        augment = build_augmenter(seed)
        rows = []
        for _, row in train_df.iterrows():
            shutil.copy2(Path(train_dir) / row['ID'], Path(oversample_dir) / row['ID'])
            rows.append({'ID': row['ID'], 'label': row['label']})
        for cls in rare_classes:
            cls_df = train_df[train_df['label'] == cls]
            n_to_generate = rare_threshold - len(cls_df)
            print(f"  {cls}: generating {n_to_generate} augmented images")
            for k in range(n_to_generate):
                src_row = cls_df.iloc[k % len(cls_df)]
                try:
                    img = np.array(Image.open(Path(train_dir) / src_row['ID']).convert('RGB'))
                except OSError:
                    continue
                src_path = Path(src_row['ID'])
                new_id = f"{src_path.stem}_aug_{k:05d}{src_path.suffix}"
                Image.fromarray(augment(img)).save(str(Path(oversample_dir) / new_id))
                rows.append({'ID': new_id, 'label': cls})
        pd.DataFrame(rows).to_csv(oversample_csv, index=False)
        print(f"Oversampled dataset saved: {len(rows)} images")
    else:
        print(f"Oversampled dataset already exists: {Path(oversample_csv).name}")

    return pd.read_csv(oversample_csv)


# ------------------------------------------------------------------
# Offline HSV-based white-cell cropping
# ------------------------------------------------------------------
def hsv_nucleus_score(img_rgb):
    """Per-pixel score in [0, 1] that is high on the purple, saturated, dark nucleus."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hdist = np.minimum(np.abs(H - 140.0), 180.0 - np.abs(H - 140.0)) / 90.0
    hscore = 1.0 - np.clip(hdist, 0.0, 1.0)
    sscore = np.clip((S - 30.0) / 225.0, 0.0, 1.0)
    vscore = 1.0 - np.clip(V / 255.0, 0.0, 1.0)
    return 0.55 * hscore + 0.25 * sscore + 0.20 * vscore


def extract_wbc_crop(img_rgb, pad=20, min_area_frac=0.0005, q=0.92):
    """Crop the image around the white cell: threshold the nucleus score, keep the large
    component closest to the image centre, dilate it to include the cytoplasm, add
    padding. Returns the original image if no reliable component is found."""
    score = hsv_nucleus_score(img_rgb)
    nuc = (score >= float(np.quantile(score, q))).astype(np.uint8)
    nuc = (cv2.medianBlur(nuc * 255, 5) > 0).astype(np.uint8)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nuc = cv2.morphologyEx(nuc, cv2.MORPH_OPEN, k1, iterations=1)
    nuc = cv2.morphologyEx(nuc, cv2.MORPH_CLOSE, k1, iterations=2)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(nuc, connectivity=8)
    if num <= 1:
        return img_rgb
    h, w = nuc.shape
    min_area = int(min_area_frac * h * w)
    cx0, cy0 = w * 0.5, h * 0.5
    best_i, best_score = None, -1e18
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx = float(stats[i, cv2.CC_STAT_LEFT] + 0.5 * stats[i, cv2.CC_STAT_WIDTH])
        cy = float(stats[i, cv2.CC_STAT_TOP] + 0.5 * stats[i, cv2.CC_STAT_HEIGHT])
        s = np.log1p(area) - 0.0008 * ((cx - cx0) ** 2 + (cy - cy0) ** 2)
        if s > best_score:
            best_score, best_i = s, i
    if best_i is None:
        return img_rgb
    kbig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    mask = cv2.dilate((lab == best_i).astype(np.uint8), kbig, iterations=1)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return img_rgb
    x1, y1 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x2, y2 = min(w, int(xs.max()) + pad), min(h, int(ys.max()) + pad)
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return img_rgb
    return img_rgb[y1:y2, x1:x2]


def precrop_directory(src_dir, dst_dir, df, desc=""):
    """Crop every image listed in `df` from `src_dir` into `dst_dir` (existing files are
    kept, so the step can be resumed)."""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(exist_ok=True)
    t0 = time.time()
    for i, row in df.iterrows():
        dst_path = dst_dir / row['ID']
        if dst_path.exists():
            continue
        try:
            img = np.array(Image.open(Path(src_dir) / row['ID']).convert('RGB'))
        except OSError:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        Image.fromarray(extract_wbc_crop(img)).save(str(dst_path))
        if (i + 1) % 2000 == 0:
            print(f"  {desc} {i + 1}/{len(df)} ({time.time() - t0:.0f}s)")
    print(f"  {desc} done: {len(df)} images in {time.time() - t0:.0f}s")


def run_precrop_pipeline(oversample_dir, precrop_train_dir, precrop_test_dir,
                         train_df_aug, test_df, test_dir, precrop_done_flag):
    """Crop the oversampled training set and the test set once (guarded by a flag file)."""
    if not Path(precrop_done_flag).exists():
        print("Pre-cropping all images (one-time step)...")
        precrop_directory(oversample_dir, precrop_train_dir, train_df_aug, desc="Train")
        precrop_directory(test_dir, precrop_test_dir, test_df, desc="Test")
        Path(precrop_done_flag).write_text("done")
    else:
        print("Pre-cropped images already exist.")


# ------------------------------------------------------------------
# Leakage-free split
# ------------------------------------------------------------------
def parent_stem(img_id):
    """Stem of the original image an (possibly augmented) image was derived from."""
    stem = Path(img_id).stem
    return stem.rsplit('_aug_', 1)[0] if '_aug_' in stem else stem


def leakage_free_split(train_df, train_df_aug, label_encoder, val_ratio=0.10, seed=SEED):
    """Stratified split on the ORIGINAL images. Augmented copies of training images stay in
    the training set; those of validation images are discarded, so the validation set
    contains original images only."""
    labels = label_encoder.transform(train_df['label'])
    train_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=val_ratio,
                                          stratify=labels, random_state=seed)
    train_stems = {Path(i).stem for i in train_df.iloc[train_idx]['ID']}
    train_fold_df = train_df_aug[train_df_aug['ID'].map(parent_stem).isin(train_stems)]
    val_fold_df = train_df.iloc[val_idx]

    leaked = {Path(i).stem for i in val_fold_df['ID']} & set(train_fold_df['ID'].map(parent_stem))
    assert not leaked, f"{len(leaked)} validation images leak into the training set"
    return train_fold_df.reset_index(drop=True), val_fold_df.reset_index(drop=True)


# ------------------------------------------------------------------
# Online preprocessing and augmentation
# ------------------------------------------------------------------
def hsv_value_clahe(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE on the HSV value channel only, so that stain hue and saturation are kept."""
    h, s, v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return cv2.cvtColor(cv2.merge([h, s, clahe.apply(v)]), cv2.COLOR_HSV2RGB)


train_aug = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.75, 1.0), ratio=(0.90, 1.10), p=0.7),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.9),
    A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
             scale=(0.85, 1.15), shear=(-8, 8), p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03, p=0.7),
    A.ToGray(p=0.2),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.GaussNoise(std_range=(0.003, 0.015), p=1.0),
    ], p=0.2),
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 32),
                    hole_width_range=(16, 32), fill=0, p=0.25),
    ToTensorV2(),
])

val_aug = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

tta_aug = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def seed_worker(_):
    """DataLoader `worker_init_fn`: gives each worker its own Albumentations random state.
    Albumentations keeps its generator inside the Compose object, so without this every
    worker would start from the same state and repeat the same draws at every epoch."""
    info = get_worker_info()
    info.dataset.aug.set_random_seed(torch.initial_seed() % 2**32)


class WBCDataset(Dataset):
    """Reads pre-cropped images, applies HSV-V CLAHE then the given Albumentations
    pipeline. Returns (image, label) when `df` has a `label` column, else the image."""

    def __init__(self, df, img_dir, label_encoder=None, augmentation=val_aug,
                 use_hsv_clahe=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.le = label_encoder
        self.aug = augmentation
        self.use_hsv_clahe = use_hsv_clahe
        self.has_labels = 'label' in df.columns
        if self.has_labels:
            self.labels = label_encoder.transform(self.df['label'])

    def __len__(self):
        return len(self.df)

    def load_image(self, idx):
        """Pre-cropped RGB image after CLAHE, before augmentation."""
        try:
            img = np.array(Image.open(self.img_dir / self.df.at[idx, 'ID']).convert('RGB'))
        except OSError:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return hsv_value_clahe(img) if self.use_hsv_clahe else img

    def __getitem__(self, idx):
        tensor = self.aug(image=self.load_image(idx))['image']
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if self.has_labels:
            return tensor, int(self.labels[idx])
        return tensor
