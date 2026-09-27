"""
Generate every figure of the report into report/figures/.

Run from the repository root:  python report/generate_figures.py
Needs the Kaggle data in data/ (with the oversampled and pre-cropped folders created by
notebooks/04_final_model.ipynb), models/best_model.pt and results/classical_ml/.
A GPU is recommended for the validation inference of the final model (~1 min).
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.deep_dataset import (IMAGENET_MEAN, IMAGENET_STD, SEED, WBCDataset,
                              hsv_nucleus_score, leakage_free_split, train_aug, val_aug)
from src.deep_train import WBCClassifier, predict

FIG_DIR = ROOT / "report" / "figures"
DATA = ROOT / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 12})

CLASS_ORDER = ['SNE', 'LY', 'MO', 'BL', 'EO', 'MY', 'BA', 'BNE', 'VLY', 'MMY', 'PMY', 'PC', 'PLY']

train_df = pd.read_csv(DATA / "train_metadata.csv")
aug_df = pd.read_csv(DATA / "train_oversampled_metadata.csv")
le = LabelEncoder().fit(train_df['label'])

def save(fig, name):
    """Vector PDF for charts; JPEG for photo grids, which would make large PDFs."""
    if name.endswith('.jpg'):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        Image.open(buf).convert('RGB').save(FIG_DIR / name, quality=90)
    else:
        fig.savefig(FIG_DIR / name, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  {name}")


def fig_class_distribution():
    orig, over = train_df['label'].value_counts(), aug_df['label'].value_counts()
    median = int(orig.median())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, counts, title, rare_color in (
            (axes[0], orig, f'Original distribution ({len(train_df):,} images)', '#e74c3c'),
            (axes[1], over, f'After oversampling ({len(aug_df):,} images)', '#2ecc71')):
        values = [counts.get(c, 0) for c in CLASS_ORDER]
        colors = [rare_color if orig[c] < median else '#3498db' for c in CLASS_ORDER]
        ax.barh(CLASS_ORDER, values, color=colors, edgecolor='white')
        for i, v in enumerate(values):
            ax.text(v + 100, i, f'{v:,}', va='center', fontsize=11)
        ax.set_xlabel('Number of images')
        ax.set_title(title)
        ax.invert_yaxis()
    axes[0].axvline(median, color='gray', linestyle='--', alpha=0.7, label=f'Median ({median})')
    axes[0].legend(fontsize=11)
    plt.tight_layout()
    save(fig, 'class_distribution.pdf')


def fig_class_examples():
    examples = [('SNE', 'Seg. neutrophil\nmulti-lobed nucleus'),
                ('LY', 'Lymphocyte\nround dark nucleus'),
                ('MO', 'Monocyte\nkidney-shaped nucleus'),
                ('EO', 'Eosinophil\nbilobed nucleus'),
                ('MY', 'Myelocyte\nround nucleus'),
                ('MMY', 'Metamyelocyte\nindented nucleus')]
    fig, axes = plt.subplots(1, 6, figsize=(16, 3))
    for k, (ax, (cls, label)) in enumerate(zip(axes, examples)):
        img_id = train_df[train_df['label'] == cls].sample(1, random_state=SEED)['ID'].iloc[0]
        ax.imshow(Image.open(DATA / "train" / img_id).convert('RGB'))
        ax.set_title(f'({"abcdef"[k]}) {label}', fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    save(fig, 'class_examples.jpg')


def fig_crop_examples():
    classes = ['SNE', 'LY', 'EO', 'BL']
    fig, axes = plt.subplots(3, len(classes), figsize=(14, 9))
    for j, cls in enumerate(classes):
        img_id = train_df[train_df['label'] == cls].sample(1, random_state=SEED)['ID'].iloc[0]
        raw = np.array(Image.open(DATA / "train" / img_id).convert('RGB'))
        crop = np.array(Image.open(DATA / "train_precropped" / img_id).convert('RGB'))
        panels = [(raw, f'{cls}: raw', None), (hsv_nucleus_score(raw), 'HSV nucleus score', 'hot'),
                  (crop, f'Cropped ({crop.shape[1]}x{crop.shape[0]})', None)]
        for i, (img, title, cmap) in enumerate(panels):
            axes[i, j].imshow(img, cmap=cmap)
            axes[i, j].set_title(title, fontsize=14)
            axes[i, j].axis('off')
    plt.tight_layout()
    save(fig, 'crop_examples.jpg')


def denormalize(tensor):
    img = tensor.permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


def fig_augmentation_gallery():
    classes, n_aug = ['PLY', 'BA', 'MY', 'SNE'], 6
    train_aug.set_random_seed(SEED)
    ds = WBCDataset(pd.concat([train_df[train_df['label'] == c].head(1) for c in classes]),
                    DATA / "train_precropped", le)
    fig, axes = plt.subplots(len(classes), n_aug + 1, figsize=(2.6 * (n_aug + 1), 2.9 * len(classes)))
    for i, cls in enumerate(classes):
        img = ds.load_image(i)
        axes[i, 0].imshow(denormalize(val_aug(image=img)['image']))
        axes[i, 0].set_title(f'{cls}: input after CLAHE', fontsize=13)
        for j in range(1, n_aug + 1):
            axes[i, j].imshow(denormalize(train_aug(image=img)['image']))
            axes[i, j].set_title(f'augmented #{j}', fontsize=13)
        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    save(fig, 'augmentation_gallery.jpg')


def final_model_validation():
    """Validation predictions of the final model checkpoint on the leakage-free split."""
    _, val_df = leakage_free_split(train_df, aug_df, le)
    ds = WBCDataset(val_df, DATA / "train_precropped", le, val_aug)
    model = WBCClassifier(num_classes=len(le.classes_), pretrained=False)
    model.load_state_dict(torch.load(ROOT / "models" / "best_model.pt",
                                     map_location='cpu')['model_state_dict'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logits, _ = predict(model, ds, device)
    return val_df, ds.labels, torch.softmax(logits, 1).numpy()


def fig_per_class_scores(y, pred):
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, zero_division=0)
    order = np.argsort(f1)
    x, w = np.arange(len(order)), 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, prec[order], w, label='Precision', color='steelblue')
    ax.bar(x, rec[order], w, label='Recall', color='darkorange')
    ax.bar(x + w, f1[order], w, label='F1', color='seagreen')
    macro = f1_score(y, pred, average='macro')
    ax.axhline(macro, color='gray', linestyle='--', alpha=0.6, label=f'Macro-F1 = {macro:.3f}')
    ax.set_xticks(x, le.classes_[order], rotation=45, ha='right')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score')
    ax.set_title('Per-class precision / recall / F1 on validation (final model)')
    ax.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    save(fig, 'per_class_f1.pdf')


def fig_confusion_matrix(y, pred):
    cm = confusion_matrix(y, pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, mat, fmt, title in ((axes[0], cm, 'd', 'Counts'),
                                (axes[1], cm_norm, '.2f', 'Row-normalized (recall)')):
        ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(len(mat)):
            for j in range(len(mat)):
                if cm[i, j]:
                    ax.text(j, i, format(mat[i, j], fmt), ha='center', va='center', fontsize=10,
                            color='white' if cm_norm[i, j] > 0.5 else 'black')
        ax.set_xticks(range(len(mat)), le.classes_, rotation=45, ha='right')
        ax.set_yticks(range(len(mat)), le.classes_)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
    fig.suptitle(f'Final model, validation (n = {len(y):,}), '
                 f'macro-F1 = {f1_score(y, pred, average="macro"):.4f}', fontsize=13)
    plt.tight_layout()
    save(fig, 'confusion_matrix.pdf')


def fig_qualitative_errors(val_df, y, probs):
    pred, conf = probs.argmax(1), probs.max(1)
    wrong = np.where(pred != y)[0]
    top = wrong[np.argsort(-conf[wrong])[:4]]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.4))
    for ax, i in zip(axes.flat, top):
        ax.imshow(Image.open(DATA / "train_precropped" / val_df.at[i, 'ID']).convert('RGB'))
        ax.set_title(f'True: {le.classes_[y[i]]} | Pred: {le.classes_[pred[i]]} '
                     f'({100 * conf[i]:.1f}%)', fontsize=12)
        ax.axis('off')
    fig.suptitle('Most confident validation errors (final model)', fontsize=13)
    plt.tight_layout()
    save(fig, 'qualitative_errors.jpg')


def feature_family(name):
    """Coarse family of a classical-pipeline feature, from its name."""
    if name.startswith('pca_'):
        return 'PCA'
    for key, family in (('_lbp_', 'LBP texture'), ('_glcm_', 'GLCM texture'), ('_hog_', 'HOG')):
        if key in name:
            return family
    if any(k in name for k in ('_hu_', '_fourier_', '_area', '_axis', '_perimeter', '_circularity',
                               '_eccentricity', '_solidity', '_convexity', '_extent', '_lobes',
                               '_diameter', 'nc_ratio')):
        return 'Shape'
    return 'Color'


def fig_feature_importance():
    df = pd.read_csv(ROOT / "results" / "classical_ml" / "feature_importance_lightgbm.csv")
    df = df.sort_values('importance', ascending=False).head(30)[::-1]
    palette = {'Color': '#4C78A8', 'LBP texture': '#F58518', 'GLCM texture': '#E45756',
               'HOG': '#54A24B', 'Shape': '#B279A2', 'PCA': '#9D9D9D'}
    families = df['feature'].map(feature_family)
    fig, ax = plt.subplots(figsize=(5.2, 6.0))
    ax.barh(df['feature'], df['importance'], color=families.map(palette))
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlabel('Importance (split count)')
    ax.set_title('Top-30 features (LightGBM split counts)', fontsize=12)
    present = [f for f in palette if f in set(families)]
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=palette[f]) for f in present],
              labels=present, loc='lower right', fontsize=9)
    plt.tight_layout()
    save(fig, 'feature_importance_ml.pdf')


def fig_iteration_timeline():
    # Validation macro-F1 of each iteration, from notebooks/03_deep_learning_journey.ipynb.
    names = ['Exp 1\n(EfficientNet\nbaseline)', 'Exp 2\n(+CBAM\n+augment)',
             'Exp 3\n(+Fusion\n+features)', 'Exp 4\n(Paper\nsegment.)',
             'Exp 5\n(ConvNeXt\n+CutMix)', 'Exp 6\n(Leakage\nfix)',
             'Exp 7\n(ResNet50\n+SpatialAtt)', 'Exp 8\n(ResNeXt101\n+MultiTask)',
             'Exp 9\n(ConvNeXt+SE\nfinal model)']
    scores = [0.552, 0.633, 0.641, 0.591, 0.822, 0.699, 0.651, 0.665, 0.705]
    labels = [f'{s:.3f}' for s in scores]
    labels[4], labels[8] = '0.822*', '0.705\n(final)'
    colors = ['#3498db'] * 3 + ['#e74c3c', '#f39c12', '#95a5a6', '#3498db', '#3498db', '#27ae60']
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(names))
    ax.bar(x, scores, color=colors, edgecolor='white', width=0.65)
    for i, (s, lab) in enumerate(zip(scores, labels)):
        ax.text(i, s + 0.008, lab, ha='center', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(5, 0.74), xytext=(4.35, 0.80), arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(4.85, 0.80, 'Leakage fix', ha='left', fontsize=11, color='red', fontstyle='italic')
    ax.text(4, 0.88, '* inflated by data leakage', ha='center', fontsize=10, color='#f39c12',
            fontstyle='italic')
    ax.axhline(0.705, color='gray', linestyle='--', alpha=0.4)
    legend = {'#3498db': 'Other iterations', '#e74c3c': 'Failed experiment',
              '#f39c12': 'Leaky validation (invalid)', '#95a5a6': 'Leakage fix',
              '#27ae60': 'Final model'}
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in legend.items()],
              fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=5, frameon=False)
    ax.set_xticks(x, names, fontsize=10.5)
    ax.set_ylabel('Validation macro-F1')
    ax.set_ylim(0, 0.95)
    ax.set_title('Validation macro-F1 across the 9 deep-learning experiments')
    plt.tight_layout()
    save(fig, 'iteration_timeline.pdf')


if __name__ == '__main__':
    print(f"Writing figures to {FIG_DIR.relative_to(ROOT)}/")
    fig_class_distribution()
    fig_class_examples()
    fig_crop_examples()
    fig_augmentation_gallery()
    fig_feature_importance()
    fig_iteration_timeline()
    val_df, y, probs = final_model_validation()
    fig_per_class_scores(y, probs.argmax(1))
    fig_confusion_matrix(y, probs.argmax(1))
    fig_qualitative_errors(val_df, y, probs)
