"""
Re-run of deep_1, the first deep-learning baseline: EfficientNet-B3 fine-tuned end to end
with class-weighted cross-entropy and a class-balanced sampler.

The original experiment used 5-fold cross-validation and was saved without outputs. This
script keeps its model, loss, sampler and augmentations but trains once, on a stratified
85/15 split and for at most 15 epochs, so that its score can be checked.

Run from the repository root:  python scripts/deep_1_rerun.py
Writes metrics, history and figures to results/deep_1_rerun/.
"""
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "deep_1_rerun"
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
PATIENCE = 5

IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(**IMAGENET),
])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(**IMAGENET),
])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class WBCDataset(Dataset):
    def __init__(self, df, img_dir, label_encoder, transform):
        self.ids = df['ID'].tolist()
        self.labels = label_encoder.transform(df['label'])
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir / self.ids[idx]).convert('RGB')
        return self.transform(img), int(self.labels[idx])


def create_model(num_classes):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(in_features, num_classes))
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for imgs, y in loader:
        imgs, y = imgs.to(device), y.to(device)
        out = model(imgs)
        total_loss += criterion(out, y).item() * imgs.size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return total_loss / len(loader.dataset), np.array(preds), np.array(labels)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = pd.read_csv(DATA / "train_metadata.csv")
    le = LabelEncoder().fit(train_df['label'])
    num_classes = len(le.classes_)
    y_all = le.transform(train_df['label'])
    train_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=0.15,
                                          random_state=SEED, stratify=y_all)
    train_ds = WBCDataset(train_df.iloc[train_idx], DATA / "train", le, train_transforms)
    val_ds = WBCDataset(train_df.iloc[val_idx], DATA / "train", le, val_transforms)
    log(f"device={device}, train={len(train_ds)}, val={len(val_ds)}")

    counts = Counter(y_all)
    class_weights = torch.tensor([len(y_all) / (num_classes * counts[i]) for i in range(num_classes)],
                                 dtype=torch.float32, device=device)
    fold_counts = Counter(train_ds.labels)
    sampler = WeightedRandomSampler([1.0 / fold_counts[y] for y in train_ds.labels],
                                    num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                            pin_memory=True)

    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_f1, best_wts, no_improve = 0.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        for imgs, y in train_loader:
            imgs, y = imgs.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        val_loss, preds, labels = evaluate(model, val_loader, criterion, device)
        val_f1 = f1_score(labels, preds, average='macro')
        scheduler.step(val_f1)
        for k, v in zip(history, (running / len(train_ds), val_loss, val_f1)):
            history[k].append(v)
        improved = val_f1 > best_f1
        if improved:
            best_f1, best_wts, no_improve = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        log(f"epoch {epoch:02d}/{NUM_EPOCHS}  train_loss {history['train_loss'][-1]:.4f}  "
            f"val_loss {val_loss:.4f}  val_f1 {val_f1:.4f}  ({time.time() - t0:.0f}s)"
            + ("  -> best" if improved else ""))
        if no_improve >= PATIENCE:
            log(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_wts)
    _, preds, labels = evaluate(model, val_loader, criterion, device)
    report = classification_report(labels, preds, target_names=le.classes_, output_dict=True,
                                   zero_division=0)
    log("\n" + classification_report(labels, preds, target_names=le.classes_, zero_division=0))
    cm = confusion_matrix(labels, preds)
    val_f1 = f1_score(labels, preds, average='macro')

    np.save(OUT / "confusion_matrix.npy", cm)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7)
    ax.set_xticks(range(num_classes), le.classes_, rotation=45, ha='right')
    ax.set_yticks(range(num_classes), le.classes_)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'deep_1 re-run, validation confusion matrix (macro-F1 = {val_f1:.4f})')
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(history['val_f1']) + 1), history['val_f1'], marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation macro-F1')
    ax.set_title('deep_1 re-run, training curve')
    fig.tight_layout()
    fig.savefig(OUT / "training_curve.png", dpi=120)
    plt.close(fig)

    summary = {
        'protocol': 'EfficientNet-B3, single stratified 85/15 split, at most 15 epochs '
                    '(original experiment: 5-fold CV)',
        'val_n': int(len(labels)),
        'val_macro_f1': round(float(val_f1), 4),
        'val_accuracy': round(float(report['accuracy']), 4),
        'epochs_run': len(history['val_f1']),
        'per_class_f1': {c: round(report[c]['f1-score'], 4) for c in le.classes_},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "classification_report.json").write_text(json.dumps(report, indent=2))
    (OUT / "history.json").write_text(json.dumps(history, indent=2))
    log(f"val macro-F1 {val_f1:.4f}, accuracy {report['accuracy']:.4f}")


if __name__ == '__main__':
    main()
