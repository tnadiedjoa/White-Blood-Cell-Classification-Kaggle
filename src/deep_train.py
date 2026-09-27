"""
Model, training loop and inference of the final deep-learning model (deep_9 no-morph):
ConvNeXt-Base + SE-style channel-attention gate + 3-layer MLP head, trained in two phases
(frozen backbone, then full fine-tuning) with Focal loss, Mixup/CutMix and mixed precision.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Base_Weights, convnext_base

from .deep_dataset import IMG_SIZE, TTA_ROUNDS, WBCDataset, tta_aug, val_aug



class WBCClassifier(nn.Module):
    """ConvNeXt-Base backbone, SE-style gate on the pooled features, MLP head.

    `n_morph_features > 0` adds a small branch for handcrafted morphology features that is
    concatenated before the head (the deep_9 with-morph ablation). The final model uses
    `n_morph_features=0`.
    """

    def __init__(self, num_classes=13, dropout=0.3, pretrained=True, n_morph_features=0):
        super().__init__()
        full_model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(full_model.features, full_model.avgpool)
        with torch.no_grad():
            self.feat_dim = self.backbone(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)).flatten(1).shape[1]

        gate_hidden = max(self.feat_dim // 8, 64)
        self.feature_gate = nn.Sequential(
            nn.Linear(self.feat_dim, gate_hidden), nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, self.feat_dim), nn.Sigmoid(),
        )

        self.use_morph = n_morph_features > 0
        morph_out = 64 if self.use_morph else 0
        self.morph_branch = nn.Sequential(
            nn.Linear(n_morph_features, 64), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(64, morph_out), nn.ReLU(inplace=True),
        ) if self.use_morph else None

        head_in = self.feat_dim + morph_out
        hidden1 = max(head_in // 2, num_classes)
        hidden2 = max(hidden1 // 2, num_classes)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden1), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(hidden2, num_classes),
        )

    def parameter_counts(self):
        count = lambda m: sum(p.numel() for p in m.parameters()) if m is not None else 0
        return {'backbone': count(self.backbone), 'attention_gate': count(self.feature_gate),
                'morph_branch': count(self.morph_branch), 'head': count(self.head),
                'total': count(self)}

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()

    def extract_features(self, x):
        """Gated backbone features (input of the head, also used by the RF hybrid)."""
        feat = self.backbone(x).flatten(1)
        return feat * self.feature_gate(feat)

    def forward(self, x, morph=None):
        feat = self.extract_features(x)
        if self.use_morph:
            feat = torch.cat([feat, self.morph_branch(morph)], dim=1)
        return self.head(feat)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = float(gamma)
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=float(label_smoothing),
                                      reduction="none")

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        return (((1.0 - torch.exp(-ce)) ** self.gamma) * ce).mean()


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[index], y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / float(H * W)
    return mixed_x, y, y[index], lam


def apply_batch_aug(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, cutmix_prob=0.5):
    """CutMix with probability `cutmix_prob`, Mixup otherwise."""
    if np.random.rand() < cutmix_prob:
        return cutmix_data(x, y, alpha=cutmix_alpha)
    return mixup_data(x, y, alpha=mixup_alpha)


def train_model_2phase(model, train_loader, val_loader, checkpoint_path, device,
                       num_epochs=60, warmup_epochs=5, patience=10,
                       head_lr=3e-4, finetune_bb_lr=1e-4, finetune_head_lr=1e-4,
                       weight_decay=1e-4, grad_clip=1.0, label_smoothing=0.05,
                       focal_gamma=2.0, mixup_alpha=0.2, cutmix_alpha=1.0, cutmix_prob=0.5,
                       rare_class_ids=(), accum_steps=1):
    """Phase 1 (`warmup_epochs`): frozen backbone, only gate + head are trained.
    Phase 2: full fine-tuning. Mixup/CutMix is skipped for batches containing a rare class.
    The best model (validation macro-F1) is saved to `checkpoint_path`; training stops
    after `patience` epochs without improvement. Returns (best model, history)."""
    device = torch.device(device)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
    rare_tensor = torch.tensor(sorted(rare_class_ids), device=device, dtype=torch.long)

    best_f1, epochs_no_improve = 0.0, 0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'lr': []}
    optimizer = scheduler = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        if epoch == 1:
            model.freeze_backbone()
            head_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and 'backbone' not in n]
            optimizer = optim.AdamW([{'params': head_params, 'lr': head_lr}],
                                    weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(warmup_epochs, 5))
        if epoch == warmup_epochs + 1:
            model.unfreeze_backbone()
            optimizer = optim.AdamW([
                {'params': [p for n, p in model.named_parameters() if 'backbone' in n],
                 'lr': finetune_bb_lr},
                {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
                 'lr': finetune_head_lr},
            ], weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max((num_epochs - warmup_epochs) // 3, 5))

        model.train()
        if epoch <= warmup_epochs:
            model.backbone.eval()
        running_loss, n_seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            contains_rare = len(rare_tensor) > 0 and torch.isin(labels, rare_tensor).any()

            with torch.amp.autocast(device.type, enabled=use_amp):
                if contains_rare:
                    loss = criterion(model(imgs), labels)
                else:
                    imgs_aug, y_a, y_b, lam = apply_batch_aug(
                        imgs, labels, mixup_alpha, cutmix_alpha, cutmix_prob)
                    outputs = model(imgs_aug)
                    loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)

            if not torch.isfinite(loss):
                print("  non-finite loss, batch skipped")
                continue
            scaler.scale(loss / accum_steps).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        train_loss = running_loss / max(n_seen, 1)

        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    outputs = model(imgs)
                    val_loss += criterion(outputs, labels).item() * imgs.size(0)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        for key, value in zip(history, (train_loss, val_loss, val_f1, current_lr)):
            history[key].append(value)
        phase = "warmup" if epoch <= warmup_epochs else "finetune"
        msg = (f"[{phase}] epoch {epoch:02d}/{num_epochs}  train_loss {train_loss:.4f}  "
               f"val_loss {val_loss:.4f}  val_f1 {val_f1:.4f}  lr {current_lr:.2e}  "
               f"({time.time() - t0:.0f}s)")

        if val_f1 > best_f1:
            best_f1, epochs_no_improve = val_f1, 0
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({'epoch': epoch, 'model_state_dict': best_model_wts, 'val_f1': best_f1},
                       str(checkpoint_path))
            print(msg + "  -> best, saved")
        else:
            epochs_no_improve += 1
            print(msg)
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model_wts)
    print(f"Best validation macro-F1: {best_f1:.4f}")
    return model, history


@torch.no_grad()
def predict(model, dataset, device, batch_size=64, num_workers=4):
    """Logits and gated features for every image of `dataset` (labels, if any, ignored).
    Supports models without the morphology branch only."""
    assert not model.use_morph, "predict() does not feed morphology features"
    device = torch.device(device)
    model.eval().to(device)
    logits, features = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True):
        imgs = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
            feat = model.extract_features(imgs)
            out = model.head(feat)
        logits.append(out.float().cpu())
        features.append(feat.float().cpu())
    return torch.cat(logits), torch.cat(features)


def predict_tta(model, df, img_dir, device, tta_rounds=TTA_ROUNDS, batch_size=64, seed=0):
    """Average of the logits of one plain pass and `tta_rounds` randomly flipped/rotated
    passes (seeded, so the result is reproducible). Returns logits (len(df), num_classes)."""
    df = df[['ID']]
    total, _ = predict(model, WBCDataset(df, img_dir, augmentation=val_aug), device, batch_size)
    for r in range(tta_rounds):
        tta_aug.set_random_seed(seed + r)
        logits, _ = predict(model, WBCDataset(df, img_dir, augmentation=tta_aug), device,
                            batch_size)
        total += logits
        print(f"  TTA round {r + 1}/{tta_rounds} done")
    return total / (1 + tta_rounds)
