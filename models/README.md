# models/

`best_model.pt` (354 MB, not versioned) holds the weights of the final model, `deep_9 no-morph`:
ConvNeXt-Base + SE-style attention gate + 3-layer MLP head
(`src.deep_train.WBCClassifier` with the default `n_morph_features=0`). It reaches a validation
macro-F1 of 0.7052 and an accuracy of 0.9367 (`results/deep_9_no_morph/summary.json`).

```python
import torch
from src.deep_train import WBCClassifier

ckpt = torch.load("models/best_model.pt", map_location="cpu")
model = WBCClassifier(pretrained=False)
model.load_state_dict(ckpt["model_state_dict"])
ckpt["epoch"], ckpt["val_f1"]   # (21, 0.7051742196484837)
```

This is the only checkpoint left from the experiments: the runs from `deep_5` to `deep_9`
all saved their best weights to this same file, so each overwrote the previous one.
`train_model_2phase` now requires an explicit `checkpoint_path`, and retraining from
`notebooks/04_final_model.ipynb` writes to `deep_9_no_morph_retrained.pt`.
