# White Blood Cell Classification

13-class classification of white blood cells in peripheral blood smear images, developed for the
IMA205 Kaggle challenge at Télécom Paris: 28,901 labelled training images, 9,634 test images,
and an extreme class imbalance (1183:1 between the largest and the smallest class). The
challenge metric is the macro-F1.

Two approaches are compared: a **classical pipeline** (segmentation, 210 handcrafted features,
gradient boosting and ensembles) and a **deep-learning pipeline** refined over nine iterations,
from an EfficientNet-B3 baseline to a ConvNeXt-Base with a channel-attention gate. The full
write-up is in [`report/report.pdf`](report/report.pdf).

The challenge is modelled on the public [WBCBench2026](https://www.kaggle.com/competitions/wbc-bench-2026/overview) benchmark.

## Results

| Approach | Validation protocol | Macro-F1 | Accuracy |
|---|---|---|---|
| Classical: LightGBM on handcrafted features | 5-fold stratified CV | 0.509 | 0.832 |
| Deep learning baseline: EfficientNet-B3 (`deep_1`) | single 85/15 split<sup>†</sup> | 0.552 | 0.716 |
| ConvNeXt-Base + MLP head, leakage fixed (`deep_6`) | 90/10 split on original images | 0.699 | 0.936 |
| **ConvNeXt-Base + SE gate + CLAHE (`deep_9 no-morph`, final model)** | 90/10 split on original images | **0.705** | **0.937** |

<sup>†</sup> The original 5-fold run of `deep_1` was saved without outputs; this is a shorter
re-run (15 epochs) of the same model, reproducible with `scripts/deep_1_rerun.py`. Its score is
noisy: an earlier unseeded run of the same script gave 0.489.

All scores are computed on held-out parts of the labelled training set, not on the Kaggle
leaderboard, and the protocols differ between rows (cross-validation vs. a single split). The
difference between `deep_6` and the final model (+0.006) is within the noise of the validation
split. Every score in this table is backed by a file in [`results/`](results/) or by the
outputs of a notebook.

## Final model

- **Preprocessing**: each image is cropped once around the white cell with an HSV nucleus
  score, then CLAHE is applied to the HSV value channel.
- **Class imbalance**: offline oversampling of the classes below the median size,
  class-balanced sampling (weights ∝ count<sup>−0.5</sup>) and Focal loss.
- **Model**: ConvNeXt-Base pretrained on ImageNet, SE-style attention gate on the pooled
  features, 3-layer MLP head (88.5 M parameters).
- **Training**: 5 epochs with a frozen backbone, then full fine-tuning with AdamW and cosine
  warm restarts; Mixup/CutMix except for batches containing rare classes (which, with the
  balanced sampler, leaves them active on only ~1 % of batches); early stopping on
  the validation macro-F1 (best epoch 21 of 31, about 70 min on one RTX 3090).
- **Validation**: stratified split on the original images; augmented copies of validation
  images are discarded. An earlier iteration that split after oversampling put copies of
  validation images in the training set and reported 0.822 instead of 0.699.
- **Inference**: average of the logits of a plain pass and 8 random flip/rotation passes.

## Repository structure

```
src/
  segmentation.py          Reinhard normalization, watershed + active-contour segmentation
  features.py              Handcrafted features: shape, colour, texture, HOG, Fourier
  classical_pipeline.py    Feature selection, 6 classifiers, cross-validation
  deep_dataset.py          Oversampling, cell cropping, CLAHE, augmentations, leakage-free split
  deep_train.py            Model, two-phase training loop, inference with TTA
notebooks/
  01_eda.ipynb                     Dataset exploration
  02_classical_ml.ipynb            Classical pipeline and its results
  03_deep_learning_journey.ipynb   The nine deep-learning iterations, with their original outputs
  04_final_model.ipynb             Final model: evaluation and test predictions
scripts/deep_1_rerun.py    Re-run of the first deep-learning baseline (EfficientNet-B3)
results/                   Metrics, logs and predictions (results/submission.csv)
report/                    LaTeX report, its figures and generate_figures.py
models/                    Trained weights (not versioned, see models/README.md)
data/                      Challenge data (not versioned)
```

## Reproducing

```bash
pip install -r requirements.txt   # Python 3.12
```

Place the challenge files in `data/`:

```
data/
  train/  test/                          images
  train_metadata.csv  test_metadata.csv  image IDs and labels
  sample_submission.csv
```

Then run the notebooks in order from the `notebooks/` folder, with any Jupyter front end
(JupyterLab, Notebook or VS Code).

- `04_final_model.ipynb` creates the oversampled and cropped image folders in `data/` on its
  first run. It loads the trained weights from `models/best_model.pt` (354 MB, not versioned);
  without them, set `RETRAIN = True` to train the model again (about 70 min on one RTX 3090).
- `02_classical_ml.ipynb` takes 8 to 10 hours on 4 CPU cores, mostly for the Stacking ensemble.
  The files in `results/classical_ml/` were exported by an earlier scripted run of the same
  functions (log in `run.log`); the notebook reproduces the same scores but does not rewrite
  them.
- `03_deep_learning_journey.ipynb` is a record of the experiments: its cells are excerpts of the
  original per-iteration notebooks, shown with their saved outputs, and are not meant to be
  re-executed.

The report figures are regenerated with `python report/generate_figures.py` from the repository
root, and the PDF with `pdflatex report && bibtex report && pdflatex report && pdflatex report`
from `report/`.

## Limitations

- Only the weights of the final model are available: every training run from `deep_5` to
  `deep_9` wrote its checkpoint to the same file, so each one overwrote the previous run's.
  Their evaluation results were saved and are unaffected.
- In the classical pipeline, SMOTE raises each rare class to 2,000 images rather than to the
  size of the majority class, to keep the SVM-based models tractable. The feature filters,
  scaler and PCA are fitted before cross-validation, which makes its CV scores slightly
  optimistic.
- The rarest classes (PLY: 11 images, PC: 68, PMY: 114) have very few validation images, so
  their per-class scores are noisy.
- The deep-learning validation split is also used for early stopping and for choosing between
  iterations, so its scores are somewhat optimistic; they come from a single split and seed.
- The Reinhard normalization of the classical pipeline uses the spread of per-image means as
  reference standard deviation instead of the pixel-level one, which compresses contrast
  before segmentation.

## Main references

- Z. Liu et al., *A ConvNet for the 2020s*, CVPR 2022.
- J. Hu, L. Shen, G. Sun, *Squeeze-and-Excitation Networks*, CVPR 2018.
- T.-Y. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
- A. Gautam, H. Bhadauria, *Classification of White Blood Cells Based on Morphological
  Features*, ICACCI 2014.
- M. Toğaçar, B. Ergen, Z. Cömert, *Classification of White Blood Cells Using Deep Features
  Obtained from Convolutional Neural Network Models Based on the Combination of Feature
  Selection Methods*, Applied Soft Computing, 2020.
- F. Rustam et al., *White Blood Cell Classification Using Texture and RGB Features of
  Oversampled Microscopic Images*, Healthcare, 2022.

The complete bibliography is in [`report/references.bib`](report/references.bib).

## Author

Théophile Nadiedjoa ([@tnadiedjoa](https://github.com/tnadiedjoa)), Télécom Paris, 2026.

The code is released under the [MIT License](LICENSE). The challenge data and the
referenced papers are not part of this repository and are not covered by this license.
