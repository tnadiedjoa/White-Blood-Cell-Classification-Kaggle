"""
Classical machine-learning pipeline: parallel handcrafted-feature extraction, feature
selection, a 6-classifier comparison under 5-fold cross-validation, and analysis of the
best model.

Known limitation: the variance/correlation filters, the scaler and the PCA are fitted on the
full training set before cross-validation, so the CV scores are slightly optimistic.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.base import clone

from imblearn.over_sampling import SMOTE

import lightgbm as lgb

from .features import extract_all_features

DEFAULT_FEATURES_DIR = Path(__file__).resolve().parent.parent / "results" / "classical_ml"


# ============================================================
# PARALLEL FEATURE EXTRACTION ON FULL DATASET
# ============================================================
def extract_features_for_row(img_id, img_dir, ref_mean, ref_std):
    """Wrapper for parallel extraction."""
    feats = extract_all_features(img_dir / img_id, ref_mean, ref_std)
    if feats is not None:
        feats['ID'] = img_id
    return feats


def extract_features_parallel(train_df, test_df, train_dir, test_dir, ref_mean, ref_std,
                               features_dir=DEFAULT_FEATURES_DIR, n_jobs=-1):
    """Extract features for the full train/test sets in parallel (joblib), cached as CSV
    in `features_dir`. Returns (train_feat_df, test_feat_df)."""
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    train_features_path = features_dir / "train_features.csv"
    test_features_path = features_dir / "test_features.csv"

    # Reuse the cached features if they include the HOG and Fourier columns
    needs_reextract = True
    if train_features_path.exists():
        cached_cols = pd.read_csv(train_features_path, nrows=0).columns
        if 'nuc_hog_0' in cached_cols and 'nuc_fourier_0' in cached_cols:
            needs_reextract = False
            print("Cached features already include HOG & Fourier. Skipping extraction.")

    if needs_reextract:
        # --- TRAIN ---
        print("Extracting TRAIN features (with HOG + Fourier)...")
        train_features = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
            delayed(extract_features_for_row)(row['ID'], train_dir, ref_mean, ref_std)
            for _, row in train_df.iterrows()
        )
        train_features = [f for f in train_features if f is not None]
        train_feat_df = pd.DataFrame(train_features)
        train_feat_df = train_feat_df.merge(train_df[['ID', 'label']], on='ID')
        print(f"Train features: {train_feat_df.shape}")

        # --- TEST ---
        print("\nExtracting TEST features (with HOG + Fourier)...")
        test_features = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
            delayed(extract_features_for_row)(row['ID'], test_dir, ref_mean, ref_std)
            for _, row in test_df.iterrows()
        )
        test_features = [f for f in test_features if f is not None]
        test_feat_df = pd.DataFrame(test_features)
        print(f"Test features: {test_feat_df.shape}")

        # Save to avoid recomputation
        train_feat_df.to_csv(train_features_path, index=False)
        test_feat_df.to_csv(test_features_path, index=False)
        print("Features saved!")
    else:
        train_feat_df = pd.read_csv(train_features_path)
        test_feat_df = pd.read_csv(test_features_path)

    return train_feat_df, test_feat_df


# ============================================================
# DATA PREPARATION & FEATURE SELECTION
#    - Remove near-zero variance
#    - Remove highly correlated features (>0.95)
#    - Add PCA components as extra features
# ============================================================
def prepare_features(train_feat_df, test_feat_df, plot=True):
    """Drop near-constant and highly correlated (>0.95) features, standardize, and append
    20 PCA components. Returns a dict with: X_sel, X_test_sel, y, le, combined_feature_names, scaler, pca.
    """
    # Feature columns (exclude ID and label)
    feature_cols = [c for c in train_feat_df.columns if c not in ['ID', 'label']]
    print(f"Initial number of features: {len(feature_cols)}")

    X = train_feat_df[feature_cols].values
    y_labels = train_feat_df['label'].values
    X_test = test_feat_df[feature_cols].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    print(f"Classes: {le.classes_}")

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Step 1: Remove near-zero variance features ---
    variances = np.var(X, axis=0)
    var_mask = variances > 1e-6
    X = X[:, var_mask]
    X_test = X_test[:, var_mask]
    feature_cols = [c for c, keep in zip(feature_cols, var_mask) if keep]
    print(f"After removing near-zero variance: {len(feature_cols)} features")

    # --- Step 2: Remove highly correlated features (>0.95) ---
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    upper_tri = np.triu(np.abs(corr_matrix), k=1)
    to_drop = set()
    for i in range(upper_tri.shape[0]):
        for j in range(i + 1, upper_tri.shape[1]):
            if upper_tri[i, j] > 0.95:
                to_drop.add(j)

    keep_mask = [i not in to_drop for i in range(X.shape[1])]
    X = X[:, keep_mask]
    X_test = X_test[:, keep_mask]
    feature_cols = [c for c, keep in zip(feature_cols, keep_mask) if keep]
    print(f"After removing correlated features (>0.95): {len(feature_cols)} features")

    # Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    # --- Step 3: Add PCA features (complementary to raw features) ---
    n_pca = min(20, X_scaled.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"PCA: {n_pca} components, explained variance = {pca.explained_variance_ratio_.sum():.3f}")

    # Combine raw + PCA features
    X_sel = np.hstack([X_scaled, X_pca])
    X_test_sel = np.hstack([X_test_scaled, X_test_pca])
    combined_feature_names = feature_cols + [f'pca_{i}' for i in range(n_pca)]
    print(f"Final feature set: {X_sel.shape[1]} features (raw + PCA)")

    # Feature importance via quick Random Forest
    print("\nComputing feature importances...")
    rf_quick = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_quick.fit(X_sel, y)
    importances = rf_quick.feature_importances_

    if plot:
        import matplotlib.pyplot as plt
        # Top 30 features
        top_idx = np.argsort(importances)[::-1][:30]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(30), importances[top_idx][::-1], color='steelblue')
        ax.set_yticks(range(30))
        ax.set_yticklabels([combined_feature_names[i] for i in top_idx][::-1], fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_title("Top 30 Features (Random Forest Importance)")
        plt.tight_layout()
        plt.show()

    return {
        'X_sel': X_sel,
        'X_test_sel': X_test_sel,
        'y': y,
        'le': le,
        'combined_feature_names': combined_feature_names,
        'scaler': scaler,
        'pca': pca,
        'importances': importances,
    }


# ============================================================
# ENSEMBLE (Stacking & Voting) — fixed hyperparameters
# ============================================================

# ---- Fixed LightGBM hyperparameters (from previous Optuna run) ----
BEST_LGBM_PARAMS = {
    'n_estimators': 1755,
    'learning_rate': 0.03428789905946424,
    'num_leaves': 54,
    'max_depth': 6,
    'min_child_samples': 17,
    'reg_alpha': 6.238522642098043,
    'reg_lambda': 1.0985817358624452,
    'subsample': 0.822106482724937,
    'colsample_bytree': 0.5114070073281863,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': 1,
    'verbose': -1,
}


# SMOTE up to the majority class (13,015 SNE images) would inflate each fold to ~135k rows and
# make the SVM-based models impractically slow; minority classes are oversampled up to
# SMOTE_CAP samples instead.
SMOTE_CAP = 2000


def capped_sampling_strategy(y_train, cap=SMOTE_CAP):
    counts = pd.Series(y_train).value_counts()
    return {cls: max(int(cnt), cap) for cls, cnt in counts.items() if cnt < cap}


def smote_resample(X_train, y_train, cap=SMOTE_CAP):
    """SMOTE (k=3, k=1 fallback for tiny classes) raising every class below `cap` to `cap`."""
    strategy = capped_sampling_strategy(y_train, cap=cap)
    if not strategy:
        return X_train, y_train
    try:
        return SMOTE(sampling_strategy=strategy, random_state=42, k_neighbors=3).fit_resample(X_train, y_train)
    except ValueError:
        return SMOTE(sampling_strategy=strategy, random_state=42, k_neighbors=1).fit_resample(X_train, y_train)


def build_classifiers(best_lgbm_params=BEST_LGBM_PARAMS):
    """LightGBM, Random Forest, Extra Trees, RBF-SVM, soft Voting and Stacking."""
    models = {
        'LightGBM_tuned': lgb.LGBMClassifier(**best_lgbm_params),
        'RF': RandomForestClassifier(n_estimators=800, class_weight='balanced', random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=800, class_weight='balanced', random_state=42, n_jobs=-1),
        'SVM_RBF': SVC(kernel='rbf', class_weight='balanced', random_state=42, C=10, gamma='scale', probability=True),
        'Voting': VotingClassifier(
            estimators=[
                ('lgbm', lgb.LGBMClassifier(**best_lgbm_params)),
                ('rf', RandomForestClassifier(n_estimators=800, class_weight='balanced', random_state=42, n_jobs=-1)),
                ('et', ExtraTreesClassifier(n_estimators=800, class_weight='balanced', random_state=42, n_jobs=-1)),
                ('svm', SVC(kernel='rbf', class_weight='balanced', random_state=42, C=10, gamma='scale', probability=True)),
            ],
            voting='soft', n_jobs=1,
        ),
        'Stacking': StackingClassifier(
            estimators=[
                ('lgbm', lgb.LGBMClassifier(**best_lgbm_params)),
                ('rf', RandomForestClassifier(n_estimators=800, class_weight='balanced', random_state=42, n_jobs=-1)),
                ('et', ExtraTreesClassifier(n_estimators=800, class_weight='balanced', random_state=42, n_jobs=-1)),
                ('svm', SVC(kernel='rbf', class_weight='balanced', random_state=42, C=10, gamma='scale', probability=True)),
            ],
            final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42, C=1.0, n_jobs=1),
            cv=5, n_jobs=1, passthrough=False,
        ),
    }
    return models


def compute_lightgbm_importance(X_sel, y, combined_feature_names, top_n=30,
                                 best_lgbm_params=BEST_LGBM_PARAMS, plot=True):
    """LightGBM split-count importance, fitted on the full training set with the tuned
    hyperparameters."""
    model = lgb.LGBMClassifier(**best_lgbm_params)
    model.fit(X_sel, y)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_n]

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(top_n), importances[top_idx][::-1], color='steelblue')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([combined_feature_names[i] for i in top_idx][::-1], fontsize=9)
        ax.set_xlabel("LightGBM importance (split count)")
        ax.set_title(f"Top {top_n} Features — LightGBM Feature Importance")
        plt.tight_layout()
        plt.show()

    return model, importances, top_idx


def run_cv_evaluation(models, X_sel, y, n_splits=5, random_state=42):
    """Stratified k-fold CV of every model, with SMOTE applied to the training folds only.
    Returns (results, skf) where `results` maps model name -> {mean, std, scores}.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        f1_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_sel, y)):
            X_tr, X_va = X_sel[train_idx], X_sel[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            X_tr_res, y_tr_res = smote_resample(X_tr, y_tr)

            m = clone(model)
            m.fit(X_tr_res, y_tr_res)
            y_pred = m.predict(X_va)
            f1 = f1_score(y_va, y_pred, average='macro')
            f1_scores.append(f1)
            print(f"  Fold {fold+1}: macro-F1 = {f1:.4f}")

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        results[name] = {'mean': mean_f1, 'std': std_f1, 'scores': f1_scores}
        print(f"  → Mean: {mean_f1:.4f} +/- {std_f1:.4f}")

    # Summary
    print("\n" + "="*60)
    print("MODEL SUMMARY (macro-F1)")
    print("="*60)
    for name, res in sorted(results.items(), key=lambda x: -x[1]['mean']):
        print(f"  {name:20s}: {res['mean']:.4f} +/- {res['std']:.4f}")

    return results, skf


# ============================================================
# BEST MODEL ANALYSIS — CONFUSION MATRIX
# ============================================================
def analyze_best_model(models, results, skf, X_sel, y, le, plot=True):
    """Out-of-fold predictions, classification report and confusion matrices of the best
    model. Returns (best_name, oof_preds)."""
    # Select the best model
    best_name = max(results, key=lambda k: results[k]['mean'])
    print(f"Best model: {best_name} (macro-F1 = {results[best_name]['mean']:.4f})")

    # Re-train with cross-validation to get OOF predictions
    oof_preds = np.zeros(len(y), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_sel, y)):
        X_tr, X_va = X_sel[train_idx], X_sel[val_idx]
        X_tr_res, y_tr_res = smote_resample(X_tr, y[train_idx])

        m = clone(models[best_name])
        m.fit(X_tr_res, y_tr_res)
        oof_preds[val_idx] = m.predict(X_va)

    # Classification report
    print("\n" + "="*60)
    print("Classification Report (OOF)")
    print("="*60)
    print(classification_report(y, oof_preds, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y, oof_preds)

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix — {best_name} (OOF, macro-F1={results[best_name]["mean"]:.4f})')
        plt.tight_layout()
        plt.show()

        # Normalized confusion matrix
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Normalized Confusion Matrix — {best_name}')
        plt.tight_layout()
        plt.show()

    # Per-class F1
    print("\nPer-class F1 scores:")
    f1_per_class = f1_score(y, oof_preds, average=None)
    for cls_name, f1_val in zip(le.classes_, f1_per_class):
        count = np.sum(y == le.transform([cls_name])[0])
        print(f"  {cls_name:4s} (n={count:5d}): F1 = {f1_val:.4f}")

    return best_name, oof_preds
