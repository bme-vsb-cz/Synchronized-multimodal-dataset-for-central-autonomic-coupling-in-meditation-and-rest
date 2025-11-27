# ========================================================================
#  RFECV Feature Selection for EEG / fNIRS (TSFRESH)
# ------------------------------------------------------------------------
#  Purpose
#    Select and rank TSFRESH features for EEG and fNIRS per participant
#    using RFECV with a RandomForest classifier. Save importance plots and
#    a CSV with the top selected features.
#
#  Data layout
#    Base folder is resolved via _last_out_root.txt or newest Processed_*.
#    Input CSVs:
#      <BASE>/TSFRESH_merged/pX_<MODALITY>_data.csv
#        where MODALITY ∈ {EEG, fNIRS}, index = measurement rows.
#
#  Pipeline
#    1) Keep rows whose index contains keywords (case-insensitive):
#         Meditation|Meditace → class “Meditation”
#         Normal|Normál       → class “Normal”
#    2) Build labels y from the index.
#    3) X cleanup: cast to numeric, replace ±inf→NaN, median impute NaNs,
#       drop zero-variance columns.
#    4) Dimensionality cut: keep top 20 features by variance.
#    5) Stratified CV: n_splits = min(5, size of minority class, N−1).
#    6) RFECV(RandomForest, n_estimators=500) → selected features +
#       importances mapped to selected columns and sorted desc.
#
#  Outputs (per participant & modality)
#    <BASE>/FeatureSelection/<MODALITY>/<pX>/
#      • selected_features_pX_<MODALITY>.csv
#      • pX_<MODALITY>_feature_importance.png
#      • pX_<MODALITY>_top_40_features.png
#
#  Requirements
#    Python 3.9+, numpy, pandas, scikit-learn, matplotlib.
#    Headless plotting is enabled via: matplotlib.use("Agg").
#
#  Usage
#    python feature_selection_EEG_fNIRS.py
#
#  Notes
#    • Files with <2 classes or <2 samples in any class are skipped.
#    • Input discovery pattern: TSFRESH_merged/p*_*_data.csv
# ========================================================================

import os, re, glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
# ----------------------------- helpers ----------------------------------
def resolve_base_dir():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    ptr_main = os.path.join(project_root, '_last_out_root.txt')

    base_dir = ''
    if os.path.isfile(ptr_main):
        with open(ptr_main, 'r', encoding='utf-8') as f:
            base_dir = f.readline().strip()

    if not base_dir or not os.path.isdir(base_dir):
        candidates = [d for d in os.listdir(project_root)
                      if d.startswith('Processed_')
                      and os.path.isdir(os.path.join(project_root, d))]
        if candidates:
            candidates.sort(key=lambda d: os.path.getmtime(os.path.join(project_root, d)), reverse=True)
            base_dir = os.path.join(project_root, candidates[0])
        else:
            raise RuntimeError(f'out_root not resolved: pointer missing and no Processed_* under {project_root}')
    return base_dir

# accepts p6, p007, pXX, pAB12 …
_FILENAME_RE = re.compile(r'^(p[0-9A-Za-z]+)_([A-Za-z]+)_data\.csv$')

def iter_merged(input_dir, wanted_modalities=None):
    for path in sorted(glob.glob(os.path.join(input_dir, 'p*_*_data.csv'))):
        name = os.path.basename(path)
        m = _FILENAME_RE.match(name)
        if not m:
            continue
        pid, modality = m.group(1), m.group(2)   # e.g., p6, EEG
        if wanted_modalities and modality not in wanted_modalities:
            continue
        yield pid, modality, path

def make_cv(y, max_cv=5):
    """Stratified CV s počtem foldů omezeným dle nejmenší třídy."""
    y = pd.Series(y)
    min_class = int(y.value_counts().min())
    if min_class < 2:
        raise ValueError("Need at least 2 samples per class for CV.")
    n_splits = min(max_cv, min_class, len(y) - 1)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def calculate_rfecv(X, y, cv):
    """RFECV + správné mapování importancí na vybrané feature názvy."""
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rfecv = RFECV(estimator=rf, cv=cv)
    rfecv.fit(X, y)

    support = rfecv.support_                      # maska vybraných sloupců
    selected_cols = X.columns[support]
    importances = rfecv.estimator_.feature_importances_  # jen pro vybrané

    order = np.argsort(importances)[::-1]
    return selected_cols[order].tolist(), importances[order].tolist()

def plot_rfecv_results(features, importances, tag, save_path):
    if len(features) == 0 or len(importances) == 0:
        raise ValueError("No features available for visualization.")

    # již seřazeno
    top_n = min(40, len(features))
    filtered_features = features[:top_n]
    filtered_importances = importances[:top_n]

    max_label_length = 40
    truncated_features = [
        lbl[:max_label_length] + '...' if len(lbl) > max_label_length else lbl
        for lbl in features
    ]
    truncated_filtered_features = truncated_features[:top_n]

    # All features
    figure_width = max(12, min(len(truncated_features) * 0.15, 20))
    plt.figure(figsize=(figure_width, 8))
    plt.bar(range(len(truncated_features)), importances, align='center')
    plt.xticks(range(len(truncated_features)), truncated_features, rotation=60, ha="right", fontsize=8)
    plt.ylabel('Feature Importance')
    plt.title(f'{tag} Feature Importance (All Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{tag}_feature_importance.png"))
    plt.close()

    # Top N
    plt.figure(figsize=(max(12, len(truncated_filtered_features) * 0.2), 8))
    plt.bar(range(len(truncated_filtered_features)), filtered_importances, align='center')
    plt.xticks(range(len(truncated_filtered_features)), truncated_filtered_features, rotation=60, ha="right", fontsize=10)
    plt.ylabel('Feature Importance')
    plt.title(f'Top {top_n} {tag} Features')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{tag}_top_{top_n}_features.png"))
    plt.close()

    return filtered_features, filtered_importances

# ----------------------------- main -------------------------------------
if __name__ == '__main__':
    BASE_DIR = resolve_base_dir()
    INPUT_DIR = os.path.join(BASE_DIR, 'TSFRESH_merged')
    if not os.path.isdir(INPUT_DIR):
        raise RuntimeError(f"Expected folder not found: {INPUT_DIR}")

    WANTED = {'EEG', 'fNIRS'}
    OUT_ROOT = os.path.join(BASE_DIR, 'FeatureSelection')
    os.makedirs(OUT_ROOT, exist_ok=True)

    try:
        for pid, modality, csv_path in iter_merged(INPUT_DIR, wanted_modalities=WANTED):
            # robust CSV load; first column is index
            df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig', index_col=0)
            if df.empty:
                print(f"Skipping empty file: {csv_path}")
                continue

            # --- filter valid rows by keywords (EN + CZ, nezachytávací skupiny) ---
            idx_s = pd.Series(df.index.astype(str), index=df.index)
            valid_mask = idx_s.str.contains(r'(?:Meditation|Meditace|Normal|Norm[áa]l)',
                                            case=False, regex=True, na=False)
            df = df.loc[valid_mask].copy()
            if df.empty:
                print(f"No valid rows after keyword filter: {csv_path}")
                continue

            # labels
            med_mask = pd.Series(df.index.astype(str)).str.contains(
                r'(?:Meditation|Meditace)', case=False, regex=True, na=False
            ).values
            y_str = np.where(med_mask, 'Meditation', 'Normal')
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y_str), index=df.index)

            # --- prepare X ----------------------------------------------------------
            X = df.apply(pd.to_numeric, errors='coerce').select_dtypes(include=[np.number])
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(X.median(numeric_only=True), inplace=True)

            # drop constant columns
            var = X.var(numeric_only=True)
            X = X.loc[:, var[var > 0].index]

            if X.empty or y.nunique() < 2:
                print(f"Not enough usable data/classes for RFECV: {csv_path}")
                continue

            # quick dimensionality reduction
            top_features = X.var(numeric_only=True).sort_values(ascending=False).head(20).index
            X_reduced = X[top_features]

            # CV dle velikosti tříd
            try:
                cv = make_cv(y, max_cv=5)
            except ValueError as e:
                print(f"{csv_path}: {e}")
                continue

            # RFECV
            sorted_features, sorted_importances = calculate_rfecv(X_reduced, y, cv=cv)

            # outputs
            tag = f"{pid}_{modality}"
            save_dir = os.path.join(OUT_ROOT, modality, pid)
            os.makedirs(save_dir, exist_ok=True)

            filtered_features, filtered_importances = plot_rfecv_results(
                sorted_features, sorted_importances, tag, save_dir
            )

            results_df = pd.DataFrame({
                "Feature": filtered_features,
                "Importance": filtered_importances
            })
            results_csv_path = os.path.join(save_dir, f"selected_features_{tag}.csv")
            results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
            print(f"Saved: {results_csv_path}")

            print(f"\nTop 10 Feature Importances for {tag}:")
            for feat, imp in zip(filtered_features[:10], filtered_importances[:10]):
                print(f"{feat}: {imp:.4f}")

        print("\nDone.")
    except Exception as e:
        print(f"Error during processing: {e}")
