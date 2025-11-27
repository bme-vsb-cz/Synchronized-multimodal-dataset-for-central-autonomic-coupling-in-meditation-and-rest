# ========================================================================
#  RFECV Feature Selection for GSR / HR / RR / Pulse (TSFRESH)
# ------------------------------------------------------------------------
#  Purpose
#    Select and rank TSFRESH features per participant and signal using
#    RFECV with a RandomForest classifier. Produce importance plots and a
#    CSV of the top selected features.
#
#  What it does
#    1) Loads merged CSVs from:
#         <BASE>/TSFRESH_merged/pX_<SIGNAL>_data.csv
#       where <BASE> is resolved via _last_out_root.txt or newest Processed_*.
#    2) Filters rows by index keywords (case-insensitive):
#         “Meditation” → class 1, otherwise “Normal” → class 0.
#    3) Cleans data: cast to numeric, drop all-NaN columns, fill NaNs with
#       column medians, drop zero-variance columns.
#    4) Sets StratifiedKFold with n_splits = min(5, size of minority class),
#       requires ≥2 samples per class.
#    5) Runs RFECV(RandomForest, class_weight='balanced_subsample',
#       n_estimators=300) on a variance-pruned top-K feature subset.
#    6) Saves bar plots (all features and Top-N) and a CSV of selected
#       features sorted by importance.
#
#  Inputs (per participant & signal)
#    TSFRESH_merged/pX_SIGNAL_data.csv   # index = measurement rows
#
#  Outputs (per participant & signal)
#    <BASE>/FeatureSelection/<SIGNAL>/<pX>/
#      • selected_features_pX_SIGNAL.csv
#      • pX_SIGNAL_feature_importance.png
#      • pX_SIGNAL_top_40_features.png
#
#  Key parameters
#    max_cv = 5                    # upper bound for CV folds
#    top_k  = min(20, n_features)  # pre-RFECV variance-based pruning
#    plots: Top-N = 40
#
#  Requirements
#    Python 3.9+, numpy, pandas, scikit-learn, matplotlib.
#    Headless plotting is enabled via: matplotlib.use("Agg")
#
#  Usage
#    python feature_selection_GSR_HR_RR_Pulse.py
#
#  Notes
#    • Skips files with <2 classes or <2 samples in any class.
#    • Folder discovery pattern: TSFRESH_merged/p*_*_data.csv
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
        cands = [d for d in os.listdir(project_root)
                 if d.startswith('Processed_') and os.path.isdir(os.path.join(project_root, d))]
        if cands:
            cands.sort(key=lambda d: os.path.getmtime(os.path.join(project_root, d)), reverse=True)
            base_dir = os.path.join(project_root, cands[0])
        else:
            raise RuntimeError(f'out_root not resolved: pointer missing and no Processed_* under {project_root}')
    return base_dir

# accepts p6, p007, pXX, pAB12 …
_FILENAME_RE = re.compile(r'^(p[0-9A-Za-z]+)_([A-Za-z]+)_data\.csv$')  # e.g., p3_GSR_data.csv

def iter_merged(input_dir, wanted_signals=None):
    for path in glob.glob(os.path.join(input_dir, 'p*_*_data.csv')):
        name = os.path.basename(path)
        m = _FILENAME_RE.match(name)
        if not m:
            continue
        pid, signal = m.group(1), m.group(2)
        if wanted_signals and signal not in wanted_signals:
            continue
        yield pid, signal, path

def calculate_rfecv(X, y, max_cv=5):
    # třídy a velikost menší třídy
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min())
    n_splits = max(2, min(max_cv, min_count))  # aspoň 2, jinak RFECV nedává smysl
    if min_count < 2:
        return None  # příliš málo dat v jedné třídě

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    rfecv = RFECV(
        estimator=rf,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1,
        step=1
    )
    rfecv.fit(X, y)

    # vybrané rysy v původním pořadí X.columns
    support_mask = rfecv.support_
    selected_cols = X.columns[support_mask]
    # importance patří jen k vybraným rysům
    imps = rfecv.estimator_.feature_importances_
    order = np.argsort(imps)[::-1]
    features_sorted = [selected_cols[i] for i in order]
    importances_sorted = [float(imps[i]) for i in order]
    return features_sorted, importances_sorted

def plot_rfecv_results(features, importances, tag, save_path):
    order = np.argsort(importances)[::-1]
    features = [features[i] for i in order]
    importances = [importances[i] for i in order]
    top_n = min(40, len(features))
    trunc = [f[:40] + '...' if len(f) > 40 else f for f in features]
    trunc_top = trunc[:top_n]; imps_top = importances[:top_n]

    os.makedirs(save_path, exist_ok=True)

    # all
    plt.figure(figsize=(max(12, len(trunc)*0.15), 8))
    plt.bar(range(len(trunc)), importances, align='center')
    plt.xticks(range(len(trunc)), trunc, rotation=60, ha="right", fontsize=8)
    plt.ylabel('Feature Importance'); plt.title(f'{tag} Feature Importance (All)')
    plt.tight_layout(); plt.savefig(os.path.join(save_path, f"{tag}_feature_importance.png")); plt.close()

    # top N
    plt.figure(figsize=(max(12, len(trunc_top)*0.2), 8))
    plt.bar(range(len(trunc_top)), imps_top, align='center')
    plt.xticks(range(len(trunc_top)), trunc_top, rotation=60, ha="right", fontsize=10)
    plt.ylabel('Feature Importance'); plt.title(f'Top {top_n} {tag} Features')
    plt.tight_layout(); plt.savefig(os.path.join(save_path, f"{tag}_top_{top_n}_features.png")); plt.close()

    return features[:top_n], imps_top

# ----------------------------- main -------------------------------------
if __name__ == '__main__':
    BASE_DIR = resolve_base_dir()
    INPUT_DIR = os.path.join(BASE_DIR, 'TSFRESH_merged')
    if not os.path.isdir(INPUT_DIR):
        raise RuntimeError(f"Expected folder not found: {INPUT_DIR}")

    WANTED = {'GSR', 'HR', 'RR', 'Pulse'}
    OUT_ROOT = os.path.join(BASE_DIR, 'FeatureSelection')
    os.makedirs(OUT_ROOT, exist_ok=True)

    try:
        for pid, signal, csv_path in iter_merged(INPUT_DIR, wanted_signals=WANTED):
            # robust CSV load; autodetect delimiter
            df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig', index_col=0)
            if df.empty:
                print(f"Skipping empty file: {csv_path}")
                continue

            # filter valid rows by keywords in the index
            idx_s = pd.Series(df.index.astype(str), index=df.index)
            # sjednocený vzor pro Normal/Meditation, case-insensitive
            valid_mask = idx_s.str.contains(r'(?:Meditation|Normal)', case=False, regex=True, na=False)
            df = df.loc[valid_mask].copy()
            if df.empty:
                print(f"No valid rows after keyword filter: {csv_path}")
                continue

            # labels
            med_mask = df.index.to_series().astype(str).str.contains(r'Meditation', case=False, regex=True, na=False).values
            y_str = np.where(med_mask, 'Meditation', 'Normal')
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y_str), index=df.index)

            # X: numerics + cleaning
            X = df.apply(pd.to_numeric, errors='coerce').select_dtypes(include=[np.number])
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            # pokud sloupec je celý NaN, median je NaN -> fill následně nedoplní; ošetříme předem
            for c in X.columns:
                if X[c].notna().sum() == 0:
                    X.drop(columns=[c], inplace=True)
            if X.empty:
                print(f"No numeric columns after cleaning: {csv_path}")
                continue
            X.fillna(X.median(numeric_only=True), inplace=True)
            var = X.var(numeric_only=True)
            X = X.loc[:, var[var > 0].index]
            if X.empty:
                print(f"No variable features after variance filter: {csv_path}")
                continue

            # třídy a min velikost kvůli CV
            class_counts = y.value_counts()
            if class_counts.size < 2:
                print(f"Not enough classes for RFECV: {csv_path} counts={class_counts.to_dict()}")
                continue
            if int(class_counts.min()) < 2:
                print(f"Skipping {csv_path}: too few samples in a class {class_counts.to_dict()}")
                continue

            # rychlé zúžení dimenze
            top_k = min(20, X.shape[1])
            top_features = X.var(numeric_only=True).sort_values(ascending=False).head(top_k).index
            X_reduced = X[top_features]

            # RFECV
            res = calculate_rfecv(X_reduced, y, max_cv=5)
            if res is None:
                print(f"Skipping {csv_path}: not enough data for CV after filtering {class_counts.to_dict()}")
                continue
            sorted_features, sorted_importances = res

            # outputs
            tag = f"{pid}_{signal}"
            save_dir = os.path.join(OUT_ROOT, signal, pid)
            os.makedirs(save_dir, exist_ok=True)

            filtered_features, filtered_importances = plot_rfecv_results(
                sorted_features, sorted_importances, tag, save_dir
            )

            results_df = pd.DataFrame({"Feature": filtered_features, "Importance": filtered_importances})
            out_csv = os.path.join(save_dir, f"selected_features_{tag}.csv")
            results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
            print(f"Saved: {out_csv}")

            print(f"\nTop 10 Feature Importances for {tag}:")
            for feat, imp in zip(filtered_features[:10], filtered_importances[:10]):
                print(f"{feat}: {imp:.4f}")

        print("\nDone.")
    except Exception as e:
        print(f"Error during processing: {e}")
