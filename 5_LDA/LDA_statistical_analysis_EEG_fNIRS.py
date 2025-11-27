# ========================================================================
#  LDA + Statistical Tests for EEG / fNIRS (FilteredBasedOnRFECV/)
# ------------------------------------------------------------------------
#  Purpose
#    Fit Linear Discriminant Analysis (LDA) on filtered EEG/fNIRS tables
#    from the FilteredBasedOnRFECV/ stage and test class separability on LD components.
#
#  What it does
#    1) Loads per-participant CSVs from:
#         <BASE>/FilteredBasedOnRFECV/<SIGNAL>/<pX>/<pX>_<SIGNAL>_filtered.csv
#       where <BASE> is resolved via _last_out_root.txt or newest Processed_*.
#    2) Infers labels from row-index keywords (case-insensitive):
#         {Meditation|Meditace} → “Meditation”
#         {Normal|Normál}      → “Normal”
#       If no keywords exist, alternates rows into Group_A / Group_B.
#    3) Cleans data (numeric cast, ±inf→NaN, drop-rows-with-NaN),
#       removes zero-variance columns, standardizes features, fits LDA.
#    4) For each LD component:
#         • Shapiro normality per class
#         • Welch t-test if both normal, else Mann–Whitney U
#    5) Saves statistics, LDA loadings, and a boxplot of LD1.
#
#  Inputs (per folder)
#    CSV: pX_EEG_filtered.csv or pX_fNIRS_filtered.csv
#
#  Outputs (per folder)
#    <BASE>/LDA/<SIGNAL>/<pX>/
#      • lda_results.csv                  # test used, p-values, significance
#      • lda_feature_contributions.csv    # LDA loadings (feature → LDk)
#      • lda_boxplot.png                  # LD1 distribution by group
#
#  Key parameters
#    alpha  = 0.05
#    WANTED = {'EEG','fNIRS'}
#
#  Requirements
#    Python 3.9+, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn.
#    For headless runs: set a non-interactive backend before importing pyplot:
#      import os; os.environ["MPLBACKEND"] = "Agg"
#
#  Usage
#    python LDA_statistical_analysis_EEG_fNIRS.py
#
#  Notes
#    • Needs ≥2 samples per class and ≥1 non-constant feature after cleaning.
#    • Folder discovery pattern: FilteredBasedOnRFECV/*/p*/p*_*_filtered.csv
#    • Number of LD components equals n_classes-1 (typically 1).
# ========================================================================

import os, re, glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# ---------- helpers ----------
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
        if not cands:
            raise RuntimeError(f'out_root not resolved under {project_root}')
        cands.sort(key=lambda d: os.path.getmtime(os.path.join(project_root, d)), reverse=True)
        base_dir = os.path.join(project_root, cands[0])
    return base_dir

FILTERED_RE = re.compile(r'^(p[0-9A-Za-z]+)_([A-Za-z]+)_filtered\.csv$')

def iter_filtered(filtered_root, wanted_signals=None):
    for path in glob.glob(os.path.join(filtered_root, '*', 'p*', 'p*_*_filtered.csv')):
        name = os.path.basename(path)
        m = FILTERED_RE.match(name)
        if not m:
            continue
        pid, signal = m.group(1), m.group(2)
        if wanted_signals and signal not in wanted_signals:
            continue
        yield pid, signal, path

# ---------- LDA + tests ----------
def lda_stat_test(data, group_col=None, groups=None, alpha=0.05, out_dir=None):
    # 1) split into groups
    if group_col is None or groups is None:
        idx = pd.Series(data.index.astype(str), index=data.index)
        has_kw  = idx.str.contains(r'(?:Meditation|Meditace|Normal|Norm[áa]l)', case=False, regex=True, na=False).any()
        if has_kw:
            med_mask = idx.str.contains(r'(?:Meditation|Meditace)', case=False, regex=True, na=False)
            group1 = data.loc[~med_mask].copy()  # Normal
            group2 = data.loc[med_mask].copy()   # Meditation
            groups = ("Normal", "Meditation")
        else:
            group1 = data.iloc[0::2, :].copy()
            group2 = data.iloc[1::2, :].copy()
            groups = ("Group_A", "Group_B")
    else:
        group1 = data[data[group_col] == groups[0]].drop(columns=[group_col])
        group2 = data[data[group_col] == groups[1]].drop(columns=[group_col])

    # 2) numerics + NaN
    group1 = group1.apply(pd.to_numeric, errors='coerce')
    group2 = group2.apply(pd.to_numeric, errors='coerce')
    group1.replace([np.inf, -np.inf], np.nan, inplace=True)
    group2.replace([np.inf, -np.inf], np.nan, inplace=True)
    group1.dropna(axis=0, how='any', inplace=True)
    group2.dropna(axis=0, how='any', inplace=True)

    if min(len(group1), len(group2)) < 2 or group1.shape[1] == 0 or group2.shape[1] == 0:
        raise ValueError("Not enough data for LDA analysis.")

    # 3) standardization
    scaler = StandardScaler()
    X1 = scaler.fit_transform(group1)
    X2 = scaler.transform(group2)

    # 4) LDA
    X = np.vstack([X1, X2])
    y = np.array([0] * len(group1) + [1] * len(group2))
    labels = np.array([groups[0]] * len(group1) + [groups[1]] * len(group2))

    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X, y)
    n_comp = X_lda.shape[1]  # pro 2 třídy typicky 1

    # 5) tests on LD components
    results = []
    for i in range(n_comp):
        d1 = X_lda[y == 0, i]
        d2 = X_lda[y == 1, i]
        if np.std(d1) == 0 or np.std(d2) == 0:
            p_value, test_used = 1.0, "Not applicable"
        else:
            p1 = stats.shapiro(d1).pvalue if len(d1) >= 3 else 0.0
            p2 = stats.shapiro(d2).pvalue if len(d2) >= 3 else 0.0
            if p1 > alpha and p2 > alpha:
                _, p_value = stats.ttest_ind(d1, d2, equal_var=False)
                test_used = "Welch t-test"
            else:
                _, p_value = stats.mannwhitneyu(d1, d2)
                test_used = "Mann–Whitney U test"
        results.append({"LDA Component": f"LD{i+1}", "Test": test_used, "p-value": p_value, "Significant": p_value < alpha})
    results_df = pd.DataFrame(results)

    # 6) saving
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        results_df.to_csv(os.path.join(out_dir, "lda_results.csv"), index=False, encoding='utf-8-sig')

        # loadings (use scalings_ if available, else coef_.T)
        load_mat = getattr(lda, "scalings_", None)
        if load_mat is None:
            load_mat = lda.coef_.T
        contrib = pd.DataFrame(load_mat, columns=[f"LD{i+1}" for i in range(load_mat.shape[1])], index=group1.columns)
        contrib.to_csv(os.path.join(out_dir, "lda_feature_contributions.csv"), encoding='utf-8-sig')

        # boxplot LD1
        lda_df = pd.DataFrame(X_lda, columns=[f"LD{i+1}" for i in range(n_comp)])
        lda_df["Group"] = labels

        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(x="Group", y="LD1", hue="Group", data=lda_df, palette="coolwarm")
        ax.set_ylabel("LD1 [-]")

        # remove legend
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        # grid
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(True, axis='y', which='major', linestyle='--', alpha=0.4)
        ax.grid(True, axis='y', which='minor', linestyle=':',  alpha=0.2)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "lda_boxplot.png"), dpi=300)
        plt.close()


    return results_df

# ---------- MAIN ----------
if __name__ == "__main__":
    BASE = resolve_base_dir()
    FILTERED_ROOT = os.path.join(BASE, 'FilteredBasedOnRFECV')
    if not os.path.isdir(FILTERED_ROOT):
        raise RuntimeError(f"Missing folder: {FILTERED_ROOT}")

    WANTED = {'EEG', 'fNIRS'}
    OUT_ROOT = os.path.join(BASE, 'LDA')
    os.makedirs(OUT_ROOT, exist_ok=True)

    for path in glob.glob(os.path.join(FILTERED_ROOT, '*', 'p*', 'p*_*_filtered.csv')):
        name = os.path.basename(path)
        m = FILTERED_RE.match(name)
        if not m:
            continue
        pid, signal = m.group(1), m.group(2)
        if signal not in WANTED:
            continue

        df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig', index_col=0)
        if df.empty:
            print(f"Skipping empty: {path}")
            continue

        out_dir = os.path.join(OUT_ROOT, signal, pid)
        try:
            _ = lda_stat_test(df, group_col=None, groups=None, alpha=0.05, out_dir=out_dir)
            print(f"Done: {pid} {signal}")
        except Exception as e:
            print(f"Error for {pid} {signal}: {e}")
