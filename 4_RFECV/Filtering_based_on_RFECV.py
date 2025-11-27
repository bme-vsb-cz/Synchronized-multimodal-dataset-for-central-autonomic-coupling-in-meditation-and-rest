# ========================================================================
#   FEATURE FILTERING – TSFRESH_merged × FeatureSelection
# ----------------------------------------------------------------------
#  Purpose:
#    Filter each merged TSFRESH dataset to the top features selected by
#    the RFECV stage. Produces per-participant, per-signal CSVs that keep
#    only the selected columns, preserving their order.
#
#  Inputs:
#    - Project pointer:  <project_root>/_last_out_root.txt
#    - Original data:    <base>/TSFRESH_merged/p<ID>_<Signal>_data.csv
#    - Selected features:<base>/FeatureSelection/**/selected_features_p<ID>_<Signal>.csv
#                         (expects a column named "Feature" or "feature")
#
#  Outputs:
#    - <base>/Filtered/<Signal>/<pID>/<pID>_<Signal>_filtered.csv
#
#  Logic:
#    1) Resolve <base> from _last_out_root.txt or the latest Processed_*.
#    2) Map all original p<ID>_<Signal>_data.csv files.
#    3) Iterate selected_features_*.csv, read feature names, de-duplicate.
#    4) Intersect with original columns and preserve the selected order.
#    5) Save filtered CSV; warn if any selected columns are missing.
#
#  Notes:
#    - Deterministic traversal (sorted glob and os.walk).
#    - UTF-8 with BOM (utf-8-sig) for Excel compatibility.
#    - Skips gracefully on missing files or empty selections.
#
#  Usage:
#    python feature_filtering.py
#
# ========================================================================
import os, re, glob
import pandas as pd

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

# selected_features_pX_Y.csv
SEL_RE = re.compile(r'^selected_features_(p[0-9A-Za-z]+)_([A-Za-z]+)\.csv$')
# pX_Y_data.csv
RAW_RE = re.compile(r'^(p[0-9A-Za-z]+)_([A-Za-z]+)_data\.csv$')

def build_original_map(tsfresh_merged_dir):
    """map[(pid, signal)] -> path to pX_Y_data.csv"""
    mapping = {}
    for p in glob.glob(os.path.join(tsfresh_merged_dir, 'p*_*_data.csv')):
        name = os.path.basename(p)
        m = RAW_RE.match(name)
        if m:
            mapping[(m.group(1), m.group(2))] = p
    return mapping

def iter_selected(featureselection_dir):
    """yield (pid, signal, path) for selected_features_pX_Y.csv in subdirs"""
    for root, _, files in os.walk(featureselection_dir):
        for f in files:
            if not f.endswith('.csv'): 
                continue
            m = SEL_RE.match(f)
            if m:
                yield m.group(1), m.group(2), os.path.join(root, f)

def read_selected_features(path):
    """returns list of feature names from 'Feature' column"""
    df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig')
    col = 'Feature' if 'Feature' in df.columns else ('feature' if 'feature' in df.columns else None)
    if col is None:
        raise ValueError(f"No 'Feature' column in {path}")
    return [x for x in df[col].astype(str).tolist() if x and x != 'nan']

def load_original(path):
    """original merged data with Measurement as index"""
    return pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig', index_col=0)

def filter_top_features(tsfresh_merged_dir, featureselection_dir, output_root):
    os.makedirs(output_root, exist_ok=True)
    raw_map = build_original_map(tsfresh_merged_dir)

    for pid, signal, sel_path in iter_selected(featureselection_dir):
        key = (pid, signal)
        raw_path = raw_map.get(key)
        if not raw_path:
            print(f"Missing original for {os.path.basename(sel_path)}")
            continue

        try:
            selected = read_selected_features(sel_path)
            if not selected:
                print(f"No features in {sel_path}")
                continue

            original = load_original(raw_path)
            keep = [c for c in selected if c in original.columns]
            if not keep:
                print(f"No overlapping columns for {pid} {signal}")
                continue

            filtered = original[keep]
            out_dir = os.path.join(output_root, signal, pid)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{pid}_{signal}_filtered.csv")
            filtered.to_csv(out_path, encoding='utf-8-sig')
            print(f"Saved: {out_path}")

        except Exception as e:
            print(f"Error for {pid} {signal}: {e}")

# ------------------ EXAMPLE RUN ------------------
if __name__ == '__main__':
    BASE = resolve_base_dir()
    TS_MERGED = os.path.join(BASE, 'TSFRESH_merged')
    FEAT_SEL = os.path.join(BASE, 'FeatureSelection')
    OUT = os.path.join(BASE, 'FilteredBasedOnRFECV')
    filter_top_features(TS_MERGED, FEAT_SEL, OUT)
