# ========================================================================
#  TSFRESH Feature Extraction for GSR / HR / RR / Pulse
# ------------------------------------------------------------------------
#  Purpose
#    Extract TSFRESH features from per-folder physiological segments and
#    aggregate them into per-participant CSVs, one file per signal.
#
#  Data layout
#    Base folder is resolved via _last_out_root.txt or newest Processed_*.
#    For each measurement folder containing:
#      else_phase_edited.mat  # matrix with columns:
#        [Time, GSR, HR, RR, Pulse, Phase]
#      Phase codes: 1=Normal, 2=Meditation, 3=Normal
#
#  Pipeline
#    1) Traverse all subfolders under <BASE>.
#    2) Load else_phase_edited.mat and validate expected shape.
#    3) Build a DataFrame and map Phase → id ∈ {"Normal","Meditation"}.
#    4) Ensure a monotonic 'time' column (fallback: row index).
#    5) Melt to long format; keep channels in {"GSR","HR","RR","Pulse"}.
#    6) Create unique_id = id + "_" + channel.
#    7) Run TSFRESH (EfficientFCParameters) with:
#         column_id="unique_id", column_sort="time", column_value="value".
#    8) For each participant (pid inferred from path like "p7"):
#         - Increment measurement counter M1, M2, …
#         - Append rows "M{m}_Meditation" and "M{m}_Normal" per signal.
#    9) Save per-participant & per-signal CSVs to:
#         <BASE>/TSFRESH_merged/<pid>_<Signal>_data.csv
#       Index name: "Measurement"; columns: TSFRESH feature names.
#
#  Outputs
#    • TSFRESH_merged/pX_GSR_data.csv
#    • TSFRESH_merged/pX_HR_data.csv
#    • TSFRESH_merged/pX_RR_data.csv
#    • TSFRESH_merged/pX_Pulse_data.csv
#
#  Requirements
#    Python 3.9+, numpy, pandas, scipy, tsfresh.
#    Adjust n_jobs in extract_features() to suit your CPU.
#
#  Usage
#    python tsfresh_extract_gsr_hr_rr_pulse.py
#    (Ensure _last_out_root.txt points to the base folder, or a
#     Processed_* folder exists at project root.)
#
#  Notes
#    • Folders missing the MAT file or with invalid shapes are skipped.
#    • If TSFRESH returns an empty set (e.g., constant signal), the folder
#      is skipped with a warning.
#    • Participant ID fallback is "pXX" if not found in the path.
# ========================================================================

import pandas as pd
import scipy.io
import os
import numpy as np
import re
from collections import defaultdict
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

# --- helpers -------------------------------------------------------------
def infer_participant_id(path: str) -> str:
    """Find 'p<number>' in any folder name in the path. Fallback to 'pXX'."""
    parts = os.path.normpath(path).split(os.sep)
    for part in reversed(parts):
        m = re.search(r'(?i)\bp[\s_.-]*0*(\d+)(?!\d)', part)
        if m:
            return f"p{int(m.group(1))}"
    return "pXX"

# ========================================================================
#   MAIN
# ========================================================================
if __name__ == '__main__':
    # --- Resolve base_dir like MATLAB (no prompts) ---
    import sys, time
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

    print(f'Using base_dir: {base_dir}')

    # Accumulators for target CSVs by participant and signal
    rows_by_key = defaultdict(list)       # key = (pid, signal) -> list[(row_label, pd.Series)]
    measure_counter = defaultdict(int)    # pid -> measurement index
    signals = ["GSR", "HR", "RR", "Pulse"]   # << CHANGED: added Pulse
    output_dir = os.path.join(base_dir, "TSFRESH_merged")
    os.makedirs(output_dir, exist_ok=True)

    # ====================================================================
    #   FOLDER LOOP — traverse all subdirectories
    # ====================================================================
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()  # stable traversal

        if 'else_phase_edited.mat' in files:
            file_path_mat = os.path.join(root, 'else_phase_edited.mat')
            print(f'\n Processing folder: {root}')

            if not os.path.exists(file_path_mat):
                print(f"File not found: {file_path_mat}")
                continue

            data = scipy.io.loadmat(file_path_mat)

            if 'else_phase_edited' not in data or not isinstance(
                data['else_phase_edited'], (list, np.ndarray)
            ):
                print(f"Invalid format in file: {file_path_mat}")
                continue

            data_matrix = data['else_phase_edited']
            expected_columns = ['Time', 'GSR', 'HR', 'RR', 'Pulse', 'Phase']

            if data_matrix.shape[1] != len(expected_columns):
                print(f"Expected {len(expected_columns)} columns, found {data_matrix.shape[1]}. Skipping {file_path_mat}.")
                continue

            df = pd.DataFrame(data_matrix, columns=expected_columns)

            # -------------------- PREPROCESSING ---------------------------
            # df = df.drop(columns=['Pulse'])       
            df['id'] = df['Phase'].replace({1: "Normal", 3: "Normal", 2: "Meditation"})
            if 'time' not in df.columns:
                df['time'] = df.index

            long_format = df.melt(
                id_vars=['id', 'time', 'Phase'],
                var_name='channel',
                value_name='value'
            ).dropna()

            long_format = long_format[long_format["channel"].isin(signals)]
            long_format["unique_id"] = long_format["id"] + "_" + long_format["channel"]

            # -------------------- FEATURES -------------------------------
            efficient_params = EfficientFCParameters()
            features = extract_features(
                long_format,
                column_id="unique_id",
                column_sort="time",
                column_value="value",
                default_fc_parameters=efficient_params,
                n_jobs=4
            )

            if features.empty:
                print(f"No features extracted in {root}. Possibly constant signals.")
                continue

            features.index.name = "Segment_Signal"

            # -------------------- AGGREGATE INTO TARGET CSVs -----------------
            pid = infer_participant_id(root)
            measure_counter[pid] += 1
            m = measure_counter[pid]

            # row order: Meditation, then Normal
            for sig in signals:
                labels = [f"Meditation_{sig}", f"Normal_{sig}"]
                rownames = [f"M{m}_Meditation", f"M{m}_Normal"]
                for lbl, rname in zip(labels, rownames):
                    if lbl in features.index:
                        rows_by_key[(pid, sig)].append((rname, features.loc[lbl]))
                    else:
                        print(f"Warning: missing row '{lbl}' for {pid} in {root}")

            # -------------------- (OPTIONAL) PER-FOLDER OUTPUT -------------
            # output_file = os.path.join(root, 'features_GSR_HR_RR_Pulse.csv')
            # features.to_csv(output_file)
            # print(f"Per-folder features saved to: {output_file}")

    # ====================================================================
    #   SAVE MERGED FILES PER PARTICIPANT & SIGNAL
    # ====================================================================
    for (pid, sig), rows in rows_by_key.items():
        if not rows:
            continue
        idx = [rname for rname, _ in rows]
        df_out = pd.DataFrame([ser for _, ser in rows], index=idx)
        df_out.index.name = "Measurement"
        out_file = os.path.join(output_dir, f"{pid}_{sig}_data.csv")
        df_out.to_csv(out_file)
        print(f"Saved merged: {out_file}")

    print("\nFeature extraction completed. Merged CSVs in TSFRESH_merged/")
