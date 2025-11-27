# ========================================================================
#  TSFRESH Feature Extraction for EEG & fNIRS (per folder → per PID)
# ------------------------------------------------------------------------
#  Purpose
#    Compute TSFRESH features from phase-edited EEG/fNIRS matrices and
#    aggregate them into per-participant CSVs (one file per modality).
#
#  Inputs (per measurement folder)
#    • EEG_phase_edited.mat
#    • fNIRS_phase_edited.mat
#      - Shape: (channels + 1) × samples
#      - Convention: last row = phase labels (1=pre, 2=meditation, 3=post)
#                    preceding rows = signal channels
#
#  Phase mapping used here
#    - Meditation  := phase == 2
#    - Normal      := phase in {1, 3}  (pre + post merged)
#
#  Pipeline
#    1) Resolve base_dir via _last_out_root.txt or newest Processed_* at project root.
#    2) For each folder containing the two MAT files:
#       a) Load matrices; split channels vs phase labels.
#       b) Merge samples by phase to two DataFrames per modality:
#          {EEG_Meditation, EEG_Normal, fNIRS_Meditation, fNIRS_Normal}.
#       c) Convert each DF to long form with:
#             id   := index // 1000  (segments of 1000 samples)
#             time := index
#       d) Run TSFRESH (EfficientFCParameters) with
#             column_id='id', column_sort='time', column_value='value'.
#          Aggregate by taking the mean across segment-level rows → one Series.
#       e) Append rows to an accumulator keyed by (participant_id, modality)
#          with labels: M{n}_Meditation, M{n}_Normal (n = measurement counter).
#    3) Write per-participant, per-modality CSVs to:
#          <base_dir>/TSFRESH_merged/{pX}_{EEG|fNIRS}_data.csv
#       - Index name: "Measurement"
#       - Encoding: UTF-8 with BOM (utf-8-sig) for Excel compatibility
#
#  Outputs (examples)
#    • TSFRESH_merged/p7_EEG_data.csv
#    • TSFRESH_merged/p7_fNIRS_data.csv
#
#  Assumptions & safeguards
#    • MAT matrices follow the “channels rows + last label row” convention.
#    • If a phase has no samples, it is skipped with a console notice.
#    • Participant ID is inferred from any 'p<number>' token in the path
#      (fallback: pXX). Measurement order follows folder traversal.
#
#  Requirements
#    Python 3.9+, numpy, pandas, scipy, tsfresh
#    (Optional) Adjust n_jobs in TSFRESH to match available CPU cores.
#
#  Usage
#    python tsfresh_extract_eeg_fnirs.py
#    # Ensure _last_out_root.txt points to the dataset root, or a
#    # Processed_* folder exists alongside this script’s parent directory.
# ========================================================================

import os
import re
import time
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.io
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

# ----------------------------- helpers ----------------------------------
def infer_participant_id(path: str) -> str:
    """Find 'p<number>' in any folder name in the path. Fallback pXX."""
    parts = os.path.normpath(path).split(os.sep)
    for part in reversed(parts):
        m = re.search(r'(?i)\bp[\s_.-]*0*(\d+)(?!\d)', part)
        if m:
            return f"p{int(m.group(1))}"
    return "pXX"

def extract_features_row(df_long: pd.DataFrame, efficient_params) -> pd.Series:
    """TSFRESH → mean across id segments; returns a single-row Series."""
    feats = extract_features(
        df_long,
        column_id='id',
        column_sort='time',
        column_value='value',
        default_fc_parameters=efficient_params,
        n_jobs=4
    )
    return feats.mean()

def to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert wide DF to long with id and time. Segment = 1000-sample blocks."""
    if df_wide.empty:
        return df_wide
    df_wide = df_wide.copy()
    df_wide['id'] = df_wide.index // 1000
    df_wide['time'] = df_wide.index
    return df_wide.melt(id_vars=['id', 'time'], var_name='channel', value_name='value').dropna()

# ========================================================================
#   MAIN EXECUTION BLOCK
# ========================================================================
if __name__ == '__main__':

    # --- Base directory ---
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

    # Accumulate rows for final CSVs by participant and modality
    rows_by_key = defaultdict(list)        # key = (pid, 'EEG'|'fNIRS') -> list[(rowname, Series)]
    measure_counter = defaultdict(int)     # pid -> M1,M2,...
    output_dir = os.path.join(base_dir, "TSFRESH_merged")
    os.makedirs(output_dir, exist_ok=True)

    efficient_params = EfficientFCParameters()

    # ====================================================================
    #   FOLDER LOOP — traverse all subdirectories
    # ====================================================================
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()  # stable order

        # Process folder only if both EEG and fNIRS files are present
        if 'EEG_phase_edited.mat' in files and 'fNIRS_phase_edited.mat' in files:

            eeg_path = os.path.join(root, 'EEG_phase_edited.mat')
            fnirs_path = os.path.join(root, 'fNIRS_phase_edited.mat')

            print(f'\n Processing folder: {root}')

            # -------------------- STEP 1: Load MATLAB data ----------------
            eeg_data = scipy.io.loadmat(eeg_path)
            fnirs_data = scipy.io.loadmat(fnirs_path)

            # The last row holds the phase labels (1, 2, 3)
            eeg_phase = eeg_data['EEG_phase_edited'][-1, :]
            fnirs_phase = fnirs_data['fNIRS_phase_edited'][-1, :]

            # All preceding rows are the signal channels
            eeg_channels = eeg_data['EEG_phase_edited'][:-1, :]
            fnirs_channels = fnirs_data['fNIRS_phase_edited'][:-1, :]

            # -------------------- STEP 2: merge phases --------------------
            def merge_phases(channels, phases, phase_values, prefix):
                mask = np.isin(phases, phase_values)
                merged_data = channels[:, mask]
                df = pd.DataFrame(
                    merged_data.T,
                    columns=[f"{prefix}_Channel_{i+1}" for i in range(merged_data.shape[0])]
                )
                return df

            # -------------------- STEP 3: phase-specific DataFrames -------
            phase_data = {
                "EEG_Meditation": merge_phases(eeg_channels,   eeg_phase,   [2.0],       "EEG"),
                "EEG_Normal":   merge_phases(eeg_channels,   eeg_phase,   [1.0, 3.0],  "EEG"),
                "fNIRS_Meditation": merge_phases(fnirs_channels, fnirs_phase, [2.0],     "fNIRS"),
                "fNIRS_Normal":   merge_phases(fnirs_channels, fnirs_phase, [1.0, 3.0],"fNIRS")
            }

            # -------------------- STEP 4: features → aggregated rows ----
            pid = infer_participant_id(root)
            measure_counter[pid] += 1
            m = measure_counter[pid]

            # EEG
            for phase_key, row_label in [("EEG_Meditation", f"M{m}_Meditation"),
                                         ("EEG_Normal",   f"M{m}_Normal")]:
                df_long = to_long(phase_data[phase_key])
                if df_long.empty:
                    print(f"Skipping {pid} {phase_key} — no data.")
                    continue
                row = extract_features_row(df_long, efficient_params)
                rows_by_key[(pid, "EEG")].append((row_label, row))

            # fNIRS
            for phase_key, row_label in [("fNIRS_Meditation", f"M{m}_Meditation"),
                                         ("fNIRS_Normal",   f"M{m}_Normal")]:
                df_long = to_long(phase_data[phase_key])
                if df_long.empty:
                    print(f"Skipping {pid} {phase_key} — no data.")
                    continue
                row = extract_features_row(df_long, efficient_params)
                rows_by_key[(pid, "fNIRS")].append((row_label, row))

            # -------------------- (optional) per-folder output -----------
            # kept disabled due to aggregated CSVs
            # def save_per_folder(df, label, prefix, save_path):
            #     if df.empty: return
            #     df_long = to_long(df)
            #     if df_long.empty: return
            #     feats = extract_features(
            #         df_long, column_id='id', column_sort='time',
            #         column_value='value', default_fc_parameters=efficient_params, n_jobs=4
            #     ).mean().to_frame().T
            #     out = os.path.join(save_path, f'{prefix}_phase_{label}_features.csv')
            #     feats.to_csv(out, index=False, encoding='utf-8-sig')
            # save_per_folder(phase_data["EEG_Meditation"], "Meditation", "eeg", root)
            # save_per_folder(phase_data["EEG_Normal"],   "Normal",   "eeg", root)
            # save_per_folder(phase_data["fNIRS_Meditation"], "Meditation","fnirs", root)
            # save_per_folder(phase_data["fNIRS_Normal"],   "Normal",  "fnirs", root)

    # ====================================================================
    #   SAVE MERGED FILES PER PARTICIPANT & MODALITY
    # ====================================================================
    for (pid, modality), rows in rows_by_key.items():
        if not rows:
            continue
        idx = [name for name, _ in rows]                  # order preserved: Mx_Meditation, Mx_Normal, ...
        df_out = pd.DataFrame([ser for _, ser in rows], index=idx)
        df_out.index.name = "Measurement"
        out_path = os.path.join(output_dir, f"{pid}_{modality}_data.csv")
        df_out.to_csv(out_path, encoding='utf-8-sig')     # UTF-8 with BOM for Excel
        print(f"Saved merged: {out_path}")

    print("\nFeature extraction for all folders completed. Merged CSVs in TSFRESH_merged/")
