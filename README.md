# Analysis code for the “Synchronized multimodal dataset for central–autonomic coupling in meditation and rest”

This repository contains the MATLAB analysis pipeline used to preprocess, segment, extract features, and run statistical models on the synchronized multimodal dataset (EEG, fNIRS, GSR, HR/HRV, RR, pulse waveform).

The dataset itself is available on Zenodo DOI: 10.5281/zenodo.17735691.  
This repository provides the code that reproduces the main processing steps and the derived outputs.

---

## 1. Code structure

### 1.1 `1_Filtration/`

- `Filtration`  
  Batch filtering of EEG, fNIRS, pulse, GSR, HR, and RR signals.

- `Filtration_with_visualization`  
  Same as above, with optional multi-channel visualization before and after filtering.

### 1.2 `2_Split_into_Meditation_Normal/`

- `Split_EEG`  
  Align EEG to GSR timeline and split into three phases: *before / meditation / after*.

- `Split_fNIRS`  
  Align fNIRS to GSR timeline and split into three phases: *before / meditation / after*.

- `Split_GSR_HR_RR_Pulse`  
  Synchronize GSR, HR/HRV, RR, pulse; segment into phases 1–3.

- `Split_EEG_visual`  
  As `Split_EEG`, with optional visualization of EEG split into three phases by markers.

- `Split_fNIRS_visual`  
  As `Split_fNIRS`, with optional visualization of fNIRS split into three phases by markers.

- `Split_GSR_HR_RR_Pulse_visual`  
  As `Split_GSR_HR_RR_Pulse`, with optional visualization of GSR/HR/RR/Pulse split into three phases by markers.

- `Preparation_for_tsfresh_library`  
  Merge phase-specific segments into datasets suitable for feature extraction with `tsfresh`.

### 1.3 `3_tsfresh/`

- `Tsfresh_EEG_fNIRS`  
  Merge phases 1 and 3 into a “Normal” condition and compute time-series features for EEG and fNIRS.

- `Tsfresh_GSR_HR_RR_Pulse`  
  Compute `tsfresh` features for GSR/HR/RR (and pulse) from `else_phase_edited`.

> Note: This step assumes an external Python environment with the `tsfresh` library installed.

### 1.4 `_4_RFECV/`

- `RFECV_EEG`  
  RFECV feature selection on EEG features; exports selected features and visualizes feature importances.

- `RFECV_RR`  
  RFECV feature selection on a chosen modality (e.g., RR); exports top features.

- `Filtering_based_on_RFECV`  
  Filters original TSFRESH CSV files to retain only features selected by RFECV.

### 1.5 `5_LDA/`

- `LDA_statistical_analysis_EEG`  
  LDA on EEG feature sets, analysis of LD1 significance, and box-plot visualization.

- `LDA_statistical_analysis_GSR`  
  LDA for two groups on GSR (or other modality), including statistics and plots.

---

## 2. MATLAB environment

- MATLAB: **R2023b Update 6 (23.2.0.2485118)** on Windows 11  
- Required Toolboxes:
  - Signal Processing Toolbox  
  - Statistics and Machine Learning Toolbox  
- Optional:
  - Parallel Computing Toolbox

Quick checks on the target machine:

```matlab
ver('matlab'); ver('signal'); ver('stats');
license('test','Signal_Toolbox');
license('test','Statistics_Toolbox');
