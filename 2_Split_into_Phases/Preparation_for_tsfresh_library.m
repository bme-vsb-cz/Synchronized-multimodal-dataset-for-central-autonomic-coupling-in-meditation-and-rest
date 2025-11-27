% ========================================================================
%   MULTI-FOLDER MERGING AND PHASE LABELING SCRIPT
% ------------------------------------------------------------------------
%   Purpose:
%       This script takes already segmented and pre-filtered biosignal data
%       (EEG, fNIRS, and other signals such as GSR, HR, RR, Pulse) and merges
%       them into unified datasets per folder. Each dataset is split into
%       three temporal phases (pre-meditation, meditation, post-meditation)
%       which are concatenated and labeled for further feature extraction
%       or machine learning analysis.
%
%   Expected input per folder:
%       - EEG_phases.mat               % contains EEG segments by phase
%       - fNIRS_phases.mat             % contains fNIRS segments by phase
%       - GSR_HR_RR_Pulse_segments.mat  % contains other signals (tables)
%
%   Output per folder:
%       - EEG_phase_edited.mat
%       - fNIRS_phase_edited.mat
%       - else_phase_edited.mat
%
%   Phase coding:
%       1 = pre-meditation, 2 = meditation, 3 = post-meditation
%
%   Notes:
%       - The script automatically resizes matrices to the smallest
%         compatible dimensions to ensure alignment across phases.
%       - The “other” signals (GSR/HR/RR/Pulse) are stored as tables,
%         therefore converted to numeric arrays before merging.
% ========================================================================

clear; close all; clc;

%% === Resolve out_root without prompts ===
this_dir    = fileparts(mfilename('fullpath'));
project_root = fileparts(this_dir);                  % shared root of both scripts
ptr_main    = fullfile(project_root, '_last_out_root.txt');
ptr_local   = fullfile(this_dir,    '_last_out_root.txt'); % fallback for old runs

out_root = '';
if exist(ptr_main,'file')
    out_root = strtrim(fileread(ptr_main));
elseif exist(ptr_local,'file')
    out_root = strtrim(fileread(ptr_local));
end

% auto-detection, no uigetdir:
if isempty(out_root) || ~exist(out_root,'dir')
    d = dir(fullfile(project_root,'Processed_*'));
    d = d([d.isdir]);
    if ~isempty(d)
        [~,ix] = max([d.datenum]);                   % latest Processed_*
        out_root = fullfile(d(ix).folder, d(ix).name);
    else
        error('out_root not resolved: pointer missing and no Processed_* under %s.', project_root);
    end
end

%% === Build list of input folders under out_root ===
required = {'GSR_HR_RR_Pulse_segments.mat','EEG_phases.mat','fNIRS_phases.mat'};

% recursive: find all files (R2016b or later)
hits = dir(fullfile(out_root, '**', 'GSR_filtered.csv'));
data_folders = unique({hits.folder});

keep = false(size(data_folders));
for i = 1:numel(data_folders)
    present = all(cellfun(@(fn) exist(fullfile(data_folders{i}, fn),'file')>0, required));
    keep(i) = present;
end
data_folders = data_folders(keep);

if isempty(data_folders)
    error('No valid data folders under: %s', out_root);
end

%% === Step 2: Loop through each input folder ===
for i = 1:length(data_folders)
    data_folder = data_folders{i};
    fprintf('\n--- Processing folder: %s ---\n', data_folder);

    % --------------------------------------------------------------------
    % 2.1 LOAD INPUT FILES
    % --------------------------------------------------------------------
    % Each folder is expected to contain three MATLAB data files:
    %   (1) EEG_phases.mat       → segmented EEG data
    %   (2) fNIRS_phases.mat     → segmented fNIRS data
    %   (3) GSR_HR_RR_Pulse_segments.mat → other physiological data
    % The script loads them as MATLAB structures.
    % --------------------------------------------------------------------
    EEG_data = load(fullfile(data_folder, 'EEG_phases.mat'));
    fNIRS_data = load(fullfile(data_folder, 'fNIRS_phases.mat'));
    else_data = load(fullfile(data_folder, 'GSR_HR_RR_Pulse_segments.mat'));

    % Extract variables from loaded structures
    % EEG signals: each is a 2D matrix (channels × samples)
    before_meditation_EEG = EEG_data.before_meditation;
    meditation_EEG      = EEG_data.meditation;
    after_meditation_EEG   = EEG_data.after_meditation;

    % fNIRS signals: each is also a 2D matrix (channels × samples)
    before_meditation_fNIRS = fNIRS_data.before_meditation;
    meditation_fNIRS      = fNIRS_data.meditation;
    after_meditation_fNIRS   = fNIRS_data.after_meditation;

    % Other modalities (GSR, HR, RR, Pulse) are stored as tables
    % → convert to numeric matrices for concatenation
    before_meditation_else = table2array(else_data.before_meditation);
    meditation_else      = table2array(else_data.meditation);
    after_meditation_else   = table2array(else_data.after_meditation);

    % --------------------------------------------------------------------
    % 2.2 SIZE ALIGNMENT ACROSS PHASES
    % --------------------------------------------------------------------
    % Purpose:
    %   Ensure that all matrices within one modality (e.g., EEG) have the
    %   same number of rows (channels) before concatenating them.
    % Reason:
    %   Some phases might contain slightly different dimensions due to
    %   earlier preprocessing or trimming. Without alignment, concatenation
    %   could fail or misalign data.
    % --------------------------------------------------------------------
    adjust_size = @(x, minRows) x(1:minRows, :);  % Helper for truncation

    % --- EEG alignment ---
    minRows = min([size(before_meditation_EEG, 1), size(meditation_EEG, 1), size(after_meditation_EEG, 1)]);
    before_meditation_EEG = adjust_size(before_meditation_EEG, minRows);
    meditation_EEG      = adjust_size(meditation_EEG, minRows);
    after_meditation_EEG   = adjust_size(after_meditation_EEG, minRows);

    % --- fNIRS alignment ---
    minRows = min([size(before_meditation_fNIRS, 1), size(meditation_fNIRS, 1), size(after_meditation_fNIRS, 1)]);
    before_meditation_fNIRS = adjust_size(before_meditation_fNIRS, minRows);
    meditation_fNIRS      = adjust_size(meditation_fNIRS, minRows);
    after_meditation_fNIRS   = adjust_size(after_meditation_fNIRS, minRows);

    % --- Other modalities alignment ---
    % These are organized column-wise (features across rows)
    minCols = min([size(before_meditation_else, 2), size(meditation_else, 2), size(after_meditation_else, 2)]);

    % Truncate all to the smallest number of columns to ensure consistency
    if size(before_meditation_else, 2) ~= minCols
        before_meditation_else = before_meditation_else(:, 1:minCols);
    end
    if size(meditation_else, 2) ~= minCols
        meditation_else = meditation_else(:, 1:minCols);
    end
    if size(after_meditation_else, 2) ~= minCols
        after_meditation_else = after_meditation_else(:, 1:minCols);
    end

    % --------------------------------------------------------------------
    % 2.3 MERGING PHASES INTO ONE MATRIX
    % --------------------------------------------------------------------
    % Combine three temporal phases into a single continuous matrix for
    % each modality. EEG and fNIRS are concatenated horizontally (same
    % channels, continuous time). “Other” data are concatenated vertically,
    % because each row corresponds to a separate sample rather than a
    % channel.
    % --------------------------------------------------------------------
    EEG_phase_edited     = [before_meditation_EEG, meditation_EEG, after_meditation_EEG];
    fNIRS_phase_edited   = [before_meditation_fNIRS, meditation_fNIRS, after_meditation_fNIRS];
    else_phase_edited = [before_meditation_else; meditation_else; after_meditation_else];

    % --------------------------------------------------------------------
    % 2.4 CREATE PHASE LABEL VECTORS
    % --------------------------------------------------------------------
    % Each final matrix should include an indicator specifying the phase
    % of every sample. This simplifies later feature extraction or
    % supervised classification by allowing direct indexing.
    % --------------------------------------------------------------------
    % EEG and fNIRS: phase labels are appended as a new last row.
    % Other modalities: labels are appended as a new last column.

    % For EEG (1D row: one label per sample/column)
    phase_row = [ ...
        ones(1, size(before_meditation_EEG, 2)), ...        % pre
        2 * ones(1, size(meditation_EEG, 2)), ...         % meditation
        3 * ones(1, size(after_meditation_EEG, 2))];         % post

    % For fNIRS (same logic)
    phase_row_fNIRS = [ ...
        ones(1, size(before_meditation_fNIRS, 2)), ...
        2 * ones(1, size(meditation_fNIRS, 2)), ...
        3 * ones(1, size(after_meditation_fNIRS, 2))];

    % For “other” data (vertical vector: one label per row/sample)
    phase_row_else = [ ...
        ones(size(before_meditation_else, 1), 1); ...
        2 * ones(size(meditation_else, 1), 1); ...
        3 * ones(size(after_meditation_else, 1), 1)];

    % --------------------------------------------------------------------
    % 2.5 APPEND PHASE LABELS TO THE MATRICES
    % --------------------------------------------------------------------
    % EEG and fNIRS → append as new last row (phase_row)
    % Other signals → append as new last column
    % --------------------------------------------------------------------
    EEG_phase_edited     = [EEG_phase_edited; phase_row];
    fNIRS_phase_edited   = [fNIRS_phase_edited; phase_row_fNIRS];
    else_phase_edited = [else_phase_edited, phase_row_else];

    % --------------------------------------------------------------------
    % 2.6 SAVE PROCESSED RESULTS
    % --------------------------------------------------------------------
    % Each processed matrix is saved into the same folder under a new name
    % with the suffix "_upravene" ("adjusted"). These files now contain
    % fully labeled and concatenated datasets for downstream analysis.
    % --------------------------------------------------------------------
    save(fullfile(data_folder, 'EEG_phase_edited.mat'), 'EEG_phase_edited');
    save(fullfile(data_folder, 'fNIRS_phase_edited.mat'), 'fNIRS_phase_edited');
    save(fullfile(data_folder, 'else_phase_edited.mat'), 'else_phase_edited');

    fprintf('Data in folder "%s" were successfully adjusted and saved.\n', data_folder);
end

%% === Step 3: Completion message ===
disp('All selected folders have been successfully processed and saved.');

%% === Helper function: multi-folder selection ===
function folders = uigetdir_multiselect()
    % ---------------------------------------------------------------
    % Purpose:
    %   This function allows the user to select multiple folders
    %   using the standard uigetdir() dialog in a loop.
    %   The selection ends when the user clicks "Cancel".
    %
    % Output:
    %   folders — a cell array of folder paths chosen by the user
    % ---------------------------------------------------------------
    folders = {};
    while true
        folder = uigetdir();
        if folder == 0
            % If user cancels selection, stop the loop
            break;
        end
        folders{end+1} = folder;
    end
end
