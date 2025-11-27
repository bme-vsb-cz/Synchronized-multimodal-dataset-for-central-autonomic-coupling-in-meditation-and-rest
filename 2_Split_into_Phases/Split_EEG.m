% ========================================================================
%   MULTI-FOLDER EEG & GSR SEGMENTATION SCRIPT
% ------------------------------------------------------------------------
%   Purpose:
%       This script processes multiple measurement folders containing EEG
%       and GSR data. It aligns EEG data with GSR timestamps and splits
%       EEG into three phases: pre-meditation, meditation, and after-
%       meditation based on event log files (EVENT_*.csv).
%
%   Input per folder:
%       - EEG_filtered.mat      (matrix variable: eeg_filtered)
%       - GSR_filtered.csv      (columns: [timestamp, value])
%       - EVENT_*.csv           (columns: [timestamp, name, type])
%
%   Output:
%       - EEG_phases.mat containing:
%             before_meditation    (EEG before meditation)
%             meditation           (EEG during meditation)
%             after_meditation     (EEG after meditation)
%
%   Notes:
%       - Folders are selected manually (multiple allowed).
%       - EEG sampling frequency: 250 Hz; GSR: 10 Hz.
% ========================================================================

%% === Clear workspace ===
clear; clc; close all;

%% === Resolve out_root without prompts ===
this_dir    = fileparts(mfilename('fullpath'));
project_root = fileparts(this_dir);                  % shared root of both scripts
ptr_main    = fullfile(project_root, '_last_out_root.txt');
ptr_local   = fullfile(this_dir,    '_last_out_root.txt'); % fallback for older runs

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
required = {'EEG_filtered.mat','GSR_filtered.csv'};

% recursive: find all files (R2016b or later)
hits = dir(fullfile(out_root, '**', 'GSR_filtered.csv'));
folders = unique({hits.folder});

keep = false(size(folders));
for i = 1:numel(folders)
    present = all(cellfun(@(fn) exist(fullfile(folders{i}, fn),'file')>0, required));
    keep(i) = present;
end
folders = folders(keep);

if isempty(folders)
    error('No valid data folders under: %s', out_root);
end


%% === Process each selected folder ===
for f = 1:length(folders)
    input_folder = folders{f};
    fprintf('\nProcessing folder: %s\n', input_folder);

    %% === Load GSR signal ===
    Fs_GSR = 10; % GSR sampling frequency [Hz]
    gsr_file = fullfile(input_folder, 'GSR_filtered.csv');
    
    if exist(gsr_file, 'file')
        % Load filtered GSR data (timestamp, value)
        gsr_data = readmatrix(gsr_file);
        ts_gsr = datetime(gsr_data(:,1) / 1000, ...
                          'ConvertFrom', 'posixtime', ...
                          'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
        gsr_values = gsr_data(:,2);
        y1 = length(gsr_values) / Fs_GSR; % Duration in seconds
    else
        warning('Missing file: GSR_filtered.csv in folder %s', input_folder);
        continue; % Skip this folder
    end

    %% === Load EEG signal ===
    eeg_file = fullfile(input_folder, 'EEG_filtered.mat');
    
    if exist(eeg_file, 'file')
        EEG = load(eeg_file);
        EEG_data = EEG.eeg_filtered;
        Fs_EEG = 250; % EEG sampling frequency [Hz]
    else
        warning('Missing file: EEG_filtered.mat in folder %s', input_folder);
        continue;
    end

    %% === Align EEG data with GSR time range ===
    % Assumption: GSR timestamps are absolute (UNIX time in ms)
    end_time = max([ts_gsr(end), ts_gsr(end)]); % Use last GSR timestamp as reference
    start_time_eeg = end_time - seconds(size(EEG_data, 2) / Fs_EEG);

    % Create EEG time vector corresponding to sample indices
    ts_eeg = start_time_eeg + seconds((0:size(EEG_data, 2)-1) / Fs_EEG);
    y2 = size(EEG_data, 2) / Fs_EEG; % EEG recording duration

    %% === Locate EVENT file (with "EVENT_" in name) ===
    event_files = dir(fullfile(input_folder, '*EVENT_*'));
    
    if isempty(event_files)
        warning('No EVENT file found in folder: %s', input_folder);
        continue; % Skip folder if no event log is available
    end

    % Use the first matching file
    event_filename = fullfile(input_folder, event_files(1).name);
    fprintf('Loading events from file: %s\n', event_filename);

    %% === Load event timestamps and labels ===
    event_data = readtable(event_filename, 'VariableNamingRule', 'preserve');
    event_times = datetime(event_data{:,1} / 1000, ...
                           'ConvertFrom', 'posixtime', ...
                           'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
    event_names = event_data{:,2};
    event_types = event_data{:,3};

    %% === Identify meditation start and end times ===
    t_start_meditation = event_times(strcmp(event_names, 'MEDITATION') & strcmp(event_types, 'START'));
    t_end_meditation   = event_times(strcmp(event_names, 'MEDITATION') & strcmp(event_types, 'END'));

    if isempty(t_start_meditation) || isempty(t_end_meditation)
        warning('Meditation start/end not found in folder: %s', input_folder);
        continue; % Skip folder if events are missing
    end

    %% === Split EEG data into phases ===
    % Logical indexing based on timestamps
    before_meditation = EEG_data(:, ts_eeg <  t_start_meditation);
    meditation          = EEG_data(:, ts_eeg >= t_start_meditation & ts_eeg <= t_end_meditation);
    after_meditation  = EEG_data(:, ts_eeg >  t_end_meditation);

    %% === Save segmented EEG data ===
    save(fullfile(input_folder, 'EEG_phases.mat'), ...
         'before_meditation', 'meditation', 'after_meditation');

    disp(['Data successfully saved in folder: ', input_folder]);
end

%% === Final message ===
disp('All folders processed successfully!');
