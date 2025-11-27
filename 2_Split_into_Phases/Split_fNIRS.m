% ========================================================================
%   MULTI-FOLDER fNIRS SEGMENTATION SCRIPT
% ------------------------------------------------------------------------
%   Purpose:
%       This script processes multiple measurement folders containing
%       pre-filtered fNIRS and GSR data. It aligns fNIRS time axes with GSR
%       timestamps and splits fNIRS signals into three temporal phases:
%       pre-meditation, meditation, and post-meditation. The segmentation
%       is based on timestamps found in the corresponding EVENT_*.csv file.
%
%   Expected input per folder:
%       - fNIRS_filtered.mat      (variable: fnirs_filtered)
%       - GSR_filtered.csv        (columns: [timestamp, value])
%       - EVENT_*.csv             (columns: [timestamp, name, type])
%
%   Output:
%       - fNIRS_phases.mat containing:
%             before_meditation    (fNIRS before meditation)
%             meditation           (fNIRS during meditation)
%             after_meditation     (fNIRS after meditation)
%
%   Notes:
%       - Folders are discovered automatically under out_root.
%       - Sampling rates: GSR = 10 Hz, fNIRS = 250 Hz.
%       - Folders missing files are skipped automatically.
% ========================================================================

%% === Clear workspace and close all figures ===
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
required = {'fNIRS_filtered.mat','GSR_filtered.csv'};

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
    Fs_GSR = 10; % Sampling frequency of GSR [Hz]
    gsr_file = fullfile(input_folder, 'GSR_filtered.csv');
    
    if exist(gsr_file, 'file')
        % Read filtered GSR signal (timestamp in ms, value)
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

    %% === Load fNIRS signal ===
    fnirs_file = fullfile(input_folder, 'fNIRS_filtered.mat');
    
    if exist(fnirs_file, 'file')
        fNIRS = load(fnirs_file);
        fNIRS_data = fNIRS.fnirs_filtered;
        Fs_fNIRS = 250; % Sampling frequency of fNIRS [Hz]
    else
        warning('Missing file: fNIRS_filtered.mat in folder %s', input_folder);
        continue;
    end

    %% === Align fNIRS data with GSR timestamps ===
    % Define the end of measurement as the last GSR timestamp
    end_time = max([ts_gsr(end), ts_gsr(end)]);
    
    % Compute start time of fNIRS measurement
    start_time_fnirs = end_time - seconds(size(fNIRS_data, 2) / Fs_fNIRS);

    % Generate full time vector for fNIRS data
    ts_fnirs = start_time_fnirs + seconds((0:size(fNIRS_data, 2)-1) / Fs_fNIRS);
    y2 = size(fNIRS_data, 2) / Fs_fNIRS; % Duration in seconds

    %% === Locate event file (containing "EVENT_" in name) ===
    event_files = dir(fullfile(input_folder, '*EVENT_*'));
    
    if isempty(event_files)
        warning('No EVENT file found in folder: %s', input_folder);
        continue; % Skip this folder
    end

    % Select first event file found in the directory
    event_filename = fullfile(input_folder, event_files(1).name);
    fprintf('Loading events from file: %s\n', event_filename);

    %% === Load event data ===
    event_data = readtable(event_filename, 'VariableNamingRule', 'preserve');
    event_times = datetime(event_data{:,1} / 1000, ...
                           'ConvertFrom', 'posixtime', ...
                           'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
    event_names = event_data{:,2};
    event_types = event_data{:,3};

    %% === Identify meditation start and end ===
    t_start_meditation = event_times(strcmp(event_names, 'MEDITATION') & strcmp(event_types, 'START'));
    t_end_meditation   = event_times(strcmp(event_names, 'MEDITATION') & strcmp(event_types, 'END'));

    if isempty(t_start_meditation) || isempty(t_end_meditation)
        warning('Meditation start or end not found in folder: %s', input_folder);
        continue; % Skip if event markers are missing
    end

    %% === Split fNIRS data into phases ===
    % Logical indexing based on event timestamps
    before_meditation = fNIRS_data(:, ts_fnirs <  t_start_meditation);                        % before meditation
    meditation        = fNIRS_data(:, ts_fnirs >= t_start_meditation & ts_fnirs <= t_end_meditation); % during meditation
    after_meditation  = fNIRS_data(:, ts_fnirs >  t_end_meditation);                          % after meditation

    %% === Save segmented data ===
    save(fullfile(input_folder, 'fNIRS_phases.mat'), ...
         'before_meditation', 'meditation', 'after_meditation');

    disp(['Data successfully saved in folder: ', input_folder]);
end

%% === Final message ===
disp('All folders processed successfully!');
