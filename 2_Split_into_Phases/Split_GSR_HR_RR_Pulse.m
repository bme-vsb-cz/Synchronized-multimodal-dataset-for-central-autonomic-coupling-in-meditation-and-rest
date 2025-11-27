% ========================================================================
%   MULTI-FOLDER GSR/HR/RR/PULSE SEGMENTATION SCRIPT
% ------------------------------------------------------------------------
%   Purpose:
%       This script processes biosignal data stored in multiple folders.
%       It synchronizes several modalities (GSR, HR, RR, and Pulse),
%       aligns their time axes, identifies meditation events, and splits
%       each signal into three temporal phases:
%           1 = pre-meditation, 2 = meditation, 3 = post-meditation.
%       The final data are saved as MATLAB tables for convenient use in
%       statistical analysis or visualization.
%
%   Expected input per folder:
%       - GSR_filtered.csv          (timestamp [ms], GSR value)
%       - HR_filtered.csv           (timestamp [s], HR value)
%       - RR_filtered.csv           (timestamp [s], RR interval)
%       - Pulse_filtered.mat        (variable: pulse_filtered)
%       - EVENT_*.csv               (event log with timestamps)
%
%   Output:
%       - GSR_HR_RR_Pulse_segments.mat
%           containing three tables:
%               before_meditation, meditation, after_meditation
%
%   Notes:
%       - Sampling rate of Pulse = 250 Hz.
%       - All time vectors are realigned to the GSR start timestamp.
%       - Each phase table contains columns:
%           Time (s), GSR (μS), HR (bpm), RR (ms), Pulse (a.u.)
% ========================================================================

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
required = {'GSR_filtered.csv','HR_filtered.csv','RR_filtered.csv','Pulse_filtered.mat'};

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


%% === Step 2: Process each input folder ===
for f = 1:length(folders)
    input_folder = folders{f};
    fprintf('\nProcessing folder: %s\n', input_folder);

    % --------------------------------------------------------------------
    % 2.1 LOAD SIGNALS
    % --------------------------------------------------------------------
    % Load pre-filtered biosignal data (GSR, HR, RR, Pulse) from files.
    % Each file type has different time resolution or units.
    % --------------------------------------------------------------------
    
    % --- GSR signal ---
    gsr_data = readmatrix(fullfile(input_folder, 'GSR_filtered.csv'));
    % Column 1 = timestamps in milliseconds (UNIX time)
    % Column 2 = filtered GSR values in microsiemens (μS)
    timestamps = datetime(gsr_data(:,1) / 1000, 'ConvertFrom', 'posixtime', ...
                          'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
    gsr_values = gsr_data(:,2);

    % --- HR signal ---
    hr_data = readmatrix(fullfile(input_folder, 'HR_filtered.csv'));
    t_hr = hr_data(:,1);     % Time in seconds (already relative or UNIX)
    hr_values = hr_data(:,2);% Heart rate in beats per minute (bpm)

    % --- RR signal ---
    rr_data = readmatrix(fullfile(input_folder, 'RR_filtered.csv'));
    t_rr = rr_data(:,1);     % Time in seconds or UNIX
    rr_values = rr_data(:,2);% RR intervals in milliseconds (ms)

    % --- Pulse signal ---
    pulse_data = load(fullfile(input_folder, 'Pulse_filtered.mat'));
    pulse_values = pulse_data.pulse_filtered; % Raw filtered pulse waveform

    % --------------------------------------------------------------------
    % 2.2 CREATE TIME VECTORS
    % --------------------------------------------------------------------
    % The pulse data have no timestamps stored, only a sampling frequency.
    % A time vector is created based on number of samples and Fs = 250 Hz.
    % --------------------------------------------------------------------
    Fs_pulse = 250; % Pulse sampling frequency [Hz]
    num_samples_pulse = length(pulse_values);
    t_pulse = linspace(0, num_samples_pulse / Fs_pulse, num_samples_pulse);

    % --------------------------------------------------------------------
    % 2.3 ALIGN ALL SIGNALS TO THE SAME TIME REFERENCE
    % --------------------------------------------------------------------
    % We use GSR timestamps as the global reference because they are
    % absolute (UNIX) and cover the full measurement duration.
    % --------------------------------------------------------------------
    start_time = timestamps(1); % First GSR timestamp
    t_gsr = seconds(timestamps - start_time); % Convert to relative time (s)
    t_rr = seconds(datetime(t_rr, 'ConvertFrom', 'posixtime') - start_time);

    % Align Pulse so that its end matches the end of the GSR signal
    % This assumes both signals were recorded simultaneously but saved
    % with different reference points.
    t_pulse = t_pulse - t_pulse(end) + t_gsr(end);

    % --------------------------------------------------------------------
    % 2.4 INTERPOLATE SIGNALS TO MATCH GSR TIMEPOINTS
    % --------------------------------------------------------------------
    % To enable synchronized analysis, HR, RR, and Pulse signals are
    % interpolated to the same time vector as GSR (t_gsr).
    % Linear interpolation is used for simplicity.
    % --------------------------------------------------------------------
    hr_values_interp = interp1(t_hr, hr_values, t_gsr, 'linear', 'extrap');
    rr_values_interp = interp1(t_rr, rr_values, t_gsr, 'linear', 'extrap');
    pulse_values_interp = interp1(t_pulse, pulse_values, t_gsr, 'linear', 'extrap');

    % --------------------------------------------------------------------
    % 2.5 LOAD EVENT FILE AND IDENTIFY MEDITATION PERIODS
    % --------------------------------------------------------------------
    % EVENT files contain labeled timestamps for key events such as
    % START and END of meditation. These will be used to segment signals
    % into pre-, during-, and post-meditation phases.
    % --------------------------------------------------------------------
    event_files = dir(fullfile(input_folder, '*EVENT_*'));
    if isempty(event_files)
        warning('No file containing "EVENT_" found in folder: %s', input_folder);
        continue; % Skip processing for this folder
    end

    % Use the first matching EVENT file
    event_filename = fullfile(input_folder, event_files(1).name);
    fprintf('Loading events from file: %s\n', event_filename);

    % Load event table (timestamp, name, type)
    event_data = readtable(event_filename, 'VariableNamingRule', 'preserve');

    % Convert timestamps (ms → s) and store as datetime array
    event_times = datetime(event_data{:,1} / 1000, 'ConvertFrom', 'posixtime', ...
                           'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
    event_names = event_data{:,2};
    event_types = event_data{:,3};

    % Compute relative event times (seconds from GSR start)
    event_sec = seconds(event_times - start_time);

    % Identify meditation start and end based on event name/type
    is_start = strcmp(event_names, 'MEDITATION') & strcmp(event_types, 'START');
    is_end   = strcmp(event_names, 'MEDITATION') & strcmp(event_types, 'END');

    % Validate presence of both markers
    if sum(is_start) == 0 || sum(is_end) == 0
        warning('Missing START or END of meditation in event file: %s', input_folder);
        continue;
    end

    % Extract start and end times (in seconds)
    t_start_meditation = event_sec(is_start);
    t_end_meditation   = event_sec(is_end);

    % --------------------------------------------------------------------
    % 2.6 CREATE LOGICAL MASKS FOR EACH PHASE
    % --------------------------------------------------------------------
    % Using GSR time vector, generate logical indices (true/false)
    % marking which samples belong to each phase.
    % --------------------------------------------------------------------
    mask_before     = t_gsr < t_start_meditation;                                   % Before meditation
    mask_meditation = (t_gsr >= t_start_meditation) & (t_gsr <= t_end_meditation);    % During meditation
    mask_after       = t_gsr > t_end_meditation;                                     % After meditation

    % --------------------------------------------------------------------
    % 2.7 CONVERT SEGMENTS INTO TABLES
    % --------------------------------------------------------------------
    % Each phase is saved as a MATLAB table with descriptive column names.
    % This makes it easier to export, visualize, or analyze later.
    % --------------------------------------------------------------------
    before_meditation = table(t_gsr(mask_before), gsr_values(mask_before), ...
        hr_values_interp(mask_before), rr_values_interp(mask_before), ...
        pulse_values_interp(mask_before), ...
        'VariableNames', {'Time_s', 'GSR_uS', 'HR_bpm', 'RR_ms', 'Pulse'});

    meditation = table(t_gsr(mask_meditation), gsr_values(mask_meditation), ...
        hr_values_interp(mask_meditation), rr_values_interp(mask_meditation), ...
        pulse_values_interp(mask_meditation), ...
        'VariableNames', {'Time_s', 'GSR_uS', 'HR_bpm', 'RR_ms', 'Pulse'});

    after_meditation = table(t_gsr(mask_after), gsr_values(mask_after), ...
        hr_values_interp(mask_after), rr_values_interp(mask_after), ...
        pulse_values_interp(mask_after), ...
        'VariableNames', {'Time_s', 'GSR_uS', 'HR_bpm', 'RR_ms', 'Pulse'});

    % --------------------------------------------------------------------
    % 2.8 SAVE RESULTS INTO A STRUCTURED MAT FILE
    % --------------------------------------------------------------------
    % Each folder receives one file “GSR_HR_RR_Pulse_segments.mat” that
    % contains three tables (before_meditation, meditation, after_meditation).
    % These are later merged with EEG/fNIRS data in other scripts.
    % --------------------------------------------------------------------
    save(fullfile(input_folder, 'GSR_HR_RR_Pulse_segments.mat'), ...
         'before_meditation', 'meditation', 'after_meditation');

    disp(['Data successfully saved in folder: ', input_folder]);
end

%% === Step 3: Final completion message ===
disp('All folders have been processed successfully!');
