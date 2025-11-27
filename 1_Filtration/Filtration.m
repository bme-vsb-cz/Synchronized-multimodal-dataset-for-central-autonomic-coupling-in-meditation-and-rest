clc; clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT OVERVIEW
% -------------------------------------------------------------------------
% Purpose:
%   Batch-process biosignal recordings (EEG, fNIRS, Pulse, GSR, HR, RR)
%   stored across multiple folders. Load raw MATLAB/CSV data, apply
%   appropriate digital filters, and save cleaned signals per folder.
%
% Input:
%   - One or more folders selected manually (uigetdir)
%   - Each folder should contain:
%       • *.mat file with matrix "Out" (all raw channels)
%       • Optional CSV files for GSR, HR, RR
%
% Output:
%   - EEG_filtered.mat
%   - fNIRS_filtered.mat
%   - Pulse_filtered.mat
%   - GSR_filtered.csv
%   - HR_filtered.csv
%   - RR_filtered.csv
%
% Processing summary:
%   1) EEG: notch (50 Hz), high-pass (0.5 Hz), low-pass (40 Hz)
%   2) fNIRS: bandpass (0.01–0.5 Hz)
%   3) Pulse: bandpass (0.7–3 Hz)
%   4) GSR: Savitzky–Golay smoothing
%   5) HR: interpolation to 0.5 s steps + smoothing
%   6) RR: interpolation + low-pass filter (cutoff 0.17 Hz)
%
% Notes:
%   - The script skips folders that lack the required files.
%   - The base sampling frequency for EEG/fNIRS/Pulse is 250 Hz; HR and RR
%     are resampled as indicated.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% === Select TOP-LEVEL folder and scan all subfolders ===
root = uigetdir(pwd, 'Select TOP-LEVEL folder with data');
if root == 0
    disp('No folder selected. Script will terminate.');
    return;
end

% find all .mat files under root and take their parent folders
hits = dir(fullfile(root, '**', '*.mat'));    % requires R2016b or later
if isempty(hits)
    disp('No .mat files found under the selected root.');
    return;
end
folders = unique({hits.folder});              % <- keeps the later loop intact


%% === HARD-CODED OUTPUT ROOT (added) ===
%out_root = 'D:\Biosignals\Processed_2025_10_20';  % <-- change as needed
%if ~exist(out_root,'dir'), mkdir(out_root); end

% === OUTPUT ROOT derived from selected root ===
root = strip(root,'right',filesep);
[parentRoot, rootName] = fileparts(root);     % parent of the picked folder

% outputs as a sibling of the picked folder (under its parent)
out_root = fullfile(parentRoot, ['Filtered_' rootName '_' datestr(now,'yyyy.mm.dd')]);
if ~exist(out_root,'dir'), mkdir(out_root); end

% Define sampling frequency and discard initial transient
fs = 250;                      
cut_off_samples = 20 * fs;     

%% === Main processing loop over all selected folders ===
for f = 1:length(folders)
    folder = folders{f};
    disp(['Processing folder: ', folder]);

    % remove trailing separator
    folder = strip(folder, 'right', filesep);
    
    % get last and penultimate names
    [parent, last] = fileparts(folder);   % last = '10.2.2025_c1'
    [~, penult]   = fileparts(parent);    % penult = 'p3'
    
    % compose target name and sanitize for Windows
    leaf = sprintf('%s_%s', penult, last);            % 'p3_10.2.2025_c1'
    leaf = regexprep(leaf, '[<>:"/\\|?*]', '_');
    
    % build output path and ensure it exists
    out_folder = fullfile(out_root, leaf);
    if ~exist(out_folder,'dir'), mkdir(out_folder); end

    % Find all .mat files in the current folder
    mat_files = dir(fullfile(folder, '*.mat'));
    if isempty(mat_files)
        disp('No .mat files found. Skipping to next folder.');
        continue;
    end
    
    %% === Process each .mat file individually ===
    for i = 1:length(mat_files)
        data_file = fullfile(folder, mat_files(i).name);
        data = load(data_file);  % Load MATLAB structure containing matrix "Out"
        
        %% === EEG Filtering ===
        disp('Processing EEG data...');
        selected_channels = 18:33;
        eeg_filtered = data.Out(selected_channels, cut_off_samples+1:end);
        d_notch = designfilt('bandstopiir', 'FilterOrder', 2, ...
            'HalfPowerFrequency1', 49, 'HalfPowerFrequency2', 51, ...
            'SampleRate', fs);
        d_highpass = designfilt('highpassiir', 'FilterOrder', 4, ...
            'HalfPowerFrequency', 0.5, 'SampleRate', fs);
        d_lowpass = designfilt('lowpassiir', 'FilterOrder', 4, ...
            'HalfPowerFrequency', 40, 'SampleRate', fs);
        for ch = 1:size(eeg_filtered,1)
            eeg_filtered(ch,:) = filtfilt(d_notch, eeg_filtered(ch,:));
            eeg_filtered(ch,:) = filtfilt(d_highpass, eeg_filtered(ch,:));
            eeg_filtered(ch,:) = filtfilt(d_lowpass, eeg_filtered(ch,:));
        end
        save(fullfile(out_folder, 'EEG_filtered.mat'), 'eeg_filtered');  % changed
        
        %% === fNIRS Filtering ===
        disp('Processing fNIRS data...');
        fnirs_filtered = data.Out(2:17, cut_off_samples+1:end);
        bandpass_low = 0.01;
        bandpass_high = 0.5;
        [b, a] = ellip(2, 0.4, 20, [bandpass_low bandpass_high] / (fs / 2), 'bandpass');
        fnirs_filtered = filtfilt(b, a, fnirs_filtered);
        save(fullfile(out_folder, 'fNIRS_filtered.mat'), 'fnirs_filtered');  % changed
        
        %% === Pulse waveform Filtering ===
        disp('Processing pulse waveform...');
        pulse_signal = data.Out(34, cut_off_samples+1:end);
        bp_cutoff = [0.7 3];
        [b_bp, a_bp] = butter(4, bp_cutoff/(fs/2), 'bandpass');
        pulse_filtered = filtfilt(b_bp, a_bp, pulse_signal);
        save(fullfile(out_folder, 'Pulse_filtered.mat'), 'pulse_filtered');  % changed
        
        %% === GSR (Galvanic Skin Response) Processing ===
        disp('Processing GSR data...');
        GSR_file = fullfile(folder, 'GSR_EB_1D_52_12_93_8E.csv');
        if isfile(GSR_file)
            gsr_data = readmatrix(GSR_file);
            gsr_filtered = sgolayfilt(gsr_data(:,2), 4, 7);
            writematrix([gsr_data(:,1), gsr_filtered], ...
                fullfile(out_folder, 'GSR_filtered.csv'));                 % changed
        else
            disp('GSR file not found in this folder.');
        end
        
        %% === HR (Heart Rate) Processing ===
        disp('Processing HR data...');
        HR_file = fullfile(folder, 'HR_EC_71_B1_97_83_8E.csv');
        if isfile(HR_file)
            hr_data = readmatrix(HR_file);
            time_s = (hr_data(:,1) - hr_data(1,1)) / 1000;
            time_regular = min(time_s):0.5:max(time_s);
            hr_interpolated = interp1(time_s, hr_data(:,2), time_regular, 'makima');
            hr_filtered = sgolayfilt(hr_interpolated, 3, 9);
            writematrix([time_regular', hr_filtered'], ...
                fullfile(out_folder, 'HR_filtered.csv'));                  % changed
        else
            disp('HR file not found in this folder.');
        end
        
        %% === RR Interval Processing ===
        disp('Processing RR intervals...');
        RR_file = fullfile(folder, 'RR_EC_71_B1_97_83_8E.csv');
        if isfile(RR_file)
            rr_data = readmatrix(RR_file);
            time_new = (rr_data(1,1)/1000:1/2:rr_data(end,1)/1000)';
            [time_unique, idx] = unique(rr_data(:,1)/1000, 'stable');
            rr_values_unique = rr_data(idx,2);
            rr_interpolated = interp1(time_unique, rr_values_unique, time_new, 'pchip');
            [b, a] = butter(2, 0.17 / (2 / 2), 'low');
            rr_filtered = filtfilt(b, a, rr_interpolated);
            writematrix([time_new, rr_filtered], ...
                fullfile(out_folder, 'RR_filtered.csv'));                  % changed
        else
            disp('RR file not found in this folder.');
        end
        
        disp(['Filtering completed for file: ', mat_files(i).name, ...
              ' in folder: ', folder]);
 
            %% === Export first EVENT_* file ===
        % --- load first EVENT_* as table and save ---
        event_files = dir(fullfile(folder, '*EVENT_*'));
        if isempty(event_files)
            warning('No EVENT_* file in: %s', folder);
        else
            [~, ord] = sort({event_files.name});   % first by name
            ev = event_files(ord(1));
        
            src = fullfile(ev.folder, ev.name);
            opts = detectImportOptions(src, 'FileType','text', 'Delimiter',{',',';','\t'}, 'VariableNamingRule','preserve');
            T = readtable(src, opts);
            writetable(T, fullfile(out_folder, 'EVENT_selected.csv'));
        end



%%
        % out_root already exists
        this_dir     = fileparts(mfilename('fullpath'));
        project_root = fileparts(this_dir);
        ptr_main     = fullfile(project_root,'_last_out_root.txt');
        fid = fopen(ptr_main,'w'); fprintf(fid,'%s\n', out_root); fclose(fid);


    end
end

disp('All selected folders have been processed successfully!');
