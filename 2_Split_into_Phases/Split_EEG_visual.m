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
%
% --- QUICK VISUALIZATION (optional, shown once after processing) ---------
%   What it shows:
%       • Top: GSR trace with vertical lines (START/END of meditation).
%       • Bottom: chosen EEG channels overlaid, with the meditation window
%         highlighted in the same time axis. Optional per-channel z-score.
%   How to use:
%       After processing, confirm the prompt → pick one processed folder
%       (must contain EEG_phases.mat) → select EEG channels to show
%       (e.g., 1,3,5 or 1:4 or 'all') and normalization Yes/No.
%   Files are NOT written; this is only for visual inspection.
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
        y1 = length(gsr_values) / Fs_GSR; %#ok<NASGU> % Duration in seconds
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
    end_time = ts_gsr(end); % last GSR timestamp as reference
    start_time_eeg = end_time - seconds(size(EEG_data, 2) / Fs_EEG);

    % Create EEG time vector corresponding to sample indices
    ts_eeg = start_time_eeg + seconds((0:size(EEG_data, 2)-1) / Fs_EEG);
    y2 = size(EEG_data, 2) / Fs_EEG; %#ok<NASGU> % EEG recording duration

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
    meditation        = EEG_data(:, ts_eeg >= t_start_meditation & ts_eeg <= t_end_meditation);
    after_meditation  = EEG_data(:, ts_eeg >  t_end_meditation);

    %% === Save segmented EEG data ===
    save(fullfile(input_folder, 'EEG_phases.mat'), ...
         'before_meditation', 'meditation', 'after_meditation');

    disp(['Data successfully saved in folder: ', input_folder]);
end

%% === Final message ===
disp('All folders processed successfully!');

%% === OPTIONAL: one-off segmentation viewer (GSR + EEG) =================
resp = questdlg('Show a segmentation visualization now?', ...
                'Segmentation Viewer', 'Yes','No','Yes');
if strcmp(resp,'Yes')
    try
        % list folders that now contain EEG_phases.mat
        cand = dir(fullfile(out_root, '**', 'EEG_phases.mat'));
        vis_folders = unique({cand.folder});
        if isempty(vis_folders)
            errordlg('No folders with EEG_phases.mat found under out_root.','No Data');
        else
            % pick one folder
            [iSel, ok] = listdlg('PromptString','Pick ONE folder to visualize:', ...
                'SelectionMode','single', 'ListString',vis_folders, 'ListSize',[700 400]);
            if ok
                input_folder = vis_folders{iSel};
                % channel selection and normalization
                ch_ans = inputdlg({'EEG channels to show (e.g. 1,3,5 or 1:4 or all):', ...
                                   'Normalize per-channel (0/1):'}, ...
                                   'Viewer options', 1, {'1:4','0'});
                if ~isempty(ch_ans)
                    ch_list = parse_channels_local(ch_ans{1}, size(load(fullfile(input_folder,'EEG_filtered.mat'),'eeg_filtered').eeg_filtered,1));
                    do_norm = str2double(ch_ans{2})~=0;
                    show_segmentation_figure(input_folder, ch_list, do_norm);
                end
            end
        end
    catch ME
        warning(ME.identifier, 'Segmentation viewer failed: %s', ME.message);
    end
end

% ========================= LOCAL FUNCTIONS ===============================

function ch = parse_channels_local(txt, nCh)
s = lower(strtrim(txt)); s = strrep(s,' ','');
if strcmp(s,'all'), ch = 1:nCh; return; end
ch = [];
parts = split(string(s), ',');
for p = parts.'
    token = char(p);
    if contains(token, ':')
        bounds = split(string(token), ':');
        if numel(bounds) == 2
            a = str2double(bounds(1)); b = str2double(bounds(2));
            if ~isnan(a) && ~isnan(b), ch = [ch, a:b]; end %#ok<AGROW>
        end
    else
        v = str2double(token);
        if ~isnan(v), ch = [ch, v]; end %#ok<AGROW>
    end
end
ch = unique(ch); ch = ch(ch>=1 & ch<=nCh);
if isempty(ch), ch = 1; end
end

function show_segmentation_figure(input_folder, ch_list, do_norm)
Fs_EEG = 250; Fs_GSR = 10; %#ok<NASGU> (Fs_GSR není nutné, čas je v ms)

% --- load inputs
S = load(fullfile(input_folder,'EEG_filtered.mat'),'eeg_filtered');
EEG = S.eeg_filtered;
G = readmatrix(fullfile(input_folder,'GSR_filtered.csv'));
ts_gsr = datetime(G(:,1)/1000,'ConvertFrom','posixtime','Format','yyyy-MM-dd HH:mm:ss.SSS');
gsr = G(:,2);

% events
ev = dir(fullfile(input_folder,'*EVENT_*'));
if isempty(ev)
    errordlg('EVENT_* not found in selected folder.','Missing events'); return;
end
T = readtable(fullfile(ev(1).folder, ev(1).name), 'VariableNamingRule','preserve');
et = datetime(T{:,1}/1000,'ConvertFrom','posixtime','Format','yyyy-MM-dd HH:mm:ss.SSS');
en = T{:,2}; ty = T{:,3};
t_start = et(strcmp(en,'MEDITATION') & strcmp(ty,'START'));
t_end   = et(strcmp(en,'MEDITATION') & strcmp(ty,'END'));
if isempty(t_start) || isempty(t_end)
    errordlg('Meditation START/END not found in events.','Missing markers'); return;
end

% align EEG time to GSR end (same logic as in segmentation)
end_time = ts_gsr(end);
t_eeg = end_time - seconds((size(EEG,2)-1)/Fs_EEG) + seconds((0:size(EEG,2)-1)/Fs_EEG);

% optional z-score per channel
if do_norm
    mu = mean(EEG,2,'omitnan'); sd = std(EEG,0,2,'omitnan'); sd(sd==0)=1;
    EEG = (EEG - mu)./sd;
    gsr = (gsr - mean(gsr))/std(gsr);
end

% figure with two tiles
f = figure('Name','GSR + EEG segmentation','Color','w');
tl = tiledlayout(f,2,1,'TileSpacing','compact','Padding','compact');

% --- GSR panel
ax1 = nexttile(tl,1);
plot(ax1, ts_gsr, gsr, 'k-'); grid(ax1,'on');
ylabel(ax1,'GSR'); title(ax1,'GSR with meditation markers');
xline(ax1, t_start, 'b--', 'START', 'LabelHorizontalAlignment','left');
xline(ax1, t_end,   'r--', 'END',   'LabelHorizontalAlignment','left');

% shade meditation interval
yl = ylim(ax1);
patch(ax1, [t_start t_end t_end t_start], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.95 1.0], 'FaceAlpha',0.2, 'EdgeColor','none');

% --- EEG panel (selected channels)
ax2 = nexttile(tl,2); hold(ax2,'on'); grid(ax2,'on');
colors = lines(numel(ch_list));
for i = 1:numel(ch_list)
    ch = ch_list(i);
    plot(ax2, t_eeg, EEG(ch,:), 'Color', colors(i,:), 'DisplayName', sprintf('EEG ch %d', ch));
end
ylabel(ax2,'EEG'); xlabel(ax2,'Time');
title(ax2,'EEG with meditation window');
xline(ax2, t_start, 'b--', 'START', 'LabelHorizontalAlignment','left');
xline(ax2, t_end,   'r--', 'END',   'LabelHorizontalAlignment','left');
yl2 = ylim(ax2);
patch(ax2, [t_start t_end t_end t_start], [yl2(1) yl2(1) yl2(2) yl2(2)], ...
      [0.9 0.95 1.0], 'FaceAlpha',0.15, 'EdgeColor','none');
legend(ax2,'show','Location','best');

linkaxes([ax1 ax2],'x');
end
