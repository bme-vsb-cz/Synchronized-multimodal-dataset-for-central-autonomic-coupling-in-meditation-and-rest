% ========================================================================
%   MULTI-FOLDER GSR/HR/RR/PULSE SEGMENTATION SCRIPT
% ------------------------------------------------------------------------
%   Purpose:
%       Process many folders with pre-filtered signals (GSR, HR, RR, Pulse).
%       Align all time axes to GSR, find MEDITATION START/END from EVENT_*,
%       split signals into three phases, and save tables per folder.
%
%   Expected input per folder:
%       - GSR_filtered.csv          (timestamp [ms], GSR value)
%       - HR_filtered.csv           (time [s], HR value)
%       - RR_filtered.csv           (time [s] OR epoch [s], RR interval)
%       - Pulse_filtered.mat        (variable: pulse_filtered, Fs = 250 Hz)
%       - EVENT_*.csv               (timestamp [ms], name, type)
%
%   Output:
%       - GSR_HR_RR_Pulse_segments.mat with tables:
%           before_meditation, meditation, after_meditation
%
%   Notes:
%       - HR time is relative seconds; RR time can be relative OR epoch.
%       - All signals are interpolated to GSR time vector for sync.
%
% --- QUICK VISUALIZATION (optional, shown once after processing) ---------
%   What it shows:
%       • Top: GSR with vertical START/END lines and highlighted window.
%       • Below: selected modalities (HR, RR, Pulse) on the same x-axis.
%       • Optional per-series z-score normalization.
%   How to use:
%       After processing confirm the prompt → pick one processed folder
%       → choose which modalities to display and normalization Yes/No.
%   No files are written by the viewer.
% ========================================================================

clear; clc; close all;

%% === Resolve out_root without prompts ===
this_dir     = fileparts(mfilename('fullpath'));
project_root = fileparts(this_dir);
ptr_main     = fullfile(project_root, '_last_out_root.txt');
ptr_local    = fullfile(this_dir,    '_last_out_root.txt');

out_root = '';
if exist(ptr_main,'file')
    out_root = strtrim(fileread(ptr_main));
elseif exist(ptr_local,'file')
    out_root = strtrim(fileread(ptr_local));
end

% auto-detection
if isempty(out_root) || ~exist(out_root,'dir')
    d = dir(fullfile(project_root,'Processed_*'));
    d = d([d.isdir]);
    if ~isempty(d)
        [~,ix] = max([d.datenum]);
        out_root = fullfile(d(ix).folder, d(ix).name);
    else
        error('out_root not resolved: pointer missing and no Processed_* under %s.', project_root);
    end
end

%% === Build list of input folders under out_root ===
required = {'GSR_filtered.csv','HR_filtered.csv','RR_filtered.csv','Pulse_filtered.mat'};
hits = dir(fullfile(out_root, '**', 'GSR_filtered.csv'));   % recursive
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

    % -------------------- 2.1 LOAD SIGNALS ------------------------------
    gsr_data = readmatrix(fullfile(input_folder, 'GSR_filtered.csv'));
    timestamps = datetime(gsr_data(:,1) / 1000, 'ConvertFrom','posixtime', ...
                          'Format','yyyy-MM-dd HH:mm:ss.SSS');
    gsr_values = gsr_data(:,2);

    hr_data = readmatrix(fullfile(input_folder, 'HR_filtered.csv'));
    t_hr = hr_data(:,1);           % seconds (relative)
    hr_values = hr_data(:,2);

    rr_data = readmatrix(fullfile(input_folder, 'RR_filtered.csv'));
    t_rr_raw = rr_data(:,1);       % seconds (relative) OR epoch seconds
    rr_values = rr_data(:,2);

    P = load(fullfile(input_folder, 'Pulse_filtered.mat'));
    pulse_values = P.pulse_filtered(:)';

    % -------------------- 2.2 CREATE TIME VECTORS -----------------------
    Fs_pulse = 250;
    Np = numel(pulse_values);
    t_pulse = (0:Np-1)/Fs_pulse;   % [s], starts at 0

    % -------------------- 2.3 ALIGN TO GSR REFERENCE --------------------
    start_time = timestamps(1);
    t_gsr = seconds(timestamps - start_time);     % [s], 0 ... T

    % RR time can be epoch seconds or relative seconds — detect:
    if max(t_rr_raw) > 1e8                      % looks like epoch seconds
        t_rr = seconds(datetime(t_rr_raw,'ConvertFrom','posixtime') - start_time);
    else                                        % treat as relative seconds
        % align RR end to GSR end
        t_rr = t_rr_raw - t_rr_raw(end) + t_gsr(end);
    end

    % Align HR and Pulse by end to GSR (both are relative seconds)
    t_hr = t_hr - t_hr(end) + t_gsr(end);
    t_pulse = t_pulse - t_pulse(end) + t_gsr(end);

    % -------------------- 2.4 INTERPOLATE TO t_gsr ----------------------
    hr_values_interp    = interp1(t_hr,    hr_values,    t_gsr, 'linear', 'extrap');
    rr_values_interp    = interp1(t_rr,    rr_values,    t_gsr, 'linear', 'extrap');
    pulse_values_interp = interp1(t_pulse, pulse_values, t_gsr, 'linear', 'extrap');

    % -------------------- 2.5 EVENTS: START/END -------------------------
    event_files = dir(fullfile(input_folder, '*EVENT_*'));
    if isempty(event_files)
        warning('No file containing "EVENT_" found in folder: %s', input_folder);
        continue;
    end
    event_filename = fullfile(input_folder, event_files(1).name);
    fprintf('Loading events from file: %s\n', event_filename);

    E  = readtable(event_filename, 'VariableNamingRule','preserve');
    event_times = datetime(E{:,1}/1000, 'ConvertFrom','posixtime', ...
                           'Format','yyyy-MM-dd HH:mm:ss.SSS');
    event_names = E{:,2};
    event_types = E{:,3};
    event_sec   = seconds(event_times - start_time);

    is_start = strcmp(event_names,'MEDITATION') & strcmp(event_types,'START');
    is_end   = strcmp(event_names,'MEDITATION') & strcmp(event_types,'END');
    if sum(is_start)==0 || sum(is_end)==0
        warning('Missing START or END of meditation in event file: %s', input_folder);
        continue;
    end
    t_start_meditation = event_sec(is_start);
    t_end_meditation   = event_sec(is_end);

    % -------------------- 2.6 PHASE MASKS -------------------------------
    mask_before     = t_gsr <  t_start_meditation;
    mask_meditation = (t_gsr >= t_start_meditation) & (t_gsr <= t_end_meditation);
    mask_after      = t_gsr >  t_end_meditation;

    % -------------------- 2.7 TABLES ------------------------------------
    before_meditation = table(t_gsr(mask_before), gsr_values(mask_before), ...
        hr_values_interp(mask_before), rr_values_interp(mask_before), ...
        pulse_values_interp(mask_before), ...
        'VariableNames', {'Time_s','GSR_uS','HR_bpm','RR_ms','Pulse'});

    meditation = table(t_gsr(mask_meditation), gsr_values(mask_meditation), ...
        hr_values_interp(mask_meditation), rr_values_interp(mask_meditation), ...
        pulse_values_interp(mask_meditation), ...
        'VariableNames', {'Time_s','GSR_uS','HR_bpm','RR_ms','Pulse'});

    after_meditation = table(t_gsr(mask_after), gsr_values(mask_after), ...
        hr_values_interp(mask_after), rr_values_interp(mask_after), ...
        pulse_values_interp(mask_after), ...
        'VariableNames', {'Time_s','GSR_uS','HR_bpm','RR_ms','Pulse'});

    % -------------------- 2.8 SAVE --------------------------------------
    save(fullfile(input_folder, 'GSR_HR_RR_Pulse_segments.mat'), ...
         'before_meditation','meditation','after_meditation');

    disp(['Data successfully saved in folder: ', input_folder]);
end

%% === Step 3: Final completion message ===
disp('All folders have been processed successfully!');

%% === OPTIONAL: one-off multi-signal viewer =============================
resp = questdlg('Show a multi-signal segmentation visualization now?', ...
                'Segmentation Viewer', 'Yes','No','Yes');
if strcmp(resp,'Yes')
    try
        % pick any folder that contains the saved segments
        cand = dir(fullfile(out_root, '**', 'GSR_HR_RR_Pulse_segments.mat'));
        vis_folders = unique({cand.folder});
        if isempty(vis_folders)
            errordlg('No folders with GSR_HR_RR_Pulse_segments.mat found under out_root.','No Data');
        else
            [iSel, ok] = listdlg('PromptString','Pick ONE folder to visualize:', ...
                'SelectionMode','single', 'ListString',vis_folders, 'ListSize',[700 400]);
            if ok
                input_folder = vis_folders{iSel};
                % choose modalities and normalization
                [iMod, ok2] = listdlg('PromptString','Select modalities to show:', ...
                    'SelectionMode','multiple', ...
                    'ListString',{'HR','RR','Pulse'}, 'InitialValue',[1 2 3], ...
                    'ListSize',[300 150]);
                if ok2
                    modes = {'HR','RR','Pulse'}; modes = modes(iMod);
                    norm_choice = questdlg('Normalize each series (z-score)?','Normalization','No','Yes','No');
                    do_norm = strcmp(norm_choice,'Yes');
                    show_segmentation_figure_multi(input_folder, modes, do_norm);
                end
            end
        end
    catch ME
        warning(ME.identifier, 'Segmentation viewer failed: %s', ME.message);
    end
end

% ========================= LOCAL FUNCTIONS ===============================

function show_segmentation_figure_multi(input_folder, modes, do_norm)
% Load everything again and draw GSR + selected modalities on one x-axis.

% --- GSR
G  = readmatrix(fullfile(input_folder,'GSR_filtered.csv'));
ts = datetime(G(:,1)/1000,'ConvertFrom','posixtime','Format','yyyy-MM-dd HH:mm:ss.SSS');
g  = G(:,2);
t0 = ts(1);
t  = seconds(ts - t0);   % relative seconds for plotting

% --- HR
H  = readmatrix(fullfile(input_folder,'HR_filtered.csv'));
thr = H(:,1); vhr = H(:,2);
thr = thr - thr(end) + t(end);     % align end to GSR end

% --- RR
R  = readmatrix(fullfile(input_folder,'RR_filtered.csv'));
trr_raw = R(:,1); vrr = R(:,2);
if max(trr_raw) > 1e8                         % epoch seconds
    trr = seconds(datetime(trr_raw,'ConvertFrom','posixtime') - t0);
else
    trr = trr_raw - trr_raw(end) + t(end);    % relative -> align end
end

% --- Pulse
P  = load(fullfile(input_folder,'Pulse_filtered.mat'));
vp = P.pulse_filtered(:)'; Fs = 250; tp = (0:numel(vp)-1)/Fs;
tp = tp - tp(end) + t(end);

% --- Events
ev = dir(fullfile(input_folder,'*EVENT_*'));
if isempty(ev)
    errordlg('EVENT_* not found in selected folder.','Missing events'); return;
end
T  = readtable(fullfile(ev(1).folder, ev(1).name), 'VariableNamingRule','preserve');
et = datetime(T{:,1}/1000,'ConvertFrom','posixtime','Format','yyyy-MM-dd HH:mm:ss.SSS');
en = T{:,2}; ty = T{:,3};
t_start = seconds(et(strcmp(en,'MEDITATION') & strcmp(ty,'START')) - t0);
t_end   = seconds(et(strcmp(en,'MEDITATION') & strcmp(ty,'END'))   - t0);
if isempty(t_start) || isempty(t_end)
    errordlg('Meditation START/END not found in events.','Missing markers'); return;
end

% optional normalization
if do_norm
    z = @(x) (x-mean(x,'omitnan'))./max(std(x,0,'omitnan'),eps);
    g = z(g); vhr = z(vhr); vrr = z(vrr); vp = z(vp);
end

% figure layout
nPanels = 1 + numel(modes);
f = figure('Name','GSR + HR/RR/Pulse segmentation','Color','w');
tl = tiledlayout(f,nPanels,1,'TileSpacing','compact','Padding','compact');

% GSR
ax1 = nexttile(tl,1);
plot(ax1, t, g, 'k-'); grid(ax1,'on'); ylabel(ax1,'GSR');
title(ax1,'GSR with meditation markers');
xline(ax1, t_start, 'b--', 'START', 'LabelHorizontalAlignment','left');
xline(ax1, t_end,   'r--', 'END',   'LabelHorizontalAlignment','left');
yl = ylim(ax1);
patch(ax1, [t_start t_end t_end t_start], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.95 1.0], 'FaceAlpha',0.2, 'EdgeColor','none');

% other modalities
panel = 2;
for m = 1:numel(modes)
    ax = nexttile(tl,panel); grid(ax,'on'); hold(ax,'on');
    switch modes{m}
        case 'HR'
            plot(ax, thr, vhr, 'b-');
            ylabel(ax,'HR [bpm]'); title(ax,'HR');
        case 'RR'
            plot(ax, trr, vrr, 'm-');
            ylabel(ax,'RR [ms]'); title(ax,'RR');
        case 'Pulse'
            plot(ax, tp, vp, 'r-');
            ylabel(ax,'Pulse [a.u.]'); title(ax,'Pulse');
    end
    xline(ax, t_start, 'b--'); xline(ax, t_end, 'r--');
    yl2 = ylim(ax);
    patch(ax, [t_start t_end t_end t_start], [yl2(1) yl2(1) yl2(2) yl2(2)], ...
          [0.9 0.95 1.0], 'FaceAlpha',0.15, 'EdgeColor','none');
    panel = panel + 1;
end

xlabel(tl,'Time [s]');
linkaxes(findall(f,'Type','axes'),'x');
end
