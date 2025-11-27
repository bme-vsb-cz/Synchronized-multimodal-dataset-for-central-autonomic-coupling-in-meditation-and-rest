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
% QUICK VISUALIZATION (RAW vs FILTERED) + PSD (PRE/POST není nutné)
% -------------------------------------------------------------------------
% Volitelná prohlížečka umí:
%   • Overlay RAW vs FILTERED v čase
%   • Navíc otevře druhé okno s Welch PSD (průměr přes vybrané kanály)
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

%% === OUTPUT ROOT derived from selected root ===
root = strip(root,'right',filesep);
[parentRoot, rootName] = fileparts(root);     % parent of the picked folder

% outputs as a sibling of the picked folder (under its parent)
out_root = fullfile(parentRoot, ['Filtered_' rootName '_' datestr(now,'yyyy.mm.dd')]);
if ~exist(out_root,'dir'), mkdir(out_root); end

% Define sampling frequency and discard initial transient
fs = 250;
cut_off_samples = 20 * fs;

% Map pro pozdější prohlížečku: out_leaf -> raw_folder
leaf2raw = containers.Map('KeyType','char','ValueType','char');

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

    % -- záznam mapy RAW -> OUT pro pozdější overlay —
    try
        leaf2raw(leaf) = folder;
    catch
        % ignore if duplicate
    end

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
        eeg_filtered   = data.Out(selected_channels, cut_off_samples+1:end);
        d_notch = designfilt('bandstopiir', 'FilterOrder', 2, ...
            'HalfPowerFrequency1', 49, 'HalfPowerFrequency2', 51, ...
            'SampleRate', fs);
        d_highpass = designfilt('highpassiir', 'FilterOrder', 4, ...
            'HalfPowerFrequency', 0.5, 'SampleRate', fs);
        d_lowpass = designfilt('lowpassiir', 'FilterOrder', 4, ...
            'HalfPowerFrequency', 40, 'SampleRate', fs);
        for ch = 1:size(eeg_filtered,1)
            eeg_filtered(ch,:) = filtfilt(d_notch,    eeg_filtered(ch,:));
            eeg_filtered(ch,:) = filtfilt(d_highpass, eeg_filtered(ch,:));
            eeg_filtered(ch,:) = filtfilt(d_lowpass,  eeg_filtered(ch,:));
        end
        save(fullfile(out_folder, 'EEG_filtered.mat'), 'eeg_filtered');

        %% === fNIRS Filtering ===
        disp('Processing fNIRS data...');
        fnirs_filtered = data.Out(2:17, cut_off_samples+1:end); 
        bandpass_low = 0.01;
        bandpass_high = 0.5;
        [b, a] = ellip(2, 0.4, 20, [bandpass_low bandpass_high] / (fs / 2), 'bandpass');
        fnirs_filtered = filtfilt(b, a, fnirs_filtered);
        save(fullfile(out_folder, 'fNIRS_filtered.mat'), 'fnirs_filtered');

        %% === Pulse waveform Filtering ===
        disp('Processing pulse waveform...');
        pulse_signal = data.Out(34, cut_off_samples+1:end);
        bp_cutoff = [0.7 3];
        [b_bp, a_bp] = butter(4, bp_cutoff/(fs/2), 'bandpass');
        pulse_filtered = filtfilt(b_bp, a_bp, pulse_signal);
        save(fullfile(out_folder, 'Pulse_filtered.mat'), 'pulse_filtered');

        %% === GSR (Galvanic Skin Response) Processing ===
        disp('Processing GSR data...');
        GSR_file = fullfile(folder, 'GSR_EB_1D_52_12_93_8E.csv');
        if isfile(GSR_file)
            gsr_data = readmatrix(GSR_file);
            gsr_filtered = sgolayfilt(gsr_data(:,2), 4, 7);
            writematrix([gsr_data(:,1), gsr_filtered], ...
                fullfile(out_folder, 'GSR_filtered.csv'));
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
                fullfile(out_folder, 'HR_filtered.csv'));
        else
            disp('HR file not found in this folder.');
        end

        %% === RR Interval Processing ===
        disp('Processing RR intervals...');
        RR_file = fullfile(folder, 'RR_EC_71_B1_97_83_8E.csv');
        if isfile(RR_file)
            rr_data = readmatrix(RR_file);
            time_new = (rr_data(1,1)/1000:1/2:rr_data(end,1)/1000)'; % 2 Hz
            [time_unique, idxU] = unique(rr_data(:,1)/1000, 'stable');
            rr_values_unique = rr_data(idxU,2);
            rr_interpolated = interp1(time_unique, rr_values_unique, time_new, 'pchip');
            [b_rr, a_rr] = butter(2, 0.17 / (2 / 2), 'low'); % fc=0.17 Hz @ Fs=2 Hz
            rr_filtered = filtfilt(b_rr, a_rr, rr_interpolated);
            writematrix([time_new, rr_filtered], ...
                fullfile(out_folder, 'RR_filtered.csv'));
        else
            disp('RR file not found in this folder.');
        end

        disp(['Filtering completed for file: ', mat_files(i).name, ...
              ' in folder: ', folder]);

        %% === Export first EVENT_* file ===
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
    end
end

%% === Write pointer to the last out_root (once) ===
this_dir     = fileparts(mfilename('fullpath'));
project_root = fileparts(this_dir);
ptr_main     = fullfile(project_root,'_last_out_root.txt');
fid = fopen(ptr_main,'w'); fprintf(fid,'%s\n', out_root); fclose(fid);

disp('All selected folders have been processed successfully!');

%% === OPTIONAL: one-off overlay viewer (RAW vs FILTERED) =================
choice = questdlg('Show a RAW vs FILTERED overlay now?', ...
                  'Quick Viewer', 'Yes','No','Yes');
if strcmp(choice,'Yes')
    try
        quick_overlay_viewer(out_root, leaf2raw, fs, cut_off_samples);
    catch ME
        warning(ME.identifier, 'Overlay viewer failed: %s', ME.message);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       LOCAL FUNCTIONS BELOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function quick_overlay_viewer(out_root, leaf2raw, fs, cutN)
    % Najdi záznamy se zpracovanými výstupy
    dd = dir(out_root); dd = dd([dd.isdir]);
    names = setdiff({dd.name},{'.','..'});
    % filtr: složky, kde existuje aspoň jeden _filtered.* soubor
    recs = {};
    for i=1:numel(names)
        of = fullfile(out_root, names{i});
        if exist(fullfile(of,'EEG_filtered.mat'),'file') || ...
           exist(fullfile(of,'fNIRS_filtered.mat'),'file') || ...
           exist(fullfile(of,'Pulse_filtered.mat'),'file') || ...
           exist(fullfile(of,'GSR_filtered.csv'),'file') || ...
           exist(fullfile(of,'HR_filtered.csv'),'file')  || ...
           exist(fullfile(of,'RR_filtered.csv'),'file')
            recs{end+1} = names{i}; %#ok<AGROW>
        end
    end
    if isempty(recs)
        errordlg('No filtered records found in out_root.','No Data');
        return;
    end

    [iSel, ok] = listdlg('PromptString','Pick ONE record to visualize:', ...
        'SelectionMode','single', 'ListSize',[600 400], 'ListString',recs);
    if ~ok, return; end
    leaf = recs{iSel};
    out_folder = fullfile(out_root, leaf);

    mods = {'EEG','fNIRS','Pulse','GSR','HR','RR'};
    [iMod, ok] = listdlg('PromptString','Pick modality:', ...
        'SelectionMode','single','ListString',mods,'ListSize',[300 170]);
    if ~ok, return; end
    modality = mods{iMod};

    % volitelná normalizace
    norm_choice = questdlg('Normalize per-channel (z-score) for overlay?', ...
                           'Normalization', 'No','Yes','No');
    do_norm = strcmp(norm_choice,'Yes');

    % kanály pro EEG/fNIRS
    ch_list = [];
    nGuess = infer_nch(out_folder, modality);
    if ismember(modality, {'EEG','fNIRS'})
        prompt = sprintf('%s channels (e.g. 1,3,5 or 1:4 or all), 1..%d:', modality, nGuess);
        a2 = inputdlg({prompt}, 'Channels', 1, {'1:4'});
        if isempty(a2), return; end
        ch_list = parse_channels(a2{1}, nGuess);
        if isempty(ch_list), ch_list = 1; end
    end

    % dohledání RAW složky z mapy
    raw_folder = '';
    if isKey(leaf2raw, leaf), raw_folder = leaf2raw(leaf); end

    switch modality
        case 'EEG'
            plot_eeg(out_folder, raw_folder, ch_list, fs, cutN, do_norm);
        case 'fNIRS'
            plot_fnirs(out_folder, raw_folder, ch_list, fs, cutN, do_norm);
        case 'Pulse'
            plot_pulse(out_folder, raw_folder, fs, cutN, do_norm);
        case 'GSR'
            plot_gsr(out_folder, raw_folder, do_norm);
        case 'HR'
            plot_hr(out_folder, raw_folder, do_norm);
        case 'RR'
            plot_rr(out_folder, raw_folder, do_norm);
    end
end

function n = infer_nch(out_folder, modality)
    n = 16;
    try
        switch modality
            case 'EEG'
                S = load(fullfile(out_folder,'EEG_filtered.mat'),'eeg_filtered');
                if isfield(S,'eeg_filtered'), n = size(S.eeg_filtered,1); end
            case 'fNIRS'
                S = load(fullfile(out_folder,'fNIRS_filtered.mat'),'fnirs_filtered');
                if isfield(S,'fnirs_filtered'), n = size(S.fnirs_filtered,1); end
        end
    catch 
    end
end

function ch = parse_channels(txt, nCh)
    s = lower(strtrim(txt));
    s = strrep(s,' ','');
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
    ch = unique(ch);
    ch = ch(ch>=1 & ch<=nCh);
end

function [Out, src] = load_first_out(raw_folder)
    Out = []; src = '';
    if isempty(raw_folder) || ~isfolder(raw_folder), return; end
    m = dir(fullfile(raw_folder,'*.mat'));
    for i = 1:numel(m)
        S = load(fullfile(m(i).folder, m(i).name), 'Out');
        if isfield(S,'Out')
            Out = S.Out; src = fullfile(m(i).folder, m(i).name); return;
        end
    end
end

function [ok, path] = find_first_file(folder, patterns)
    ok = false; path = '';
    if isempty(folder) || ~isfolder(folder), return; end
    for i = 1:numel(patterns)
        g = dir(fullfile(folder, patterns{i}));
        if ~isempty(g)
            path = fullfile(g(1).folder, g(1).name);
            ok = true; return;
        end
    end
end

function [raw,filt,t] = align_series(raw, filt, fs)
    L = min(size(raw,2), size(filt,2));
    raw  = raw(:,1:L);
    filt = filt(:,1:L);
    t    = (0:L-1)/fs;
end

function X = zscore_rows(X)
    mu = mean(X,2,'omitnan');
    sd = std(X,0,2,'omitnan'); sd(sd==0) = 1;
    X = (X - mu)./sd;
end

% --------- Plotters (EEG / fNIRS multi-channel; others single) ----------
function plot_eeg(out_folder, raw_folder, ch_list, fs, cutN, do_norm)
    idx = 18:33;
    S = load(fullfile(out_folder,'EEG_filtered.mat'),'eeg_filtered');
    if ~isfield(S,'eeg_filtered'), errordlg('Missing EEG_filtered.mat'); return; end
    filt = S.eeg_filtered;

    raw = [];
    if ~isempty(raw_folder)
        [Out,~] = load_first_out(raw_folder);
        if ~isempty(Out)
            raw = Out(idx,:);
            if size(raw,2) > cutN, raw = raw(:,cutN+1:end); end
            [raw, filt, t] = align_series(raw, filt, fs);
        end
    end
    if isempty(raw)
        t = (0:size(filt,2)-1)/fs;
    end

    ch_list = unique(ch_list);
    ch_list = ch_list(ch_list>=1 & ch_list<=size(filt,1));
    if isempty(ch_list), ch_list = 1; end

    if do_norm
        if ~isempty(raw), raw = zscore_rows(raw); end
        filt = zscore_rows(filt);
    end

    n = numel(ch_list); ncols = min(3,n); nrows = ceil(n/ncols);
    f = figure('Name','EEG Overlay','Color','w');
    tl = tiledlayout(f,nrows,ncols,'TileSpacing','compact','Padding','compact');
    title(tl,'EEG Overlay (RAW vs FILTERED)');
    for j=1%n
        ch = ch_list(j);
        ax = nexttile(tl);
        if ~isempty(raw), plot(ax,t,raw(ch,:),'k-','DisplayName','RAW'); hold(ax,'on'); else, hold(ax,'on'); end
        plot(ax,t,filt(ch,:),'r-','DisplayName','FILTERED');
        grid(ax,'on'); xlabel(ax,'Time [s]'); ylabel(ax,sprintf('EEG ch %d',ch));
        if j==1, legend(ax,'show','Location','best'); end
    end

    % === PSD (RAW vs FILTERED), 0.5–40 Hz ===
    raw_eeg = [];
    if ~isempty(raw), raw_eeg = raw(ch_list,:); end
    plot_psd_avg(raw_eeg, filt(ch_list,:), fs, [0.5 100], ...
        sprintf('EEG Welch PSD (avg ch = [%s])', num2str(ch_list)), 'PSD [dB/Hz]');
end

function plot_fnirs(out_folder, raw_folder, ch_list, fs, cutN, do_norm)
    idx = 2:17;
    S = load(fullfile(out_folder,'fNIRS_filtered.mat'),'fnirs_filtered');
    if ~isfield(S,'fnirs_filtered'), errordlg('Missing fNIRS_filtered.mat'); return; end
    filt = S.fnirs_filtered;

    raw = [];
    if ~isempty(raw_folder)
        [Out,~] = load_first_out(raw_folder);
        if ~isempty(Out)
            raw = Out(idx,:);
            if size(raw,2) > cutN, raw = raw(:,cutN+1:end); end
            [raw, filt, t] = align_series(raw, filt, fs);
        end
    end
    if isempty(raw)
        t = (0:size(filt,2)-1)/fs;
    end

    ch_list = unique(ch_list);
    ch_list = ch_list(ch_list>=1 & ch_list<=size(filt,1));
    if isempty(ch_list), ch_list = 1; end

    if do_norm
        if ~isempty(raw), raw = zscore_rows(raw); end
        filt = zscore_rows(filt);
    end

    n = numel(ch_list); ncols = min(3,n); nrows = ceil(n/ncols);
    f = figure('Name','fNIRS Overlay','Color','w');
    tl = tiledlayout(f,nrows,ncols,'TileSpacing','compact','Padding','compact');
    title(tl,'fNIRS Overlay (RAW vs FILTERED)');
    for j=1:n
        ch = ch_list(j);
        ax = nexttile(tl);
        if ~isempty(raw), plot(ax,t,raw(ch,:),'k-','DisplayName','RAW'); hold(ax,'on'); else, hold(ax,'on'); end
        plot(ax,t,filt(ch,:),'r-','DisplayName','FILTERED');
        grid(ax,'on'); xlabel(ax,'Time [s]'); ylabel(ax,sprintf('fNIRS ch %d',ch));
        if j==1, legend(ax,'show','Location','best'); end
    end

    % === PSD (RAW vs FILTERED), 0.01–0.5 Hz ===
    raw_fn = [];
    if ~isempty(raw), raw_fn = raw(ch_list,:); end
    plot_psd_avg(raw_fn, filt(ch_list,:), fs, [0.01 1], ...
        sprintf('fNIRS Welch PSD (avg ch = [%s])', num2str(ch_list)), 'PSD [dB/Hz]');
end

function plot_pulse(out_folder, raw_folder, fs, cutN, do_norm)
    P = load(fullfile(out_folder,'Pulse_filtered.mat'),'pulse_filtered');
    if ~isfield(P,'pulse_filtered'), errordlg('Missing Pulse_filtered.mat'); return; end
    filt = P.pulse_filtered(:)';

    raw = [];
    if ~isempty(raw_folder)
        [Out,~] = load_first_out(raw_folder);
        if ~isempty(Out)
            raw = Out(34,:);
            if numel(raw) > cutN, raw = raw(cutN+1:end); end
            L = min(numel(raw), numel(filt));
            raw  = raw(1:L); filt = filt(1:L);
        end
    end

    if do_norm
        if ~isempty(raw), raw = (raw-mean(raw))/std(raw); end
        filt = (filt-mean(filt))/std(filt);
    end

    t = (0:numel(filt)-1)/fs;
    figure('Name','Pulse Overlay','Color','w');
    if ~isempty(raw), plot(t,raw,'k-','DisplayName','RAW'); hold on; else, hold on; end
    plot(t,filt,'r-','DisplayName','FILTERED'); grid on;
    xlabel('Time [s]'); ylabel('Pulse [a.u.]'); title('Pulse Overlay (RAW vs FILTERED)');
    legend('show','Location','best');

    % === PSD (RAW vs FILTERED), 0.5–10 Hz ===
    raw_pl = [];
    if ~isempty(raw), raw_pl = raw; end
    plot_psd_avg(raw_pl, filt, fs, [0.5 10], 'Pulse Welch PSD', 'PSD [dB/Hz]');
end

function plot_gsr(out_folder, raw_folder, do_norm)
    fp = fullfile(out_folder,'GSR_filtered.csv');
    if ~exist(fp,'file'), errordlg('Missing GSR_filtered.csv'); return; end
    F = readmatrix(fp); tf = (F(:,1)-F(1,1))/1000; vf = F(:,2);

    [okR, rp] = find_first_file(raw_folder, {'GSR*.csv'});
    figure('Name','GSR Overlay','Color','w');
    if okR
        R = readmatrix(rp); tr = (R(:,1)-R(1,1))/1000; vr = R(:,2);
        if do_norm, vr = (vr-mean(vr))/std(vr); end
        plot(tr,vr,'k-','DisplayName','RAW'); hold on;
    else
        hold on;
        vr = []; %#ok<NASGU>
    end
    if do_norm, vf = (vf-mean(vf))/std(vf); end
    plot(tf,vf,'r-','DisplayName','FILTERED'); grid on;
    xlabel('Time [s]'); ylabel('GSR [a.u.]'); title('GSR Overlay (RAW vs FILTERED)');
    legend('show','Location','best');

    % === PSD (RAW vs FILTERED), 0–0.5 Hz ===
    dt = median(diff(tf)); FsGSR = 1/max(dt, eps);
    raw_g = [];
    if exist('R','var') && ~isempty(R)
        raw_g = (R(:,2)).';  % bez re-samplingu
    end
    plot_psd_avg(raw_g, (vf(:)).', FsGSR, [0 0.5], 'GSR Welch PSD', 'PSD [dB/Hz]');
end

function plot_hr(out_folder, raw_folder, do_norm)
    fp = fullfile(out_folder,'HR_filtered.csv');
    if ~exist(fp,'file'), errordlg('Missing HR_filtered.csv'); return; end
    F = readmatrix(fp); tf = F(:,1); vf = F(:,2);

    [okR, rp] = find_first_file(raw_folder, {'HR*.csv'});
    figure('Name','HR Overlay','Color','w');
    if okR
        R = readmatrix(rp); tr = (R(:,1)-R(1,1))/1000; vr = R(:,2);
        if do_norm, vr = (vr-mean(vr))/std(vr); end
        plot(tr,vr,'k-','DisplayName','RAW'); hold on;
    else
        hold on;
        vr = []; %#ok<NASGU>
    end
    if do_norm, vf = (vf-mean(vf))/std(vf); end
    plot(tf,vf,'r-','DisplayName','FILTERED'); grid on;
    xlabel('Time [s]'); ylabel('HR [bpm]'); title('HR Overlay (RAW vs FILTERED)');
    legend('show','Location','best');

    % === PSD (RAW vs FILTERED), 0.04–0.5 Hz ===
    dt = median(diff(tf)); FsHR = 1/max(dt, eps);   % typ. 2 Hz
    raw_h = [];
    if exist('R','var') && ~isempty(R)
        raw_h = (R(:,2)).';
    end
    plot_psd_avg(raw_h, (vf(:)).', FsHR, [0.04 0.5], 'HR Welch PSD', 'PSD [dB/Hz]');
end

function plot_rr(out_folder, raw_folder, do_norm)
    fp = fullfile(out_folder,'RR_filtered.csv');
    if ~exist(fp,'file'), errordlg('Missing RR_filtered.csv'); return; end
    F = readmatrix(fp); tf = F(:,1); vf = F(:,2);

    [okR, rp] = find_first_file(raw_folder, {'RR*.csv'});
    figure('Name','RR Overlay','Color','w');
    if okR
        R = readmatrix(rp); tr = (R(:,1)-R(1,1))/1000; vr = R(:,2);
        if do_norm, vr = (vr-mean(vr))/std(vr); end
        plot(tr,vr,'k-','DisplayName','RAW'); hold on;
    else
        hold on;
        vr = []; %#ok<NASGU>
    end
    if do_norm, vf = (vf-mean(vf))/std(vf); end
    plot(tf,vf,'r-','DisplayName','FILTERED'); grid on;
    xlabel('Time [s]'); ylabel('RR [s]'); title('RR Overlay (RAW vs FILTERED)');
    legend('show','Location','best');

    % === PSD (RAW vs FILTERED), 0.04–0.5 Hz ===
    dt = median(diff(tf)); FsRR = 1/max(dt, eps);   % typ. 2 Hz
    raw_r = [];
    if exist('R','var') && ~isempty(R)
        raw_r = (R(:,2)).';
    end
    plot_psd_avg(raw_r, (vf(:)).', FsRR, [0.04 0.5], 'RR Welch PSD', 'PSD [dB/Hz]');
end

% ---------- Helper: averaged Welch PSD over channels ---------------------
function plot_psd_avg(raw, filt, fs, frange, ttl, ylab)
% raw/filt: [nCh x N] nebo [1 x N]; raw může být []
% fs: vzorkovací frekvence; frange: [fmin fmax]
    if isempty(filt), return; end
    if isvector(filt), filt = filt(:).'; end
    if ~isempty(raw) && isvector(raw), raw = raw(:).'; end

    % volba délky okna
    if frange(2) <= 0.6
        wlen = min(max(round(64*fs), 256), size(filt,2));   % GSR/HRV
    else
        wlen = min(max(round(2*fs), 256), size(filt,2));    % EEG/Pulse
    end
    nover = floor(0.5*wlen);

    % akumulace přes kanály
    nCh = size(filt,1);
    accF = []; accR = [];
    for k = 1:nCh
        xF = detrend(filt(k,:),0);
        [PF, f] = pwelch(xF, hamming(wlen,'periodic'), nover, [], fs);
        if isempty(accF), accF = zeros(size(PF)); end
        accF = accF + PF;

        if ~isempty(raw)
            row = min(k,size(raw,1));
            xR = detrend(raw(row,:),0);
            [PR, ~] = pwelch(xR, hamming(wlen,'periodic'), nover, [], fs);
            if isempty(accR), accR = zeros(size(PR)); end
            accR = accR + PR;
        end
    end
    PFm = accF / nCh;
    if ~isempty(raw), PRm = accR / nCh; end

    idx = f >= frange(1) & f <= frange(2);
    figure('Name', ttl, 'Color','w');
    plot(f(idx), 10*log10(PFm(idx)), 'r-', 'LineWidth',1.2, 'DisplayName','FILTERED'); hold on; grid on;
    if ~isempty(raw)
        plot(f(idx), 10*log10(PRm(idx)), 'k--', 'LineWidth',1.0, 'DisplayName','RAW');
    end
    xlabel('Frequency [Hz]'); ylabel(ylab);
    legend('Location','best'); title(ttl);
    ax = gca; ax.XMinorGrid = 'on'; ax.YMinorGrid = 'on'; ax.Box = 'off';
end
