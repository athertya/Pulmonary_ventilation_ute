%% Inspect Jiaji NEX=2 Ventilation Outputs
clearvars; clc; close all;

%% Set Paths
base_dir   = 'F:\UTE_Lab\Pulmonary_ventilation_ute\gitdata';
input_mat  = fullfile(base_dir, 'img_Jj_2025_12_17.mat');
result_dir = fullfile(base_dir, 'nex2_results');

%% 1. Load Raw 4D Input (NEX=2: Index 3 = Exp, Index 4 = Ins)
fprintf('Loading raw 4D input data...\n');
raw_data  = load(input_mat);
var_names = fieldnames(raw_data);
img_4d    = double(raw_data.(var_names{1}));

img_exp_raw = img_4d(:,:,:,3); % Fixed
img_ins_raw = img_4d(:,:,:,4); % Moving

%% 2. Load Processed Results
fprintf('Loading generated ventilation results...\n');
reg_data = load(fullfile(result_dir, 'img_ins_registered.mat'));
sm_exp   = load(fullfile(result_dir, 'img_exp_smooth.mat'));
sm_ins   = load(fullfile(result_dir, 'img_ins_smooth.mat'));
rv_data  = load(fullfile(result_dir, 'rv_map.mat'));
sv_data  = load(fullfile(result_dir, 'sv_map.mat'));

img_ins_reg = double(reg_data.img_ins_reg);
img_exp_sm  = double(sm_exp.img_exp_smooth);
img_ins_sm  = double(sm_ins.img_ins_smooth);
rv          = double(rv_data.rv);
sv          = double(sv_data.sv);

%% 3. Combine into 4D Inspection Array
% Dim 4 channels:
% 1: Raw Ins | 2: Raw Exp | 3: Sm Ins | 4: Sm Exp | 5: Reg Ins | 6: SV | 7: RV
vol_sz = size(img_exp_raw);
img_check = zeros([vol_sz, 7]);
img_check(:,:,:,1) = img_ins_raw;
img_check(:,:,:,2) = img_exp_raw;
img_check(:,:,:,3) = img_ins_sm;
img_check(:,:,:,4) = img_exp_sm;
img_check(:,:,:,5) = img_ins_reg;
img_check(:,:,:,6) = sv;
img_check(:,:,:,7) = rv;

fprintf('Done! Array "img_check" ready: [%d x %d x %d x %d]\n', size(img_check));

