%% script to analyze Beck's data
src_dir = 'D:\Documents\SpatialPathology';
IMG_DIR = fullfile(src_dir,'OriginalImages');
%addpath(genpath('C:\Users\luong_nguyen\Documents\GitHub\color-deconvolution'));
%addpath(genpath('C:\Users\luong_nguyen\Documents\GitHub\HE-tumor-object-segmentation'));
%% preprocessing script for deep learning
%preprocessing_script

%% normalize color
im_list = dir(fullfile(IMG_DIR,'*.tif'));
im_list = {im_list.name}';
norm_dir = fullfile(src_dir,'Normalized_Images','NormalizedImages_Khan');
if ~exist(norm_dir,'dir')
    mkdir(norm_dir);
end

%target_im = imread('target.tif');
% for i = 1:length(im_list)
%    imname = im_list{i}(1:end-4);
%    source_image = imread(fullfile(IMG_DIR,[imname '.tif']));
%    tic
%    %nim = NormLuong(source_image,target_im);
%    nim = NormSCDLeeds(source_image, target_im);
%    toc
%    imwrite(nim,fullfile(norm_dir,[imname '.tif']));
% end

%% calculate PMI
pmi_features_dir = fullfile(src_dir,'PMI_features');
% if ~exist(pmi_features_dir,'dir'); mkdir(pmi_features_dir); end
% parfor i = 1:length(im_list)
%    tic;
%    imname = im_list{i}(1:end-4);
%    nim = imread(fullfile(norm_dir,[imname '.tif']));
%    features = calculate_PMI_features_tile(nim, ones(size(nim,1), size(nim,2)));
%    parsave(fullfile(pmi_features_dir,[imname '.mat']),features);
%    toc
% end

%% tSNE
% read in labels for all the images
T = readtable(fullfile(src_dir,'lists','names_labels_udh_dcis_hospitals.csv'),'Delimiter',',','ReadVariableNames',true);
%T = readtable(fullfile(src_dir,'lists','names_labels_4_classes_hospitals.csv'),'Delimiter',',','ReadVariableNames',true);

im_list = T.ImageNames;
labels = T.Classes;
all_features = cell(length(im_list),1);
for i = 1:length(im_list)
    imname = im_list{i};
    load(fullfile(pmi_features_dir,[imname '.mat']));
    all_features{i} = features;
end

all_features = cat(1, all_features{:});
%all_features_mat = all_features(:,[1:8 54:58]);

mean_features = mean(all_features,1);
sd_features = std(all_features,[],1);

features_scaled = (all_features - repmat(mean_features,[size(all_features,1),1]))./...
    repmat(sd_features,[size(all_features,1),1]);

features_scaled_mat = features_scaled(:,[1:8 54:58]);
tne_dir = 'C:\Users\luong_nguyen\Documents\GitHub\tSNE_matlab';
addpath(genpath(tne_dir));
no_dims = 3; initial_dims = 10; perpexity = 30;
D = squareform(pdist(features_scaled_mat,'cosine'));
mappedX = tsne(features_scaled_mat,[],no_dims,initial_dims, perpexity);
x = mappedX(:,1); y = mappedX(:,2); z = mappedX(:,3);

gu = unique(labels); 
figure; h =  gscatter(x,y,labels);
for k = 1:numel(gu)
  set(h(k),'ZData',z(labels == gu(k))); 
end
view(3)

%% SVM
% cv partition
cv = cvpartition(labels,'k',5);
rng(1);
t = templateSVM('Standardize',1);
%[~,numerical_labels] = ismember(labels,component_list);
Md1 = fitcecoc(features_scaled_mat,labels,'Learners',t);
CVMd1 = crossval(Md1);
%pool = parpool;
options = statset('UseParallel',1);
oosLoss = kfoldLoss(CVMd1,'Options',options);
oofLabel = kfoldPredict(CVMd1,'Options',options);
ConfMat = confusionmat(labels,oofLabel);
cm = ConfMat./repmat(sum(ConfMat,2),[1 size(ConfMat,1)]);
%plotConfMat(cm,{'UDH', 'DCIS 1', 'DCIS 2', 'DCIS 3'},4);
plotConfMat(cm,{'UDH', 'DCIS'},2);

%% calculate the mask
%segmented_dir = fullfile(src_dir,'SegmentedImages');


