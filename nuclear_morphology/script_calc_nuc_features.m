% script to calculate all nuclei 
%src_dir = 'N:\SpatialPathology';
src_dir = 'D:\Documents\SpatialPathology\';
%segmented_dir = fullfile(src_dir, 'SegmentedImages');
%segmented_dir = fullfile(src_dir, 'SegmentedImages_Nathan');
%im_dir = fullfile(src_dir,'OriginalImages');
npy_dir = 'C:\Users\luong_nguyen\Documents\GitHub\npy-matlab';
addpath(genpath(npy_dir)); addpath(genpath(pwd));
im_dir = fullfile(src_dir,'subset_segmented_masked_normalized');
%T = readtable(fullfile(src_dir,'UDH_DCIS_table_new.csv'),'Delimiter',',','ReadVariableNames',true);
%numImages = length(T.CaseName);
% T.numCells = cell(numImages,1); % number of cells/image
%nuc_feature_dir = fullfile(src_dir,'NucFeatures_mat_Oct30');
imageNames = dir(fullfile(im_dir,'*.jpg'));
imageNames = {imageNames.name}';
numImages = length(imageNames);
nuc_feature_dir = fullfile(src_dir,'NucFeatures_mat_ADH');
if ~exist(nuc_feature_dir, 'dir'); mkdir(nuc_feature_dir); end
% num_cells = 0;
%imageNames = T.CaseName;
% numCells = cell(numImages,1);

images_with_issues = {};
count = 0;
for i = 78:numImages
    imname = imageNames{i}(1:end-4);
    %mask_name = fullfile(segmented_dir, [imname '.tiff']);
    mask_name = fullfile(im_dir, [imname '_nucleimask.png']);
    segmented_image = imread(mask_name);
    
    if sum(double(segmented_image(:)))/numel(segmented_image) < 1e-3
        count = count + 1;
        images_with_issues{count} = imname;
        fprintf('Image %s has some issue\n', imname);
        continue;
    end
    output_file = fullfile(nuc_feature_dir, [imname,'.mat']);
    if exist(output_file,'file'); continue; end
    im_rgb = imread(fullfile(im_dir,[imname '.jpg']));
    t1 = tic;
    features = nucleiFeatures(segmented_image, im_rgb);
    fprintf('Done with image %s in %.2f seconds\n',imname, toc(t1));
    
    writeNPY(features,output_file);
    output_matfile = fullfile(nuc_feature_dir, [imname,'.mat']);
    parsave(output_matfile,features);
%     T.CellID{i} = [num_cells + 1, num_cells + size(features, 1)];
%     num_cells = num_cells + size(features, 1);
%     numCells{i} = size(features, 1);
end

%T2 = readtable(fullfile(src_dir,'lists','names_labels_4_classes_hospitals.csv'),'Delimiter',',','ReadVariableNames',true);
%T.Classes4types = T2.Classes; 
%T.numCells = cell2mat(numCells);

all_features = cell(numImages, 1);
for i = 1:numImages
   all_features{i} = readNPY(fullfile(nuc_feature_dir, [imageNames{i} '.npy']));
%    numCells{i} = size(all_features{i},1);
end
% T.numCells = cell2mat(numCells);
% T.end_indx = cumsum(T.numCells);
% T.start_indx = cat(1,[1],T.end_indx(1:end-1)+1);
all_features = cat(1, all_features{:});
fprintf('There are %d nan\n',sum(isnan(all_features(:))));
writeNPY(all_features, fullfile(nuc_feature_dir,'all_nuc_features.npy'));
% writetable(T,fullfile(src_dir,'UDH_DCIS_table_new.csv'),'delimiter',',');

%{
%% find out the nan
images_with_nan = {};
num_images_with_nan = 0;
for i = 1:numImages
   features =  readNPY(fullfile(nuc_feature_dir, [imageNames{i} '.npy']));
   if sum(isnan(features(:))) > 0
       num_images_with_nan = num_images_with_nan + 1;
       images_with_nan{num_images_with_nan} = imageNames{i};
       fprintf('Image %s of id %d has %d nan values\n', imageNames{i},i, sum(isnan(features(:))));
   end
end

% for i = [17 19]%:length(images_with_nan)
%    imname = images_with_nan{i};
%    segmented_image = imread(fullfile(segmented_dir, [imname '_Binary.tif']));
%    im_rgb = imread(fullfile(im_dir,[imname '.tif']));
%    T = tic;
%    features = nucleiFeatures(segmented_image, im_rgb);
%    fprintf('Done with image %s in %.2f seconds\n',imname, toc(T));
%    if sum(isnan(features(:))) > 0
%        fprintf('something is still wrong with %s at ID %d \n', imname, i);
%    end
%    output_file = fullfile(nuc_feature_dir, [imname,'.npy']);
%    writeNPY(features,output_file);
% end

% all_x_y_areas = cell(numImages,1);
% parfor i = 1:numImages
%    imname = imageNames{i};
%    segmented_image = imread(fullfile(segmented_dir, [imname '_Binary.tif']));
%    T1 = tic;
%    features = getXYArea(segmented_image);
%    all_x_y_areas{i} = features;
%    fprintf('Done with image %s in %.2f seconds\n',imname, toc(T1));
% end
% 
% all_x_y_areas = cat(1, all_x_y_areas{:});
% writeNPY(all_x_y_areas,fullfile(nuc_feature_dir, ['cells_x_y_areas','.npy']));

%% nuclear feature names
morphology_feat_names = {'Area', 'Perim', 'Width','Height', 'Major','Minor',...
    'Extent','Euler', 'Ecc','Orientatition','Solidity','equivDiam'};
int_feats = {'mean','median','variance','kurtosis', 'skewness'};
haral_feats ={'correlation','clusterShade','clusterProm','energy',...
    'entropy','contrast','homogeneity'};
grl_feats = {'RP','SRE','LRE','GLN','RLN','LGLRE','HGRLE','SGLGLE',...
    'SRHGLE', 'LRLGLE','LRHGLE'};

int_text_feas = cat(2, int_feats, haral_feats, grl_feats);
chan_names = {'R','G','B','HSV','Lab','Luv','HE','BR'};

chan_name_combo = cell(length(chan_names),1);

for i = 1:length(chan_names)
    chan_feat_names = cell(length(int_text_feas),1);
    for j = 1: length(int_text_feas)
        chan_feat_names{j} = [int_text_feas{j} '_' chan_names{i}];
    end
    chan_name_combo{i} = chan_feat_names;
end

chan_name_combo = cat(1,chan_name_combo{:});
all_var_names = cat(1,morphology_feat_names', chan_name_combo);
% indices
%% calculate the mean
core_data = readtable(fullfile(src_dir,'ComputedFeatures_Label.csv'),...
    'delimiter',',','ReadVariableNames',true);

mean_features = cell(length(core_data.CaseName),1);
std_features = cell(length(core_data.CaseName),1); 
for i = 1:length(core_data.CaseName)
    train_eg = i < 117;
    casename = core_data.CaseName{i};
    
    if train_eg
        match_id = strncmp({[casename ' ']}, T.CaseName, length(casename) + 1);
    else
        match_id = strcmp(casename, T.CaseName);
    end
    match_id = find(match_id);
    features = cell(length(match_id),1);
    for j = 1:length(match_id)
       features{j} = readNPY(fullfile(nuc_feature_dir, [T.CaseName{match_id(j)} '.npy'])); 
    end
    features = cat(1,fatures{:});
    mean_features{i} = mean(features, 1);
    std_features{i} = std(features,1);
end

mean_features = cat(1,mean_features{:});
std_features = cat(1,std_features{:});

indx_mean = find(strncmp('Mean_', core_data.Properties.VariableNames,5));
mean_feature_core = table2array(core_data(:,indx_mean));
indx_std = find(strncmp('SD_', core_data.Properties.VariableNames,3));
std_feature_core = table2array(core_data(:,indx_std));


figure(2);
for i = 18:22
subplot(3,2,i-18); plot(mean_feature_core(:,i), mean_features(:,i), '.');axis equal;
%xlim([0 max(mean_feature_core(:,i))]);ylim([0 max(mean_feature_core(:,i))]);
title(all_var_names{i});
hold on; plot(1:max(mean_features(:,i)), 1:max(mean_features(:,i))); hold off;
end
%% overlay image and segmentation
T = readtable(fullfile(src_dir,'UDH_DCIS.csv'),'Delimiter',',','ReadVariableNames',true);
overlay_dir = fullfile(src_dir,'OverlayImages');
if ~exist(overlay_dir,'dir'); mkdir(overlay_dir); end
for i = 1:length(T.CaseName)
    imname = T.CaseName{i};
    segmented_image = imread(fullfile(segmented_dir, [imname '_Binary.tif']));
    im_rgb = imread(fullfile(im_dir,[imname '.tif']));
    overlayIm = uint8(repmat(double(segmented_image)./255,[1 1 3]).*double(im_rgb));
    imwrite(overlayIm,fullfile(overlay_dir,[imname '.tif']));
end


%% kmeans clustering on cells and visualize them 
all_features = readNPY(fullfile(nuc_feature_dir,'all_nuc_features.npy'));
T = readtable(fullfile(src_dir,'UDH_DCIS_table_new.csv'),'delimiter',',');
train_indx = (T.Hospital == 1);

train_data = cell(sum(train_indx),1);
start_indices = T.start_indx(train_indx);
end_indices = T.end_indx(train_indx);
for i = 1:sum(train_indx)
    train_data{i} = all_features(start_indices(i):end_indices(i),:);   
end

train_data = cat(1, train_data{:});
% train_data = train_data(:,1:9);
mean_data = mean(train_data,1);
std_data = std(train_data,1);

train_data_scaled = (train_data - repmat(mean_data,[size(train_data,1),1]))./...
    repmat(std_data,[size(train_data,1),1]);
n_clusters = 3;
%train_data = train_data - repmat(mean(train_data,1),[size(train_data,1),1]);
% kmeans clustering 
pool = parpool;
stream = RandStream('mlfg6331_64');  % Random number stream
options = statset('UseParallel',1,'UseSubstreams',1,...
    'Streams',stream);
[idx, C, sumd, D] = kmeans(train_data_scaled,n_clusters, 'Options',options,'Distance','correlation','Replicate',5,'MaxIter',200);

% identify the cluster which cells belong to
data = all_features;
data_scaled = (data - repmat(mean_data,[size(data,1),1]))./...
    repmat(std_data,[size(data,1),1]);

pw_dist = pdist2(data_scaled,C,'correlation');
[~,all_idx] = min(pw_dist,[],2);

% visualize the different cell types
kmeans_viz_dir = fullfile(src_dir,'kmeans_viz');
if ~exist(kmeans_viz_dir,'dir'); mkdir(kmeans_viz_dir); end

numImages = length(T.CaseName);

for i = 1:numImages
   tic;
   imname = T.CaseName{i};
   segmented_image = imread(fullfile(segmented_dir, [imname '_Binary.tif']));
   im_rgb = imread(fullfile(im_dir,[imname '.tif']));
   cls_im = all_idx(T.start_indx(i):T.end_indx(i));
   CC = bwconncomp(segmented_image);
   % calculate the area to filter out non nuclei (200-400 pixels)
   areas = cell2mat(cellfun(@length, CC.PixelIdxList, 'UniformOutput',false));
   max_area = 4000; min_area = 200;
   filtered_objects = (areas >= min_area) & (areas <= max_area);
   CC.PixelIdxList = CC.PixelIdxList(filtered_objects);
   CC.NumObjects = sum(filtered_objects);  
   
   for j = 1:n_clusters
       PixelLists = CC.PixelIdxList(cls_im == j);
       [rr,cc] = ind2sub(CC.ImageSize, cat(1,PixelLists{:}));
       mask = zeros(CC.ImageSize);
       mask(cat(1,PixelLists{:})) = 1;
       output_mask = uint8(repmat(mask,[1 1 3]).*double(im_rgb));
       imwrite(output_mask,fullfile(kmeans_viz_dir,[imname '_cls_' num2str(j) '.tif']));
   end
   fprintf('Done with image %s in %.2f seconds\n',imname, toc);
end

% can also collect the proportion of each cluster type and put it in the
% table 

cell_type_prop = zeros(numImages, n_clusters);
for i = 1:numImages 
   cls_im = all_idx(T.start_indx(i):T.end_indx(i));
   for j = 1:n_clusters
       cell_type_prop(i,j) = sum(cls_im ==j)/T.numCells(i);
   end
end

mean_cell_type_prop = zeros(4,n_clusters);

for i = 1:length(unique(T.ClassGrade))
   grade_indx = T.ClassGrade == (i-1);
   mean_cell_type_prop(i,:) = mean(cell_type_prop(grade_indx,:),1);
end

%% print out train, test list and label for Burak
writetable(table(T.CaseName,T.ClassGrade>0, T.Hospital),fullfile(src_dir,'test_train_2classes.txt'),'delimiter',',','WriteVariableNames',false);
writetable(table(T.CaseName,T.ClassGrade, T.Hospital),fullfile(src_dir,'test_train_4classes.txt'),'delimiter',',','WriteVariableNames',false);

%% print out the cells position, area, classes for burak
all_x_y_areas = readNPY(fullfile(nuc_feature_dir, ['cells_x_y_areas','.npy']));
output_dir = fullfile(src_dir,'objList');
if ~exist(output_dir,'dir'); mkdir(output_dir); end

for i = 1:numImages
   imname = T.CaseName{i}; 
   x_y_area = all_x_y_areas(T.start_indx(i):T.end_indx(i),:);
   cls_im = all_idx(T.start_indx(i):T.end_indx(i));
   
   f = fopen(fullfile(output_dir,[imname '.txt']),'w');
   fprintf(f,'%d\n',length(cls_im));
   for j = 1:length(cls_im)
       fprintf(f,'%d %d %d %.1f %.1f\n',j,x_y_area(j,3), cls_im(j), x_y_area(j,1),x_y_area(j,2));
   end
   fclose(f);
%    writetable(table([1:length(cls_im)]', x_y_area(:,3), cls_im, x_y_area(:,1), x_y_area(:,2)),...
%        fullfile(output_dir,[imname '.txt']),'WriteVariableNames',false);
end


%% use Burak code to find the cluster center
C_pca = initializeClusterVectorsUsingPrincipalComponent(train_data_scaled , n_clusters );
pw_dist_pca = pdist2(data_scaled,C_pca);
[~,all_idx_pca] = min(pw_dist_pca,[],2);

% visualize the different cell types
kmeans_viz_dir = fullfile(src_dir,'kmeans_viz_pca');
if ~exist(kmeans_viz_dir,'dir'); mkdir(kmeans_viz_dir); end

numImages = length(T.CaseName);

for i = 1:numImages
   tic;
   imname = T.CaseName{i};
   segmented_image = imread(fullfile(segmented_dir, [imname '_Binary.tif']));
   im_rgb = imread(fullfile(im_dir,[imname '.tif']));
   cls_im = all_idx_pca(T.start_indx(i):T.end_indx(i));
   CC = bwconncomp(segmented_image);
   % calculate the area to filter out non nuclei (200-400 pixels)
   areas = cell2mat(cellfun(@length, CC.PixelIdxList, 'UniformOutput',false));
   max_area = 4000; min_area = 200;
   filtered_objects = (areas >= min_area) & (areas <= max_area);
   CC.PixelIdxList = CC.PixelIdxList(filtered_objects);
   CC.NumObjects = sum(filtered_objects);  
   
   for j = 1:n_clusters
       PixelLists = CC.PixelIdxList(cls_im == j);
       [rr,cc] = ind2sub(CC.ImageSize, cat(1,PixelLists{:}));
       mask = zeros(CC.ImageSize);
       mask(cat(1,PixelLists{:})) = 1;
       output_mask = uint8(repmat(mask,[1 1 3]).*double(im_rgb));
       imwrite(output_mask,fullfile(kmeans_viz_dir,[imname '_cls_' num2str(j) '.tif']));
   end
   fprintf('Done with image %s in %.2f seconds\n',imname, toc);
end

%% load the calculated nuclei mean
tmp = load(fullfile(src_dir,'im_nuc_features.mat'));
mean_area = tmp.im_nuc_features(:,1);
figure; histogram(mean_area);

core_data = readtable(fullfile(src_dir,'ComputedFeatures_Label.csv'),...
    'delimiter',',','ReadVariableNames',true);

% train set and test set are matched differently
mean_area_core = core_data.Mean_Area;
mean_area_ex = zeros(size(mean_area_core));
for i = 1:length(core_data.CaseName)
    train_eg = i < 117;
    casename = core_data.CaseName{i};
    if train_eg
        match_id = strncmp({[casename ' ']}, T.CaseName, length(casename) + 1);
    else
        match_id = strcmp(casename, T.CaseName);
    end
    mean_area_ex(i) = mean(tmp.im_nuc_features(match_id,1));
end


%}





