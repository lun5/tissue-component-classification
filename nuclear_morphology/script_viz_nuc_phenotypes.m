% script to calculate all nuclei 
src_dir = 'N:\SpatialPathology';
segmented_dir = fullfile(src_dir, 'SegmentedImages');
%segmented_dir = fullfile(src_dir, 'SegmentedImages_Nathan');
im_dir = fullfile(src_dir,'OriginalImages');
npy_dir = 'C:\Users\luong_nguyen\Documents\GitHub\npy-matlab';
addpath(genpath(npy_dir)); addpath(genpath(pwd));
T = readtable(fullfile(src_dir,'UDH_DCIS_table_new.csv'),'Delimiter',',','ReadVariableNames',true);
numImages = length(T.CaseName);
% T.numCells = cell(numImages,1); % number of cells/image
nuc_feature_dir = fullfile(src_dir,'NucFeatures_mat_Oct30');
%nuc_feature_dir = fullfile(src_dir,'NucFeatures_mat_Jan28');
if ~exist(nuc_feature_dir, 'dir'); mkdir(nuc_feature_dir); end
% num_cells = 0;
imageNames = T.CaseName;

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

imNames = {'D2','D10','U50'};
for i = 1:numImages
   tic;
   imname = T.CaseName{i};
   
   if ~ismember(imname, imNames)
       continue;
   end
   
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
   output_mask = zeros(size(im_rgb));
   
   for j = 1:n_clusters
       PixelLists = CC.PixelIdxList(cls_im == j);
       [rr,cc] = ind2sub(CC.ImageSize, cat(1,PixelLists{:}));
       mask = zeros(CC.ImageSize);
       mask(cat(1,PixelLists{:})) = 1;
       output_mask(:,:,j) = mask;
   end
   imwrite(output_mask,fullfile(kmeans_viz_dir,[imname '.tif']));
   fprintf('Done with image %s in %.2f seconds\n',imname, toc);
end