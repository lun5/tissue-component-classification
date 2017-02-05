%% script to convert from masks to individual nuclei
function features = nucleiFeatures(segmented_image, im_rgb)
%% INPUT: image in RGB, binary mask
%% OUTPUT: 1st order and 2nd order features 
%% morphology features
% use bwconncomp to get the connected components
%segmented_image = segmented_image > 0; % convert to logical 
CC = bwconncomp(segmented_image);
if CC.NumObjects == 0
    features = [];
    return;
end
% calculate the area to filter out non nuclei (200-400 pixels)
areas = cell2mat(cellfun(@length, CC.PixelIdxList, 'UniformOutput',false));
max_area = 1000; min_area = 50;% 4000 and 200 for 40x.
filtered_objects = (areas >= min_area) & (areas <= max_area);
CC.PixelIdxList = CC.PixelIdxList(filtered_objects);
CC.NumObjects = sum(filtered_objects);
% use regionprops to calculate 1st order features
properties = {'Centroid','Area','BoundingBox','ConvexArea','Eccentricity','EquivDiameter'...
    'Extent','FilledArea','MajorAxisLength', 'MinorAxisLength', 'Orientation',...
    'Perimeter','Solidity','EulerNumber'}; 
morphology_features = regionprops(CC, properties);
bounding_width = arrayfun(@(x) x.BoundingBox(3), morphology_features);
bounding_height = arrayfun(@(x) x.BoundingBox(4), morphology_features);
x_coords = arrayfun(@(x) x.Centroid(1), morphology_features);
y_coords = arrayfun(@(x) x.Centroid(2), morphology_features);

bounding_box = arrayfun(@(x) x.BoundingBox, morphology_features,'UniformOutput',false);
morphology_features = cell2mat(arrayfun(@(x) [x.Area, x.Perimeter, ...
    x.MajorAxisLength,x.MinorAxisLength, x.Extent,x.EulerNumber, x.Eccentricity, x.Orientation,...
     x.Solidity,x.EquivDiameter], morphology_features,'UniformOutput',false));
morphology_features = cat(2,x_coords, y_coords, morphology_features(:,1:2),...
    bounding_width, bounding_height,morphology_features(:,3:end));
clear bounding_width bounding_height 
% x, y, Area, Major, Minor, Ecc, Orientation, Perimeter, solidity, width, height

%% 8 different channels
im_rgb = double(im_rgb);
c1 = im_rgb(:,:,1); % red
c2 = im_rgb(:,:,2); % green
c3 = im_rgb(:,:,3); % blue
hsv = rgb2hsv(im_rgb); c4 = hsv(:,:,3); % V channel in HSV %v = max(max(r,g),b); 
lab = rgb2lab(uint8(im_rgb)); c5 = lab(:,:,1); % L channel in lab space
luv = rgb2luv(im_rgb); c6 = luv(:,:,1); % L channel in Luv space 
[stain_density, ~, ~] = calc_stain_density(im_rgb, 0); % H channel in stain density
c7 = reshape(stain_density(1,:), size(im_rgb,1), size(im_rgb,2));
c7(c7>1) = 1;
c8 = ((100*c3)./(1+c1+c2))./(1+c1+c2+c3)*256; % blue ratio
c8(c8>255) = 255;
% I don't think they normalize the features, this doesn't sound so right
all_channels = {c1,c2,c3,c4,c5,c6,c7,c8};
clear c1 c2 c3 c4 c5 c6 c7 c8 hsv lab luv stain_density
% make the non nuclei area to be the lowest values in each channels
for i = 1:length(all_channels)
   channel = all_channels{i}; 
   if i < 7
       channel(segmented_image == 0) = max(channel(:));
   else
       channel(segmented_image == 0) = min(channel(:));
   end
   all_channels{i} = reshape(channel,size(im_rgb,1), size(im_rgb,2));
end
% %% mean, median, standard deviation, skewness and kurtosis for each nuclei
% first_order_stats = cellfun(@(x) firstOrder(x, CC), all_channels, 'UniformOutput', false);
% % use graycomatrix and graycoprops to calculate 2nd order features
% first_order_stats = cat(2,first_order_stats{:});
% colMean = nanmean(first_order_stats);
% [~,col] = find(isnan(first_order_stats));
% first_order_stats(isnan(first_order_stats)) = colMean(col);
% 
% % 8 color channels, have to loop through all of them
% 
% %% second order statistics: runlength and Haralick
% second_order_stats = cellfun(@(x) secondOrder(x, bounding_box), all_channels, 'UniformOutput', false);
% second_order_stats = cat(2,second_order_stats{:});
% colMean = nanmean(second_order_stats);
% [~,col] = find(isnan(second_order_stats));
% second_order_stats(isnan(second_order_stats)) = colMean(col);

%% texture features
texture_features = cellfun(@(x) calcIntTextureFeatures(x, CC, bounding_box, segmented_image),...
    all_channels, 'UniformOutput', false);
texture_features = cat(2,texture_features{:});
colMean = nanmean(texture_features);
[~,col] = find(isnan(texture_features));
texture_features(isnan(texture_features)) = colMean(col);

features = cat(2, morphology_features,texture_features);
end

function first_order_stats = intensityFeats(channel, CC)
% INPUT: connected component list of nuclei and channel values
% OUTPUT: first_order_stats array of size num_nuclei x 5
% mean, median, standard deviation, skewness and kurtosis for each nuclei

pix_list = CC.PixelIdxList;
mean_intensity = cell2mat(cellfun(@(x) mean(channel(x)), pix_list,'UniformOutput',false));
median_intensity = cell2mat(cellfun(@(x) median(channel(x)), pix_list,'UniformOutput', false));
var_intensity = cell2mat(cellfun(@(x) var(channel(x)), pix_list,'UniformOutput', false));
skewness_intensity = cell2mat(cellfun(@(x) skewness(channel(x)), pix_list,'UniformOutput', false));
kurtosis_intensity = cell2mat(cellfun(@(x) kurtosis(channel(x)), pix_list,'UniformOutput', false));
first_order_stats = cat(1,mean_intensity, median_intensity,var_intensity,...
    kurtosis_intensity,skewness_intensity)';
end

function second_order_stats = textureFeats(channel, BoundingBox, mask)

low = min(channel(:));
high = max(channel(:));
offset = [0 1; -1 1; -1 0; -1 -1];
% set the regions outside segmentation mask to be NAN
old_channel = channel;
channel(~mask) = nan;

% separate whole image into nuclear boxes
nuc_im = cellfun(@(x) imcrop(channel, x), BoundingBox,'UniformOutput', false);
% calculate the gray level co-occurence matrix for each nuclei, take mean
% over 4 offsets
glcms = cellfun(@(x) round(mean(graycomatrix(x, 'GrayLimits',[low high], ...
    'Offset', offset,'Symmetric',true),3)),...
    nuc_im, 'UniformOutput',false);
harr_features = cellfun(@(x) GLCM_Features1(x,0),glcms, 'UniformOutput',false);
% only select some of these output
harr_features = cell2mat(cellfun(@(x) [x.corrp, x.cshad, x.cprom, x.energ, ...
    x.entro, x.contr,x.homop], harr_features, 'UniformOutput',false)); % no inertia
% run length features

if max(old_channel(:)) == unique(unique((old_channel(~mask))))
    channel(~mask) = Inf;
else
    channel(~mask) = -Inf;
end
nuc_im = cellfun(@(x) imcrop(channel, x), BoundingBox,'UniformOutput', false);

grayrlm = cellfun(@(x) grayrlmatrix(x, 'GrayLimits',[low high], ...
    'Offset', [1 2 3 4]'),...
    nuc_im, 'UniformOutput',false);
runlength_features = cell2mat(cellfun(@(x) mean(grayrlprops(x),1),grayrlm, 'UniformOutput',false));
second_order_stats = cat(2, harr_features,runlength_features);
end

function texture_features = calcIntTextureFeatures(channel, CC, BoundingBox, mask)
    first_order_stats = intensityFeats(channel,CC);
    second_order_stats = textureFeats(channel,BoundingBox,mask);
    texture_features = cat(2,first_order_stats,second_order_stats);
end

% Features computed 
% Autocorrelation: [2]                      (out.autoc)
% Contrast: matlab/[1,2]                    (out.contr)
% Correlation: matlab                       (out.corrm)
% Correlation: [1,2]                        (out.corrp)
% Cluster Prominence: [2]                   (out.cprom)
% Cluster Shade: [2]                        (out.cshad)
% Dissimilarity: [2]                        (out.dissi)
% Energy: matlab / [1,2]                    (out.energ)
% Entropy: [2]                              (out.entro)
% Homogeneity: matlab                       (out.homom)
% Homogeneity: [2]                          (out.homop)
% Maximum probability: [2]                  (out.maxpr)
% Sum of sqaures: Variance [1]              (out.sosvh)
% Sum average [1]                           (out.savgh)
% Sum variance [1]                          (out.svarh)
% Sum entropy [1]                           (out.senth)
% Difference variance [1]                   (out.dvarh)
% Difference entropy [1]                    (out.denth)
% Information measure of correlation1 [1]   (out.inf1h)
% Informaiton measure of correlation2 [1]   (out.inf2h)
% Inverse difference (INV) is homom [3]     (out.homom)
% Inverse difference normalized (INN) [3]   (out.indnc) 
% Inverse difference moment normalized [3]  (out.idmnc)
%

% feature_names = {'Area', 'Major', 'Minor', 'Ecc', 'Orientation','Perimeter',
% 'solidity', 'width', 'height', 'mean', 'median', variance, kurtosis, skewness, 
% contrast, correlation, cluster shade,Cluster
% Prominence,energy,entropy,homogeneity,SRE, LRE, GLN, RLN,  RP, LGRE, HGRE,
%SGLGE, SRHGE, LRLGE,  LRHGE }



