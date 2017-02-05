%% script to convert from masks to individual nuclei
function features = getXYArea(segmented_image)
%% INPUT: image in RGB, binary mask
%% OUTPUT: 1st order and 2nd order features 
%% morphology features
% use bwconncomp to get the connected components
CC = bwconncomp(segmented_image);
% calculate the area to filter out non nuclei (200-400 pixels)
areas = cell2mat(cellfun(@length, CC.PixelIdxList, 'UniformOutput',false));
max_area = 4000; min_area = 200;
filtered_objects = (areas >= min_area) & (areas <= max_area);
CC.PixelIdxList = CC.PixelIdxList(filtered_objects);
CC.NumObjects = sum(filtered_objects);
% use regionprops to calculate 1st order features
properties = {'Centroid','Area'}; 
morphology_features = regionprops(CC, properties);
x_coords = arrayfun(@(x) x.Centroid(1), morphology_features);
y_coords = arrayfun(@(x) x.Centroid(2), morphology_features);
areas = arrayfun(@(x) x.Area, morphology_features);
features = cat(2,x_coords, y_coords, areas);
