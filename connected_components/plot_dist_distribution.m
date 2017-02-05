%% plot the distribution of distances between superpixel
% Luong Nguyen 3/8/2016

IMG_DIR = 'Z:\HEproject\object_proposals';
imlist = dir(fullfile(IMG_DIR,'adjDela','*_se1_*'));
imlist = {imlist.name}';
num_images = length(imlist);
param_string = '_se1_minNuc3_minStr5_minLum5';
num_neighbors = 10;
%outdir = fullfile(IMG_DIR, ['new_cc_nn_' num2str(num_neighbors)]);
outdir = fullfile(IMG_DIR, ['new_cc_nearest_neighbor_' num2str(num_neighbors)]);
%outdir = fullfile(IMG_DIR, ['cc_nearest_neighbor_' num2str(num_neighbors)]);
obj_type = 1;
num_comps = 20; 
plot_flag = 1;
top_centers = cell(num_images,1);
top_radii = cell(num_images,1);
im_size = [2048 2048];

parfor i = 1:num_images
    imname = imlist{i}(1:end-36);
    data = load(fullfile(outdir,[imname '_distances_to_nn.mat']),'dist_values');
    dist_values = data.dist_values;
    h = figure; histogram(dist_values,'Normalization','probability');
    xlim([0 100]); ylim([0 0.15]);
    print(h,'-dpng',fullfile(outdir,[imname '_type_' num2str(obj_type) 'histogram_dist.png']));
    close all; 
end
