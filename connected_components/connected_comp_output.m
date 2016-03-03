%% Luong Nguyen 3/2/16
% print out object proposals/connected components for WSI
GIT_DIR = '/home/lun5/github/tissue-component-classification';
addpath(genpath(GIT_DIR));
IMG_DIR ='/home/lun5/CVPR_images/breast_wsi';
imlist = dir(fullfile(IMG_DIR,'adjDela','*_se1_*'));
imlist = {imlist.name}';
%imlist = {'tp10-867-1','tp10-876-1'};
num_images = length(imlist);
myVars = {'x','y','k'}; % coordinates of the polygon surroung roi
for i = 2:num_images
   if i == 6
      continue;
   end
   tic;
   imname = imlist{i}(1:end-36);
   fprintf('Working on image %s...',imname);
   I = imread(fullfile(IMG_DIR,'images',[imname '.tif']));
   T = toc; fprintf('take %.2f to read\n',T);tic;
   nrow = size(I,1); ncol = size(I,2);
   cca_list = dir(fullfile(IMG_DIR,'connected_comp',[imname '*.mat']));
   cca_list = {cca_list.name}';
   num_comp = length(cca_list);
   fprintf('Number of components is %d\n',num_comp);
   for j = 1:num_comp
      cca_name = cca_list{j};
      fprintf('working on %s\n',cca_name);
      bb_var = load(fullfile(IMG_DIR,'connected_comp',cca_name),myVars{:});
      mask = poly2mask(bb_var.x(bb_var.k),bb_var.y(bb_var.k),nrow,ncol);
      min_y = max(0,min(bb_var.y)); max_y = min(max(bb_var.y),nrow);
      min_x = max(0,min(bb_var.x)); max_x = min(max(bb_var.x),ncol);
      if min_y >= max_y || min_x >= max_x
         continue;
      end
      I_crop = I(min_y:max_y, min_x:max_x,:);
      mask_crop = mask(min_y:max_y, min_x:max_x);
      area = sum(mask_crop(:));
      se = strel('disk',floor(area^(1/9)),4);
      boundary_map = edge(mask_crop);
      boundary_map = imdilate(boundary_map,se);
      bdry_edge_map_im = I_crop.*uint8(repmat(~boundary_map,[1 1 3]));
      imwrite(bdry_edge_map_im,fullfile(IMG_DIR,'conn_comp_im',[cca_name(1:end-4) '.tif']),'Resolution',300);
   end
   T = toc;
   fprintf('Done with printing out conn comp of %s in %.2f seconds\n', imname, T);
end
