% This part only concern tiling the image with different size
function [all_tiles_names, all_tiles_labels] = ...
    tiling_images(all_im_names, all_labels, tile_size, input_dir, output_dir)

% INPUT:
% -all_im_names
% -tile_size: default 256
% -input_dir: where input images are stored
% -output_dir: where output images will be saved

% OUTPUT:
% -seg: segmentation mask of the superpixel type
% -indx_sp_cl: indices of superpixels belonging to each clusters
% Luong Nguyen 3/13/2016
if isempty(tile_size); tile_size = 256; end;
cut_corner_pixels = 300;
omit_corner_tiles = ceil(cut_corner_pixels/tile_size);
all_tiles_names = cell(length(all_im_names),1);
all_tiles_labels = cell(length(all_im_names),1);
for i = 1:length(all_im_names)
   imname = all_im_names{i};
   tiles_count = 0;
   if ~exist(fullfile(input_dir,[imname '.tiff']),'file')      
       im = imread(fullfile(input_dir,[imname '.tif']));
   else
       im = imread(fullfile(input_dir,[imname '.tiff']));
   end
   % change image name to omit the space
   
   %im = im(cut_corner_pixels:end,1:end-cut_corner_pixels,:);
   imname = strrep(imname,' ','-');
   % tile the image
   [nrow, ncol,nc] = size(im);
   wholeBlockRows = floor(nrow/tile_size);
   blockVectorR = [tile_size*ones(1,wholeBlockRows),rem(nrow,tile_size)];
   wholeBlockCols = floor(ncol/tile_size);
   blockVectorC = [tile_size*ones(1,wholeBlockCols),rem(ncol,tile_size)];
   splitImage = mat2cell( im, blockVectorR,blockVectorC, nc);
   tiles_coords_r = [0 cumsum(blockVectorR)];
   tiles_coords_c = [0 cumsum(blockVectorC)];
   all_tiles_names{i} = cell(numel(splitImage),1);
   all_tiles_labels{i}(1:numel(splitImage),:) = all_labels(i);
   % update the train/test list
   % cut corner
   if size(splitImage,1) < omit_corner_tiles || size(splitImage,2) < omit_corner_tiles
       continue;
   end
   for r = 1:size(splitImage,1) - omit_corner_tiles
       for c = 1:size(splitImage,2) - omit_corner_tiles
           current_tile = splitImage{r,c};
           if size(current_tile,1) < tile_size || size(current_tile,2) < tile_size
               continue;
           end
           current_gray = sum(current_tile,3)./3;
           % omit empty tiles
           if (sum(current_gray(:) > 210)/numel(current_gray)) >= 0.7
               continue;
           end
           tile_name = sprintf('%s_%d_%d_%d_%d.tif',...
               imname,tiles_coords_r(r), tiles_coords_r(r) + blockVectorR(r),...
               tiles_coords_c(c), tiles_coords_c(c) + blockVectorC(c));
           tiles_count = tiles_count + 1;
           all_tiles_names{i}{tiles_count} = tile_name;
           imwrite(current_tile,fullfile(output_dir,tile_name));
       end
   end
   
   if tiles_count == 0
      all_tiles_names{i} = [];
      all_tiles_labels{i} = [];
      continue;
   end
   non_empty_cells = ~cellfun('isempty',all_tiles_names{i});
   all_tiles_labels{i} = all_tiles_labels{i}(non_empty_cells);   
   all_tiles_names{i} = all_tiles_names{i}(non_empty_cells);
   fprintf('Done with %s\n',imname);
end

end
