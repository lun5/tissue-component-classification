function [bdry_im, seg] = object_proposal_tile(image, bb_coords, adj_map,num_comps,...
    max_num_neighbors, min_superpixSize)
% compute the top connected components from the object maps in Burak's
% output. 

% INPUT:
% -image: tile image
% -num_neighbors: number of neighbors to identify the median distances
% between objects (default 15)
% -num_comp: number of top connected component to report (default 10)
% -min_superpixel: minimum number of superpixels in a components (50)
% -min_superpixSize: minimum area of superpixel (default 30 pixels)

% OUTPUT:
% -seg: segmentation mask of the superpixel type
% -indx_sp_cl: indices of superpixels belonging to each clusters
% Luong Nguyen 3/13/2016
max_mum_neighbors_default = 35;
num_comps_default = 10; 
min_superpixSize_default = 30;

if nargin < 3
    error('Please input: image, bounding box coords, adjacency map');
elseif nargin < 4
    max_num_neighbors = max_mum_neighbors_default;
    num_comps = num_comps_default;
    min_superpixSize = min_superpixSize_default;
elseif nargin < 5
    max_num_neighbors = max_mum_neighbors_default;
    min_superpixSize = min_superpixSize_default;
elseif nargin < 6
    min_superpixSize = min_superpixSize_default;
end
     
% read in image, mask, adjacency map
nrows = size(image,1); ncols = size(image,2);
%adj_map = dlmread([imname param_string '_adjDela'],',',0,0);

%adj_map = adj_map(2:end,:);
%obj_types = adj_map(:,3);
%obj_coords = adj_map(:,[5 4]);
%obj_radii = sqrt(adj_map(:,2)./pi);

% find the centers and radii of the object of certain type
%indx_type = obj_types == obj_type;
%num_superpx = sum(indx_type);

%adj_obj = adj_map(indx_type,:); 
num_superpx = size(adj_map,1);
adj_map_coords = adj_map(:,[5 4]) - repmat([bb_coords(1) bb_coords(3)],num_superpx,1) + 1;
adj_map_radii = sqrt(adj_map(:,2)./pi);
adj_map_area = adj_map(:,2);
% slow way can be fast if we calculate the index first
indx_1 = cell(num_superpx,1);
indx_2 = cell(num_superpx,1);
dist_values = cell(num_superpx,1);
%tic;
for i = 1:num_superpx
   id1 = adj_map(i,1);
   num_neighbors_input = adj_map(i,6);% number of neighbors
   % filter out small objects
   if adj_map_area(i) <= min_superpixSize
       continue;
   end
   if num_neighbors_input > 0
       neighbor_indx = adj_map(i,7:(6+num_neighbors_input));       
       % filter out non existent neighbors, small area, and of same type
       neighbor_indx = find(ismember(adj_map(:,1),neighbor_indx) & adj_map(:,3) == 1 &...
           adj_map_area > min_superpixSize);
       
       neighbor_indx_obj = adj_map(neighbor_indx,1)';
       num_neighbors = length(neighbor_indx);
       if num_neighbors > 0
           distances = zeros(1, num_neighbors);
           for j = 1:num_neighbors
               % CHANGE FOR FEATURES
               distances(j) = norm(adj_map_coords(adj_map(:,1) == id1,:) - ...
                   adj_map_coords(neighbor_indx(j),:));
           end
           % limit the number of neighbor to be fewer than max_num_neighbors
           num_neighbors = min(max_num_neighbors, num_neighbors);       
           [sort_dists, sort_indx] = sort(distances);
           indx_1{i} = repmat(id1,[1, num_neighbors]);
           indx_2{i} = neighbor_indx_obj(sort_indx(1:num_neighbors));
           dist_values{i} = sort_dists(1:num_neighbors); %repmat(distances,[1, 2]);
       end       
   end
end
%T = toc; fprintf('Indexing done in %.2f seconds\n',T);
% omit empty distances
indx_empty = cellfun(@isempty,dist_values);
indx_1(indx_empty) = [];
indx_2(indx_empty) = [];
dist_values(indx_empty) = [];
% Elapsed time is 0.411493 seconds.
%tic;
indx_1 = cat(2,indx_1{:});
indx_2 = cat(2,indx_2{:});
new_indx_1 = arrayfun(@(x) find(adj_map(:,1) == x), indx_1);
new_indx_2 = arrayfun(@(x) find(adj_map(:,1) == x), indx_2);

dist_values = cat(2,dist_values{:});
% CHANGE FOR FEATURES
med_dist = median(unique(dist_values)); 
% save the distance values
%save(fullfile(outdir,[imname '_distances_to_nn.mat']),'dist_values');
indx = dist_values >= 2*med_dist;%prctile(dist_values,75);
sigma = 1.5*med_dist;
dist_new = exp(- dist_values.^2./(2*sigma^2));
dist_new(indx) = 0;
similarities = sparse([new_indx_1,new_indx_2],[new_indx_2 new_indx_1],[dist_new dist_new],num_superpx,num_superpx);
[~,C] = graphconncomp(similarities);
%fprintf('Done with connected components in %.2f seconds\n',toc);

% components_areas = zeros(max(C),1);
% for i = 1:length(components_areas)
%     indx_cl = C == i;
%     components_areas(i) = sum(obj_radii(indx_cl).^2);
% end
%tic;
components_numelts = zeros(max(C),1);
for i = 1:length(components_numelts)
    indx_cl = C == i;
    components_numelts(i) = sum(indx_cl);
end
[~,indx] = sort(components_numelts,'descend');
% take all the components greater than min_area
%num_comps = max(sum(components_numelts > min_superpix),num_comps);
top_centers = cell(num_comps,1);
top_radii = cell(num_comps,1);
for i = 1:num_comps
   indx_cl = C == indx(i);
   top_centers{i} = adj_map_coords(indx_cl,:);
   top_radii{i} = adj_map_radii(indx_cl);
end
%fprintf('Done with choosing components greater than min area in %.2f seconds\n',toc);
%if plot_flag
seg = zeros(nrows,ncols);
for i = 1:num_comps
    %tic;
    x = top_centers{i}(:,1); y = top_centers{i}(:,2);
    if length(x) == 1
        ang=0:0.01:2*pi;
        xp=floor(top_radii{i}*cos(ang)'+x);
        yp=floor(top_radii{i}*sin(ang)'+y);
        k = boundary(xp,yp);
        mask = poly2mask(xp(k),yp(k),nrows, ncols);
    elseif length(x) <= 10
        extended_x = [x; min(x+top_radii{i},nrows); max(x - top_radii{i},0)];
        extended_y = [y; min(y+top_radii{i},nrows); max(y - top_radii{i},0)];
        k = boundary(extended_x,extended_y,0.5);
        mask = poly2mask(extended_x(k),extended_y(k),nrows, ncols);
    else
        k = boundary(x,y,0.5);
        mask =  poly2mask(x(k),y(k),nrows, ncols);
    end
    % avoid overlapping
    if sum(seg(mask)>0)/sum(mask(:)) < 0.3
        seg(mask) = i;
    else
        mask_diff = ((seg - mask) == -1);
        seg(mask_diff) = i;
    end
    %toc;
end

se = strel('disk',4,4);
bdry = seg2bdry(seg,'imageSize');
bdry = imdilate(bdry,se) > 0;
r = image(:,:,1); g = image(:,:,2); b = image(:,:,3);
r(bdry) = 0; g(bdry) = 255; b(bdry) = 0;
bdry_im = cat(3,r,g,b);

end