% script to make figures for presentation
IMDIR = 'Z:\ADH_Jeff';
imname = 'flat_1050497_level2.tif';

he = imread(fullfile(IMDIR, imname));
figure; imshow(he);

% crop white part out of the image
rect = getrect;
he = imcrop(he,rect);
figure; imshow(he);

% get polygon for regions
num_regions = 10;
regions = cell(num_regions,1);
xi = cell(num_regions,1); yi = cell(num_regions,1);
for i = 1:num_regions
    [regions{i}, xi{i}, yi{i}] = roipoly(he);
end

all_regions = cat(3,regions{:});
all_regions = sum(all_regions,3);
%all_regions(all_regions == 0) = 2;
%xi = cat(1,xi{:}); yi = cat(1,yi{:});

rgb = he;
imshow(rgb);
I = rgb2gray(he);
hold on
h = imshow(I.*1.3); % Save the handle; we'll need it later
%hold off
alpha_data = ~all_regions;
set(h, 'AlphaData', alpha_data);
for i = 1:num_regions
    plot(xi{i},yi{i},'-','Color','b','LineWidth',3);
end


rgb = imread('peppers.png');
imshow(rgb);
I = rgb2gray(rgb);
hold on
h = imshow(I); % Save the handle; we'll need it later
hold off

[M,N] = size(I);
block_size = 50;
P = ceil(M / block_size);
Q = ceil(N / block_size);
alpha_data = checkerboard(block_size, P, Q) > 0;
alpha_data = alpha_data(1:M, 1:N);
set(h, 'AlphaData', alpha_data);

