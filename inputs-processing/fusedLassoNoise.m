%% fused lasso for H&E image in sine(hue) and cosine(hue) channels
% Luong Nguyen 2/22/16

tiles_dir = fullfile('Z:\','TilesForLabeling_tiff_renamed'); %window
%tiles_dir = '/Users/lun5/Research/data/TilesForLabeling_tiff_renamed'; %mac
%tiles_dir =  '/home/lun5/HEproject/TilesForLabeling_tiff_renamed'; %linux

imname = 'dRfMkOErZY.tif';
imname = '8ghygsmwjy.tif';
raw_image = imread(fullfile(tiles_dir, imname));

rotation_matrix = load(fullfile(pwd,'DanTrainingData','rotation_matrix_tp10-867-1.mat'),'rotation_matrix');
im_rgb = double(raw_image)./255;
X = reshape(im_rgb,[size(im_rgb,1)*size(im_rgb,2),size(im_rgb,3)]);
rotated_coordinates = rotation_matrix.rotation_matrix*X';

r = im_rgb(:,:,1);
theta = angle(rotated_coordinates(2,:) + 1i*rotated_coordinates(3,:)); %hue
im_theta = reshape(theta,size(r));
sat = sqrt(rotated_coordinates(2,:).^2 + rotated_coordinates(3,:).^2);
im_sat = reshape(sat,size(r));
brightness = rotated_coordinates(1,:);
im_brightness = reshape(brightness,size(r));

indx = 750;
im_sine = sin(im_theta);
im_cosine = cos(im_theta);

figure; plot(im_sine(indx,:),'r-');%hold on;axis tight;
%plot(im_cosine(indx,:),'g-');
set(gca,'FontSize',20);
%legend('br','sat*cos(h)', 'sat*sin(h)');

% fused Lasso 
%% Part b: 1d fused lasso problem
y = im_sine(indx,:)';
lamb = 1;
ndim = size(y,1);
%[ii jj] = getLocalPairs(im_size, 1,0,[]); % index of adjacent pixels
%minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(ii) - betas(jj),1)

cvx_begin
variable betas(ndim)
minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(1:end-1) - betas(2:end),1)
cvx_end

figure; plot(y,'k-'); hold on;
plot(betas,'r-'); hold off;

% 2. solutions as we vary lambda
lambdas = logspace(1,-2,100);
num_changepoints = zeros(length(lambdas),1);
beta_vec = zeros( length(y), length(lambdas));
for i = 1:length(lambdas)
    lamb = lambdas(i);
    cvx_begin
    variable betas(ndim)
    minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(1:end-1) - betas(2:end),1)
    cvx_end
    beta_vec(:,i) = betas;
    num_changepoints(i) = sum(abs(betas(1:end-1) - betas(2:end)) > 1e-8);
end

figure; plot(lambdas,num_changepoints,'r-','LineWidth',3);
xlabel('\lambda'); ylabel('No change points'); set(gca,'FontSize',16);


figure; imshow(im_rgb);
axis off; set(gca,'FontSize',20);
hold on; plot(1:size(im_rgb,1), ones(size(im_rgb,1),1)*indx,'k-','LineWidth',3)

figure; plot(y,'b-'); hold on;
plot(beta_vec(:,1),'r-','LineWidth',3); hold off;
axis([0 size(im_rgb,1) -1 1]);
legend('sine image', 'fused lasso');set(gca,'FontSize',20);

%%
% smooth sat*sin (3rd rotated), and sat*cos (2rd rotated)
im_sat_cos = reshape(rotated_coordinates(2,:),size(r));
y = im_sat_cos(indx,:)';
lamb = 1;
ndim = size(y,1);
%[ii jj] = getLocalPairs(im_size, 1,0,[]); % index of adjacent pixels
%minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(ii) - betas(jj),1)

cvx_begin
variable betas(ndim)
minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(1:end-1) - betas(2:end),1)
cvx_end

figure; plot(y,'b-'); hold on;
plot(betas,'r-','LineWidth',3); hold off;
axis([0 size(im_rgb,1) -.3 .3]);
legend('sat*sine image', 'fused lasso');set(gca,'FontSize',20);

%%
% smooth theta
y = im_theta(indx,:)';
lamb = 10;
ndim = size(y,1);
%[ii jj] = getLocalPairs(im_size, 1,0,[]); % index of adjacent pixels
%minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(ii) - betas(jj),1)

cvx_begin
variable betas(ndim)
minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(1:end-1) - betas(2:end),1)
cvx_end

figure; plot(y,'b-'); hold on;
plot(betas,'r-','LineWidth',3); hold off;
axis([0 size(im_rgb,1) -pi pi]);
legend('raw theta', 'fused lasso');set(gca,'FontSize',20);

%%
% smooth cosine
indx = 300;
y = im_cosine(indx,:)';
lamb = 10;
ndim = size(y,1);
%[ii jj] = getLocalPairs(im_size, 1,0,[]); % index of adjacent pixels
%minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(ii) - betas(jj),1)

cvx_begin
variable betas(ndim)
minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(1:end-1) - betas(2:end),1)
cvx_end

figure; plot(y,'b-'); hold on;
plot(betas,'r-','LineWidth',3); hold off;
axis([0 size(im_rgb,1) -1 1]);
legend('raw cosine', 'fused lasso');set(gca,'FontSize',20);

figure; imshow(im_rgb);
axis off; set(gca,'FontSize',20);
hold on; plot(1:size(im_rgb,1), ones(size(im_rgb,1),1)*indx,'k-','LineWidth',3)

%% set up a loop to smooth out the image horizontally
down_size = 8;
im_size = size(r);
im_sine_small = im_sine(1:down_size:end,1:down_size:end);
im_cosine_small = im_cosine(1:down_size:end,1:down_size:end);
im_sat_small = im_sat(1:down_size:end,1:down_size:end);%im_size(1)/2
im_theta_small = im_theta(1:down_size:end,1:down_size:end);

im_size_small = size(im_sat_small);
im_brightness_small = im_brightness(1:down_size:end,1:down_size:end);
cosine_smooth = zeros(im_size_small);
sine_smooth =  zeros(im_size_small);
sat_smooth = zeros(im_size_small);
br_smooth = zeros(im_size_small);

lamb = 2;lamb_sat = 1; lamb_br = 2;
ndim = im_size_small(1);
T = tic;
for indx = 1:im_size_small(1)
    y = im_sine_small(indx,:)';
    cvx_begin
    variable betas(ndim)
    minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(1:end-1) - betas(2:end),1)
    cvx_end
    sine_smooth(indx,:) = betas';
    
    y = im_cosine_small(indx,:)';
    cvx_begin
    variable betas(ndim)
    minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(1:end-1) - betas(2:end),1)
    cvx_end
    cosine_smooth(indx,:) = betas';
    
    y = im_sat_small(indx,:)';
    cvx_begin
    variable betas(ndim)
    minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb_sat*norm(betas(1:end-1) - betas(2:end),1)
    cvx_end
    sat_smooth(indx,:) = betas';
    
    y =  im_brightness_small(indx,:)';
    cvx_begin
    variable betas(ndim)
    minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb_br*norm(betas(1:end-1) - betas(2:end),1)
    cvx_end
    br_smooth(indx,:) = betas';
end
t = toc(T); fprintf('Done in %.2f seconds \n', t);

% now vertical
T = tic;
for indx = 1:im_size_small(1)
    y = sine_smooth(:,indx);
    cvx_begin
    variable betas(ndim)
    minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(1:end-1) - betas(2:end),1)
    cvx_end
    sine_smooth(:,indx) = betas;
    
    y = cosine_smooth(:,indx);
    cvx_begin
    variable betas(ndim)
    minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(1:end-1) - betas(2:end),1)
    cvx_end
    cosine_smooth(:,indx) = betas;
    
    y = sat_smooth(:,indx);
    cvx_begin
    variable betas(ndim)
    minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb_sat*norm(betas(1:end-1) - betas(2:end),1)
    cvx_end
    sat_smooth(:,indx) = betas;
    
    y = br_smooth(:,indx);% im_brightness_small(:,indx);
    cvx_begin
    variable betas(ndim)
    minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb_br*norm(betas(1:end-1) - betas(2:end),1)
    cvx_end
    br_smooth(:,indx) = betas;
end
t = toc(T); fprintf('Done in %.2f seconds \n', t);

[ii jj] = getLocalPairs(im_size_small, 1,0,[]); % index of adjacent pixels
y = im_cosine_small(:);

ndim = length(y);
cvx_begin
variable betas(ndim)
minimize 1/2*pow_pos(norm(y-betas,2),2) + lamb*norm(betas(ii) - betas(jj),1)
cvx_end


theta_smooth = atan2(sine_smooth, cosine_smooth);
rotated_coordinate_smooth = zeros(3,im_size_small(1)*im_size_small(2));
rotated_coordinate_smooth(2,:) = im_sat_small(:).*cosine_smooth(:);
rotated_coordinate_smooth(3,:) = im_sat_small(:).*sine_smooth(:);
rotated_coordinate_smooth(1,:) = im_brightness_small(:);
rgb_smooth = rotation_matrix.rotation_matrix\rotated_coordinate_smooth;
rgb_im_smooth = reshape(rgb_smooth,[im_size_small(1) im_size_small(2) 3]);

theta_smooth = angle(cosine_smooth + 1i*sine_smooth);
rotated_coordinate_smooth = zeros(3,im_size_small(1)*im_size_small(2));
rotated_coordinate_smooth(2,:) = sat_smooth(:).*cosine_smooth(:);
rotated_coordinate_smooth(3,:) = sat_smooth(:).*sine_smooth(:);
rotated_coordinate_smooth(1,:) = br_smooth(:);%  im_brightness_small(:);
rgb_smooth = (rotation_matrix.rotation_matrix\rotated_coordinate_smooth)';
rgb_im_smooth = reshape(rgb_smooth,[im_size_small(1) im_size_small(2) 3]);

%check what is wrong with the reconstruction
rotated_coordinate_smooth(2,:) = im_sat_small(:).*im_cosine_small(:);
rotated_coordinate_smooth(3,:) = im_sat_small(:).*im_sine_small(:);
rotated_coordinate_smooth(1,:) = im_brightness_small(:);
rgb_smooth = (rotation_matrix.rotation_matrix\rotated_coordinate_smooth)';
rgb_im_smooth = reshape(rgb_smooth,[im_size_small(1) im_size_small(2) 3]);




