function features = calculate_PMI_features_tile(image, mask)

rotation_matrix = load('rotation_matrix_tp10-867-1.mat','rotation_matrix');
rotation_matrix = rotation_matrix.rotation_matrix;

im_rgb = double(image)./255;
nrows = size(im_rgb,1); ncols = size(im_rgb,2);
X = reshape(im_rgb,[nrows*ncols,3]);
rotated_coordinates = rotation_matrix*X';

%% cluster using von Mises
theta = angle(rotated_coordinates(2,:) + 1i*rotated_coordinates(3,:));

% initial parameters for bivariate von mises
numClusters = 3;
opts_mixture = struct('maxiter',20,'noise',1,'mask',mask>0);
[ mu_hat_polar,~, kappa_hat,~, prior_probs,conv] =  moVM([cos(theta); sin(theta)]',numClusters,opts_mixture);

% sample pixels
Nsamples=10000; opts.model_half_space_only = 0; opts.sig = 3;
theta_im = reshape(theta,nrows,ncols);
[F,~,~] = sampleF(theta_im,Nsamples,opts,mask);
init_params = struct('theta_hat',mu_hat_polar,'kappa_hat',kappa_hat);
init_params.prior_probs = prior_probs;
features = [mu_hat_polar,kappa_hat, init_params.prior_probs(1:2)]; % 3+3+2 = 8 features
[ params,~, prior_probs] = mixture_of_bivariate_VM(F,9,init_params);
features = [features,params.mu',params.nu', params.kappa1', params.kappa2', params.kappa3',prior_probs(1:5),sum(mask(:))/4e6]; % 8 + 9*5 + 5 + 1 = 
end