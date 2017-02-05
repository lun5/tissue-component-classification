%% script for preprocessing images for deep learning
src_dir = 'D:\Documents\SpatialPathology';
IMG_DIR = fullfile(src_dir,'OriginalImages');

%% need to generate the test and train list from the xlxs file
data_fname = 'BreastCancerCases_UDH_DCIS_167.csv';
T = readtable(fullfile(src_dir,'lists',data_fname),'Delimiter',',','ReadVariableNames',true);
train_set_indx = strcmp(T.Hospital,'MGH');
test_set_indx = strcmp(T.Hospital,'BIDMC');
udh_case_id = strcmp(T.Diagnosis,'UDH');
dcis_case_id = strcmp(T.Diagnosis,'DCIS');

% weird naming convention, both tif and tiff
tif_im_names = dir(fullfile(IMG_DIR,'*.tif'));
tif_im_names = {tif_im_names.name}';
tif_im_names = cellfun(@(x) x(1:end-4), tif_im_names,'UniformOutput',false);

tiff_im_names = dir(fullfile(IMG_DIR,'*.tiff'));
tiff_im_names = {tiff_im_names.name}';
tiff_im_names = cellfun(@(x) x(1:end-5), tiff_im_names,'UniformOutput',false);
all_im_names = cat(1,tif_im_names,tiff_im_names);

% match the image names from the list to the folder
im_names_csv = T.CaseDigital;
class_labels = -ones(length(all_im_names),1); % 0 is UDH, 1 is DCIS
hospital_ids = -ones(length(all_im_names),1); 
% if i can't find a match then it will be discarded

% train set and test set are matched differently
for i = 1:length(im_names_csv)
    train_eg = train_set_indx(i);
    casename = im_names_csv{i};
    if train_eg
        match_id = strncmp({[casename ' ']}, all_im_names, length(casename) + 1);
    else
        match_id = strncmp({casename},all_im_names,length(casename));
    end
    class_id = dcis_case_id(i);
    class_labels(match_id) = class_id;
    hospital_ids(match_id) = train_eg;
end
% discard images that are not matched to any cases
non_matched_im = class_labels == -1;
all_im_names(non_matched_im) = [];
class_labels(non_matched_im) = [];
hospital_ids(non_matched_im) = [];
[~,sorted_indx] = sort(class_labels,'ascend');
all_im_names = all_im_names(sorted_indx);
class_labels = class_labels(sorted_indx);
hospital_ids = hospital_ids(sorted_indx);
names_labels_T = table(all_im_names,class_labels,hospital_ids,...
    'VariableNames',{'ImageNames','Classes','Hospitals'});
writetable(names_labels_T,'names_labels_udh_dcis_hospitals.csv');
%indices = crossvalind('Kfold',class_labels,5);
%test_im_indx = (indices == 1); train_im_indx = ~test_im_indx;

train_im_indx = ~test_im_indx;
train_im_name = all_im_names(train_im_indx);
test_im_name = T.CaseDigital(test_set_indx);
test_im_indx = ismember(all_im_names,test_im_name);
train_im_indx(non_matched_im) = [];
train_im_indx = train_im_indx(sorted_indx);test_im_indx = ~train_im_indx;
% chopping into tiles
train_dir = fullfile(src_dir,'train_grades');
test_dir =  fullfile(src_dir,'test_grades');
tile_size = 256;

%if ~exist(train_dir,'dir'); mkdir(train_dir); end
%if ~exist(test_dir,'dir'); mkdir(test_dir); end

%f_train = fopen('train.txt','w');
%f_test = fopen('test.txt','w');

%max_numDCIS = 1145;
%count_DCIS = 0;
% I should print out all the files before putting them into test/train list

tile_names = {};
labels = {};
split_list = {};
im_names = {};
tiles_count = 0;
for i = 1:length(all_im_names)
   imname = all_im_names{i};
   if ~exist(fullfile(IMG_DIR,[imname '.tiff']),'file')      
       im = imread(fullfile(IMG_DIR,[imname '.tif']));
   else
       im = imread(fullfile(IMG_DIR,[imname '.tiff']));
   end
   % change image name to omit the space
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
   
   % update the train/test list
   train_eg = train_im_indx(i);
   for r = 1:size(splitImage,1)
       for c = 1:size(splitImage,2)
           current_tile = splitImage{r,c};
           if size(current_tile,1) < tile_size || size(current_tile,2) < tile_size
               continue;
           end
           tile_name = sprintf('%s_%d_%d_%d_%d.tif',...
               imname,tiles_coords_r(r), tiles_coords_r(r) + blockVectorR(r),...
               tiles_coords_c(c), tiles_coords_c(c) + blockVectorC(c));
           tiles_count = tiles_count + 1;
           tile_names{tiles_count} = tile_name;
           im_names{tiles_count} = imname;
           labels{tiles_count} = class_labels(i);
           split_list{tiles_count} = train_eg;
           %if train_eg
           %    fprintf(f_train,'%s %d\n',tile_name, class_labels(i));
               %imwrite(current_tile,fullfile(train_dir,tile_name)); 
           %else
           %    fprintf(f_test,'%s %d\n',tile_name, class_labels(i));
               %imwrite(current_tile,fullfile(test_dir,tile_name)); 
           %end                    
       end
   end
   fprintf('Done with %s\n',imname);
end

%% evaluate classification results
%gtT = readtable('test.txt','Delimiter',' ','ReadVariableNames',false);
%outputT = readtable('cls_results_beck_2classes_balanced.txt','Delimiter',' ','ReadVariableNames',false);

gtT = readtable('test_grades.txt','Delimiter',' ','ReadVariableNames',false);
outputT = readtable('cls_results_beck_4classes.txt','Delimiter',' ','ReadVariableNames',false);

test_im_names = all_im_names(test_im_indx);
test_labels = class_labels(test_im_indx);
% find indices of tiles from the same image
output_labels = zeros(size(test_labels));
gt_labels = zeros(size(test_labels));
for i = 1:length(test_im_names)
    imname = test_im_names{i};
    imname = strrep(imname,' ','-');
    match_id = strncmp({imname},outputT.Var1,length(imname));
    output_labels(i) = mode(outputT.Var2(match_id));
    gt_labels(i) = mode(gtT.Var2(match_id));
end

acc = sum(test_labels == output_labels)/length(test_labels);

%%


