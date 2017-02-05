%% script for preprocessing images for deep learning
src_dir = 'X:\SpatialPathology\';
IMG_DIR = fullfile(src_dir,'OriginalImages');

% preprocessing 4 classes
data_fname = 'BreastCancerCases_UDH_DCIS_167_grades.csv';
T = readtable(fullfile(src_dir,'lists',data_fname),'Delimiter',',','ReadVariableNames',true);
T.Grade(isnan(T.Grade)) = 0;
T.Grade(T.Grade > 1 & T.Grade < 2) = 1;
T.Grade(T.Grade > 2 & T.Grade < 3) = 2;
train_set_indx = strcmp(T.Hospital,'MGH');
test_set_indx = strcmp(T.Hospital,'BIDMC');

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
classGrade = -ones(length(all_im_names),1); % 0 is UDH, 1 is DCIS
classes = cell(length(all_im_names),1);
hospital_ids = -ones(length(all_im_names),1); % 1 is MGH, 0 is BIDMC
% if i can't find a match then it will be discarded

% train set and test set are matched differently
for i = 1:length(im_names_csv)
    train_eg = train_set_indx(i);
    casename = im_names_csv{i};
    if train_eg
        match_id = strncmp({[casename ' ']}, all_im_names, length(casename) + 1);
    else
        match_id = strcmp(casename, all_im_names);
    end
    classGrade(match_id) =  T.Grade(i);
    classes(match_id) = T.Diagnosis(i);
    hospital_ids(match_id) = train_eg;
end

non_matched_im = classGrade == -1;
all_im_names(non_matched_im) = [];
classGrade(non_matched_im) = [];
hospital_ids(non_matched_im) = [];
classes(non_matched_im) = [];
im_id = 1:length(all_im_names);
% [~,sorted_indx] = sort(classGrade,'ascend');
% all_im_names = all_im_names(sorted_indx);
% classGrade = classGrade(sorted_indx);
% hospital_ids = hospital_ids(sorted_indx);
names_labels_T = table(im_id', all_im_names,classes,classGrade,hospital_ids,...
    'VariableNames',{'CaseID','CaseName','Class','ClassGrade','Hospital'});
%writetable(names_labels_T,'names_labels_4_classes_hospitals.csv');
writetable(names_labels_T,fullfile(src_dir, 'UDH_DCIS.csv'));
%% tiling images
tile_size = 256;
output_dir = fullfile(src_dir,['tiles' num2str(tile_size)]);
if ~exist(output_dir,'dir'); mkdir(output_dir); end;
% [all_tiles_names, all_tiles_labels] = ...
%   tiling_images(all_im_names, class_labels, tile_size, IMG_DIR, output_dir);
%save(sprintf('tiles_names_%d.mat',tile_size),'all_tiles_names');
%save(sprintf('tiles_labels_%d.mat',tile_size),'all_tiles_labels');

load(fullfile(src_dir,'lists',sprintf('tiles_labels_%d.mat',tile_size)));
%% create train/test set
%indices = crossvalind('Kfold',class_labels,5);
%save(sprintf('indices_grade_tile_%d.mat',tile_size),'indices');
% We already know the train/test list: MGH-train_indx_set
load(fullfile(src_dir,'lists',sprintf('tiles_names_%d.mat',tile_size)));
train_im_indx = hospital_ids == 1;
test_im_indx = hospital_ids == 0;

% load(sprintf('indices_grade_tile_%d.mat',tile_size),'indices');
% test_im_indx = (indices == 1); train_im_indx = ~test_im_indx;

%f_train = fopen(sprintf('train_grades_%d.txt',tile_size));
%f_test = fopen(sprintf('test_grades_%d.txt',tile_size)); 
%train_tiles_id = cell(length(all_tiles_names),1);
all_tiles_Class = cell(length(all_tiles_names),1);
all_tiles_ClassGrade = cell(length(all_tiles_names),1);
ClassLabel = strcmp('DCIS',classes);
% recalculate the tile labels
for i = 1:length(all_tiles_names)
    all_tiles_Class{i} = ones(size(all_tiles_names{i}))*ClassLabel(i);
    all_tiles_ClassGrade{i} = ones(size(all_tiles_names{i}))*classGrade(i);
end

non_empty_cells = ~cellfun('isempty',all_tiles_names);
all_tiles_Class = all_tiles_Class(non_empty_cells);
all_tiles_names = all_tiles_names(non_empty_cells);
all_tiles_ClassGrade = all_tiles_ClassGrade(non_empty_cells);

train_tiles_id = cell(length(all_tiles_names));
for i = 1:length(all_tiles_names)
   train_eg = train_im_indx(i);  
   tiles_names = all_tiles_names{i};
   if isempty(tiles_names)
       train_tiles_id{i} = [];
   else
       train_tiles_id{i}(1:length(tiles_names),:) = train_eg;
   end
end

all_tiles_names = cat(1,all_tiles_names{:});
all_tiles_Class = cat(1,all_tiles_Class{:});
all_tiles_ClassGrade = cat(1,all_tiles_ClassGrade{:});
train_tiles_id = logical(cat(1,train_tiles_id{:}));
non_empty_ind = ~cellfun('isempty',all_tiles_names);
all_tiles_names = all_tiles_names(non_empty_ind);
all_tiles_Class = all_tiles_Class(non_empty_ind);
all_tiles_ClassGrade = all_tiles_ClassGrade(non_empty_ind);
train_tiles_id = train_tiles_id(non_empty_ind);
tiles_T = table(all_tiles_names,all_tiles_Class, train_tiles_id,...
    'VariableNames',{'TileNames','Classes','train_id'});

train_tiles_T = table(all_tiles_names(train_tiles_id),all_tiles_Class(train_tiles_id));
writetable(train_tiles_T,fullfile(src_dir,'lists',sprintf('train_%d.txt',tile_size)),'WriteVariableNames',false,'Delimiter',' ');
test_tiles_T = table(all_tiles_names(~train_tiles_id),all_tiles_Class(~train_tiles_id));
writetable(test_tiles_T,fullfile(src_dir,'lists',sprintf('test_%d.txt',tile_size)),'WriteVariableNames',false,'Delimiter',' ');

train_tiles_T = table(all_tiles_names(train_tiles_id),all_tiles_ClassGrade(train_tiles_id));
writetable(train_tiles_T,fullfile(src_dir,'lists',sprintf('train_grade_%d.txt',tile_size)),'WriteVariableNames',false,'Delimiter',' ');
test_tiles_T = table(all_tiles_names(~train_tiles_id),all_tiles_ClassGrade(~train_tiles_id));
writetable(test_tiles_T,fullfile(src_dir,'lists',sprintf('test_grade_%d.txt',tile_size)),'WriteVariableNames',false,'Delimiter',' ');

% test_dir = fullfile(output_dir,'test');
% if ~exist(test_dir,'dir'); mkdir(test_dir); end;
% for i = 1:length(test_tiles_T.Var1)
%    movefile(fullfile(output_dir,test_tiles_T.Var1{i}),test_dir); 
% end
% 
% train_dir = fullfile(output_dir,'train');
% if ~exist(train_dir,'dir'); mkdir(train_dir); end;
% 
% movefile(fullfile(output_dir,'*.tif'),train_dir);

%% accuracy
% need to calculate the proportion of tiles of each grades in the image
% use this proportion to make the prediction or to plot the auc

test_im_indx = (indices == 1); train_im_indx = ~test_im_indx;
gtT = readtable('test_grade_256.txt','Delimiter',' ','ReadVariableNames',false);
outputT = readtable('cls_results_beck_4classes_256.txt','Delimiter',' ','ReadVariableNames',false);

test_im_names = all_im_names(test_im_indx);
test_labels = classGrade(test_im_indx);
% find indices of tiles from the same image
output_labels = -ones(size(test_labels));
gt_labels = -ones(size(test_labels));
for i = 1:length(test_im_names)
    imname = test_im_names{i};
    imname = strrep(imname,' ','-');
    match_id = strncmp({imname},outputT.Var1,length(imname));
    if sum(match_id) == 0; continue; end
    output_labels(i) = mode(outputT.Var2(match_id));
    gt_labels(i) = mode(gtT.Var2(match_id));
end

test_labels(output_labels == -1) = [];
output_labels(output_labels == -1) = []; gt_labels(output_labels == -1) = [];
acc = sum(test_labels == output_labels)/length(output_labels);

%% create the train, test list
T = readtable(fullfile(src_dir,'UDH_DCIS_table_new.csv'),'Delimiter',',','ReadVariableNames',true);
tile_size = 128;
output_dir = fullfile(src_dir,['tiles' num2str(tile_size)]);
tiles_name = dir(fullfile(output_dir,'*.tif'));
tiles_name = {tiles_name.name}';

arranged_tiles_name = cell(length(T.CaseName),1);
tiles_Class = cell(length(T.CaseName),1);
tiles_ClassGrade = cell(length(T.CaseName),1);
tiles_Hospital = cell(length(T.CaseName),1);

for i = 1:length(T.CaseName)
   imname = T.CaseName{i};
   imname = strrep(imname,' ','-');
   match_id = strncmp(imname, tiles_name, length(imname));
   tile_count = sum(match_id);
   if tile_count > 0
       arranged_tiles_name{i} = tiles_name(match_id);
       tiles_Class{i} = repmat(T.Class{i},[tile_count, 1]);
       tiles_ClassGrade{i} = ones(tile_count,1)*T.ClassGrade(i);
       tiles_Hospital{i} =  ones(tile_count,1)*T.Hospital(i);
   end
end

arranged_tiles_name = cat(1, arranged_tiles_name{:});
% tiles_Class = cat(1, tiles_Class{:});
tiles_ClassGrade = cat(1, tiles_ClassGrade{:});
tiles_Hospital = cat(1, tiles_Hospital{:});

tilesT = table(arranged_tiles_name,tiles_ClassGrade, tiles_Hospital);
tilesT.Properties.VariableNames = {'TileName','ClassGrade','Hospital'};
writetable(tilesT, fullfile(src_dir,'lists',['tiles', tile_size '.csv']),...
    'delimiter',' ','WriteVariableNames',true);   

%% create train test table
tilesT.ClassLabel = double(tilesT.ClassGrade > 0);
writetable(table(tilesT.TileName(tilesT.Hospital == 1),tilesT.ClassLabel(tilesT.Hospital == 1)),...
    fullfile(src_dir,'lists',['train_' num2str(tile_size) '_bin.txt']),...
    'delimiter',' ','WriteVariableNames',false);
writetable(table(tilesT.TileName(tilesT.Hospital == 0),tilesT.ClassLabel(tilesT.Hospital == 0)),...
    fullfile(src_dir,'lists',['test_' num2str(tile_size) '_bin.txt']),...
    'delimiter',' ','WriteVariableNames',false);

writetable(table(tilesT.TileName(tilesT.Hospital == 1),tilesT.ClassGrade(tilesT.Hospital == 1)),...
    fullfile(src_dir,'lists',['train_' num2str(tile_size) '_4way.txt']),...
    'delimiter',' ','WriteVariableNames',false);
writetable(table(tilesT.TileName(tilesT.Hospital == 0),tilesT.ClassGrade(tilesT.Hospital == 0)),...
    fullfile(src_dir,'lists',['test_' num2str(tile_size) '_4way.txt']),...
    'delimiter',' ','WriteVariableNames',false);

%% calculate the accuracy
% need to calculate the proportion of tiles of each grades in the image
% use this proportion to make the prediction or to plot the auc
T = readtable(fullfile(src_dir,'UDH_DCIS_table_new.csv'),'Delimiter',',','ReadVariableNames',true);
tile_size = 256;

% binary results
table_name = fullfile(src_dir,'lists',['cls_results_' num2str(tile_size) '_4way.txt']);
%table_name = fullfile(src_dir,'lists',['cls_results_' num2str(tile_size) '_bin.txt']);

resultT = readtable(table_name,'ReadVariableNames',false,'Delimiter',' ');
resultT.Properties.VariableNames = {'TileName','ClassLabel','softmax'};

num_classes = 4;
test_images = T.CaseName(T.Hospital == 0);
probas_classes = zeros(length(test_images),num_classes);
for i = 1:length(test_images)
   imname = test_images{i};
   imname = strrep(imname,' ','-');
   match_id = strncmp(imname, resultT.TileName, length(imname));
   tile_count = sum(match_id);
   if tile_count > 0
       for j = 1:num_classes
           probas_classes(i,j) = sum(resultT.ClassLabel(match_id) == j-1)/tile_count;
       end
   end
end


T.ClassLabel = double(T.ClassGrade > 0);
gtLabel = T.ClassGrade(T.Hospital == 0);
[fpr, tpr, thres, auc] = perfcurve(gtLabel, probas_classes(:,2),1);
figure; plot(fpr,tpr);




