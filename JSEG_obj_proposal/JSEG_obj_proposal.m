% generate output at OIS for best images in the test set
% Luong Nguyen 07/23/2015
% this is to test why the performanc isn't very good
% UPDATE: 10/12/15
% UPDATE: 2/29/16
% Use JSEG ms results as object proposals. 
%% mac
% githubdir = '/Users/lun5/Research/github/HE-tumor-object-segmentation';
% addpath(genpath(githubdir)); cd(githubdir)
% seismdir = '/Users/lun5/Research/github/seism'; addpath(genpath(seismdir));
% DATA_DIR = '/Users/lun5/Research/data/';
% IMG_DIR = fullfile(DATA_DIR,'TilesForLabeling_tiff_renamed','test');%'/home/lun5/HEproject/data/images/test';
% GT_DIR = fullfile(DATA_DIR,'groundTruth','groundTruth_512_512_reannotated','best_images_july30');%fullfile(DATA_DIR,'data','groundTruth_512_512');
% RESULTS_DIR = fullfile(DATA_DIR,'evaluation_results','eval_reannotated');
%% window
%githubdir = 'C:\Users\luong_nguyen\Documents\GitHub\HE-tumor-object-segmentation'; % window
%seismdir = 'C:\Users\luong_nguyen\Documents\GitHub\seism'; 
%addpath(genpath(seismdir)); cd(githubdir)
%DATA_DIR = 'Z:\';
%IMG_DIR = 'Z:\Tiles_512\Test';
%GT_DIR = 'Z:\HEproject\data\groundTruth_512_512';
%RESULTS_DIR = fullfile(DATA_DIR,'HEproject','evaluation_results','Isola_lowres_accurate');
%gt_display = fullfile(DATA_DIR,'groundTruth','groundTruth_reannotated_display');
%outDir = fullfile(RESULTS_DIR,'best_boundary');

%% linux
% githubdir = '/home/lun5/github/HE-tumor-object-segmentation';
% addpath(genpath(githubdir));cd(githubdir);
% seismdir = '/home/lun5/github/seism'; addpath(genpath(seismdir));% linux
% bsrdir = '/home/lun5/github/BSR/grouping';addpath(genpath(bsrdir));
% DATA_DIR ='/home/lun5/HEproject/'; % linux
% IMG_DIR = '/home/lun5/HEproject/data/Tiles_512/Test';

%IMG_DIR = '/home/lun5/HEproject/data/Tiles_512/';
githubdir = 'C:\Users\luong_nguyen\Documents\GitHub\HE-tumor-object-segmentation'; % window
seismdir = 'C:\Users\luong_nguyen\Documents\GitHub\seism'; 
addpath(genpath(seismdir)); cd(githubdir)
DATA_DIR = 'Z:\HEproject';
IMG_DIR = 'Z:\Tiles_512\Test';
% GT_DIR = 'Z:\HEproject\data\groundTruth_512_512';
% RESULTS_DIR = fullfile(DATA_DIR,'evaluation_results','EGB');
%GT_DIR = fullfile(DATA_DIR,'groundTruth','groundTruth_512_512_reannotated_Oct', 'best_images_july30');
%GT_DIR = fullfile(DATA_DIR,'groundTruth','groundTruth_512_512');

GT_DIR = 'Z:\TilesForLabeling_bestImages\groundTruth_512_512_reannotated_Oct\best_images_july30';
RESULTS_DIR = cell(12,1);
RESULTS_DIR{1} = fullfile(DATA_DIR,'evaluation_results','eval_PMI_hue_offset');
RESULTS_DIR{2} = fullfile(DATA_DIR,'evaluation_results','eval_PJoint_hue_fullscale');
RESULTS_DIR{3} = fullfile(DATA_DIR,'evaluation_results','Isola_lowres_accurate');
RESULTS_DIR{4} = fullfile(DATA_DIR,'evaluation_results','Isola_speedy');
RESULTS_DIR{5} = fullfile(DATA_DIR,'evaluation_results','bsr');
RESULTS_DIR{6} = fullfile(DATA_DIR,'evaluation_results','JSEG','new_params','scale1');
RESULTS_DIR{7} = fullfile(DATA_DIR,'evaluation_results','JSEG','new_params','scale2');
RESULTS_DIR{8} = fullfile(DATA_DIR,'evaluation_results','ncut_multiscale_1_6');
RESULTS_DIR{9} = fullfile(DATA_DIR,'evaluation_results','EGB','seism_params');
RESULTS_DIR{10} = fullfile(DATA_DIR,'evaluation_results','QuadTree');
RESULTS_DIR{11} = fullfile(DATA_DIR,'evaluation_results','MeanShift');
RESULTS_DIR{12} = fullfile(DATA_DIR,'evaluation_results','GraphRLM','new_params');

method_names = {'H&E-hue-PMI','H&E-hue-PJoint',...
    'Isola-lowres-acc','Isola-speedy','gPb',...
   'JSEG-ss','JSEG-ms','NCut','EGB','QuadTree','MShift','GraphRLM'};
%RESULTS_DIR = fullfile(DATA_DIR,'evaluation_results','Isola_multiscale');
img_list = dirrec(GT_DIR,'.mat');
for med = 1:length(RESULTS_DIR)
    fprintf('\n\nCalculate best segmentation for methods %s...\n',method_names{med}); T = tic;
    EV_DIR = fullfile(RESULTS_DIR{med},'ev_txt');
    
    eval_bdry_img = dlmread(fullfile(EV_DIR,'eval_bdry_img.txt'));
    bdry_outDir = fullfile(RESULTS_DIR{med},'best_boundary_300_pad');
    if ~exist(bdry_outDir,'dir')
        mkdir(bdry_outDir)
    end    
    best_bdry_thres = eval_bdry_img(:,2);
    best_bdry_F = eval_bdry_img(:,4);
    %fprintf('Number of bdry thres is %d\n',length(best_bdry_thres));
    eval_Fop_img = dlmread(fullfile(EV_DIR,'eval_Fop_img.txt'));
    Fop_outDir = fullfile(RESULTS_DIR{med},'best_Fop_300_pad');
    if ~exist(Fop_outDir,'dir')
        mkdir(Fop_outDir)
    end    
    best_Fop_thres = eval_Fop_img(:,2);   
    best_Fop_F = eval_Fop_img(:,4); 
    %fprintf('Number of Fop thres is %d\n',length(best_Fop_thres));
    %IMG_EXT = '.tif';
    %img_list = dirrec(IMG_DIR,IMG_EXT);
    UCM_DIR = fullfile(RESULTS_DIR{med},'ucm2');
    SEG_DIR = fullfile(RESULTS_DIR{med},'segmented_images');
    parfor i = 1:numel(img_list)
        [~,im_name,~] = fileparts(img_list{i}); im_name = lower(im_name);       
        bdry_outFile = fullfile(bdry_outDir,[im_name, '.tif']);
        Fop_outFile = fullfile(Fop_outDir,[im_name, '.tif']);
        %if ~exist(bdry_outFile,'file') || ~exist(Fop_outFile,'file')
            %I = imread(img_list{i});
            I = imread(fullfile(IMG_DIR,[im_name '.tif']));
            if exist(UCM_DIR,'dir')
                tmp = load(fullfile(UCM_DIR,[im_name '.mat']));
                ucm2 = tmp.data; ucm2 = ucm2(3:2:end,3:2:end);
                bdry_thr = best_bdry_thres(i);
                bdry_edge_map = (ucm2>=bdry_thr);
                Fop_thr = best_Fop_thres(i);
                Fop_edge_map = (ucm2>=Fop_thr);
            else
                tmp = load(fullfile(SEG_DIR,[im_name '.mat']));
                segs = tmp.data;
                bdry_thr = floor(best_bdry_thres(i));
                bdry_edge_map = edge(segs{bdry_thr});
                Fop_thr = floor(best_Fop_thres(i));
                Fop_edge_map = edge(segs{Fop_thr});
            end
            bdry_edge_map = imdilate(bdry_edge_map, strel('disk',1));
            bdry_edge_map_im = I.*uint8(repmat(~bdry_edge_map,[1 1 3]));
            pad_im = padarray(bdry_edge_map_im,[60 60],255,'both');
            pad_im = insertText(pad_im,[150 5],method_names{med},'FontSize',50,'BoxColor','white');
            pad_im = insertText(pad_im,[240 570],...
                sprintf('%.2f',best_bdry_F(i)),'FontSize',50,'BoxColor','white');
            imwrite(pad_im,bdry_outFile,'Resolution',300);
            
            Fop_edge_map = imdilate(Fop_edge_map, strel('disk',1));
            Fop_edge_map_im = I.*uint8(repmat(~Fop_edge_map,[1 1 3]));
            pad_im = padarray(Fop_edge_map_im,[60 60],255,'both');
            pad_im = insertText(pad_im,[150 5],method_names{med},'FontSize',50,'BoxColor','white');
            pad_im = insertText(pad_im,[240 570],...
                sprintf('%.2f',best_Fop_F(i)),'FontSize',50,'BoxColor','white');
            imwrite(pad_im,Fop_outFile,'Resolution',300);
            %imwrite(label2rgb(labels),fullfile(output_dir,'seg_im',[im_name, '_' num2str(q_thresh), '_seg.jpg']));
        %end        
    end
    t = toc(T); fprintf('done: %1.2f sec\n', t);
end
disp('Done');