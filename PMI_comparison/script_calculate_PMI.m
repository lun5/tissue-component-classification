%  script to calculate PMI for each segment

mainfolder =pwd ;
dirData = dir( fullfile(mainfolder,'*.tif') );      %# Get the data for the current directory
fileList = {dirData.name}';  %'# Get a list of the files

noOfTiles=0;
param_string = '_se1_minNuc3_minStr5_minLum5';

for fileIndex=length(fileList)    
    imageName = fileList{fileIndex}(1:end-4);
    load(fullfile(mainfolder,[imageName param_string '_segmentationTilesResult.mat']));
    cntim=1;
    adj_map = dlmread([imageName param_string '_adjDela'],',',0,0);
    adj_map(:,[5,4]) = adj_map(:,[5,4]) + 500; 
    for i=1:size(seg,1)
        ss = seg{i};
        if ~isempty(ss)
            tmpim = imread(fullfile(mainfolder,[imageName '.tif']),...
                'PixelRegion', {[ss(3) ss(4)],[ss(1) ss(2)]});
            tmpim_gray = mean(tmpim, 3);
            if sum(tmpim_gray(:) > 210)/numel(tmpim_gray) < 0.7
                indx =  adj_map(:,3) == 1 & adj_map(:,5) >= ss(1) & adj_map(:,5) <= ss(2) & ...
                    adj_map(:,4) >= ss(3) & adj_map(:,4) <= ss(4);
                adj_map_tile = adj_map(indx,:);
                %tic;
                [bdry_im, mask] = object_proposal_tile(tmpim, ss, adj_map_tile,1,30,40);
                
                % calculate the features               
                %fprintf('Done with seg %d in %.2f seconds\n',cntim, toc);
                figure; imshow(bdry_im);
                pause(3);
                close all;
                %tic; 
                features = calculate_PMI_features_tile(tmpim, mask);
                %fprintf('Done with feature calculation seg %d in %.2f seconds\n',cntim,toc);
                %if sum(mean(Datam))<650 &&  sum(var(Datam))> 500
                %imwrite(tmpim,fullfile(mainfolder,[imageName '_seg_' num2str(cntim) '.jpg'],'jpg'));
                imwrite(bdry_im,fullfile(mainfolder,[imageName '_bdry_' num2str(cntim) '.jpg']));
                save(fullfile(mainfolder,[imageName '_features_' num2str(cntim) '.mat']),'features');
                cntim=cntim+1;
                noOfTiles=noOfTiles+1;                
            end
        end
    end    
end