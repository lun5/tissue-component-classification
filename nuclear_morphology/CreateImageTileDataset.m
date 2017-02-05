
mainfolder =pwd ;
dirData = dir( fullfile(mainfolder,'*.tif') );      %# Get the data for the current directory
dirIndex = [dirData.isdir];  %# Find the index for directories
fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files

if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(mainfolder,x),...  %# Prepend path to files
        fileList,'UniformOutput',false);
end


dirData = dir( fullfile(mainfolder,'*segmentationTilesResult*') );      %# Get the data for the current directory
dirIndex = [dirData.isdir];  %# Find the index for directories
fileListResult = {dirData(~dirIndex).name}';  %'# Get a list of the files

if ~isempty(fileListResult)
    fileListResult = cellfun(@(x) fullfile(mainfolder,x),...  %# Prepend path to files
        fileListResult,'UniformOutput',false);
end


noOfTiles=0;

for fileIndex=1:length(fileList)
    
    imageName = fileList{fileIndex};
    load(fileListResult{fileIndex});
    cntim=1;
    for i=1:size(seg)
        ss = seg{i};
        if ~isempty(ss)
            tmpim = imread(imageName,'PixelRegion', {[ss(3) ss(4)],[ss(1) ss(2)]});
            [mrows, mcols, xx] =size(tmpim);

            Datam = double([reshape(tmpim(:,:,1),mrows*mcols,1) reshape(tmpim(:,:,2),mrows*mcols,1) reshape(tmpim(:,:,3),mrows*mcols,1)]);
            
            if sum(mean(Datam))<650 &&  sum(var(Datam))> 500
                
                
                imwrite(tmpim,[imageName(1:end-4) '_seg' num2str(cntim) '.jpg'],'jpg');
                cntim=cntim+1;
                noOfTiles=noOfTiles+1;
                
            end
        end
    end
    
end