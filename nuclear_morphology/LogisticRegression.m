%% statistical classification for UDH DCIS
data_dir = 'X:\SpatialPathology';

%% Beck's data
data_fname = 'ComputedFeatures_Label.csv';
dataT = readtable(fullfile(data_dir, data_fname),'Delimiter',',','ReadVariableNames',true);
data = table2array(dataT(:,1:end-4));
data_train = data(1:116,:);
data_test = data(117:end,:);
ClassLabel = strncmp('DCIS',dataT.Class,3);
ClassLabel_train = ClassLabel(1:116);
ClassLabel_test = ClassLabel(117:end);

PredictorNames = dataT.Properties.VariableNames(1:end-4);
% normalize the data
% default is always true in matlab
X = data_train;
y = ClassLabel_train;
tic;
[B,FitInfo] = lassoglm(X,y,'binomial',...
    'NumLambda',10,'CV',9,'PredictorNames',PredictorNames);
toc
lassoPlot(B,FitInfo,'PlotType','CV');

% use the lambda value with minimum deviance plus one standard deviation point. 
indx = FitInfo.Index1SE;
B0 = B(:,indx);
nonzeros = sum(B0 ~= 0)
PredictorNames(B0 ~=0);
% build the model 
cnst = FitInfo.Intercept(indx);
B1 = [cnst;B0];
preds = glmval(B1,X,'logit');

% alternative way that will get me scores
predictors = find(B0);
mld = fitglm(X, y, 'linear','Distribution','binomial',...
    'PredictorVars',predictors);%, 'VarNames',cat(2,PredictorNames,'ClassLabel'));
scores = mld.Fitted.Probability;
[fpr, tpr, thres, auc] = perfcurve(y, scores,1);


