close all;clear;
clc;

% Ucitavanje podataka 
dataset = importdata('messidor_features.arff');
data = dataset.data;
features = [1,2,3,4,8,9,10,16,17,18,19];
data = data(:,features);

X = data(:,1:end-1)';
X = (X-mean(X,2))./(sqrt(var(X')))';
Y = data(:,end)';
Y(Y==0)=-1;

Xtrainval = X(:, 1:round(0.85*size(X,2)));  %zajedno train i val
ytrainval = Y(1:round(0.85*size(X,2))); 
Xtrain = X(:, 1:round(0.7*size(X,2)));  %zajedno train i val
ytrain = Y(1:round(0.7*size(X,2))); 
Xval = X(:, round(0.7*size(X,2))+1:round(0.85*size(X,2))); 
yval = Y(round(0.7*size(X,2))+1:round(0.85*size(X,2)));
Xtest = X(:, round(0.85*size(X,2))+1:end); 
ytest = Y(round(0.85*size(X,2))+1:end);

%% Jedan trening

structure = 5;
reg = 0;

net = newff(X, Y, structure,{'tansig'});
net.divideParam.trainRatio = 0.8; 
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;
%net.divideFcn = ''; %nema podele
%net.performParam.regularization = reg;
net.trainParam.epochs = 500;
net.trainParam.goal = 1e-6; % ciljana greska
net.performFcn = 'msereg';
%net.trainParam.show = 5; % na koliko epoha da prikazuje rezultate 


% Obucavanje mreze
[net, tr] = train(net, Xtrainval, ytrainval);
%[net, tr] = train(net, Xtrain, ytrain);

%%

threshold = -0.3;

% Validacija
yout = sim(net, X(:,tr.valInd));
%yout = sim(net, Xval);
yout(yout < threshold) = -1; 
yout(yout >= threshold) = 1;
Mv = confusionmat(Y(tr.valInd), yout);
%Mv = confusionmat(yval, yout);
accv = trace(Mv)/(sum(sum(Mv)));

yout_confusion = zeros(2,length(yout));
yout_confusion(1,:) = (yout < 0)*1;
yout_confusion(2,:) = (yout >= 0)*1;
Yv_confusion = zeros(2,length(Y(tr.valInd)));
Yv_confusion(1,:) = (Y(tr.valInd) < 0)*1;
Yv_confusion(2,:) = (Y(tr.valInd) >= 0)*1;
%Yv_confusion = zeros(2,length(yval));
%Yv_confusion(1,:) = (yval < 0)*1;
%Yv_confusion(2,:) = (yval >= 0)*1;
figure
plotconfusion(Yv_confusion,yout_confusion)

yout = sim(net, X(:,tr.trainInd));
yout(yout < threshold) = -1; 
yout(yout >= threshold) = 1;
Mt = confusionmat(Y(tr.trainInd), yout);
acct = trace(Mt)/(sum(sum(Mt)));

yout_confusion = zeros(2,length(yout));
yout_confusion(1,:) = (yout < 0)*1;
yout_confusion(2,:) = (yout >= 0)*1;
Yt_confusion = zeros(2,length(Y(tr.trainInd)));
Yt_confusion(1,:) = (Y(tr.trainInd) < 0)*1;
Yt_confusion(2,:) = (Y(tr.trainInd) >= 0)*1;
figure
plotconfusion(Yt_confusion,yout_confusion)


%% Test set

yout = sim(net, Xtest);
yout(yout < threshold) = -1; 
yout(yout >= threshold) = 1;
M = confusionmat(ytest, yout);
acc_t = trace(M)/(sum(sum(M)))

yout_confusion = zeros(2,length(yout));
yout_confusion(1,:) = (yout < 0)*1;
yout_confusion(2,:) = (yout >= 0)*1;
Y_confusion = zeros(2,length(ytest));
Y_confusion(1,:) = (ytest < 0)*1;
Y_confusion(2,:) = (ytest >= 0)*1;
figure
plotconfusion(Y_confusion,yout_confusion)