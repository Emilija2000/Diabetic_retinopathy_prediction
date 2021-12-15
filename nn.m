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

%% Uticaj razlicitih velicina - strukture, reg parametar

%struc = 1:30;
structure = [10 10];
reg = [0.001 0.003 0.01 0.03 0.1 0.3 0.5];
%reg = 0.01;
L = length(reg);
%L=length(struc);
acc_v = zeros(L,1);
acc_tr = zeros(L,1);

max_acc = 0;
for i = 1:L
    try_num = 20;
    acc_j = 0;
    acc_j_max = 0;
    accj = 0;
    acctj = 0;
    %structure = struc(i);
    for j = 1:try_num
        net = newff(X, Y, structure);
        %net.divideParam.trainRatio = 0.8; 
        %net.divideParam.valRatio = 0.2;
        %net.divideParam.testRatio = 0;
        net.divideFcn = ''; %nema podele
        net.performParam.regularization = reg(i);
        net.trainParam.epochs = 500;
        net.trainParam.goal = 1e-6; % ciljana greska
        %net.trainParam.show = 5; % na koliko epoha da prikazuje rezultate 
        net.performFcn = 'msereg';

        % Obucavanje mreze
        %[net, tr] = train(net, Xtrainval, ytrainval);
        [net, tr] = train(net, Xtrain, ytrain);
        
        % Validacija
        %yout = sim(net, X(:,tr.valInd));
        yout = sim(net, Xval);
        yout(yout < 0) = -1; 
        yout(yout >=0) = 1;
        %M = confusionmat(Y(tr.valInd), yout);
        M = confusionmat(yval, yout);
        acc_j = trace(M)/(sum(sum(M)));
        
        yout = sim(net, X(:,tr.trainInd));
        yout(yout < 0) = -1; 
        yout(yout >=0) = 1;
        M = confusionmat(Y(tr.trainInd), yout);
        acct_j = trace(M)/(sum(sum(M)));
        
        accj = accj+acc_j;
        acctj = acctj+acct_j;
        
        if(acc_j > acc_j_max)
            acc_j_max = acc_j;
            net_best_j = net;
            tr_best_j = tr;
            acct_j_max = acct_j;
        end
    end
    if(acc_j_max > max_acc)
        net_best = net_best_j;
        tr_best = tr_best_j;
        max_acc = acc_j_max;
        best_i = i;
    end
    %prosecne
    accj_v(i) = accj/try_num; 
    accj_tr(i) = acctj/try_num;
    %maksimalne
    acc_v(i) = acc_j_max;
    acc_tr(i) = acct_j_max;
end

max_acc

%% 

figure
plot(struc, accj_tr)
hold on
plot(struc, accj_v)
legend('trening','validacija')
xlabel('Broj neurona u skrivenom sloju')
ylabel('Tacnost klasifikacije')

%%

figure
r = reg(1:end-1)';
plot(r, accj_tr(1:end-1))
hold on
plot(r, accj_v(1:end-1))
legend('trening','validacija')
xlabel('Regularizacioni parametar')
ylabel('Tacnost klasifikacije')

%% Jedan trening

structure = [5];
reg = 0;

net = newff(X, Y, structure);
net.divideParam.trainRatio = 0.8; 
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;
%net.divideFcn = ''; %nema podele
net.performParam.regularization = reg;
net.trainParam.epochs = 500;
%net.trainParam.goal = 1e-6; % ciljana greska
%net.trainParam.show = 5; % na koliko epoha da prikazuje rezultate 
net.performFcn = 'msereg';

% Obucavanje mreze
[net, tr] = train(net, Xtrainval, ytrainval);
%[net, tr] = train(net, Xtrain, ytrain);

% Validacija
yout = sim(net, X(:,tr.valInd));
%yout = sim(net, Xval);
yout(yout < 0) = -1; 
yout(yout >=0) = 1;
Mv = confusionmat(Y(tr.valInd), yout);
%M = confusionmat(yval, yout);
accv = trace(Mv)/(sum(sum(Mv)));

yout = sim(net, X(:,tr.trainInd));
yout(yout < 0) = -1; 
yout(yout >=0) = 1;
Mt = confusionmat(Y(tr.trainInd), yout);
acct = trace(Mt)/(sum(sum(Mt)));


%% Test set

yout = sim(net_best, Xtest);
yout(yout < 0) = -1; 
yout(yout >=0) = 1;
M = confusionmat(ytest, yout);
acc_t = trace(M)/(sum(sum(M)))

yout_confusion = zeros(2,length(yout));
yout_confusion(1,:) = (yout < 0)*1;
yout_confusion(2,:) = (yout >= 0)*1;
Y_confusion = zeros(2,length(ytest));
Y_confusion(1,:) = (ytest < 0)*1;
Y_confusion(2,:) = (ytest >= 0)*1;
figure
plotconfusion(yout_confusion,Y_confusion)