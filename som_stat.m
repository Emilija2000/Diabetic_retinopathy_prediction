close all;clear;
clc;

% Ucitavanje podataka 
dataset = importdata('messidor_features.arff');
data = dataset.data;


%% Pretprocesiranje podataka

% Nema nedostajucih ni txt vrednosti
data = data(data(:,1)==1,2:end); %izbacuju se podaci sa losim kvalitetom

Rs = corrcoef(data);
figure
heatmap(Rs);
title('Personov koeficijent korelacije svih podataka'); 

%% Izbor 10 atributa

%features = [1,2,3,4,13,14,15,16,17,18,19];

features = [1,2,3,4,8,9,10,16,17,18,19];
discrete = [1,0,0,0,0,0,0,0,0,1];
numDifValues = [2,10,10,10,5,5,5,15,15,2];
data = data(:,features);

% Za slucaj da ne izbacujemo nista
% discrete = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1];
% numDifValues = [2,2,10,10,5,5,5,15,15,10,10,10,10,10,10,10,10,10,2];

[num_samples, num_attributes] = size(data); 
num_attributes = num_attributes -1; 
num_classes = length(unique(data(:, end))); 

p0 = sum(data(:, end) == 0)/num_samples;
p1 = sum(data(:, end) == 1)/num_samples; 


%% Korelaciona analiza i IG

Rs = corrcoef(data);
figure
heatmap(Rs);
title('Personov koeficijent korelacije na izabranih 10 obelezja'); 

K = num_attributes;
rii = sum(sum(Rs(1:end-1,1:end-1))) - trace(Rs(1:end-1,1:end-1));
rii = rii/(K*K-K);
rzi = mean(Rs(:,end));
r = K*rzi/sqrt(K +K*(K-1)*rii);
disp('Korelacioni koef:')
disp(r)

Info_D =-p0*log2(p0) -p1*log2(p1); 

IG = zeros(num_attributes,1);
for i = 1:num_attributes
    IG(i) = informationGain(Info_D, data(:,i),data(:,end),numDifValues(i),discrete(i));
end
disp('IG:')
disp(IG)


%% LDA

format long e

s0 = data(data(:,end)==0,1:end-1);
m0 = mean(s0,1)';
s0 = cov(s0);
s1 = data(data(:,end)==1,1:end-1);
m1 = mean(s1,1)';
s1 = cov(s1);
Sw = p0*s0+p1*s1;
M = p0*m0+p1*m1;
Sb = p0*(m0-M)*(m0-M)' + p1*(m1-M)*(m1-M)';
Sm = Sw+Sb;

S = inv(Sw)*Sb;
[Phi,Lambda] = eig(S);
Lambda = diag(real(Lambda));
Phi = real(Phi);

disp('Sopstvene vrednosti: ');
disp(Lambda)

%% Prikaz podataka sa izabranim obelezjima

% Izbor po LDA

% 3D
At = [Phi(:,1),Phi(:,2),Phi(:,3)];
X = data(:,1:end-1)*At;
Y = data(:,end);

figure;
X0 = X(Y==0,:);
plot3(X0(:,1),X0(:,2),X0(:,3),'bo')
hold on
X1 = X(Y==1,:);
plot3(X1(:,1),X1(:,2),X1(:,3),'r*')

% 2D
At = [Phi(:,1),Phi(:,2)];
X = data(:,1:end-1)*At;
Y = data(:,end);

figure;
X0 = X(Y==0,:);
plot(X0(:,1),X0(:,2),'bo')
hold on;
X1 = X(Y==1,:);
plot(X1(:,1),X1(:,2),'r*')


%% Linearni klasifikator na bazi zeljenog izlaza - 2D LDA

At = [Phi(:,1),Phi(:,2)];
X = data(:,1:end-1)*At;
Y = data(:,end);

Xob = X(1:floor(size(X,1)*0.7), :);
Xt = X((floor(size(X,1)*0.7)+1):end, :);

Yob = Y(1:floor(size(Y,1)*0.7), :);
Yt = Y((floor(size(Y,1)*0.7)+1):end, :);

X0 = Xob(Yob==0,:);
X1 = Xob(Yob==1,:);

Z = [-ones(1, length(X0)) ones(1,length(X1)); -X0' X1'];
Gama=[ones(length(X0),1); ones(length(X1),1)];
W = (Z*Z')^(-1)*Z*Gama;
v0 = W(1);
V = [W(2);W(3)];

figure;
X0 = X(Y==0,:);
plot(X0(:,2),X0(:,1),'bo')
hold on;
X1 = X(Y==1,:);
plot(X1(:,2),X1(:,1),'r*')
xp = [min(X(:,2)) max(X(:,2))];
plot(xp,(-v0-V(2)*xp)/V(1),'k--')
hold off

Yout = (Xt*V+v0 > log(p0/p1))*1; 

M = confusionmat(Yt, Yout);
disp('Linearni klasifikator na bazi zeljenog izlaza - LDA 2D')
acc = trace(M)/(sum(sum(M)))


%% Linearni klasifikator na bazi zeljenog izlaza - svi atributi
X = data(:,1:end-1);
Y = data(:,end);

Xob = X(1:floor(size(X,1)*0.7), :);
Xt = X((floor(size(X,1)*0.7)+1):end, :);

Yob = Y(1:floor(size(Y,1)*0.7), :);
Yt = Y((floor(size(Y,1)*0.7)+1):end, :);

X0 = Xob(Yob==0,:);
X1 = Xob(Yob==1,:);

Z = [-ones(1, length(X0)) ones(1,length(X1)); -X0' X1'];
Gama=[ones(length(X0),1); ones(length(X1),1)];
W = (Z*Z')^(-1)*Z*Gama;
v0 = W(1);
V = W(2:end);

Yout = (Xt*V+v0 > log(p0/p1))*1; 

M = confusionmat(Yt, Yout);

disp('Linearni klasifikator na bazi zeljenog izlaza - LDA svi atr')
acc = trace(M)/(sum(sum(M)))

%% Izbor po IG 2D

X = data(:,[2,3]);
Y = data(:,end);

figure;
X0 = X(Y==0,:);
plot(X0(:,1),X0(:,2),'bo')
hold on;
X1 = X(Y==1,:);
plot(X1(:,1),X1(:,2),'r*')

Xob = X(1:floor(size(X,1)*0.7), :);
Xt = X((floor(size(X,1)*0.7)+1):end, :);

Yob = Y(1:floor(size(Y,1)*0.7), :);
Yt = Y((floor(size(Y,1)*0.7)+1):end, :);

X0 = Xob(Yob==0,:);
X1 = Xob(Yob==1,:);

Z = [-ones(1, length(X0)) ones(1,length(X1)); -X0' X1'];
Gama=[ones(length(X0),1); ones(length(X1),1)];
W = (Z*Z')^(-1)*Z*Gama;
v0 = W(1);
V = W(2:end);

Yout = (Xt*V+v0 > log(p0/p1))*1; 

M = confusionmat(Yt, Yout);

disp('Linearni klasifikator na bazi zeljenog izlaza - IG 2D')
acc = trace(M)/(sum(sum(M)))

%% Izbor po IG 3D

X = data(:,[2,3,4]);
Y = data(:,end);

figure;
X0 = X(Y==0,:);
plot3(X0(:,1),X0(:,2),X0(:,3),'bo')
hold on;
X1 = X(Y==1,:);
plot3(X1(:,1),X1(:,2),X1(:,3),'r*')


Xob = X(1:floor(size(X,1)*0.7), :);
Xt = X((floor(size(X,1)*0.7)+1):end, :);

Yob = Y(1:floor(size(Y,1)*0.7), :);
Yt = Y((floor(size(Y,1)*0.7)+1):end, :);

X0 = Xob(Yob==0,:);
X1 = Xob(Yob==1,:);

Z = [-ones(1, length(X0)) ones(1,length(X1)); -X0' X1'];
Gama=[ones(length(X0),1); ones(length(X1),1)];
W = (Z*Z')^(-1)*Z*Gama;
v0 = W(1);
V = W(2:end);

Yout = (Xt*V+v0 > log(p0/p1))*1; 

M = confusionmat(Yt, Yout);

disp('Linearni klasifikator na bazi zeljenog izlaza - IG 3D')
acc = trace(M)/(sum(sum(M)))
