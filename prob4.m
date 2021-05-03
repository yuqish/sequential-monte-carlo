clc;
clear;
close all;

load stations.mat;
load RSSI-measurements.mat;


N = 10000;
n = 500;
tau1 = zeros(1,n); % vector of estimates x1
tau2 = zeros(1,n); % vector of estimates x2

%parameters
v=90;
eta=3;
S0=[1/5 1/5 1/5 1/5 1/5]';
P=1/20*[16 1 1 1 1; 1 16 1 1 1; 1 1 16 1 1; 1 1 1 16 1; 1 1 1 1 16];
dt=0.5;
alpha=0.6;

phi_tilda=[1 dt dt^2/2; 0 1 dt; 0 0 alpha];
phi_tilda_z=[dt^2/2; dt; 0];
phi_tilda_w=[dt^2/2; dt; 1];

phi=[phi_tilda zeros(3,3); zeros(3,3) phi_tilda];
phi_z=[phi_tilda_z zeros(3,1); zeros(3,1) phi_tilda_z];
phi_w=[phi_tilda_w zeros(3,1); zeros(3,1) phi_tilda_w];

p = @(x,y) mvnpdf(y,[v-10*eta*log10(sqrt((x(1,:)-pos_vec(1,1)).^2+(x(4,:)-pos_vec(2,1)).^2));
    v-10*eta*log10(sqrt((x(1,:)-pos_vec(1,2)).^2+(x(4,:)-pos_vec(2,2)).^2));
    v-10*eta*log10(sqrt((x(1,:)-pos_vec(1,3)).^2+(x(4,:)-pos_vec(2,3)).^2));
    v-10*eta*log10(sqrt((x(1,:)-pos_vec(1,4)).^2+(x(4,:)-pos_vec(2,4)).^2));
    v-10*eta*log10(sqrt((x(1,:)-pos_vec(1,5)).^2+(x(4,:)-pos_vec(2,5)).^2));
    v-10*eta*log10(sqrt((x(1,:)-pos_vec(1,6)).^2+(x(4,:)-pos_vec(2,6)).^2))]',1.5^2*eye(6)); % observation density, for weights



X0_1=sqrt(500)*randn(1,N);
X0_2=sqrt(5)*randn(1,N);
X0_3=sqrt(5)*randn(1,N);
X0_4=sqrt(200)*randn(1,N);
X0_5=sqrt(5)*randn(1,N);
X0_6=sqrt(5)*randn(1,N);
X0=[X0_1;X0_2;X0_3;X0_4;X0_5;X0_6];
w = p(X0,Y(:,1)'); %initialize weight
%tau
tau1(1) = X0(1,:)*w/sum(w);
tau2(1) = X0(4,:)*w/sum(w);


Z=[0 3.5 0 0 -3.5; 0 0 3.5 -3.5 0];
Z_seed=rand(1,N);
indexes=floor(Z_seed./0.2)+1;
Zn=Z(:,indexes); %initial Zn: Z0

Xarray=zeros(6,N,n);
Xarray(:,:,1)=X0;
X=X0;
w_array=zeros(N,n);
w_array(:,1)=w;
ESS=(sum(w.^2)/sum(w)^2)^(-1)
figure(2);
hist(w)
most_prob_actions=zeros(1,n-1);
for i=1:n-1
    %resample
    index = randsample(N,N,true,w);
    X=X(:,index);
    Zn=Zn(:,index);
    indexes=indexes(:,index);
    most_prob_actions(i)=mode(indexes);
    Wn=[0.5*randn;0.5*randn];
    %update X
    X=phi*X+phi_z*Zn+phi_w*Wn;
    %update weight
    w = p(X,Y(:,i+1)');
    %ESS
    if (mod((i+1),100) == 0)
        ESS=(sum(w.^2)/sum(w)^2)^(-1)
        figure(i+1);
        hist(w);
    end
    %tau
    tau1(i+1) = sum(X(1,:))/N;
    tau2(i+1) = sum(X(4,:))/N;
    
    %update Z
    for j = 1:5
        temp=indexes;
        indexes(temp==j) = randsample(5,sum(temp==j),true,P(j,:));
        Zn(:,temp==j) = Z(:,indexes(temp==j));
    end
    Xarray(:,:,i+1)=X;
    w_array(:,i+1)=w;
end

figure(1);
plot(pos_vec(1,:),pos_vec(2,:),'*');hold on;
plot(tau1,tau2);

% for i=1:n
%     figure(1);
%     plot(tau1(i),tau2(i),'r.');hold on;
%     pause(0.01)
% end

[GC,GR]=groupcounts(most_prob_actions');
for i=1:length(GR)
    switch GR(i)
        case 1
            fprintf('no action: %d\n',GC(1));
        case 2
            fprintf('east: %d\n',GC(2));
        case 3
            fprintf('north: %d\n',GC(3));
        case 4
            fprintf('south: %d\n',GC(4));
        case 5
            fprintf('west: %d\n',GC(5));
    end
end


%log-weights normalization
function w_out = log_normalize(log_w_in)
    L=max(log_w_in);
    w_bar=exp(log_w_in-L);
    w_out=w_bar;
end