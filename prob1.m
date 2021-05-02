clc;
clear;
close all;

S0=[1/5 1/5 1/5 1/5 1/5]';
P=1/20*[16 1 1 1 1; 1 16 1 1 1; 1 1 16 1 1; 1 1 1 16 1; 1 1 1 1 16];
dt=0.5;
alpha=0.6;
%S=P^3*S0

phi_tilda=[1 dt dt^2/2; 0 1 dt; 0 0 alpha];
phi_tilda_z=[dt^2/2; dt; 0];
phi_tilda_w=[dt^2/2; dt; 1];

phi=[phi_tilda zeros(3,3); zeros(3,3) phi_tilda];
phi_z=[phi_tilda_z zeros(3,1); zeros(3,1) phi_tilda_z];
phi_w=[phi_tilda_w zeros(3,1); zeros(3,1) phi_tilda_w];

X0_1=sqrt(500)*randn;
X0_2=sqrt(5)*randn;
X0_3=sqrt(5)*randn;
X0_4=sqrt(200)*randn;
X0_5=sqrt(5)*randn;
X0_6=sqrt(5)*randn;
X0=[X0_1;X0_2;X0_3;X0_4;X0_5;X0_6];

Z=[0 3.5 0 0 -3.5; 0 0 3.5 -3.5 0];
Z_seed=rand;
index=floor(Z_seed/0.2)+1;
Zn=Z(:,index); %initial Zn: Z0

N=50;
Xarray=zeros(6,N);
Xarray(:,1)=X0;

for i=1:N-1
    Wn=[0.5*randn;0.5*randn];
    Xarray(:,i+1)=phi*Xarray(:,i)+phi_z*Zn+phi_w*Wn;
    Zn_newPr=cumsum(P(:,index));
    Z_seed=rand;
    %table reading
    for j=1:5
        if (Z_seed<=Zn_newPr(j))
            index=j;
            break;
        end
    end
    Zn=Z(:,index);
end

figure(1);
hold on;
for i=1:N
    plot(Xarray(1,i),Xarray(4,i),'r.', 'MarkerSize', 10)
    axis([-200 200 -200 200])
    pause(.1)
end
