% Starting from a bound state, slowly move the potential and study the
% properties of the bound state, using TSSP method to perform the time
% evolution, varying the parameter v.

clear;
tic;

global sigma
sigma = 0.05;
% tol = 1e-11;
v = [0.2 0.1 0.05];
dt = 0.0001;
dx = 0.005;
T = 0.45;
t = 0:dt:T;
L = 10;
x = -L:dx:L-dx;
nt = length(t);
nx = length(x);
phi0 = zeros(nt,nx);
V = zeros(nt,nx);
std_phi = zeros(nt,1);
mean_phi = zeros(nt,1);

fname = 'ground_state_sigma0.05_dt0.0005_L10_dx0.005_tol1e-08.mat';
load(fname);
phi0(1,:) = phi;
std_phi(1) = std(x,abs(phi0(1,:)));
mean_phi(1) = wmean(x,abs(phi0(1,:)),dx);

miu = zeros(1,nx);
pha2 = zeros(1,nx);
for i = 1:nx
    miu(i) = 2*pi*(-nx/2+i-1)/(2*L);
    pha2(i) = exp(-1i*dt*miu(i)^2/2);
end

figure;
for n = 1:length(v)
    for i = 2:nt
        phi1 = exp(-1i*dt*f(x+v(n)*(t(i)-dt))/2).*phi0(i-1,:);
%         phi1f = phi1*exp(-1i*(x'+L)*miu);
%         phi2 = pha2.*phi1f*exp(1i*miu'*(x+L))/nx;
        phi1f = nufft(phi1,x+L,miu/(2*pi));
        phi2 = nufft(pha2.*phi1f,-miu/(2*pi),x+L)/nx;    
        
        phi0(i,:) = exp(-1i*dt*f(x+v(n)*(t(i)-dt/2))/2).*phi2;
        temp = abs(phi0(i,:));
        mean_phi(i) = wmean(x,abs(phi0(i,:)),dx);
        std_phi(i) = std(x-mean_phi(i),abs(phi0(i,:)));
    end
    plot(t,std_phi);
    hold on
end
% std_phi = sqrt(var_phi);
toc;

fname = ['v_sigma',num2str(sigma),'_dt',num2str(dt),'_T',num2str(T),'_L',num2str(L),'_dx',num2str(dx),'_v',num2str(v),'.mat'];
save(fname','std_phi','mean_phi','-v7.3');


function y = f(x)
    global sigma;
    y = -exp(-(x/sigma).^2/2)/(sqrt(2*pi)*sigma);
end

function y = var1(phi,x)
    y = 0;
    len = length(x);
    for i = 1:len
        y = y + x(i)^2*phi(i);
    end
    y = y/len;
end

function y = wmean(x,phi,dx)
    y = 0;
    len = length(x);
    for i = 1:len
        y = y + x(i)*phi(i);
    end
    y = y*dx;
end