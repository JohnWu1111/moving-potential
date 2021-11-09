% Starting from a bound state, slowly move the potential and study the
% properties of the bound state, using TSSP method to perform the time
% evolution, varying the parameter dt.

clear;
tic;

global sigma
sigma = 0.05;
% tol = 1e-11;
v = 0.05;
dt = [0.001 0.0005 0.0002 0.0001];
dx = 0.005;
T = 0.45;
L = 10;
x = -L:dx:L-dx;
nx = length(x);
ndt = length(dt);
t = 0:dt(end):T;
nt = length(t);
area = zeros(nt,ndt);
std_phi = zeros(nt,ndt);
mean_phi = zeros(nt,ndt);

fname = 'ground_state_sigma0.05_dt0.0005_L10_dx0.005_tol1e-08.mat';
load(fname);

figure;
for n = 1:ndt
    t = 0:dt(n):T;
    nt = length(t);
    phi0 = zeros(nt,nx);
    phi0(1,:) = phi;

    std_phi(1,n) = std(x,abs(phi0(1,:)).^2);
    mean_phi(1,n) = wmean(x,abs(phi0(1,:)).^2,dx);
    
    miu = zeros(1,nx);
    pha2 = zeros(1,nx);
    for i = 1:nx
        miu(i) = 2*pi*(-nx/2+i-1)/(2*L);
        pha2(i) = exp(-1i*dt(n)*miu(i)^2/2);
    end
    
    for i = 2:nt
        phi1 = exp(-1i*dt(n)*f(x+v*(t(i)-dt(n)))/2).*phi0(i-1,:);
%         phi1f = phi1*exp(-1i*(x'+L)*miu);
%         phi2 = pha2.*phi1f*exp(1i*miu'*(x+L))/nx;
        phi1f = nufft(phi1,x+L,miu/(2*pi));
        phi2 = nufft(pha2.*phi1f,-miu/(2*pi),x+L)/nx;  
        
        phi0(i,:) = exp(-1i*dt(n)*f(x+v*(t(i)-dt(n)/2))/2).*phi2;
        temp = abs(phi0(i,:)).^2;
        area(i,n) = sum(temp)*dx;
        mean_phi(i,n) = wmean(x,abs(phi0(i,:)),dx);
        std_phi(i,n) = std(x-mean_phi(i,n),abs(phi0(i,:)));
    end
    plot(t,std_phi(1:nt,n));
    hold on
end

toc;

function y = f(x)
    global sigma;
    y = -exp(-(x/sigma).^2/2)/(sqrt(2*pi)*sigma);
end

function y = wmean(x,phi,dx)
    y = 0;
    len = length(x);
    for i = 1:len
        y = y + x(i)*phi(i);
    end
    y = y*dx;
end