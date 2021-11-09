% Starting from a bound state, slowly move the potential and study the
% properties of the bound state, using TSSP method to perform the time
% evolution, varying the parameter dt, with GPU boost.

clear;
tic;

global sigma
sigma = 0.05;
% tol = 1e-11;
v = 0.05;
% dt = [0.001 0.0005 0.0002 0.0001];
dt = [0.001 0.0005];
dx = 0.005;
T = 10;
L = 10;
x = -L:dx:L-dx;
nx = length(x);
ndt = length(dt);
t = 0:dt(end):T;
nt = length(t);
std_phi = zeros(nt,ndt);
mean_phi = zeros(nt,ndt);

fname = 'ground_state_sigma0.05_dt0.0005_L10_dx0.005_tol1e-08.mat';
load(fname);

coeff = (-1).^(0:(nx-1));
coeff = gpuArray(coeff);

x = gpuArray(x);
dt = gpuArray(dt);
miu = zeros(1,nx);
pha2 = zeros(1,nx);
miu = gpuArray(miu);
pha2 = gpuArray(pha2);
mean_phi = gpuArray(mean_phi);

figure;
for n = 1:ndt
    t = 0:dt(n):T;
    nt = length(t);
    t = gpuArray(t);
    std_phi = gpuArray(std_phi);
    phi0 = gpuArray(phi);

    std_phi(1,n) = std(x,abs(phi0).^2);
%     mean_phi(1,n) = wmean(x,abs(phi0),dx); 
    mean_phi(1,n) = x*abs(phi0).^2'*dx;
    
    for i = 1:nx
        miu(i) = 2*pi*(-nx/2+i-1)/(2*L);
        pha2(i) = exp(-1i*dt(n)*miu(i)^2/2);
    end
    
    for i = 2:nt
        phi1 = exp(1i*dt(n)*(exp(-((x+v*(t(i)-dt(n)))/sigma).^2/2)/(sqrt(2*pi)*sigma))/2).*phi0;
%         phi1f = phi1*exp(-1i*(x'+L)*miu);
%         phi2 = pha2.*phi1f*exp(1i*miu'*(x+L))/nx;
        phi1f = fft(coeff.*phi1);
        phi2 = coeff.*ifft(pha2.*phi1f);
        
        phi0 = exp(1i*dt(n)*(exp(-((x+v*(t(i)-dt(n)/2))/sigma).^2/2)/(sqrt(2*pi)*sigma))/2).*phi2;
        temp = abs(phi0).^2;
%         mean_phi(i,n) = wmean(x,abs(phi0),dx);
        mean_phi(i,n) = x*temp'*dx;
        std_phi(i,n) = std(x-mean_phi(i,n),temp);
    end

    t = gather(t);
    std_phi = gather(std_phi);
    plot(t,std_phi(1:nt,n));
    hold on
end

toc;

% fname = ['result_sigma',num2str(sigma),'_dt',num2str(dt),'_T',num2str(T),'_L',num2str(L),'_dx',num2str(dx),'_v',num2str(v),'.mat'];
% save(fname,'phi00','std_phi','mean_phi','-v7.3');

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