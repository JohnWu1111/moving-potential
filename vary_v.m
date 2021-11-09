% Starting from a bound state, slowly move the potential and study the
% properties of the bound state, using TSSP method to perform the time
% evolution, varying the parameter v.

clear;
tic;

global sigma
sigma = 0.05;
% tol = 1e-11;
v = [0.2 0.1 0.05];
dt = 0.001;
dx = 0.005;
T = 2;
t = 0:dt:T;
L = 10;
x = -L:dx:L-dx;
nt = length(t);
nx = length(x);
std_phi = zeros(nt,1);
mean_phi = zeros(nt,1);

fname = 'ground_state_sigma0.05_dt0.0005_L10_dx0.005_tol1e-08.mat';
load(fname);

miu = zeros(1,nx);
pha2 = zeros(1,nx);
for i = 1:nx
    miu(i) = 2*pi*(-nx/2+i-1)/(2*L);
    pha2(i) = exp(-1i*dt*miu(i)^2/2);
end

coeff = (-1).^(0:(nx-1));
coeff = gpuArray(coeff);

x = gpuArray(x);
dt = gpuArray(dt);
miu = gpuArray(miu);
pha2 = gpuArray(pha2);
mean_phi = gpuArray(mean_phi);

figure;
for n = 1:length(v)
    t = gpuArray(t);
    phi0 = gpuArray(phi);
    std_phi = gpuArray(std_phi);
    std_phi(1) = std(x,abs(phi0));
    mean_phi(1) = wmean(x,abs(phi0),dx);
    
    for i = 2:nt
        phi1 = exp(1i*dt*(exp(-((x+v(n)*(t(i)-dt))/sigma).^2/2)/(sqrt(2*pi)*sigma))/2).*phi0;
%         phi1f = phi1*exp(-1i*(x'+L)*miu);
%         phi2 = pha2.*phi1f*exp(1i*miu'*(x+L))/nx;
        phi1f = fft(coeff.*phi1);
        phi2 = coeff.*ifft(pha2.*phi1f);   
        
        phi0 = exp(1i*dt*(exp(-((x+v(n)*(t(i)-dt/2))/sigma).^2/2)/(sqrt(2*pi)*sigma))/2).*phi2;
        temp = abs(phi0);
        mean_phi(i) = x*temp.^2'*dx;
        std_phi(i) = std(x-mean_phi(i),temp);
    end
    
    t = gather(t);
    std_phi = gather(std_phi);
    
    plot(t,std_phi);
    hold on
end
% std_phi = sqrt(var_phi);
toc;

% fname = ['v_sigma',num2str(sigma),'_dt',num2str(dt),'_T',num2str(T),'_L',num2str(L),'_dx',num2str(dx),'_v',num2str(v),'.mat'];
% save(fname','std_phi','mean_phi','-v7.3');


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