% Use imaginary time revolution to recognize the ground state of the
% Schrodinger equation. TSSP method is employed to perform time revolution.

clear;
tic;

global sigma
sigma = 0.05;
tol = 1e-9;
ep = 1;
dt = 0.0005;
dx = 0.005;
t0 = 0:dt:100;
L = 30;
x = -L:dx:L-dx;
nt0 = length(t0);
nx = length(x);
phi0 = f0(x);

pha1 = zeros(1,nx);
miu = zeros(1,nx);
pha2 = zeros(1,nx);

phi0 = gpuArray(phi0);
x = gpuArray(x);

for i = 1:nx
    pha1(i) = exp(-dt*f(x(i))/(2*ep));
    miu(i) = 2*pi*(-nx/2+i-1)/(2*L);
    pha2(i) = exp(-dt*miu(i)^2*ep/2);
end
coeff = (-1).^(0:(nx-1));

pha1 = gpuArray(pha1);
miu = gpuArray(miu);
pha2 = gpuArray(pha2);
coeff = gpuArray(coeff);
tol = gpuArray(tol);
dx = gpuArray(dx);

i = 2;
i = gpuArray(i);
nt0 = gpuArray(nt0);
while i <= nt0
    comp = phi0(nx/2);
    phi1 = pha1.*phi0;
%     phi1f = phi1*cos((x'+L)*miu);
%     phi2 = pha2.*phi1f*cos(miu'*(x+L));
    phi1f = fft(coeff.*phi1);
    phi2 = coeff.*ifft(pha2.*phi1f);
    
    temp = pha1.*phi2;
    temp1 = abs(temp);
    s = sum(temp1.^2);
    temp = temp./sqrt(s*dx);
    phi0 = temp; 
    
    if abs(abs(phi0(nx/2)) - comp) < tol
        count = i;
        break;
    end
    i = i + 1;
end

nt0
count

phi = abs(phi0);

phi = gather(phi);
x = gather(x);
fname = ['ground_state_sigma',num2str(sigma),'_dt',num2str(dt),'_L',num2str(L),'_dx',num2str(dx),'_tol',num2str(tol),'.mat'];
save(fname,'phi','-v7.3');

%figure;
%plot(x,phi);
%hold on
%plot(x,f0(x));

%fname = ['ground_state_sigma',num2str(sigma),'_dt',num2str(dt),'.png'];
%saveas(gcf, fname)

toc;

function y = f(x)
    global sigma;
    y = -exp(-(x/sigma)^2/2)/(sqrt(2*pi)*sigma);
end

% function y = f(x)
%     y = x^2;
% end

function y = f0(x)
    len = length(x);
    y = zeros(1,len);
    for i = 1:len
        if x(i) >= 0
            y(i) = exp(-x(i));
        else
            y(i) = exp(x(i));
        end
    end
end