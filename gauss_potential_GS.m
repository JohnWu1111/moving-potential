% Use imaginary time revolution to recognize the ground state of the
% Schrodinger equation. TSSP method is employed to perform time revolution.

clear;
tic;

global sigma
sigma = 0.05;
tol = 1e-11;
ep = 1;
dt = 0.001;
dx = 0.01;
t0 = 0:dt:50;
L = 10;
x = -L:dx:L-dx;
nt0 = length(t0);
nx = length(x);
phi0 = zeros(nt0,nx);
phi0(1,:) = f0(x);

pha1 = zeros(1,nx);
miu = zeros(1,nx);
pha2 = zeros(1,nx);
for i = 1:nx
    pha1(i) = exp(-dt*f(x(i))/(2*ep));
    miu(i) = 2*pi*(-nx/2+i-1)/(2*L);
    pha2(i) = exp(-dt*miu(i)^2*ep/2);
end

for i = 2:nt0
    phi1 = pha1.*phi0(i-1,:);
    phi1f = phi1*cos((x'+L)*miu);
    phi2 = pha2.*phi1f*cos(miu'*(x+L));
    
    phi0(i,:) = pha1.*phi2;
    temp = abs(phi0(i,:));
    s = sum(temp.^2);
    phi0(i,:) = phi0(i,:)./sqrt(s*dx);
    
    if abs(abs(phi0(i,1)) - abs(phi0(i-1,1))) < tol
        count = i;
        break;
    end
end
phi00 = abs(phi0);
phi = phi00(count,:);
fname = ['ground_state_sigma',num2str(sigma),'_dt',num2str(dt),'.mat'];
save(fname,'phi','-v7.3');

figure;
plot(x,phi);
hold on
plot(x,f0(x));

fname = ['ground_state_sigma',num2str(sigma),'_dt',num2str(dt),'.png'];
saveas(gcf, fname)

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