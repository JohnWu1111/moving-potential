tic;

dt = 1e-4;
dx = 5e-2;
t = 0:dt:1;
x = -500:dx:500;
nt = length(t);
nx = length(x);
k = dt/dx^2;
global alpha
alpha = 1;

y = zeros(nt,nx);
delta = zeros(nt,nx);
num = zeros(nt,nx);
for i = 1:nt
    delta(i,i+10000) = 1e3;
end
y(1,:) = f0(x);
for i = 2:nt
    y(i,1) = -((y(i-1,2)-2*y(i-1,1))/(2*(dx)^2)+alpha*delta(i,1)*y(i-1,1))*(-1i*dt)+y(i-1,1);
    for j = 2:nx-1
        y(i,j) = -((y(i-1,j+1)-2*y(i-1,j)+y(i-1,j-1))/(2*(dx)^2)+alpha*delta(i,j)*y(i-1,j))*(-1i*dt)+y(i-1,j);
    end
    y(i,nx) = -((-2*y(i-1,nx)+y(i-1,nx-1))/(2*(dx)^2)+alpha*delta(i,nx)*y(i-1,nx))*(-1i*dt)+y(i-1,nx);
end

for i = 1:nt
    for j = 1:nx
        num(i,j) = abs(y(i,j))^2;
    end
end

figure;
imagesc(num)

toc;

function y = f0(x)
    global alpha
    len = length(x);
    y = zeros(1,len);
    for i = 1:len
        if x(i) >= 0
            y(i) = exp(-alpha*x(i));
        else
            y(i) = exp(alpha*x(i));
        end
    end
end