% Compute the Rayleigh damping constants 
% M_{2}=alpha*M_{1}+beta*M_{3}
% where M_{i} are the system matrices 
% and "alpha" and "beta" are the constants that need to be computed
clear
% here we choose the frequencies and damping constants
w1=500;
w2=1000;
zeta1=0.2;
zeta2=0.2;
A=[1 w1^2;
    1 w2^2];
b=[2*w1*zeta1; 2*w2*zeta2];
x=inv(A)*b;
alpha=x(1)
beta=x(2)


% Check the damping at other frequencies
ww=1:1:10000;
for i=1:10000
zz(i)=0.5*(x(1)*(1/ww(i))+x(2)*ww(i));    
end
plot(ww,zz)