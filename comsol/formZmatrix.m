% Listing of Zernike Polynomials 
% taken from 
% http://www.opt.indiana.edu/vsg/library/vsia/vsia-2000_taskforce/tops4_2.html
% see this link for a thorough description
function Z=formZmatrix(rho, theta)
syms r
syms q
a=[
1

2*r*sin(q)

2*r*cos(q)

sqrt(6)*(r^2)*sin(2*q)

sqrt(3)*(2*r^2-1)

sqrt(6)*(r^2)*cos(2*q)

sqrt(8)*(r^3)*sin(3*q)

sqrt(8)*(3*r^3-2*r)*sin(q)

sqrt(8)*(3*r^3-2*r)*cos(q)

sqrt(8)*(r^3)*cos(3*q)

sqrt(10)*(r^4)*sin(4*q)

sqrt(10)*(4*r^4-3*r^2)*sin(2*q)

sqrt(5)*(6*r^4-6*r^2+1)

sqrt(10)*(4*r^4-3*r^2)*cos(2*q)

sqrt(10)*(r^4)*cos(4*q)

sqrt(12)*(r^4)*sin(5*q)

sqrt(12)*(5*r^5-4*r^3)*sin(3*q)

sqrt(12)*(10*r^5-12*r^3+3*r)*sin(q)
	
sqrt(12)* (10*r^5-12*r^3+3*r)*cos(q)

sqrt(12)*(5*r^5-4*r^3)*cos(3*q)

sqrt(12)*(r^5)*cos(5*q)

sqrt(14)*(r^6)*sin(6*q)

sqrt(14)*(6*r^6-5*r^4)*sin(4*q)

sqrt(14)*(15*r^6-20*r^4+6*r^2)*sin(2*q)

sqrt(7)*(20*r^6-30*r^4+12*r^2-1)

sqrt(14)*(15*r^6-20*r^4+6*r^2)*cos(2*q)

sqrt(14)*(6*r^6-5*r^4)*cos(4*q)

sqrt(14)*(r^6)*cos(6*q)

4*(r^7)*sin(7*q)

4*(7*r^7-6*r^5)*sin(5*q)

4*(21*r^7-30*r^5+10*r^3)*sin(3*q)

4*(35*r^7-60*r^5+30*r^3-4*r)*sin(q)
];

Z=zeros(numel(rho),numel(a));
for i=1:numel(rho)
  Z(i,:)=double(subs(a,{r,q},{rho(i),theta(i)}))';
end

end