% First try to simulate the unscalled model
load matrices
C=Cr1;
[n,~]=size(M1);
[~,m]=size(B);
[r,~]=size(C);

% this is the unscalled model
invM1=inv(M1);
A1=[zeros(n,n) eye(n,n); -invM1*M3 -invM1*M2];
B1= [zeros(n,m); invM1*B];
C1 = [C zeros(r,n)];
x0=rand(2*n,1);

% get the scales
max(sum(M1,2))
min(sum(M1,2))

max(sum(M2,2))
min(sum(M2,2))

max(sum(M3,2))
min(sum(M3,2))

% time scale
c1=10^(-5);
% state scale
c2=10^(-6);

M1s=M1*c2/(c1^2);
M2s=M2*c2/c1;
M3s=M3*c2;
Cs=C*c2;

max(sum(M1s,2))
min(sum(M1s,2))

max(sum(M2s,2))
min(sum(M2s,2))

max(sum(M3s,2))
min(sum(M3s,2))

invM1s=inv(M1s);
A1s=[zeros(n,n) eye(n,n); -invM1s*M3s -invM1s*M2s];
B1s= [zeros(n,m); invM1s*B];
C1s = [Cs zeros(r,n)];

% similarity transformation
SI=[c2*speye(n,n) zeros(n,n); zeros(n,n) (c2/c1)*speye(n,n)]
x0s=inv(SI)*x0;
Toriginal=0.0001;
Adoriginal=inv(eye(2*n,2*n)-Toriginal*A1);
Tscaled=Toriginal/c1;
Adscaled=inv(eye(2*n,2*n)-Tscaled*A1s);
clear Xs
clear Xo

for i=1:40
   if i==1
       Xs{1,i}=x0s;
       Xo{1,i}=x0;
   else
       Xs{1,i}= Adscaled*Xs{1,i-1};
       Xo{1,i}= Adoriginal*Xo{1,i-1};
   end
    
 end
 
Xs=cell2mat(Xs);
Xo=cell2mat(Xo);
Xo_fs=SI*Xs;
plot(Xo_fs(400,:),'r');
hold on 
plot(Xo(400,:),'k');
normest(Xo_fs-Xo)/normest(Xo)

% descriptor state-space model, unscaled and scaled matrices 
E=[eye(size(M1)) zeros(size(M1));
    zeros(size(M1)) M1];
A=[zeros(size(M1)) eye(size(M1)); -M3 -M2];
B1=[zeros(size(B)); B];
C1 = [C zeros(r,n)];
E=sparse(E)
A=sparse(A)
B1=sparse(B1)
C1=sparse(C1)

Es=[eye(size(M1s)) zeros(size(M1s));
    zeros(size(M1s)) M1s];
As=[zeros(size(M1s)) eye(size(M1s)); -M3s -M2s];
B1=[zeros(size(B)); B];
C1s = [Cs zeros(r,n)];
Es=sparse(Es)
As=sparse(As)
B1=sparse(B1)
C1s=sparse(C1s)



% % try another way of scalling the matrices 
% [p1,q1,r1]=find(M1);
% [p3,q3,r3]=find(M3);
% 
% max(abs(r1))
% min(abs(r1))
% 
% max(abs(r3))
% min(abs(r3))





