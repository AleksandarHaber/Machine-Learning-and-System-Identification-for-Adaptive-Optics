import com.comsol.model.*
import com.comsol.model.util.*
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       These are the standard settings, do not alter them
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = ModelUtil.create('Model');
model.modelNode.create('comp1');
model.geom.create('geom1', 2);
model.mesh.create('mesh1', 'geom1');
model.geom('geom1').create('c1', 'Circle');

% we can also manually adjust the radius
plate_radius=1;
model.geom('geom1').feature('c1').set('r',mat2str(plate_radius));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Here we create a list of points for actuation and spring foundation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here we generate points inside of the circle of radius 1
pts_circle=[];
% for certain actuator spacing it is good to increase this initial radius
% and later on, the actuators outside 1m radius are neglected
radius_act=1
% this is the actuator spacing
stepp=0.2;
[X,Y]=meshgrid([-radius_act:stepp:radius_act],[-radius_act:stepp:radius_act])
pts=[X(:) Y(:)];
indx=1;
[s1,~]=size(pts);
%plot(pts(:,1),pts(:,2),'x');
for i=1:s1
  if sqrt(pts(i,1)^2+pts(i,2)^2)<1 
     pts_circle(indx,:)=pts(i,:);
     indx=indx+1;
  end
end
% here you can visualize the points
% plot(pts_circle(:,1),pts_circle(:,2),'x');

% Here we define the COMSOL points on the basis of the previously defined
% points
string_pt='pt';
[s1,s2]=size(pts_circle);
for i=1:s1
        chr = int2str(i);
        model.geom('geom1').create(strcat(string_pt,chr), 'Point');    
        xValue=pts_circle(i,1);
        yValue=pts_circle(i,2);
        model.geom('geom1').feature(strcat(string_pt,chr)).setIndex('p', mat2str(xValue), 0, 0);
        model.geom('geom1').feature(strcat(string_pt,chr)).setIndex('p', mat2str(yValue), 1, 0);
       % model.geom('geom1').feature(strcat(string_pt,chr)).setIndex('p',
       % '0', 2, 0); % this is for the 3D case- I do not need it
end
model.geom('geom1').run;
model.geom('geom1').run('fin');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            end of the point definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do not alter this
model.material.create('mat1', 'Common', 'comp1');
model.physics.create('plate', 'Plate', 'geom1');

% here, the boundary conditions are created
model.physics('plate').create('fix1', 'Fixed', 1);
model.physics('plate').feature('fix1').selection.set([1 2 3 4]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Here we define the point loads and the mass spring foundation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

string_pl='pl';

for i=1:s1
    chr = int2str(i);
    xValue=pts_circle(i,1);
    yValue=pts_circle(i,2);        
    model.physics('plate').create(strcat(string_pl,chr), 'PointLoad', 0);
    model.physics('plate').feature(strcat(string_pl,chr)).selection.set(mphselectcoords(model,'geom1',[xValue yValue]','point',...
'radius',0.003));
    model.physics('plate').feature(strcat(string_pl,chr)).set('Fp', {'0' '0' '1'});    
end

%  set the mass-spring foundation and the point masses
%  set the stiffness and damping constants
stiffness = int2str(10^4);
damping = int2str(500);
mass=int2str(0.3);

% spring-damper foundation
string_spf='spf';

% point mass
string_pm='pm';
for i=1:s1
    chr = int2str(i);
    xValue=pts_circle(i,1);
    yValue=pts_circle(i,2);        
    model.physics('plate').create(strcat(string_spf,chr), 'SpringFoundation0', 0);
    model.physics('plate').feature(strcat(string_spf,chr)).selection.set(mphselectcoords(model,'geom1',[xValue yValue]','point',...
'radius',0.003));
    model.physics('plate').create(strcat(string_pm,chr), 'PointMass', 0);
    model.physics('plate').feature(strcat(string_pm,chr)).selection.set(mphselectcoords(model,'geom1',[xValue yValue]','point',...
'radius',0.003));
    % model.physics('plate').feature(strcat(string_pl,chr)).set('Fp', {'0' '0' '1'});    
    % this was original
     model.physics('plate').feature(strcat(string_spf,chr)).set('kSpring', {stiffness; '0'; '0'; '0'; stiffness; '0'; '0'; '0'; stiffness});
     model.physics('plate').feature(strcat(string_spf,chr)).set('DampTot', {damping; '0'; '0'; '0'; damping; '0'; '0'; '0'; damping});  % here I used zero damping, change this in the final model
     % model.physics('plate').feature(strcat(string_spf,chr)).set('DampTot',{'1000'; '0'; '0'; '0'; '1000'; '0'; '0'; '0'; '1000'}); 
     model.physics('plate').feature(strcat(string_pm,chr)).set('pointmass', mass);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  material constants and the mesh size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
model.material('mat1').propertyGroup('def').set('youngsmodulus', '9.03*1e10');
model.material('mat1').propertyGroup('def').set('poissonsratio', '0.24');
model.material('mat1').propertyGroup('def').set('density', '2530');
model.physics('plate').prop('d').set('d', '0.003[m]');
% uncomment this is to disable the 3D formulation
% if the 3D formulation is used, then we have 6 degrees of freedom
% if the 3D formulation is not being used, then we have 3 degrees of
% freedom
model.physics('plate').prop('ShellAdvancedSettings').set('Use3DFormulation', '0');
model.mesh('mesh1').autoMeshSize(9);
% 9 - extremely coarse  
% 8 - extra coarse
model.mesh('mesh1').run;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.study.create('std3');
model.study('std3').create('time', 'Transient');
model.sol.create('sol3');
model.sol('sol3').study('std3');
model.sol('sol3').attach('std3');
model.sol('sol3').create('st1', 'StudyStep');
model.sol('sol3').create('v1', 'Variables');
model.sol('sol3').create('t1', 'Time');
model.sol('sol3').feature('t1').create('fc1', 'FullyCoupled');
model.sol('sol3').feature('t1').feature.remove('fcDef');
model.study('std3').feature('time').set('tlist', 'range(0,0.0001,0.3)');
model.sol('sol3').attach('std3');
model.sol('sol3').feature('v1').feature('comp1_u').set('scalemethod', 'manual');
model.sol('sol3').feature('v1').feature('comp1_u').set('scaleval', '1e-2*2.8284271247461903');
model.sol('sol3').feature('t1').set('timemethod', 'genalpha');
model.sol('sol3').feature('t1').set('tlist', 'range(0,0.0001,0.3)');
model.sol('sol3').runAll;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               here we extract the matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% extract the matrices and the vectors
MA2 = mphmatrix(model ,'sol3', ...
'Out', {'K','L', 'M','N','D','E','Kc','Lc', 'Dc','Ec','Null', 'Nullf','ud','uscale'},...
'initmethod','sol','initsol','zero');

% K- stiffness matrix
% L- load vector 
% M- constraint vector
% N-constraint Jacobian 
% D- damping matrix
% E- mass matrix 
% NF-Constraint force Jacobian 
% NP-Optimization constraint Jacobian 
% MP- Optimizatioin constraint vector
% MLB- lower bound constraint vector 
% MUB - upper bound constrainty vector

% Kc - eliminated stifness matrix 
% Dc - eliminated damping matrix 
% Ec - eliminated mass matrix 
% Null - constraint null-space basis 
% Nullf- constraint force null-space basis 
% ud - particular solution ud
% uscale-scale vector

% this is the example how the eliminated solution is being mapped into the
% true solution
% Uc = MA2.Null*(MA2.Kc\MA2.Lc);
% U0 = Uc+MA2.ud;
% U1 = U0.*MA2.uscale;

[n,n]=size(MA2.Ec)
r=n/3;
% construct the B matrix on the basis of the vector 
m=nnz(MA2.Lc);
B=sparse(n,m);
positionsB=find(MA2.Lc~=0);
for i=1:m
   B(positionsB(i),i)=1;
end

% extract the M1,M2, and M3 matrices, THESE MATRICES ARE EXPORTED
M1 = MA2.Ec;
M2 = MA2.Dc;
M3 = MA2.Kc;

info = mphxmeshinfo(model);
% info.dofs - this tells us about the meaning of the degrees of freedom
info.fieldnames
info.fieldndofs

info.nodes.dofnames
info.nodes.dofs    % 3  \times  numbers of nodes matrix, the third row is the row for the w component
info.nodes.coords  % coordinates of the nodes
info.dofs.dofnames % This corresponds to 0,1,2, with 2 being the w coordinate

w_indices=find(info.dofs.nameinds==2);
eliminated_dofs_indices=find(sum(MA2.Null,2)==0); % find the dofs that belong to the boundary domains.
internal_domain_indices=find(sum(MA2.Null,2)>0.5);

% eliminate the indices
[tmp1,tmp2]=size(info.dofs.nameinds);
source_indices=1:tmp1;
source_indices=source_indices';
indx=1;

% new_vector is the vector of internal domain indices after eliminating the
% degrees of freedom that correspond to the boundary

% original_indices_keep_them are the original indices of the internal
% domain, just for check

clear original_indices_keep_them;
clear new_vector;
for i=1:numel(source_indices)
    if numel(find(eliminated_dofs_indices==source_indices(i)))==0  %if the index is not in the set of the indices that need to be eliminated
     offset=numel(find(eliminated_dofs_indices<source_indices(i)));
     original_indices_keep_them(indx)=source_indices(i);
     new_vector(indx)=source_indices(i)-offset;
     indx=indx+1;
    end
end

%driver test code for the above code
%source_indices=1:15;
%eliminated_dofs_indices=[ 4 5 6 10 11 12] to be eliminated

% now map w_indices such that we know in the eliminated vector the
% positions of w_indices
indx=1;
clear  new_vector_w;
clear original_indices_keep_them_w;
for i=1:numel(w_indices)
    if numel(find(eliminated_dofs_indices==w_indices(i)))==0  %if the index is not in the set of the indices that need to be eliminated
     offset=numel(find(eliminated_dofs_indices<w_indices(i)));
     original_indices_keep_them_w(indx)=w_indices(i);
     new_vector_w(indx)=w_indices(i)-offset;
     indx=indx+1;
    end
end

% here we interpolate and visualize the Zernike modes using the discretization points
coordinate_eliminated=info.dofs.coords(:,original_indices_keep_them_w);
x_values =coordinate_eliminated(1,:);
y_values =coordinate_eliminated(2,:);
[theta,rad] = cart2pol(x_values,y_values);
idx = rad<=1;
z = nan(size(rad));
z(idx) = zernfun(5,-3,rad(idx),theta(idx)); % here you choose the Zernike polynomial, see the function zernfun.m for a complete description
%z_fem=zernfun(2,0,rad,theta);
%view(-90,90)
%scatter3(x_values,y_values,z)
% this is the 3D plot- just for check
%figure(1)
%plot3(x_values,y_values,z,'x')
%[xq,yq,vq]=interp_zern(x_values, y_values, z, 0.8, 0.01);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Here we find indices of coordinates that are inside of the circle of the
%  radius r1- this is necessary for defining zernike modes inside of the
%  radius r1, that is smaller than the maximal radius of the mirror. This
%  is for the case when we do not want to use the whole mirror aperture
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r1=0.6;
coordinate_eliminated_r1=[];
idx=1;
for i=1:r
    if sqrt(coordinate_eliminated(1,i)^2+coordinate_eliminated(2,i)^2)<= r1
    coordinate_eliminated_r1_idx(idx)=i;
    idx=idx+1;
    end
end

% this is the output matrix- THIS MATRIX IS EXPORTED
Cr1=sparse(numel(coordinate_eliminated_r1_idx),n);

for i=1:numel(coordinate_eliminated_r1_idx)
    Cr1(i,3*coordinate_eliminated_r1_idx(i))=1;
end

coordinate_eliminated_r1=coordinate_eliminated(:,coordinate_eliminated_r1_idx);
XY_scale=1.6;
x_values_r1 =XY_scale*coordinate_eliminated_r1(1,:);
y_values_r1 =XY_scale*coordinate_eliminated_r1(2,:);
[theta,rad] = cart2pol(x_values_r1,y_values_r1);
idx = rad<=1;
z_r1 = nan(size(rad));
z_r1(idx) = zernfun(2,-2,rad(idx),theta(idx)); % here you choose the Zernike polynomial, see the function zernfun.m for a complete description
figure(1)
%plot3(x_values,y_values,z,'x')
[xq_r1,yq_r1,vq_r1]=interp_zern((1/XY_scale)*x_values_r1,(1/XY_scale)*y_values_r1, z_r1, 0.4, 0.01);

% Zmap is the matrix that maps Zernike coefficients into the global surface
% deformations
Zmap=formZmatrix(rad(idx), theta(idx)) %THIS MATRIX IS EXPORTED
zern_coeff=pinv(Zmap)*z_r1';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here we solve a linear system of equations to compute the steady-state
% input- for the case of observing all points inside of the radius r1 - the
% matrix Cr1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

z_scaled_r1=z_r1.*10^(-4) % scale the zernike modes to be in the micrometer range
LHS_r1 = [zeros(n,1); z_scaled_r1']; % form the LHS
% data matrix 
Adata_r1=sparse([M3 B; Cr1 sparse(numel(coordinate_eliminated_r1_idx),m)]);
spparms('spumoni',2)
solutionSteadyState_r1=Adata_r1\LHS_r1; 

% slowly converges- the system is ill-conditioned
%[solutionSteadyState_r1,flag,relres,iter,resvec]=lsqr(Adata_r1,LHS_r1,10^(-8),20000)

stateComputedSteady_r1=solutionSteadyState_r1(1:n,1);
inputComputedSteady_r1=solutionSteadyState_r1(n+1:end,1);
wcomputed_r1=stateComputedSteady_r1(3*coordinate_eliminated_r1_idx,1);

figure(3)
plot(wcomputed_r1)
hold on 
plot(z_scaled_r1,'r')
error_r1=wcomputed_r1-z_scaled_r1'
error_rel_r1=norm(error_r1)/norm(wcomputed_r1)
figure
hold on
plot(error_r1)

[xq2,yq2,vq2]=interp_zern( (1/(XY_scale))* x_values_r1,  (1/(XY_scale))* y_values_r1, wcomputed_r1, 0.55, 0.01);
[xq3,yq3,vq3]=interp_zern( (1/(XY_scale))* x_values_r1, (1/(XY_scale))* y_values_r1,z_scaled_r1, 0.55, 0.01);
[xq4,yq4,vq4]=interp_zern( (1/(XY_scale))* x_values_r1, (1/(XY_scale))* y_values_r1, error_r1, 0.55, 0.01);

% get the input coordinates
inputIndices_r1=find((MA2.Null*B*inputComputedSteady_r1)~=0);
inputCoordinates_r1=info.dofs.coords(:,inputIndices_r1);
x_input_r1=inputCoordinates_r1(1,:);
y_input_r1=inputCoordinates_r1(2,:);
figure()
plot3(x_input_r1,y_input_r1, inputComputedSteady_r1,'x')
pointsize = 100;
figure()
scatter(x_input_r1,y_input_r1, pointsize, inputComputedSteady_r1, 'filled')

% =[Z_{1}^{-1}   Z_{2}^{0} Z_{3}^{-1} Z_{4}^{-2} Z_{5}^{-3} Z_{6}^{-2} Z_{7}^{-3}]
% error values for the actuator spacing of 0.2, number of inputs 69
error_values_0_2=[0.0023 0.0052 0.0127 0.0189  0.0518 0.1288 0.2348]

% error values for the actuator spacing of 0.1, number of inputs 305
error_values_0_1=[4.3817e-07 1.1508e-06 1.5689e-04 4.6810e-04 9.9117e-04 0.0070 0.0123]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               here we extract the matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%save matrices M1 M2 M3 B Cr1 Zmap

% M1\ddot{z}+M2\dot{z}+M_{3}z=Bu
% y=pinv(Zmap)*Crl*z
% M1 is the mass matrix 
% M2 is the damping matrix 
% M3 is the stiffness matrix 
% B is the control input matrix of the model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               here the main code stops
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here we solve a linear system of equations to compute the steady-state
% input- for the case of observing all points inside of the aperture - the matrix C 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% form the C matrix, measure every displacement at the discretization point
[n,n]=size(MA2.Ec)
C=sparse(n/3,n);
r=n/3;
for i=1:r
   C(i,3*i)=1;
end



z_scaled=z.*10^(-5) % scale the zernike modes to be in the micrometer range
LHS = [zeros(n,1); z_scaled']; % form the LHS
% data matrix 
Adata=sparse([M3 B; C sparse(r,m)]);
spparms('spumoni',2)
solutionSteadyState=Adata\LHS; 

% this is an alternative version
%[solutionSteadyState,flag,relres,iter,resvec]=lsqr(Adata,LHS,10^(-3),4000)

stateComputedSteady=solutionSteadyState(1:n,1);
inputComputedSteady=solutionSteadyState(n+1:end,1);
wcomputed=stateComputedSteady(new_vector_w,1);

figure(3)
plot(wcomputed)
hold on 
plot(z_scaled,'r')
error=wcomputed-z_scaled'
error_rel=norm(error)/norm(wcomputed)
figure
hold on
plot(error)

[xq2,yq2,vq2]=interp_zern(x_values,  y_values, wcomputed, 0.8, 0.01);
[xq3,yq3,vq3]=interp_zern(x_values,  y_values, z_scaled,  0.8, 0.01);
[xq4,yq4,vq4]=interp_zern(x_values,  y_values, error,     0.8, 0.01);


% % get the input coordinates
% inputIndices=find((MA2.Null*B*inputComputedSteady)~=0);
% inputCoordinates=info.dofs.coords(:,inputIndices);
% x_input=inputCoordinates(1,:);
% y_input=inputCoordinates(2,:);
% figure()
% plot3(x_input,y_input, inputComputedSteady,'x')
% pointsize = 100;
% figure()
% scatter(x_input,y_input, pointsize, inputComputedSteady, 'filled')
% 
% % an alternative way 
% [X1,Y1]=meshgrid([-1:stepp:1],[-1:stepp:1])
% v_int = griddata(x_input,y_input, inputComputedSteady,X1,Y1);
% pcolor(X1,Y1,v_int), shading interp
% axis square, colorbar
% hold on 
% plot(pts_circle(:,1),pts_circle(:,2),'x');




