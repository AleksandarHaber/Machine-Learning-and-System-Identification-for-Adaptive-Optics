function [xq,yq,vq]=interp_zern(x_values, y_values, z, rad, intStep)
% this function interpolates the Zernike polynomials and plots the results
% input parameters: 
% x_values - x coordinates of the original data 
% y_values - y coordinates of the original data 
% z        - z coordinates of the original data
% rad      - cutoff radius for displaying the results

[xq,yq] = meshgrid(-1:intStep:1, -1:intStep:1);
vq = griddata(x_values,y_values,z,xq,yq);
figure(2)
[s1,~]=size(xq);
for i=1:s1
    for j=1:s1
        if (xq(i,j)^2+yq(i,j)^2)>rad^2
            vq(i,j)=NaN;
        end
    end
end
pcolor(xq,yq,vq), shading interp
axis square, colorbar
%title('Zernike function Z_5^1(r,\theta)')
end