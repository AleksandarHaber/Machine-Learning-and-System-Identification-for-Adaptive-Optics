These are the source codes to generate data and Deformable Mirror models for the two papers:

Subspace Identification (SI) paper 
Machine Learning (ML) paper 

Explanation of the files:

 - The file "frequency_response_and_eigenfrequencies_undamped.mph" is a COMSOL file used to generate
Figures 1 and 2 in the paper. This file contains both the frequency domain and eigenfrequency studies. 

 - The files "frequency_response_P1" and "frequency_response_P2" are the frequency responses at the 
corresponding points when the force is acting at the point P1(-0.3,-0.3). This data is used to generate
graphs in Fig. 1b)

-  The file "frequency_response_data.m" contains the vectors of data from the files "frequency_response_P1" and "frequency_response_P2"

-  The file "frequency_response.m" contains the code to generate the plots in Fig.1b) in the SI paper.

-  The file "rayleigh_damping.m" is used to compute the Rayleigh damping constants.

-  The file "frequency_response_and_eigenfrequencies_rayleigh_damping.mph" is the COMSOL file used to compute the Rayleigh damped model.
    NOTE- I had to remove this file since it is above 100MB and I cannot post it on GitHub. Contact me if you need this file


-  The files "Pi_damped_time_response.txt", i=1,2,3 are the time responses of the damped mirror (Rayleigh damping). 

-  The file  "time_response_damped_rayleigh.m" contains all the time and deformation data extracted from the files "Pi_damped_time_response.txt"

-  The files  "frequency_response_Pi_damped_rayleigh", i=1,2 contains the amplitudes

-  The files  "frequency_response_Pi_damped_rayleigh_phase" i=1,2 contains the phases 

-  The file   "frequency_response_damped_rayleigh.m" contains the data originally stored in the files "frequency_response_Pi_damped_rayleigh" and "frequency_response_Pi_damped_rayleigh_phase"

-  The file   "frequency_response_damped_computations_plot.m" contains the code used to generate Figure 3.

-  The file   "frequency_response_and_eigenfrequencies_spring_foundation.m" was used to generate the Figures 5,6 and 7 in the SI paper.
   
   NOTE- I had to remove this file since it is above 100MB and I cannot post it on GitHub. Contact me if you need this file
	

-  The file   "frequency_response_and_eigenfrequencies_spring_foundation.mph" is a COMSOL file used to generate the first version of the file "frequency_response_and_eigenfrequencies_spring_foundation.m". Later on, this MATLAB was significantly modified.

-  The file   "extract_matrices_new.m" is used to generate figures 2,3,4 in the ML paper and to extract the system matrices for both papers.

-  The file   "zernfun.m" contains a function used to generate Zernike polynomials.

-  The file   "interp_zern.m" contains a function that interpolates the Zernike polynomimals and plots a graph.

-  The file   "scaling_simulation_check.m" is used to verify that the scaling procedure works. 

-  The file   "formZmatrix.m" is used to form the Zmap matrix that maps zernike coefficients into displacements

-  The files  "matricesi.mat", i=1,2,3,4,5 are extracted matrices for the following parameters:
               
	       matrices1.mat   actuator spacing=0.2, radius=1, stiffness=10^4, damping=500, mass=0.3, mesh size=9
	       matrices2.mat   same parameters as in matrices1.mat except mesh size=8
	       matrices3.mat   same parameters as in matrices1.mat except actuator spacing=0.1
	       matrices4.mat   same parameters as in matrices1.mat except stiffness=10^5 and damping=3000
               matrices5.mat   same parameters as in matrices4 except actuator spacing=0.1

-  The file   "seed_code.m" is used to develop the code "extract_matrices.m". Also this files contains many other comments 
that explain the codes and it can be used to develop new codes.

