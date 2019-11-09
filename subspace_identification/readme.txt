The codes used in the Subspace Identification (SI) paper. 


"final_pbsid_identification_noise.py"     - file containing the code used to identify the DM model in the case when the measurements are corrupted by noise
                                          - this file is used to generate the figures 11 and 12 in the SI paper. 

"final_pbsid_identification_no_noise.py"  - file containing the code used to identify the DM model in the case when the measurements are not corrupted by noise
                                          - this file is used to generate the figures 8,9,10 in the SI paper. 

"functionsSID.py" - file containing the subspace identification functions.


	       - "matrices1.mat"   actuator spacing=0.2, radius=1, stiffness=10^4, damping=500, mass=0.3, mesh size=9
	       - "matrices2.mat"   same parameters as in matrices1.mat except mesh size=8
	       - "matrices3.mat"   same parameters as in matrices1.mat except actuator spacing=0.1
	       - "matrices4.mat"   same parameters as in matrices1.mat except stiffness=10^5 and damping=3000
               - "matrices5.mat"   same parameters as in matrices4 except actuator spacing=0.1