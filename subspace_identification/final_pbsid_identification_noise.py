"""
Subspace Identification of a Deformable Mirror Model 
Author: Aleksandar Haber 
Date: January - November 2019

In this file, we identify a model when the measurements are corrupted by the measurement noise.
The data generating model has the following form:

M_{1}\ddot{z}+M_{2}\dot{z}+M_{3}z=B_{f}u   (1)
y=C_{f} z                                  (2)

where z is the displacement vector, u is the control force vector, y is the system output 
M_{1}, M_{2}, and M_{3} are the mass, damping, and stiffness matrices
B_{f} is the control matrix 
C_{f} is the output matrix

The model (1)-(2) is written as a descriptor state-space model 

Ed\dot{x}=Ad x +Bd u
y = Cd x 

and discretized using the backward Euler method. The discretized model has the following form 

x_{k} = A x_{k-1} + B u_{k} (3)
y_{k} = C x_{k}             (4)

this is the data-generating model. 

We identify a Kalman innovation state-space model

x_{k+1}=Aid x_{k}+Bid u_{k}+Kid (y_{k}-Cx_{k})
y_{k}=Cid x_{k}+e_{k}

where x_{k}\in \mathbb{R}^{n} is the state, u_{k}\in \mathbb{R}^{r} is the input vector,
y_{k}\in \mathbb{R}^{r}, A,B,K, and C are the system matrices

In this file, we assume that the output y_{k} of the data generating model is 
not corrupted by the measurement noise.

In the validation step we simulate the model

x_{k+1}=Aid x_{k}+Bid u_{k}
y_{k}=Cid x_{k}

that is, we omit the innovation e_{k}=y_{k}-Cx_{k}

"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy import linalg
from scipy.io import loadmat


from functionsSID import estimateMarkovParameters
from functionsSID import estimateModel
from functionsSID import estimateInitial
from functionsSID import systemSimulate
from functionsSID import modelError
from functionsSID import whiteTest
from functionsSID import portmanteau
from functionsSID import estimateInitial_K
from functionsSID import systemSimulate_Kopen
from functionsSID import systemSimulate_Kclosed


# load the descriptor state-space matrices from the file
matrices = loadmat('matrices1.mat')
Bf=np.matrix(matrices['B'].toarray())
Cr1=np.matrix(matrices['Cr1'].toarray())
M1=np.matrix(matrices['M1'].toarray())
M2=np.matrix(matrices['M2'].toarray())
M3=np.matrix(matrices['M3'].toarray())
Zmap=np.matrix(matrices['Zmap'])
pinvZmap=np.linalg.pinv(Zmap)
Cf=pinvZmap*Cr1;
n=M1.shape[0]
m=Bf.shape[1]
r=Cf.shape[0]

#define the descriptor state-space matrices
Ed=np.block([ 
    [np.identity(n),          np.zeros(shape=(n,n))], 
    [np.zeros(shape=(n,n)),   M1 ]
    ])

Ad=np.block([
        [np.zeros(shape=(n,n)), np.identity(n)],
        [-M3,                   -M2]
        ])

Bd=np.block([
        [np.zeros(shape=(n,m))],
        [Bf]
        ])
Cd=np.block([[Cf, np.zeros(shape=(r,n))]])

# this is the sampling constant
h=0.1*10**(-3)

Ed=np.asmatrix(Ed) #Unlike numpy.matrix, asmatrix does not make a copy if the input is already a matrix or an ndarray. Equivalent to matrix(data, copy=False).
Ad=np.asmatrix(Ad)
Bd=np.asmatrix(Bd)
Cd=np.asmatrix(Cd)

# model discretization
invM=inv(Ed-h*Ad)
A=np.matmul(invM,Ed);
B=h*np.matmul(invM,Bd);
C=Cd;
#D=np.asmatrix(np.zeros(shape=(C.shape[0],B.shape[1])))

# total time steps used for the identification
time_steps=100

###############################################################################
#      generation of the identification and validation data
###############################################################################

# identification data
# initial state for system simulation
x0id=0.1*np.random.randn(A.shape[0],1)
# input sequence for identification
Uid=1*np.random.randn(B.shape[1],time_steps)
#Uid_test=np.ones((B.shape[1],time_steps))
# outputs witout the noise
Yid_no_noise,Xid = systemSimulate(A,B,C,Uid,x0id)

# validation data
x0val=0.1*np.random.randn(A.shape[0],1)
Uval=1*np.random.randn(B.shape[1],time_steps)
Yval_no_noise,Xval = systemSimulate(A,B,C,Uval,x0val)

# generate a measurement noise for the identification sequence
SNR_target=20  # approximate signal to noise ratio
power_signal_no_noise=(1/time_steps)*np.linalg.norm(Yid_no_noise[0,:],2)**2
power_noise=power_signal_no_noise/SNR_target
noise=np.sqrt(power_noise)*np.random.randn(C.shape[0],time_steps)
# add noise to the identification data
Yid=Yid_no_noise+noise

# generate a measurement noise for the validation sequence
power_signal_no_noise_val=(1/time_steps)*np.linalg.norm(Yval_no_noise[0,:],2)**2
power_noise_val=power_signal_no_noise_val/SNR_target
noise_val=np.sqrt(power_noise_val)*np.random.randn(C.shape[0],time_steps)
# add noise to the validation data
Yval=Yval_no_noise+noise_val

# check the SNR ratio
power_noise_test=(1/time_steps)*np.linalg.norm(noise[0,:],2)**2
power_singnal_with_noise=(1/time_steps)*np.linalg.norm(Yid[0,:],2)**2
SNR_test=power_singnal_with_noise/power_noise_test
plt.plot(Yid_no_noise[0,:1000],'k')
plt.plot(Yid[0,:1000],'r')
plt.show()

plt.plot(Yval_no_noise[0,:1000],'k')
plt.plot(Yval[0,:1000],'r')
plt.show()



###############################################################################
#                      Estimation of the VARX model
###############################################################################
# maximal value of the past window
max_past=15
# estimated Markov matrices
Markov_matrices=[]
# Data matrices
Z_matrices=[]
# Akaike value
Akaike_value=[]
time_computations_Markov=[]

import time

# search for the "best" value of the past window
for i in np.arange(1,max_past+1):
   t0=time.time()
   print(i)
   Markov_tmp,Z, Y_p_p_l =estimateMarkovParameters(Uid,Yid,i)
   Markov=Markov_tmp
   Error=Y_p_p_l-np.matmul(Markov,Z)
   Cov=(1/(time_steps-i))*np.matmul(Error,Error.T)
   #Akaike_value.append(np.log(np.linalg.det(Cov))+(2/time_steps)*(Markov.shape[0]*Markov.shape[1]))  # it is not 2/time_steps, since we use less time steps to estimate the model.
   Akaike_value.append(np.log(np.linalg.det(Cov))+(2/(time_steps-i))*(Markov.shape[0]*Markov.shape[1]))
   Markov_matrices.append(Markov)
   Z_matrices.append(Z)
   print(Akaike_value[-1])
   t1=time.time()
   time_computations_Markov.append(t1-t0)
   print(t1-t0)
   

plt.figure()
plt.plot(np.arange(1,max_past+1),Akaike_value, linewidth=2)
plt.xlabel('past horizon - p')
plt.ylabel('AIC(p)')
#plt.savefig('AIC_long1.png')


state_order=35
rel_error=[]
vaf_values=[]
past_values=np.array([5])

for i in past_values:
    # estimate the final model
    Aid,Atilde,Bid,Kid,Cid,s_singular,X_p_p_l = estimateModel(Uid,Yid,Markov_matrices[i-1],Z_matrices[i-1],i,i,state_order)          

    # open loop validation of  x_{k+1}=Ax_{k}+Bu_{k},  y_{k}=Cx_{k}  
    # estimate the initial state
    h=50
    # estimate x0 for the open loop model
    x0est=estimateInitial(Aid,Bid,Cid,Uval,Yval,h)
    # estimate x0 for the open loop innovation model
    #x0est=estimateInitial_K(Atilde,Bid,Cid,Kid,Uval,Yval,h)
    # simulate the open loop model 
    Yval_prediction,Xval_prediction = systemSimulate(Aid,Bid,Cid,Uval,x0est)
    # simulate the open loop 
    #Yval_prediction,Xval_prediction = systemSimulate(Aid,Bid,Cid,Uval,x0est)
    #Yval_prediction,Xval_prediction = systemSimulate_Kopen(Atilde,Bid,Cid,Kid,Uval,x0est,np.array([Yval[:,0]]).T)
    # compute the errors
    relative_error_percentage, vaf_error_percentage, Akaike_error = modelError(Yval,Yval_prediction,r,m,15)
    rel_error.append(relative_error_percentage)
    vaf_values.append(vaf_error_percentage)
    print('Final model relative error %f and VAF value %f' %(relative_error_percentage, vaf_error_percentage))
    

# plot the prediction and the real output 
plt.plot(Yval[0,:300],'k')
plt.plot(Yval_prediction[0,:300],'r')


###############################################################################
# error results, open-loop validation on the basis of the simulation of 
# x_{k+1}=Ax_{k}+Bu_{k}, y_{k}=Cx_{k}

error_n_10=np.array([61.73784275640332, 61.68320493872652, 61.647492646400245, 61.61255732666312, 61.61574490443951, 61.57316251025201, 61.61094756838996, 61.68361398744785, 61.696897499816394, 61.75815490423734, 61.832204578716144, 61.90315699410216, 62.09549936650165, 62.27500397515479, 62.65622269600151, 62.68600106409974])
vaf_n_10=np.array([61.884387717856185, 61.95182228487064, 61.99586650412028, 62.03892779668651, 62.034999798710366, 62.08745658486096, 62.04091139725103, 61.95131765447528, 61.934928388971485, 61.85930302824225, 61.76778476935794, 61.67999154163542, 61.44148958424793, 61.21823879894455, 60.741977574690665, 60.70465270591687])


error_n_30=np.array([54.29065902259113, 54.310708734530145, 54.42981565996065, 54.456898483287944, 54.66798905785879, 54.79713823571187, 54.99453069112724, 55.27202043374658, 55.55293590399011, 55.82291878229903, 56.113411798008215, 56.48611531685258, 56.98720466381795, 57.37048605213823, 58.14694324502029, 58.86527054922467])
vaf_n_30=np.array([70.52524342892745, 70.50346916753031, 70.37395167222702, 70.3444620758087, 70.11410972369832, 69.97273641176284, 69.75601594062664, 69.450037571715, 69.13871312447166, 68.83801738624847, 68.51285016387152, 68.09318776411232, 67.52458504604127, 67.08627330141411, 66.1893299126039, 65.34879923166584])


error_n_100=np.array([56.198506149096, 56.49937247844585, 56.740770349353184, 56.91186952009547, 57.260475608491014, 57.57523692769132, 57.84927748950521, 58.03033530625045, 58.33566136075643, 58.57956707159448, 58.90476667299885, 59.08706341612691, 59.37562938385709, 59.98400290947469, 60.395507633923586, 60.89887686088258])
vaf_n_100=np.array([68.4172790661002, 68.07820909541834, 67.80484980161962, 67.61039107727629, 67.21237933089405, 66.85092092720208, 66.53461093942225, 66.32480184244143, 65.9695061360315, 65.68434321704562, 65.30228463199563, 65.08718936858597, 64.74534635270845, 64.01919394956133, 63.52382657640678, 62.9132679708306])


error_n_160=np.array([57.20185251615868, 57.9639935810376, 58.65165978863113, 59.24930587081555, 59.85181341182005, 60.37884829001767, 60.85384708861506, 61.390834540028486, 61.84526924823868, 62.329042473732024, 62.927163008898646, 63.0298987078594, 63.42239053285941, 64.02330998143908, 64.26115652215475, 65.04840488521029])
vaf_n_160=np.array([67.27948068719631, 66.40175448137433, 65.5998280403867, 64.89519753826542, 64.17760431316677, 63.54394679171031, 62.96809294515455, 62.31165434478845, 61.75162671612864, 61.1509046430771, 60.40172155651498, 60.27231868876983, 59.776003790974656, 59.01015779020562, 58.70503762435128, 57.687050218897504])

plt.plot(past_values,error_n_10,'r')
plt.plot(past_values,error_n_30,'b')
plt.plot(past_values,error_n_100,'k')
plt.plot(past_values,error_n_160,'m')
plt.savefig('error_past_value.eps')
plt.show()

plt.plot(past_values,vaf_n_10,'r')
plt.plot(past_values,vaf_n_30,'b')
plt.plot(past_values,vaf_n_100,'k')
plt.plot(past_values,vaf_n_160,'m')
plt.savefig('vaf_past_value.eps')
plt.show()


# plot the prediction and the true value    
plt.plot(Yval[0,:200],'k',linewidth=2)
plt.plot(Yval_prediction[0,:200],'r',linewidth=2)
plt.savefig('output_prediction0.eps')
plt.show()

# residual test
corr_matrices_model_Kopen = whiteTest(Yval,Yval_prediction)

# Choose the correlation indices
idx1=10
idx2=10

# extract the correlation vector
vector_corr_Kopen=[]
for matrix in corr_matrices_model_Kopen:
    vector_corr_Kopen.append(matrix[idx1,idx2])
    

# entries that are outside of the two standard-error limits
entries_outside=[x for x in vector_corr_Kopen if np.abs(x) >= (2/(np.sqrt(Yval.shape[1])))]
len(entries_outside)/len(vector_corr_Kopen)

plt.figure()
plt.plot(vector_corr_Kopen,'b')
plt.plot((2/(np.sqrt(Yval.shape[1])))*np.ones(Yval.shape[1]),'r--')
plt.plot(-(2/(np.sqrt(Yval.shape[1])))*np.ones(Yval.shape[1]),'r--')
plt.xlabel('Lag')
plt.ylabel('Correlation value')
plt.ylim((-0.2,0.2))
#plt.savefig('corr130_short_modelB.eps')
plt.show()

# percentages of entries that are outside the bounds 
# corr 1-1-0.02530120481927711, corr 1-5 - 0.016867469879518072
# corr 1-15-0.01967871485943775, corr 1-30 - 0.01325301204819277
###############################################################################


















###############################################################################







