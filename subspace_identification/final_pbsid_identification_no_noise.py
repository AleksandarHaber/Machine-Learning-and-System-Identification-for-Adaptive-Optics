"""

Subspace Identification of a Deformable Mirror Model 
Author: Aleksandar Haber 
Date: January - November 2019



In this file, we identify a model when the measurements are not corrupted by the measurement noise.
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
h=5*10**(-3)
# total time steps used for the identification
time_steps=1200

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

###############################################################################
#  simulate the data generating model to obtain the identification and 
#  the validation data
###############################################################################
# initial state for system simulation - used for the identification
x0id=0.1*np.random.randn(A.shape[0],1)
# input sequence that is used for system identification
Uid=np.random.randn(B.shape[1],time_steps)
#Uid=np.ones((B.shape[1],time_steps)) # this is for the step response analysis
# identification output and state
Yid,Xid = systemSimulate(A,B,C,Uid,x0id)

# initial state for system simulation - used for validation
x0val=0.1*np.random.randn(A.shape[0],1)
# input sequence that is used for system validation
Uval=np.random.randn(B.shape[1],time_steps)
# validation output and state
Yval,Xval = systemSimulate(A,B,C,Uval,x0val)

#plt.plot(Yid[0,:100],'r')
###############################################################################
#               end of simulation
###############################################################################

###############################################################################
#                      Estimation of the VARX model
###############################################################################
# past value
past_value=10

import time
t0=time.time()
# estimate the Markov parameters
Markov,Z, Y_p_p_l =estimateMarkovParameters(Uid,Yid,past_value)
Error=Y_p_p_l-np.matmul(Markov,Z)
t1=time.time()
print(t1-t0)
###############################################################################
#                      Estimation of the final model
###############################################################################
# identification model order
model_order=40
Aid,Atilde,Bid,Kid,Cid,s_singular,X_p_p_l = estimateModel(Uid,Yid,Markov,Z,past_value,past_value,model_order)          

# open loop validation of  x_{k+1}=Ax_{k}+Bu_{k},  y_{k}=Cx_{k}  
# estimate the initial state
window=40 # window for estimating the initial state
x0est=estimateInitial(Aid,Bid,Cid,Uval,Yval,window)

# simulate the open loop model 
Yval_prediction,Xval_prediction = systemSimulate(Aid,Bid,Cid,Uval,x0est)

# compute the errors
relative_error_percentage, vaf_error_percentage, Akaike_error = modelError(Yval,Yval_prediction,r,m,30)
print('Final model relative error %f and VAF value %f' %(relative_error_percentage, vaf_error_percentage))

# plot the prediction and the real output 
plt.plot(Yval[0,:100],'k',label='Real output')
plt.plot(Yval_prediction[0,:100],'r',label='Prediction')
plt.legend()
plt.xlabel('Time steps')
plt.ylabel('Predicted and real outputs')
plt.show()

# compute the eigenvalues
eigen_A=linalg.eig(A)[0]
eigen_Aid=linalg.eig(Aid)[0]

# this is a detailed graph
#plt.figure(figsize=(7,7))
#plt.plot(eigen_A.real,eigen_A.imag,'or',linewidth=1, markersize=9, label='Original system')
#plt.plot(eigen_Aid.real,eigen_Aid.imag,'xk',linewidth=1, markersize=12, markeredgewidth= 2, label='Identified system')
#plt.title('Eigenvalues of the identified and original systems')
#plt.xlabel('Real part')
#plt.ylabel('Imaginary part')
#plt.legend()
#plt.savefig('eigenvalues.png')
#plt.show()

# plot the eigenvlaues
plt.figure(figsize=(6,6))
plt.plot(eigen_A.real,eigen_A.imag,'or',linewidth=1, markersize=9)
plt.plot(eigen_Aid.real,eigen_Aid.imag,'xk',linewidth=1, markersize=13, markeredgewidth= 2)
plt.title('Eigenvalues of the identified and original systems')
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.legend()
#plt.savefig('eigenvalues_h0005_n200.eps')
plt.show()


plt.plot(state_order,error,'or-',linewidth=2,markersize=9)
plt.plot(state_order,vaf,'sk-',linewidth=2,markersize=9) 
#plt.savefig('vaf_error_0_005.eps')
plt.show()

# plot the singular values
plt.figure()
plt.plot(s_singular,'xk',markersize=5)
plt.yscale('log')
#plt.savefig('singular_0_005.eps')
plt.show()


###############################################################################
#                   some results
###############################################################################

# discretization constant 5*10**(-3)
# conditions past=10 and future=10, window for estimating the initial state in the validation-40, time_steps=1200
state_order=np.array([5,  20, 35,  50,  65,  80, 95, 110, 125, 140 ])
error=np.array([62.288115063459834,  24.66615893424684, 10.492955176288715, 3.6205921295767536, 2.4034249963569763, 1.99819849274286,  2.359069435164046,   2.71245398123384,  2.866046159326907, 2.9079129865228794]  )
vaf=np.array([61.201907218411876, 93.91580603430475,   98.89897891668396,  99.86891312631246,   99.94223548286887,  99.960072027836,   99.94434791400076,   99.92642593399688, 99.91785779412608, 99.91544042062812])

# discretization constant 1*10**(-3)
# conditions past=15 and future=15, window for estimating the initial state in the validation-60, time_steps=2000

state_order=np.array([5,  20, 35,  50,  65,  80, 95, 110, 125, 140 ])
error=np.array([65.32277966448768, 43.03812249190754,   31.907947724703302, 21.05728428675633, 12.921182548541498,  2.9821732090140776,   2.0441328591263916, 1.138414395280608,   0.8674635489534611,  0.8754088706501165])
vaf=np.array([57.32934456904795, 81.47720012371562, 89.81882871997601,      95.56590778466725,  98.33043041547268,    99.91106642951438,  99.9582152085424,    99.98704012664618,  99.99247506991237,   99.99233659309186])

# discretization constant h=0.5*10**(-3)
# conditions past=15 and future=15, window for estimating the initial state in the validation-60, time_steps=2000
state_order=np.array([5,  20, 35,  50,  65,  80, 95, 110, 125, 140 ])
error=np.array([87.076584, 74.638547, 84.416081, 38.605859, 32.474764, 25.568138, 19.593143,  12.736097, 9.757194, 5.792968      ])
vaf=np.array([24.176685, 44.290873, 28.739253, 85.095876, 89.453897,    93.462703, 96.161088, 98.377918, 99.047972, 99.664415        ])





