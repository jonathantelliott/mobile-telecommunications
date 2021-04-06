import autograd.numpy as np
#import autograd.scipy.special as special
#import numpy as np
#import scipy.special as special
from scipy.special import expi
from autograd.extend import primitive, defvjp

import demand.coefficients as coef

conv_factor = 1000. # =1. if MB, =1000. if GB

throttle_lim = 500. / conv_factor
    
def Q_L(Q):
    return 128. / 1000. # convert to Mbps

def correct_units(xbar, Q, QL):
    xbar = xbar / conv_factor
    Q = Q / 8. / conv_factor # bps -> Bps
    QL = QL / 8. / conv_factor # bps -> Bps
    return xbar, Q, QL

@primitive
def Ei(x):
    return expi(x)

# derivative of Ei(.)
defvjp(Ei, lambda ans, x: lambda g: g * np.exp(x) / x)

def E_x(ds, theta, X, Q, xbar, yc):
    d = coef.theta_di(ds, theta, yc)
    Q = Q[:,:,np.newaxis] # add i index
    QL = Q_L(Q)
    xbar = xbar[:,:,np.newaxis]
    xbar, Q, QL = correct_units(xbar, Q, QL)
    c = coef.theta_c(ds, theta, yc)
    
    int_1_1 = np.exp(-d * c * (xbar + 1.) / Q)
    int_1_2 = np.exp(-d * c * (xbar + 1.) / Q) / (c * d) * -Q * (d * c * (xbar + 1.) / Q + 1.)
    int_1_3 = -np.exp(-d * c / Q)
    int_1_4 = -np.exp(-d * c / Q) / (c * d) * -Q * (d * c / Q + 1.)
    int_1 = int_1_1 + int_1_2 + int_1_3 + int_1_4
    
    int_2_1 = -xbar * np.exp(-d * c * (xbar + 1.) / QL)
    int_2_2 = xbar * np.exp(-d * c * (xbar + 1.) / Q)
    int_2 = int_2_1 + int_2_2
    
    int_3_1 = -np.exp(-d * c * (xbar + 1.) / QL)
    int_3_2 = -np.exp(-d * c * (xbar + 1.) / QL) / (c * d) * -QL * (d * c * (xbar + 1.) / QL + 1.)
    int_3 = int_3_1 + int_3_2
    
    w_throttle = int_1 + int_2 + int_3
    
    int_2_wo_throttle = int_2_2
    
    wo_throttle = int_1 + int_2_wo_throttle
    
    return (np.tile(xbar,(1,1,w_throttle.shape[2])) < throttle_lim) * wo_throttle + (np.tile(xbar,(1,1,w_throttle.shape[2])) >= throttle_lim) * w_throttle

def E_u(ds, theta, X, Q, xbar, yc):
    d = coef.theta_di(ds, theta, yc)
    Q = Q[:,:,np.newaxis] # add i index
    QL = Q_L(Q)
    xbar = xbar[:,:,np.newaxis]
    xbar, Q, QL = correct_units(xbar, Q, QL)
    c = coef.theta_c(ds, theta, yc)
    
    int_1_1 = np.exp(-d * c * (xbar + 1.) / Q) / (d * Q) * -(d * c * (xbar + 1.) + Q) * np.log(xbar + 1.)
    int_1_2 = Ei(-d * c * (xbar + 1.) / Q) / d
    int_1_3 = np.exp(-d * c * (xbar + 1.) / Q) / (d * Q) * d * c * xbar
    int_1_4 = -Ei(-d * c / Q) / d
    int_1 = int_1_1 + int_1_2 + int_1_3 + int_1_4
    
    int_2_1 = np.exp(-d * c * (xbar + 1.) / QL) / Q * c * xbar
    int_2_2 = np.exp(-d * c * (xbar + 1.) / QL) / (d * Q) * -np.log(xbar + 1.) * (d * Q * c * (xbar + 1.) / QL + Q)
    int_2_3 = -np.exp(-d * c * (xbar + 1.) / Q) / Q * c * xbar
    int_2_4 = -np.exp(-d * c * (xbar + 1.) / Q) / (d * Q) * -np.log(xbar + 1.) * (d * c * (xbar + 1.) + Q)
    int_2 = int_2_1 + int_2_2 + int_2_3 + int_2_4
    
    int_3_1 = -np.exp(-d * c * (xbar + 1.) / QL) / (Q * QL) * -c * (-QL * xbar + Q + Q * xbar)
    int_3_2 = -np.exp(-d * c * (xbar + 1.) / QL) / (d) * -(d * c * (xbar + 1.) / QL + 1.) * np.log(xbar + 1.)
    int_3_3 = -Ei(-d * c * (xbar + 1.) / QL) / d
    int_3_4 = -np.exp(-d * c * (xbar + 1.) / QL) / QL * c * (xbar + 1.)
    int_3 = int_3_1 + int_3_2 + int_3_3 + int_3_4
    
    w_throttle = int_1 + int_2 + int_3
    
    int_2_wo_throttle = int_2_3 + int_2_4
    
    wo_throttle = int_1 + int_2_wo_throttle
    
    return (np.tile(xbar,(1,1,w_throttle.shape[2])) < throttle_lim) * wo_throttle + (np.tile(xbar,(1,1,w_throttle.shape[2])) >= throttle_lim) * w_throttle