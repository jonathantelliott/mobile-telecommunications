import autograd.numpy as np
#import numpy as np

p0 = 0
pz = 1
v = 2
O = 3
d0 = 4
dz = 5
c = 6
sigma = 7 # note that this is imputed

income_conv_factor = 10000.

def theta_pi(ds, theta, yc):
    return np.exp(theta[p0] + theta[pz] * yc / income_conv_factor)

def theta_di(ds, theta, yc):
    return np.exp(theta[d0] + theta[dz] * yc / income_conv_factor)

def theta_c(ds, theta, yc):
    return np.exp(theta[c])
