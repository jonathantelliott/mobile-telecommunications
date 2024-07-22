# %%
# Import packages
import autograd.numpy as np
#import numpy as np

# %%
# Location of demand parameters
p0 = 0
pz = 1
v = 2
O = 3
d0 = 4
dz = 5
c = 6
sigma = 7

# %%
# Units for income
income_conv_factor = 10000.

# %%
# Parameters as function of vector of demand parameter \theta
def theta_pi(ds, theta, yc):
    """
        Return the heterogeneous price coefficient
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        yc : ndarray
            (M,J,I) array of incomes for each market, consumer type

    Returns
    -------
        pi : ndarray
            (M,J,I) array of price coefficients
    """
    
    pi = np.exp(theta[p0] + theta[pz] * yc / income_conv_factor)
    return pi

def theta_di(ds, theta, yc):
    """
        Return the heterogeneous data consumption coefficient
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        yc : ndarray
            (M,J,I) array of incomes for each market, consumer type

    Returns
    -------
        di : ndarray
            (M,J,I) array of data consumption coefficient
    """
    
    di = np.exp(theta[d0] + theta[dz] * yc / income_conv_factor)
    return di

def theta_c(ds, theta, yc):
    """
        Return the opportunity cost of time coefficient
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        yc : ndarray
            (M,J,I) array of incomes for each market, consumer type

    Returns
    -------
        c_ : float
            opportunity cost of time coefficient
    """
    
    c_ = np.exp(theta[c])
    return c_

def theta_sigma(ds, theta, yc):
    """
        Return the nesting parameter
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        yc : ndarray
            (M,J,I) array of incomes for each market, consumer type

    Returns
    -------
        s_ : float
            nesting parameter
    """
    
    s_ = np.exp(theta[sigma]) / (1.0 + np.exp(theta[sigma]))
    return s_
