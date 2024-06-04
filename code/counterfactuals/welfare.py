# %%
import numpy as np
import scipy.misc as misc

import counterfactuals.infrastructurefunctions as infr

import demand.coefficients as coef
import demand.blpextension as blp

# %%
def consumer_surplus(ds, xis, theta, include_logit_shock=True):
    """
        Return the individual consumer surplus associated with the product characteristics
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        include_logit_shock : bool
            determine whether or not to include logit shocks in the consumer surplus calculation

    Returns
    -------
        cs_i : ndarray
            (M,I) individual consumer surplus by market and income quantile
    """
    
    # Determine inidividual surplus
    p, E_u, v, O = blp.transformX(ds, theta, ds.data)
    theta_v = theta[coef.v]
    theta_O = theta[coef.O]
    sigma = theta[coef.sigma]
    deltas = theta_v * v + theta_O * O + xis
    yc = blp.ycX(ds, theta, ds.data)
    mus = blp.mu_ijm(ds, theta, p, E_u, yc)
    deltamu = deltas[:,:,np.newaxis] + mus
    
    if include_logit_shock: # if including logit shocks in the computation, then this is E[max_j {u_j}]
        I_g = (1. - sigma) * misc.logsumexp(deltamu / (1. - sigma), axis=1) # this is (1-sigma) * log(sum exp((delta + mu) / (1-sigma)))
        inclusive_val = np.logaddexp(0., I_g) # inclusive value, including outside option
        cs_mi = (1. / coef.theta_pi(ds, theta, yc))[:,0,:] * inclusive_val # M x I (consumer types)
    else: # else, this is sum of V_j * s_j
        shares = blp.s_mji(ds, theta, ds.data, xis) # M x J x I
        avg_V = np.sum(shares * deltamu, axis=1) # M x I
        cs_mi = (1. / coef.theta_pi(ds, theta, yc))[:,0,:] * avg_V
    
    return cs_mi

def agg_consumer_surplus(ds, xis, theta, pop, include_logit_shock=True, include_pop=True, market_weights=None):
    """
        Return the aggregate consumer surplus associated with the product characteristics
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        pop : ndarray
            (M,) array of market populations
        include_logit_shock : bool
            determine whether or not to include logit shocks in the consumer surplus calculation
        include_pop : bool
            determines whether or not to include population in the consumer surplus measure
        market_weights : ndarray
            (M,) array of weights for each market (or None if no weights)

    Returns
    -------
        cs : float
            aggregate consumer surplus
    """
    
    # Get consumer surplus by type
    cs_mi = consumer_surplus(ds, xis, theta, include_logit_shock=include_logit_shock)
    
    # Aggregate across markets and types
    pop_use = pop if include_pop else np.ones((cs_mi.shape[0],))
    if market_weights is not None:
        pop_use = pop_use * market_weights
    cs_i = np.sum(cs_mi * pop_use[:,np.newaxis] * (1. / float(cs_mi.shape[1])), axis=0)
    cs = np.sum(cs_i)
    
    return cs
    
def producer_surplus(ds, xis, theta, pop, market_size, R, c_u, c_R, include_pop=True, market_weights=None):
    """
        Return the producer surplus associated with the product characteristics and infrastructure investment
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        pop : ndarray
            (M,) array of market populations
        market_size : ndarray
            (M,) array of market sizes (in km^2)
        R : ndarray
            (M,) array of investment radii (in km)
        c_u : ndarray
            (J,) array of per-user costs
        c_R : ndarray
            (M,F) array of per-tower costs
        include_pop : bool
            determines whether or not to include population in the consumer surplus measure
        market_weights : ndarray
            (M,) array of weights for each market (or None if no weights)

    Returns
    -------
        ps : float
            aggregate consumer surplus
    """
    
    # Determine total profits
    shares = blp.s_mj(ds, theta, ds.data, xis) * pop[:,np.newaxis]
    if market_weights is not None:
        shares = shares * market_weights[:,np.newaxis]
    pidx = ds.chars.index(ds.pname)
    profits = np.sum(shares * (ds.data[:,:,pidx] - c_u[np.newaxis,:]))
    
    # Determine investment cost
    stations = infr.num_stations(R, market_size[:,np.newaxis])
    investment_cost = np.sum(stations * c_R)
    if market_weights is not None:
        investment_cost = np.sum(stations * c_R * market_weights[:,np.newaxis])
    
    # Calculate producer surplus
    sum_pop = np.sum(pop)
    if market_weights is not None:
        sum_pop = np.sum(pop * market_weights)
    ps = profits - investment_cost if include_pop else (profits - investment_cost) / sum_pop
    
    return ps

def total_surplus(ds, xis, theta, pop, market_size, R, c_u, c_R, include_logit_shock=True, include_pop=True, market_weights=None):
    """
        Return the total surplus associated with the product characteristics and infrastructure investment
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        pop : ndarray
            (M,) array of market populations
        market_size : ndarray
            (M,) array of market sizes (in km^2)
        R : ndarray
            (M,) array of investment radii (in km)
        c_u : ndarray
            (J,) array of per-user costs
        c_R : ndarray
            (M,F) array of per-tower costs
        include_logit_shock : bool
            determine whether or not to include logit shocks in the consumer surplus calculation
        include_pop : bool
            determines whether or not to include population in the consumer surplus measure
        market_weights : ndarray
            (M,) array of weights for each market (or None if no weights)

    Returns
    -------
        ts : float
            aggregate total surplus
    """
    
    cs = agg_consumer_surplus(ds, xis, theta, pop, include_logit_shock=include_logit_shock, include_pop=include_pop, market_weights=market_weights)
    ps = producer_surplus(ds, xis, theta, pop, market_size, R, c_u, c_R, include_pop=include_pop, market_weights=market_weights)
    ts = cs + ps
    
    return ts
    