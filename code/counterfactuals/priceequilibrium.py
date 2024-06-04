# %%
import copy

import numpy as np
from numpy.lib.index_tricks import CClass
from scipy.optimize import fsolve

import counterfactuals.transmissionequilibrium as transeq

import demand.blpextension as blp

# %%
def s_jacobian_p(p, cc, ds, xis, theta, num_stations, pop, symmetric=False, impute_MVNO={'impute': False}, q_0=None, eps=0.01, full=True, market_weights=None, calc_q_carefully=False):
    """
        Return the Jacobian of the share function with respect to prices, based on two-sided numerical derivative
    
    Parameters
    ----------
        p : ndarray
            (J,) array of prices at which taking derivative
        cc : ndarray
            (M,F) array of channel capacity in Mb/s
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        num_stations : ndarray
            (M,F) array of number of stations in each market
        pop : ndarray
            (M,) array of market populations
        symmetric : bool
            specifies whether the equilibrium solving for is symmetric (quicker to compute)
        impute_MVNO : dict
            dict with
            'impute' : bool (whether to impute the Qs for MVNO)
            'firms_share' (optional) : ndarray ((F-1,) array of whether firms share qualities with MVNOs)
            'include' (optional) : bool (whether to include MVNO Q in returned Q)
        q_0 : ndarray
            (M,F) array of initial guess of q
        eps : float
            size of perturbation to measure derivative
        full : bool
            determines whether taking full or partial elasticity
        market_weights : ndarray
            (M,) array of weights for each market (or None if no weights)

    Returns
    -------
        p_deriv : ndarray
            (J,J) array of firm's pricing FOC
    """
    
    # Number of firms
    firms, firm_counts = np.unique(ds.firms, return_counts=True)
    num_firms = firms.shape[0]
    M = ds.data.shape[0]
    
    # Obtain indices
    pidx = ds.chars.index(ds.pname)
    qidx = ds.chars.index(ds.qname)
    
    # Create price arrays, JxJ (axis1=product prices, axis2=with respect to which product)
    p_high = p[:,np.newaxis] + np.identity(p.shape[0]) * eps
    p_low = p[:,np.newaxis] - np.identity(p.shape[0]) * eps

    # Derivative for each product
    p_deriv = np.zeros(p_high.shape)
    for j in range(p.shape[0]):
        # Add price arrays to product characteristics
        ds_high = copy.deepcopy(ds) # this method might be slow b/c requires copying all other attributes too--might be better to rewrite some demand functions instead
        ds_low = copy.deepcopy(ds)
        if symmetric:
            ds_high.data[:,:,pidx] = np.concatenate((p_high[:,j], np.tile(p, (num_firms - 1,))))[np.newaxis,:]
            ds_low.data[:,:,pidx] = np.concatenate((p_low[:,j], np.tile(p, (num_firms - 1,))))[np.newaxis,:]
        else:
            ds_high.data[:,:,pidx] = p_high[:,j][np.newaxis,:]
            ds_low.data[:,:,pidx] = p_low[:,j][np.newaxis,:]

        # Calculate the equilibrium download speeds and add to product characteristics
        # Go through market-by-market b/c faster to compute Jacobians that way (wouldn't need to do that if could calculate transmission equilibrium Jacobians analytically, which is in theory possible)
        ds_high_temp = copy.deepcopy(ds_high)
        ds_low_temp = copy.deepcopy(ds_low)
        F = num_firms
        if impute_MVNO['impute']:
            if not impute_MVNO['include']:
                F = F - 1 # don't need to keep track of MVNO quality
        q_high = np.zeros((M,F))
        q_low = np.zeros((M,F))
        for m in range(ds_high.data.shape[0]):
            select_m = np.arange(M) == m
            ds_high_temp.data = ds_high.data[select_m,:,:]
            ds_low_temp.data = ds_low.data[select_m,:,:]
            if not full:  # if not taking into account impact on download speed, just keep the same prices for determining download speeds
                ds_high_temp.data[:,:,pidx] = p[np.newaxis,:] if not symmetric else np.tile(p, (num_firms,))[np.newaxis,:]
                ds_low_temp.data[:,:,pidx] = p[np.newaxis,:] if not symmetric else np.tile(p, (num_firms,))[np.newaxis,:]
            q_high[m,:] = transeq.q(cc[select_m,:], ds_high_temp, xis[select_m,:], theta, num_stations[select_m,:], pop[select_m], impute_MVNO=impute_MVNO, q_0=q_0, calc_carefully=calc_q_carefully)
            q_low[m,:] = transeq.q(cc[select_m,:], ds_low_temp, xis[select_m,:], theta, num_stations[select_m,:], pop[select_m], impute_MVNO=impute_MVNO, q_0=q_0, calc_carefully=calc_q_carefully)
        ds_high.data[:,:,qidx] = np.repeat(q_high, firm_counts, axis=1) # only works b/c products in order
        ds_low.data[:,:,qidx] = np.repeat(q_low, firm_counts, axis=1) # only works b/c products in order

        # Calculate demand for each product
        shares_pop_high = blp.s_mj(ds_high, theta, ds_high.data, xis) * pop[:,np.newaxis]
        shares_pop_low = blp.s_mj(ds_high, theta, ds_low.data, xis) * pop[:,np.newaxis]
        if market_weights is not None:
            shares_pop_high = shares_pop_high * market_weights[:,np.newaxis]
            shares_pop_low = shares_pop_low * market_weights[:,np.newaxis]
        s_high = np.sum(shares_pop_high, axis=0)
        s_low = np.sum(shares_pop_low, axis=0)

        # Calculate Jacobian for jth price
        select_firms = np.ones(s_high.shape[0], dtype=bool)
        if symmetric:
            select_firms[p.shape[0]:] = False
        p_deriv[:,j] = (s_high[select_firms] - s_low[select_firms]) / (2. * eps)

    # Return Jacobian
    return p_deriv

def p_foc(p, c_u, cc, ds, xis, theta, num_stations, pop, symmetric=False, impute_MVNO={'impute': False}, q_0=None, eps=0.01, market_weights=None, calc_q_carefully=False):
    """
        Return the FOCs of the pricing function, based on two-sided numerical derivative
    
    Parameters
    ----------
        p : ndarray
            (J,) array of prices at which taking derivative
        c_u : ndarray
            (J,) array of product marginal costs
        cc : ndarray
            (M,F) array of channel capacity in Mb/s
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        num_stations : ndarray
            (M,F) array of number of stations in each market
        pop : ndarray
            (M,) array of market populations
        symmetric : bool
            specifies whether the equilibrium solving for is symmetric (quicker to compute)
        impute_MVNO : dict
            dict with
            'impute' : bool (whether to impute the Qs for MVNO)
            'firms_share' (optional) : ndarray ((F-1,) array of whether firms share qualities with MVNOs)
            'include' (optional) : bool (whether to include MVNO Q in returned Q)
        q_0 : ndarray
            (M,F) array of initial guess of q
        eps : float
            size of perturbation to measure derivative
        market_weights : ndarray
            (M,) array of weights for each market (or None if no weights)

    Returns
    -------
        foc : ndarray
            (J,) array of firms' pricing FOCs
    """
    
    firms = np.unique(ds.firms)
    select_firms = np.ones(ds.firms.shape[0], dtype=bool)
    select_firms_unique = np.ones(firms.shape[0], dtype=bool)
    
    # Alter variables if looking at symmetric equilibrium
    if symmetric:
        # Expand variables
        num_firms = firms.shape[0]
        xis = np.tile(xis, (1,num_firms))
        cc = np.tile(cc, (1,num_firms))
        num_stations = np.tile(num_stations, (1,num_firms))
        # Only select the first firm
        select_firms_unique[1:] = False
        select_firms[p.shape[0]:] = False

    # Solve for shares for each product (summing across markets)
    shares_pop = blp.s_mj(ds, theta, ds.data, xis) * pop[:,np.newaxis]
    if market_weights is not None:
        shares_pop = shares_pop * market_weights[:,np.newaxis]
    shares = np.sum(shares_pop, axis=0)[select_firms]

    # Solve for Jacobian of shares with respect to prices
    Jac = s_jacobian_p(p, cc, ds, xis, theta, num_stations, pop, symmetric=symmetric, impute_MVNO=impute_MVNO, q_0=q_0, eps=eps, market_weights=market_weights, calc_q_carefully=calc_q_carefully)

    # Determine FOCs
    foc = np.zeros(p.shape)
    for f, firm in enumerate(firms[select_firms_unique]):
        firm_cond = ds.firms[select_firms] == firm
        Jac_firm = Jac[np.ix_(firm_cond, firm_cond)]
        shares_firm = shares[firm_cond]
        foc[firm_cond] = shares_firm + np.matmul(Jac_firm, p[firm_cond] - c_u[firm_cond])

    return foc

def price_elast(p, cc, ds, xis, theta, num_stations, pop, symmetric=False, impute_MVNO={'impute': False}, q_0=None, eps=0.01, full=True):
    """
        Return the elasticity of shares with respect to prices
    
    Parameters
    ----------
        p : ndarray
            (J,) array of prices at which taking derivative
        cc : ndarray
            (M,F) array of channel capacity in Mb/s
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        num_stations : ndarray
            (M,F) array of number of stations in each market
        pop : ndarray
            (M,) array of market populations
        symmetric : bool
            specifies whether the equilibrium solving for is symmetric (quicker to compute)
        impute_MVNO : dict
            dict with
            'impute' : bool (whether to impute the Qs for MVNO)
            'firms_share' (optional) : ndarray ((F-1,) array of whether firms share qualities with MVNOs)
            'include' (optional) : bool (whether to include MVNO Q in returned Q)
        q_0 : ndarray
            (M,F) array of initial guess of q
        eps : float
            size of perturbation to measure derivative
        full : bool
            determines whether taking full or partial elasticity

    Returns
    -------
        elast : ndarray
            (J,) array of elasticities
    """
    
    firms = np.unique(ds.firms)
    select_firms = np.ones(ds.firms.shape[0], dtype=bool)
    select_firms_unique = np.ones(firms.shape[0], dtype=bool)
    
    # Alter variables if looking at symmetric equilibrium
    if symmetric:
        # Expand variables
        num_firms = firms.shape[0]
        xis = np.tile(xis, (1,num_firms))
        cc = np.tile(cc, (1,num_firms))
        num_stations = np.tile(num_stations, (1,num_firms))
        # Only select the first firm
        select_firms_unique[1:] = False
        select_firms[p.shape[0]:] = False

    # Solve for shares for each product (summing across markets)
    shares = np.sum(blp.s_mj(ds, theta, ds.data, xis) * pop[:,np.newaxis], axis=0)[select_firms]
    
    # Solve for Jacobian of shares with respect to prices
    Jac = s_jacobian_p(p, cc, ds, xis, theta, num_stations, pop, symmetric=symmetric, impute_MVNO=impute_MVNO, q_0=q_0, eps=eps, full=full)
    
    # Determine elasticities
    elast = np.zeros(p.shape)
    for f, firm in enumerate(firms[select_firms_unique]):
        firm_cond = ds.firms[select_firms] == firm
        Jac_firm = Jac[np.ix_(firm_cond, firm_cond)]
        shares_firm = shares[firm_cond]
        p_firm = p[firm_cond]
        partial_share_partial_price = np.sum(Jac_firm, axis=0) # sum over product shares
        elast[firm_cond] = partial_share_partial_price * (p_firm / shares_firm)
        
    return elast
