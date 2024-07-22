# %%
# Import packages
import autograd.numpy as np
import autograd.scipy.special as special
#import numpy as np
#import scipy.special as special

import demand.coefficients as coef
import demand.iteration as iteration
import demand.dataexpressions as de

# %%
# Functions related to demand

def mu_ijm(ds, theta, p, E_u, yc):
    """
        Return the heterogeneous component of utilities
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        p : ndarray
            (M,J) array of prices
        E_u : ndarray
            (M,J,I) array of expected utility of data consumption for each market, product, consumer type
        yc : ndarray
            (M,J,I) array of incomes for each market, consumer type

    Returns
    -------
        heterogeneous_component : ndarray
            (M,J,I) array of heterogeneous component of utilities for each market, product, consumer type
    """
    
    theta_p = coef.theta_pi(ds, theta, yc)
    heterogeneous_component = -theta_p * p[:,:,np.newaxis] + E_u
    return heterogeneous_component

def mutilde(ds, theta, X, p, E_u, v, O, yc):
    """
        Return the transformed heterogeneous component of utilities at firm-level based on paper's grouped products BLP extension
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        X : ndarray
            (M,J) array of market-product characteristics
        p : ndarray
            (M,J) array of prices
        E_u : ndarray
            (M,J,I) array of expected utility of data consumption for each market, product, consumer type
        v : ndarray
            (M,J) array of unlimited voice dummies
        O : ndarray
            (M,J) array of Orange dummies
        yc : ndarray
            (M,J,I) array of incomes for each market, consumer type

    Returns
    -------
        mutilde_ifm : ndarray
            (M,J,I) array of transformed heterogeneous component of utilities for each market, product, consumer type
    """
    
    theta_v = theta[coef.v]
    theta_O = theta[coef.O]
    sigma = coef.theta_sigma(ds, theta, yc)
    mutilde_ijm = (theta_v * (v[:,:,np.newaxis] - ds.Xbar(v[:,:,np.newaxis])) + theta_O * (O[:,:,np.newaxis] - ds.Xbar(O[:,:,np.newaxis])) + mu_ijm(ds, theta, p, E_u, yc)) / (1. - sigma)
    # go from MxJ matrix to MxF, summing
    mutilde_ifm = np.tile(mutilde_ijm[:,:ds.J_O], (1,X.shape[0],1)) * ds.relevant_markets(X)[:,:ds.J_O*X.shape[0],np.newaxis] # for J_O
    for f in np.unique(ds.firms[~ds.Oproducts]):
        mutilde_f = special.logsumexp(mutilde_ijm[:,ds.firms==f,:], axis=1, keepdims=True)
        mutilde_ifm = np.concatenate((mutilde_ifm, mutilde_f), axis=1)
    return mutilde_ifm

def use_f_idxs(ds, X, nonOxis):
    """
        Return bools with whether to solve for the xi
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        X : ndarray
            (M,J) array of market-product characteristics
        nonOxis : None or dict
            None (in case of solving for all) or dictionary telling which firms to impute and what xi to impute for them

    Returns
    -------
        idxs : ndarray
            (J,) array of whether to solve for xi for each product (flattened)
    """
    
    idxs = np.ones(ds.Fnum(X), dtype=bool)
    if nonOxis is not None:
        argsortidxs = np.zeros(len(nonOxis['firms']))
        for i in range(len(nonOxis['firms'])):
            firmidx = ds.ftj(X)[0,np.argwhere(ds.firms == (nonOxis['firms'][i]))[0][0]].astype(int)
            argsortidxs[i] = firmidx
            idxs[firmidx] = False
    return idxs
    
def normalized_deltas(ds, theta, X, nonOxis, v, O, mutilde_ifm, yc):
    """
        Return the deltas implied by imputed xis
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        X : ndarray
            (M,J) array of market-product characteristics
        nonOxis : None or dict
            None (in case of solving for all) or dictionary telling which firms to impute and what xi to impute for them
        v : ndarray
            (M,J) array of unlimited voice dummies
        O : ndarray
            (M,J) array of Orange dummies
        mutilde_ifm : ndarray
            (M,J,I) array of transformed heterogeneous component of utilities for each market, product, consumer type
        yc : ndarray
            (M,J,I) array of incomes for each market, consumer type

    Returns
    -------
        norm : ndarray
            (F_impute,) array of logsumexp of implied deltas
        normdelta : ndarray
            (F_impute,) array of implied deltas
    """
    
    theta_v = theta[coef.v]
    theta_O = theta[coef.O]
    sigma = coef.theta_sigma(ds, theta, yc)
    idxs = np.ones(ds.Fnum(X), dtype=bool)
    norm = np.array([])
    normdelta = np.array([])
    if nonOxis is not None:
        argsortidxs = np.array([])
        nonOdeltas = np.array([])
        for i in range(len(nonOxis['firms'])):
            nonOdelta = (theta_v * np.mean(v[:,ds.firms==(nonOxis['firms'][i])]) + theta_O * np.mean(O[:,ds.firms==(nonOxis['firms'][i])]) + nonOxis['xis'][i]) / (1. - sigma)
            firmidx = ds.ftj(X)[0,np.argwhere(ds.firms == (nonOxis['firms'][i]))[0][0]].astype(int)
            normdelta = np.concatenate((normdelta, np.array([nonOdelta])))
            argsortidxs = np.concatenate((argsortidxs, np.array([firmidx])))
            if nonOdeltas.shape[0] == 0:
                nonOdeltas = (nonOdelta + mutilde_ifm[:,firmidx,:])[np.newaxis,:,np.newaxis,:]
            else:
                nonOdeltas = np.concatenate((nonOdeltas, (nonOdelta + mutilde_ifm[:,firmidx,:])[np.newaxis,:,np.newaxis,:]), axis=0)
            idxs[firmidx] = False
        normdelta = normdelta[np.argsort(argsortidxs)] # make sure that these deltas are sorted in the order needed for reconstructing delta array
        norm = special.logsumexp(nonOdeltas, axis=0)
    return norm, normdelta

def sbar_f(deltatilde_f, mutilde_ifm, norm, sigma, relevantmarkets, mktweights, yc):
    """
        Return market shares for grouped products
    
    Parameters
    ----------
        deltatilde_f : ndarray
            (F~,) array of transformed mean utilities
        mutilde_ifm : ndarray
            (M,F~,I) array of transformed heterogeneous component of utilities for each market, product, consumer type
        norm : ndarray
            (F_impute,) array of logsumexp of implied deltas
        sigma : float
            nesting parameter
        relevantmarkets : ndarray
            (M,F~,1) array of whether a group (F~ axis, could be true product or grouped) is present in a particular market
        mktweights : ndarray
            (M,F~,1) array of weights for each market (based on market sizes)
        yc : ndarray
            (M,J,I) array of incomes for each market, consumer type

    Returns
    -------
        s_f : ndarray
            (F~,) array of shares
    """
    
    deltamu = deltatilde_f[np.newaxis,:,np.newaxis] + mutilde_ifm
    I_g = (1. - sigma) * special.logsumexp(deltamu, axis=1, keepdims=True, b=relevantmarkets) # this is (1-sigma) * log(sum_{everything} relevant_in_m * exp((delta + mu) / (1-sigma))), so we're not including irrelevant markets, and normalized values aren't here b/c there are none (if there are, will be taken care of in the overriding next statement)
    if norm.shape[0] > 1:
        I_g = (1. - sigma) * np.logaddexp(norm, special.logsumexp(deltamu, axis=1, keepdims=True, b=relevantmarkets)) # for when we have norm, this is (1-sigma) * log(sum_{everything} relevant_in_m * exp((delta + mu) / (1-sigma)) + sum_{stuff in norm} exp((delta + mu) / (1-sigma))), so we're not including irrelevant markets, the normalized are always relevant
    I = np.logaddexp(0., I_g)
    s_mfi = np.exp(deltamu + (sigma / (sigma - 1.)) * I_g - I) # see Grigolon and Verboven (2015) Review of Economics and Statistics, gets rid of exp overflow problems
    s_fi = np.sum(mktweights * s_mfi, axis=0) # average across markets, weighting by population, this is an F x I matrix
    s_f = np.matmul(s_fi, (np.ones(yc.shape[2]) / float(yc.shape[2]))) # we have quantiles, so the weights when taking the integral are all equal
    return s_f

def contractionmap(deltatilde_f, mutilde_ifm, norm, sigma, relevantmarkets, mktweights, yc, log_shares):
    """
        Return output of contraction mapping of paper's grouped products BLP extension
    
    Parameters
    ----------
        deltatilde_f : ndarray
            (F~,) array of transformed mean utilities
        mutilde_ifm : ndarray
            (M,F~,I) array of transformed heterogeneous component of utilities for each market, product, consumer type
        norm : ndarray
            (F_impute,) array of logsumexp of implied deltas
        sigma : float
            nesting parameter
        relevantmarkets : ndarray
            (M,F~,1) array of whether a group (F~ axis, could be true product or grouped) is present in a particular market
        mktweights : ndarray
            (M,F~,1) array of weights for each market (based on market sizes)
        yc : ndarray
            (M,J,I) array of incomes for each market, consumer type
        log_shares : ndarray
            (F~,) array of logged shares

    Returns
    -------
        res : ndarray
            (F~,) array of output of contraction mapping
    """
    
    res = deltatilde_f + (log_shares - np.log(sbar_f(deltatilde_f, mutilde_ifm, norm, sigma, relevantmarkets, mktweights, yc))) # using MVNO is the last entry (last firm), (1 - sigma) dampening term not needed here
    return res

def ycX(ds, theta, X):
    """
        Return the incomes, stored in X
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        X : ndarray
            (M,J) array of market-product characteristics

    Returns
    -------
        yc : ndarray
            (M,J,I) array of incomes for each market, consumer type
    """
    
    yc1idx = ds.dim3.index(ds.demolist[0])
    yclastidx = ds.dim3.index(ds.demolist[-1])
    yc = X[:,:,yc1idx:yclastidx+1]
    return yc

def transformX(ds, theta, X): # transform dlim column to E[\int_0^1 v(.)] as specified (q column untouched b/c we will never use it)
    """
        Return the product characteristics, stored in X
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        X : ndarray
            (M,J) array of market-product characteristics

    Returns
    -------
        p : ndarray
            (M,J) array of prices
        expected_data_util : ndarray
            (M,J,I) array of expected utility from data consumption
        v : ndarray
            (M,J) array of unlimited voice dummies
        O : ndarray
            (M,J) array of Orange dummies
    """
    
    pidx = ds.chars.index(ds.pname)
    p = X[:,:,pidx]
    qidx = ds.chars.index(ds.qname)
    Q = X[:,:,qidx]
    dlimidx = ds.chars.index(ds.dlimname)
    dlim = X[:,:,dlimidx]
    expected_data_util = de.E_u(ds, theta, X, Q, dlim, ycX(ds, theta, X))
    vidx = ds.chars.index(ds.vunlimitedname)
    v = X[:,:,vidx]
    Oidx = ds.chars.index(ds.Oname)
    O = X[:,:,Oidx]
    return p, expected_data_util, v, O
            
def deltatilde(theta, ds, X, nonOxis):
    """
        Return grouped mean utilities based on paper's grouped products BLP extension
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        X : ndarray
            (M,J) array of market-product characteristics
        nonOxis : None or dict
            None (in case of solving for all) or dictionary telling which firms to impute and what xi to impute for them

    Returns
    -------
        delta : ndarray
            (F~,) array of grouped mean utilities
    """
    
    idxs = use_f_idxs(ds, X, nonOxis)
    log_shares = np.log(ds.fshares(X)[idxs])
    p, E_u, v, O = transformX(ds, theta, X)
    yc = ycX(ds, theta, X)
    mutilde_ifm = mutilde(ds, theta, X, p, E_u, v, O, yc)
    relevantmarkets = ds.relevant_markets(X)[:,idxs,np.newaxis]
    mktweights = ds.marketweights(X)[:,idxs,np.newaxis]
    norm = normalized_deltas(ds, theta, X, nonOxis, v, O, mutilde_ifm, yc)[0]
    mutilde_ifm = mutilde_ifm[:,idxs,:] # only want the ones for which searching for deltas
    contraction = lambda x: contractionmap(x, mutilde_ifm, norm, coef.theta_sigma(ds, theta, yc), relevantmarkets, mktweights, yc, log_shares)
    max_evaluations_squarem = 500
    max_evaluations_simple = 1000
    if coef.theta_sigma(ds, theta, yc) > 0.8:
        max_evaluations_squarem = 2000
        max_evaluations_simple = 2000
    initdeltas = np.zeros(np.sum(idxs))
    delta, success = iteration.squarem_iterator(initdeltas, contraction, max_evaluations_squarem, iteration.linf_norm, iteration.safe_norm, 1e-12, 1, 0.9, 1.1, 3.0)
    if success:
        return delta
    else:
        print('SQUAREM failed to converge. Now trying simple iteration.')
        delta, success = iteration.simple_iterator(delta, contraction, max_evaluations_simple, iteration.linf_norm, iteration.safe_norm, 1e-12)
        if success:
            return delta
        else:
            print('Failed to converge.')
            raise ValueError('deltatilde failed to converge.')

def xi(ds, theta, X, nonOxis):
    """
        Return xis
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        X : ndarray
            (M,J) array of market-product characteristics
        nonOxis : None or dict
            None (in case of solving for all) or dictionary telling which firms to impute and what xi to impute for them

    Returns
    -------
        xis : ndarray
            (M,J) array of xis
    """
    
    deltatildes_wonormalizeddeltas = deltatilde(theta, ds, X, nonOxis)
    p, E_u, v, O = transformX(ds, theta, X)
    yc = ycX(ds, theta, X)
    normdelta = normalized_deltas(ds, theta, X, nonOxis, v, O, mutilde(ds, theta, X, p, E_u, v, O, yc), yc)[1]
    # put the deltas together
    deltatildes = np.concatenate((deltatildes_wonormalizeddeltas, normdelta))
    deltatildes = deltatildes[ds.ftj(X).astype(int)]
    theta_v = theta[coef.v]
    theta_O = theta[coef.O]
    sigma = coef.theta_sigma(ds, theta, yc)
    xis = deltatildes * (1. - sigma) - theta_v * ds.Xbar(v[:,:,np.newaxis])[:,:,0] - theta_O * ds.Xbar(O[:,:,np.newaxis])[:,:,0]
    return xis

def s_mji(ds, theta, X, xis):
    """
        Return consumer type-specific market shares
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        X : ndarray
            (M,J) array of market-product characteristics
        xis : ndarray
            (M,J) array of xis

    Returns
    -------
        s_ijm : ndarray
            (M,J,I) array of consumer type-specific market shares
    """
    
    p, E_u, v, O = transformX(ds, theta, X)
    theta_v = theta[coef.v]
    theta_O = theta[coef.O]
    yc = ycX(ds, theta, X)
    sigma = coef.theta_sigma(ds, theta, yc)
    deltas = theta_v * v + theta_O * O + xis
    mus = mu_ijm(ds, theta, p, E_u, yc)
    deltamu = deltas[:,:,np.newaxis] + mus
    I_g = (1. - sigma) * special.logsumexp(deltamu / (1. - sigma), axis=1, keepdims=True) # this is (1-sigma) * log(sum exp((delta + mu) / (1-sigma)))
    I = np.logaddexp(0., I_g)
    s_ijm = np.exp(deltamu / (1. - sigma) + (sigma / (sigma - 1.)) * I_g - I) # see Grigolon and Verboven (2015) Review of Economics and Statistics
    return s_ijm

def s_mj(ds, theta, X, xis):
    """
        Return market shares
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        X : ndarray
            (M,J) array of market-product characteristics
        xis : ndarray
            (M,J) array of xis

    Returns
    -------
        s_jm : ndarray
            (M,J) array of market shares
    """
    
    s_ijm = s_mji(ds, theta, X, xis)
    typeweights = np.ones(s_ijm.shape[2]) / float(s_ijm.shape[2]) # b/c uniformly distributed types
    s_jm = np.matmul(s_ijm, typeweights) # evaluating integral, returns MxJ
    return s_jm

def elast_constraint(ds, theta, X, xis=None, nonOxis=None):
    """
        Return the price elasticity with respect a proportional increase in price of all Orange products
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        X : ndarray
            (M,J) array of market-product characteristics
        xis : ndarray
            (M,J) array of xis
        nonOxis : None or dict
            None (in case of solving for all) or dictionary telling which firms to impute and what xi to impute for them

    Returns
    -------
        elast : ndarray
            (M,) array of Orange price elasticities
    """
    
    if xis is None:
        xis = xi(ds, theta, X, nonOxis)
    shares = s_mj(ds, theta, X, xis)
    shares_O = np.sum(shares[:,ds.Oproducts], axis=1, keepdims=True)
    Delta = np.ones(len(ds.dim3))
    pidx = ds.chars.index(ds.pname)
    Delta[pidx] = 1.01 # direct array assignment fine b/c theta doesn't enter
    Delta = np.tile(Delta, (ds.J_O,1))
    Delta = np.vstack((Delta, np.tile(np.ones(len(ds.dim3)), (ds.J - ds.J_O, 1))))
    shares_new = s_mj(ds, theta, X * Delta[np.newaxis,:,:], xis)
    shares_new_O = np.sum(shares_new[:,ds.Oproducts], axis=1, keepdims=True)
    elast = (shares_new_O - shares_O) / (0.01 * shares_O)
    return elast

def div_ratio_numdenom(ds, theta, X, xis=None, nonOxis=None):
    """
        Return the diversion ratio with respect a proportional increase in price of all Orange products
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about markets
        theta : ndarray
            (K,) array of demand parameters
        X : ndarray
            (M,J) array of market-product characteristics
        xis : ndarray
            (M,J) array of xis
        nonOxis : None or dict
            None (in case of solving for all) or dictionary telling which firms to impute and what xi to impute for them

    Returns
    -------
        div : ndarray
            (M,) array of diversion ratios
    """
    
    if xis is None:
        xis = xi(ds, theta, X, nonOxis)
        
    # Original shares
    shares = s_mj(ds, theta, X, xis)
    shares_Org = np.sum(shares[:,ds.Oproducts], axis=1, keepdims=True)
    shares_0 = 1.0 - np.sum(shares, axis=1, keepdims=True)
    
    # Determine \partial q_k / \partial P_ORG
    sum_dq0_dpORG = 0.0
    sum_dqORG_dpORG = 0.0
    for j in range(ds.J_O): # this works since Orange products are ordered first
        Delta = np.zeros(len(ds.dim3))
        pidx = ds.chars.index(ds.pname)
        p_eps = 0.01 # size of our numerical differentiation
        Delta[pidx] = p_eps # direct array assignment fine b/c theta doesn't enter
        Delta = np.tile(Delta, (ds.J_O,1)) * (np.arange(ds.J_O) == j)[:,np.newaxis] # only j gets the additional p_eps
        Delta = np.vstack((Delta, np.tile(np.zeros(len(ds.dim3)), (ds.J - ds.J_O, 1))))
        shares_new = s_mj(ds, theta, X + Delta[np.newaxis,:,:], xis)
        shares_new_Org = np.sum(shares_new[:,ds.Oproducts], axis=1, keepdims=True)
        shares_new_0 = 1.0 - np.sum(shares_new, axis=1, keepdims=True)
        p_j = X[0,j,pidx]
        sum_dq0_dpORG = sum_dq0_dpORG + (shares_new_0 - shares_0) / p_eps * p_j
        sum_dqORG_dpORG = sum_dqORG_dpORG + (shares_new_Org - shares_Org) / p_eps * p_j
    
    return sum_dq0_dpORG, sum_dqORG_dpORG

def div_ratio(ds, theta, X, xis=None, nonOxis=None):
    sum_dq0_dpORG, sum_dqORG_dpORG = div_ratio_numdenom(ds, theta, X, xis=xis, nonOxis=nonOxis)
     
    # Determine diversion ratio
    div = -sum_dq0_dpORG / sum_dqORG_dpORG
    
    return div
