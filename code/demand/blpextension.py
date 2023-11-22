import autograd.numpy as np
import autograd.scipy.misc as misc
#import numpy as np
#import scipy.misc as misc

import demand.coefficients as coef
import demand.iteration as iteration
import demand.dataexpressions as de

def mu_ijm(ds, theta, p, E_u, yc):
    theta_p = coef.theta_pi(ds, theta, yc)
    return -theta_p * p[:,:,np.newaxis] + E_u

def mutilde(ds, theta, X, p, E_u, v, O, yc):
    theta_v = theta[coef.v]
    theta_O = theta[coef.O]
    sigma = coef.theta_sigma(ds, theta, yc)
    mutilde_ijm = (theta_v * (v[:,:,np.newaxis] - ds.Xbar(v[:,:,np.newaxis])) + theta_O * (O[:,:,np.newaxis] - ds.Xbar(O[:,:,np.newaxis])) + mu_ijm(ds, theta, p, E_u, yc)) / (1. - sigma)
    # go from MxJ matrix to MxF, summing
    mutilde_ifm = np.tile(mutilde_ijm[:,:ds.J_O], (1,X.shape[0],1)) * ds.relevant_markets(X)[:,:ds.J_O*X.shape[0],np.newaxis] # for J_O
    for f in np.unique(ds.firms[~ds.Oproducts]):
        mutilde_f = misc.logsumexp(mutilde_ijm[:,ds.firms==f,:], axis=1, keepdims=True)
        mutilde_ifm = np.concatenate((mutilde_ifm, mutilde_f), axis=1)
    return mutilde_ifm

def use_f_idxs(ds, X, nonOxis):
    idxs = np.ones(ds.Fnum(X), dtype=bool)
    if nonOxis is not None:
        argsortidxs = np.zeros(len(nonOxis['firms']))
        for i in range(len(nonOxis['firms'])):
            firmidx = ds.ftj(X)[0,np.argwhere(ds.firms == (nonOxis['firms'][i]))[0][0]].astype(int)
            argsortidxs[i] = firmidx
            idxs[firmidx] = False
    return idxs
    
def normalized_deltas(ds, theta, X, nonOxis, v, O, mutilde, yc):
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
            #nonOdelta = theta_v * np.mean(v[:,ds.firms==(nonOxis['firms'][i])]) + theta_O * np.mean(O[:,ds.firms==(nonOxis['firms'][i])]) + nonOxis['xis'][i]
            nonOdelta = (theta_v * np.mean(v[:,ds.firms==(nonOxis['firms'][i])]) + theta_O * np.mean(O[:,ds.firms==(nonOxis['firms'][i])]) + nonOxis['xis'][i]) / (1. - sigma)
            firmidx = ds.ftj(X)[0,np.argwhere(ds.firms == (nonOxis['firms'][i]))[0][0]].astype(int)
            normdelta = np.concatenate((normdelta, np.array([nonOdelta])))
            argsortidxs = np.concatenate((argsortidxs, np.array([firmidx])))
            if nonOdeltas.shape[0] == 0:
                nonOdeltas = (nonOdelta + mutilde[:,firmidx,:])[np.newaxis,:,np.newaxis,:]
            else:
                nonOdeltas = np.concatenate((nonOdeltas, (nonOdelta + mutilde[:,firmidx,:])[np.newaxis,:,np.newaxis,:]), axis=0)
            idxs[firmidx] = False
        normdelta = normdelta[np.argsort(argsortidxs)] # make sure that these deltas are sorted in the order needed for reconstructing delta array
        #norm = misc.logsumexp(nonOdeltas / (1. - sigma), axis=0) # this is an M x newaxis x I matrix
        norm = misc.logsumexp(nonOdeltas, axis=0) # this is an M x newaxis x I matrix
    return norm, normdelta

def sbar_f(deltatilde_f, mutilde_ifm, norm, sigma, relevantmarkets, mktweights, yc):
    deltamu = deltatilde_f[np.newaxis,:,np.newaxis] + mutilde_ifm
    # expdeltamu = np.exp(deltamu / (1. - sigma)) # not necessary unless first s_mfi definition is uncommented
    #I_g = (1. - sigma) * misc.logsumexp(deltamu / (1. - sigma), axis=1, keepdims=True, b=relevantmarkets) # this is (1-sigma) * log(sum_{everything} relevant_in_m * exp((delta + mu) / (1-sigma))), so we're not including irrelevant markets, and normalized values aren't here b/c there are none (if there are, will be taken care of in the overriding next statement)
    I_g = (1. - sigma) * misc.logsumexp(deltamu, axis=1, keepdims=True, b=relevantmarkets) # this is (1-sigma) * log(sum_{everything} relevant_in_m * exp((delta + mu) / (1-sigma))), so we're not including irrelevant markets, and normalized values aren't here b/c there are none (if there are, will be taken care of in the overriding next statement)
    if norm.shape[0] > 1:
        #I_g = (1. - sigma) * np.logaddexp(norm, misc.logsumexp(deltamu / (1. - sigma), axis=1, keepdims=True, b=relevantmarkets)) # for when we have norm, this is (1-sigma) * log(sum_{everything} relevant_in_m * exp((delta + mu) / (1-sigma)) + sum_{stuff in norm} exp((delta + mu) / (1-sigma))), so we're not including irrelevant markets, the normalized are always relevant
        I_g = (1. - sigma) * np.logaddexp(norm, misc.logsumexp(deltamu, axis=1, keepdims=True, b=relevantmarkets)) # for when we have norm, this is (1-sigma) * log(sum_{everything} relevant_in_m * exp((delta + mu) / (1-sigma)) + sum_{stuff in norm} exp((delta + mu) / (1-sigma))), so we're not including irrelevant markets, the normalized are always relevant
    I = np.logaddexp(0., I_g)
    #s_mfi = expdeltamu / np.exp(I_g / (1. - sigma)) * np.exp(I_g) / np.exp(I) # see Grigolon and Verboven (2015) Review of Economics and Statistics
    #s_mfi = np.exp(deltamu / (1. - sigma) + (sigma / (sigma - 1.)) * I_g - I) # see Grigolon and Verboven (2015) Review of Economics and Statistics, should be equivalent to the line above but gets rid of exp overflow problems
    s_mfi = np.exp(deltamu + (sigma / (sigma - 1.)) * I_g - I) # see Grigolon and Verboven (2015) Review of Economics and Statistics, should be equivalent to the line above but gets rid of exp overflow problems
    s_fi = np.sum(mktweights * s_mfi, axis=0) # average across markets, weighting by population, this is an F x I matrix
    s_f = np.matmul(s_fi, (np.ones(yc.shape[2]) / float(yc.shape[2]))) # we have quantiles, so the weights when taking the integral are all equal
    return s_f

def contractionmap(deltatilde_f, mutilde_ifm, norm, sigma, relevantmarkets, mktweights, yc, log_shares):
    #return deltatilde_f + (1. - sigma) * (log_shares - np.log(sbar_f(deltatilde_f, mutilde_ifm, norm, sigma, relevantmarkets, mktweights, yc))) # using MVNO is the last entry (last firm)
    return deltatilde_f + (log_shares - np.log(sbar_f(deltatilde_f, mutilde_ifm, norm, sigma, relevantmarkets, mktweights, yc))) # using MVNO is the last entry (last firm), (1 - sigma) dampening term not needed here, as shown in proof in appendix

def ycX(ds, theta, X):
    yc1idx = ds.dim3.index(ds.demolist[0])
    yclastidx = ds.dim3.index(ds.demolist[-1])
    return X[:,:,yc1idx:yclastidx+1]

def transformX(ds, theta, X): # transform dlim column to E[\int_0^1 v(.)] as specified (q column untouched b/c we will never use it)
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
        #print("Failed to converge.")
        #raise ValueError('deltatilde failed to converge.')
        print('SQUAREM failed to converge. Now trying simple iteration.')
        delta, success = iteration.simple_iterator(delta, contraction, max_evaluations_simple, iteration.linf_norm, iteration.safe_norm, 1e-12)
        if success:
            return delta
        else:
            print('Failed to converge.')
            raise ValueError('deltatilde failed to converge.')

def xi(ds, theta, X, nonOxis):
    deltatildes_wonormalizeddeltas = deltatilde(theta, ds, X, nonOxis)
    p, E_u, v, O = transformX(ds, theta, X)
    yc = ycX(ds, theta, X)
    normdelta = normalized_deltas(ds, theta, X, nonOxis, v, O, mutilde(ds, theta, X, p, E_u, v, O, yc), yc)[1]
    # put the deltas together - could use the fact that it's all the nonO's and MVNO is last
    deltatildes = np.concatenate((deltatildes_wonormalizeddeltas, normdelta))
    deltatildes = deltatildes[ds.ftj(X).astype(int)]
    theta_v = theta[coef.v]
    theta_O = theta[coef.O]
    sigma = coef.theta_sigma(ds, theta, yc)
    xis = deltatildes * (1. - sigma) - theta_v * ds.Xbar(v[:,:,np.newaxis])[:,:,0] - theta_O * ds.Xbar(O[:,:,np.newaxis])[:,:,0]
    return xis

def s_mji(ds, theta, X, xis):
    p, E_u, v, O = transformX(ds, theta, X)
    theta_v = theta[coef.v]
    theta_O = theta[coef.O]
    yc = ycX(ds, theta, X)
    sigma = coef.theta_sigma(ds, theta, yc)
    deltas = theta_v * v + theta_O * O + xis
    mus = mu_ijm(ds, theta, p, E_u, yc)
    deltamu = deltas[:,:,np.newaxis] + mus
    #expdeltamu = np.exp(deltamu / (1. - sigma)) # not necessary unless first s_ijm definition is uncommented
    I_g = (1. - sigma) * misc.logsumexp(deltamu / (1. - sigma), axis=1, keepdims=True) # this is (1-sigma) * log(sum exp((delta + mu) / (1-sigma)))
    I = np.logaddexp(0., I_g)
    #s_ijm = expdeltamu / np.exp(I_g / (1. - sigma)) * np.exp(I_g) / np.exp(I) # see Grigolon and Verboven (2015) Review of Economics and Statistics
    s_ijm = np.exp(deltamu / (1. - sigma) + (sigma / (sigma - 1.)) * I_g - I) # see Grigolon and Verboven (2015) Review of Economics and Statistics, should be the same as above, gets rid of overflow error
    return s_ijm

def s_mj(ds, theta, X, xis):
    s_ijm = s_mji(ds, theta, X, xis)
    typeweights = np.ones(s_ijm.shape[2]) / float(s_ijm.shape[2]) # b/c uniformly distributed types
    s_jm = np.matmul(s_ijm, typeweights) # evaluating integral, returns MxJ
    return s_jm

def elast_constraint(ds, theta, X, xis=None, nonOxis=None):
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
