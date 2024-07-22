import autograd.numpy as np
#import numpy as np

import demand.demandfunctions as blp
import demand.coefficients as coef

import moments as moments

def g(ds, theta, X, avg_price_el, div_ratio, nonOxis=None, all_markets=True):
    xis = blp.xi(ds, theta, X, nonOxis)
    # evaluate the set of moments and combine the moments into a MxK matrix (M observations, K moments)
    M = ds.num_markets_moms
    if not all_markets:
        M = X.shape[0]
    moms = np.zeros((M,0))
    for moment in moments.moments:
        mom = moment(ds, X, xis, theta, avg_price_el, div_ratio, all_markets=all_markets)
        moms = np.concatenate((moms, mom), axis=1)
    return moms

def g_n(ds, theta, X, g, avg_price_el, div_ratio):
    obs = g(ds, theta, X, avg_price_el, div_ratio)
    print('Moments: ' + str(np.mean(obs, axis=0)))
    return np.mean(obs, axis=0)

def objfct(ds, theta, X, W, g, avg_price_el, div_ratio):
    print('theta_pi: ' + str(np.mean(coef.theta_pi(ds, theta, blp.ycX(ds, theta, X)), axis=(0,1))))
    print('theta_di: ' + str(np.mean(coef.theta_di(ds, theta, blp.ycX(ds, theta, X)), axis=(0,1))))
    print('theta_c: ' + str(coef.theta_c(ds, theta, blp.ycX(ds, theta, X))))
    print('theta_sigma: ' + str(coef.theta_sigma(ds, theta, blp.ycX(ds, theta, X))))
    gn = g_n(ds, theta, X, g, avg_price_el, div_ratio)
    obj = 1.0 / 2.0 * np.matmul(np.matmul(gn.T, W), gn)
    return obj
