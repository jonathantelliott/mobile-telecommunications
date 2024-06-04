import autograd.numpy as np
#import numpy as np

import demand.blpextension as blp
import demand.gmm as gmm

def V(G, W, S):
    return np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(G.T, W), G)), G.T), W), S), W), G), np.linalg.inv(np.matmul(np.matmul(G.T, W), G)))

def G_n(ds, theta, X, avg_price_el, div_ratio, N, K, eps, firm_num_MVNO=5):
    g_jac = np.zeros((N, K, len(theta)))
    xis = blp.xi(ds, theta, X, None)
    nonOxis = xis[0,np.unique(ds.firms, return_index=True)[1][1:]]
    firm_nums = np.array([2,3,4,5])
    if firm_num_MVNO not in ds.firms:
        firm_nums = np.array([2,3,4])
    for n in range(N):
        g_0 = gmm.g(ds, theta, X[ds.markets_moms,:,:][n,:,:][np.newaxis,:,:], avg_price_el, div_ratio, nonOxis={'firms': firm_nums, 'xis': nonOxis}, all_markets=False)
        for i in range(len(theta)):
            epsilon = np.zeros(len(theta))
            epsilon[i] = eps
            grad = (gmm.g(ds, theta + epsilon, X[ds.markets_moms,:,:][n,:,:][np.newaxis,:,:], avg_price_el, div_ratio, nonOxis={'firms': firm_nums, 'xis': nonOxis}, all_markets=False) - g_0) / eps
            g_jac[n, :, i] = grad
    return np.nanmean(g_jac, axis=0)