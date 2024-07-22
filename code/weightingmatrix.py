import autograd.numpy as np
#import numpy as np

import demand.demandfunctions as blp

import gmm as gmm

def W(ds, thetahat, K, avg_price_el, div_ratio, firm_num_MVNO=5):
    N = ds.num_markets_moms
    opdm = np.zeros((N, K, K)) # outer product demeaned moments
    ghat = gmm.g_n(ds, thetahat, ds.data, gmm.g, avg_price_el, div_ratio)
    xis = blp.xi(ds, thetahat, ds.data, None)
    if firm_num_MVNO in ds.firms:
        nonOxis = xis[0,np.unique(ds.firms, return_index=True)[1][1:-1]]
    else:
        nonOxis = xis[0,np.unique(ds.firms, return_index=True)[1][1:]]
    for n in range(N):
        demeanedmoment = gmm.g(ds, thetahat, ds.data[ds.markets_moms,:,:][n,:,:][np.newaxis,:,:], avg_price_el, div_ratio, nonOxis={'firms': np.array([2,3,4]), 'xis': nonOxis}, all_markets=False) - ghat
        opdm[n,:,:] = np.outer(demeanedmoment, demeanedmoment)
    Shat = np.nanmean(opdm, axis=0)
    print('Conditioning number: ')
    print(str(np.linalg.cond(Shat)))
    What = np.linalg.inv(Shat)
    print('Weighting matrix: ')
    print(str(What))
    return What