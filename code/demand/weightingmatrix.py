import autograd.numpy as np
#import numpy as np

import demand.blpextension as blp
import demand.gmm as gmm

def W(ds, thetahat, K, avg_price_el, sigma):
    N = ds.data.shape[0]
    opdm = np.zeros((N, K, K)) # outer product demeaned moments
    ghat = gmm.g_n(ds, thetahat, ds.data, gmm.g, avg_price_el, sigma)
    xis = blp.xi(ds, np.concatenate((thetahat, np.array([sigma]))), ds.data, None)
    nonOxis = xis[0,np.unique(ds.firms, return_index=True)[1][1:-1]]
    for n in range(N):
        demeanedmoment = gmm.g(ds, thetahat, ds.data[n,:,:][np.newaxis,:,:], avg_price_el, sigma, nonOxis={'firms': np.array([2,3,4]), 'xis': nonOxis}) - ghat
        opdm[n,:,:] = np.outer(demeanedmoment, demeanedmoment)
    Shat = np.nanmean(opdm, axis=0)
    print('Conditioning number: ')
    print(str(np.linalg.cond(Shat)))
    What = np.linalg.inv(Shat)
    print('Weighting matrix: ')
    print(str(What))
    return What