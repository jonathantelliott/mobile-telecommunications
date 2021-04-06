import autograd.numpy as np
#import numpy as np

import demand.dataexpressions as de
import demand.blpextension as blp
import demand.coefficients as coef

# Match price elasticity
def M_price_el_impute(ds, X, xis, theta, avg_price_el):
    return blp.elast_constraint(ds, theta, X, xis=xis) - avg_price_el

# Predicted data consumed = observed data consumed
def M_data_cons(ds, X, xis, theta, avg_price_el):
    # determine E[x*] and shares
    qidx = ds.chars.index(ds.qname)
    Q = X[:,:,qidx]
    dlimidx = ds.chars.index(ds.dlimname)
    dlim = X[:,:,dlimidx]
    Ex = de.E_x(ds, theta, X, Q, dlim, blp.ycX(ds, theta, X)) # M x J x I
    s_ijm = blp.s_mji(ds, theta, X, xis) # M x J x I
    
    # calculate weights from the shares of adoption of product j by i times weight of i
    num_i = s_ijm.shape[2]
    weights = s_ijm * (np.ones(num_i) / num_i)[np.newaxis,np.newaxis,:] # only works b/c quantiles, uniformly distributed
    predicted_dbar = np.sum(Ex * weights, axis=2) / np.sum(weights, axis=2) # weighted average across i
    
    # difference between predicted and recorded
    dbaridx = ds.dim3.index(ds.dbarname)
    mom = np.mean(predicted_dbar[:,ds.Oproducts] - (X[:,ds.Oproducts,dbaridx] / de.conv_factor), axis=1, keepdims=True)
    return mom

# Difference in predicted data consumed and observed data consumed shouldn't vary by quality
def M_data_cons_q(ds, X, xis, theta, avg_price_el):
    # determine E[x*] and shares
    qidx = ds.chars.index(ds.qname)
    Q = X[:,:,qidx]
    dlimidx = ds.chars.index(ds.dlimname)
    dlim = X[:,:,dlimidx]
    Ex = de.E_x(ds, theta, X, Q, dlim, blp.ycX(ds, theta, X)) # M x J x I
    s_ijm = blp.s_mji(ds, theta, X, xis) # M x J x I
    
    # calculate weights from the shares of adoption of product j by i times weight of i
    num_i = s_ijm.shape[2]
    weights = s_ijm * (np.ones(num_i) / num_i)[np.newaxis,np.newaxis,:] # only works b/c quantiles, uniformly distributed
    predicted_dbar = np.sum(Ex * weights, axis=2) / np.sum(weights, axis=2) # weighted average across i
    
    # difference between predicted and recorded
    dbaridx = ds.dim3.index(ds.dbarname)
    
    qidx = ds.chars.index(ds.qname)
    q_jm = X[:,ds.Oproducts,qidx]
    
    mom = np.mean((predicted_dbar[:,ds.Oproducts] - (X[:,ds.Oproducts,dbaridx] / de.conv_factor)) * (q_jm/10.), axis=1, keepdims=True)
    return mom

# Demand shock should be uncorrelated with population density (instrument for quality since quality endogenous but affected by pop dens)
def M_pop_dens(ds, X, xis, theta, avg_price_el):
    xis_O = xis[:,ds.Oproducts]
    popdensidx = ds.dim3.index(ds.popdensname)
    popdens_m = X[:,ds.Oproducts,popdensidx]
    mom = np.mean(xis_O * popdens_m / 1000., axis=1, keepdims=True)
    return mom

# Demand shock should be uncorrelated with data limit characteristic (helps to identify theta_x)
def M_dlim(ds, X, xis, theta, avg_price_el):
    xis_O = xis[:,ds.Oproducts]
    dlimidx = ds.chars.index(ds.dlimname)
    dlim_jm = X[:,ds.Oproducts,dlimidx]
    mom = np.mean(xis_O * dlim_jm / de.conv_factor, axis=1, keepdims=True)
    return mom

# Demand shock should be uncorrelated with unlimited voice dummy
def M_vlim(ds, X, xis, theta, avg_price_el):
    xis_O = xis[:,ds.Oproducts]
    vidx = ds.chars.index(ds.vunlimitedname)
    v_jm = X[:,ds.Oproducts,vidx]
    mom = np.mean(xis_O * v_jm, axis=1, keepdims=True)
    return mom

# Demand shock should be uncorrelated with Orange dummy
def M_O(ds, X, xis, theta, avg_price_el):
    xis_O = xis[:,ds.Oproducts]
    Oidx = ds.chars.index(ds.Oname)
    O_jm = X[:,ds.Oproducts,Oidx]
    mom = np.mean(xis_O * O_jm, axis=1, keepdims=True)
    return mom

# Demand shock should be uncorrelated with median income (helps to identify the price heterogeneity term)
def M_yc5(ds, X, xis, theta, avg_price_el):
    xis_O = xis[:,ds.Oproducts]
    yc2idx = ds.dim3.index(ds.demolist[1])
    yc5idx = ds.dim3.index(ds.demolist[4])
    yc8idx = ds.dim3.index(ds.demolist[7])
    yc2_jm = X[:,ds.Oproducts,yc2idx]
    yc5_jm = X[:,ds.Oproducts,yc5idx]
    yc8_jm = X[:,ds.Oproducts,yc8idx]
    #mom2 = np.mean(xis_O * yc2_jm / np.mean(yc2_jm), axis=1, keepdims=True)
    mom5 = np.mean(xis_O * yc5_jm / coef.income_conv_factor, axis=1, keepdims=True)
    #mom8 = np.mean(xis_O * yc8_jm / np.mean(yc8_jm), axis=1, keepdims=True)
    #return np.hstack((mom2,mom5,mom8))
    return mom5

# Difference between predicted and observed data consumption should be uncorrelated with median income (helps to identify the data heterogeneity term)
def M_data_cons_yc5(ds, X, xis, theta, avg_price_el):
    yc2idx = ds.dim3.index(ds.demolist[1])
    yc5idx = ds.dim3.index(ds.demolist[4])
    yc8idx = ds.dim3.index(ds.demolist[7])
    yc2_jm = X[:,ds.Oproducts,yc2idx]
    yc5_jm = X[:,ds.Oproducts,yc5idx]
    yc8_jm = X[:,ds.Oproducts,yc8idx]
    
    # determine E[x*] and shares
    qidx = ds.chars.index(ds.qname)
    Q = X[:,:,qidx]
    dlimidx = ds.chars.index(ds.dlimname)
    dlim = X[:,:,dlimidx]
    Ex = de.E_x(ds, theta, X, Q, dlim, blp.ycX(ds, theta, X)) # M x J x I
    s_ijm = blp.s_mji(ds, theta, X, xis) # M x J x I
    
    # calculate weights from the shares of adoption of product j by i times weight of i
    num_i = s_ijm.shape[2]
    weights = s_ijm * (np.ones(num_i) / num_i)[np.newaxis,np.newaxis,:] # only works b/c quantiles, uniformly distributed
    predicted_dbar = np.sum(Ex * weights, axis=2) / np.sum(weights, axis=2) # weighted average across i
    
    # difference between predicted and recorded
    dbaridx = ds.dim3.index(ds.dbarname)
    #mom2 = np.mean(yc2_jm / np.mean(yc2_jm) * ((predicted_dbar[:,ds.Oproducts] - (X[:,ds.Oproducts,dbaridx] / de.conv_factor)) / (X[:,ds.Oproducts,dbaridx] / de.conv_factor)), axis=1, keepdims=True)
    mom5 = np.mean(yc5_jm / coef.income_conv_factor * (predicted_dbar[:,ds.Oproducts] - (X[:,ds.Oproducts,dbaridx] / de.conv_factor)), axis=1, keepdims=True)
    #mom8 = np.mean(yc8_jm / np.mean(yc8_jm) * ((predicted_dbar[:,ds.Oproducts] - (X[:,ds.Oproducts,dbaridx] / de.conv_factor)) / (X[:,ds.Oproducts,dbaridx] / de.conv_factor)), axis=1, keepdims=True)
    #return np.hstack((mom2,mom5,mom8))
    return mom5
    return mom

# Demand shock should be uncorrelated with average download speed for other firms within that commune
def M_avg_q_other(ds, X, xis, theta, avg_price_el):
	xis_O = xis[:,ds.Oproducts]
	qidx = ds.chars.index(ds.qname)
	q_jm = np.mean(X[:,~ds.Oproducts,qidx], axis=2, keepdims=True)
	mom = np.mean(xis_O * q_jm, axis=1, keepdims=True)
	return mom


moments = (M_price_el_impute, M_data_cons, M_dlim, M_vlim, M_O, M_yc5, M_data_cons_yc5, M_pop_dens)
K = 8
