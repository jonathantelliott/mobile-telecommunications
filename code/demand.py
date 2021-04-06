#*******************************************************
#* Demand Estimation                                   *
#*******************************************************

#----------------------------------------------------------------
#   Import
#----------------------------------------------------------------

import autograd.numpy as np
#import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sys

import paths

import demand.demandsystem as demsys
import demand.estimation as est
import demand.moments as moments
import demand.variancematrix as vm
import demand.dataexpressions as de
import demand.blpextension as blp
import demand.coefficients as coef

#----------------------------------------------------------------
#   Process Data
#----------------------------------------------------------------
        
df = pd.read_csv(f"{paths.data_path}demand_estimation_data_new.csv")
df_agg = pd.read_csv(f"{paths.data_path}agg_data.csv")

prodfirms = df[['j_new','opcode']].values
products, prod_idx = np.unique(prodfirms[:,0], return_index=True)
print(products)
firms = prodfirms[prod_idx,1]
products = products.astype(int)
firms = firms.astype(int)

# Aggregate shares
npagg = np.array(df_agg[df_agg['month'] == 24])[0][1:]

# Adjust aggregate shares so Orange aggregate == weighted Orange market shares
s_O = np.sum(df[(df['month']==24) & (df['j_new']<=5)]['customer']) / np.sum(df[(df['month']==24) & (df['j']==1)]['msize'])
npagg_orig = np.copy(npagg)
Oinagg = 3
npagg[Oinagg] = s_O
notO = np.ones(npagg.shape[0], dtype=bool)
notO[Oinagg] = False
npagg[notO] = (1. - (s_O - npagg_orig[Oinagg]) / np.sum(npagg[notO])) * npagg[notO]

# Reconstruct aggregate shares b/c we are using a different indexing
npaggshare = np.zeros(len(npagg) - 2)
prepaidcontractinagg = 6
for i in range(len(npagg)):
    if i == Oinagg:
        npaggshare[0] = npagg[i]
    elif i < Oinagg:
        if i != 0 and i != prepaidcontractinagg:
            npaggshare[i] = npagg[i]
    elif i != prepaidcontractinagg:
        npaggshare[i - 1] = npagg[i]
s_notinclude = npagg[0] + npagg[-1]

share_of_outside_option = 0.1

# Set up demand system
chars = {'names': ['p', 'q_ookla', 'dlim', 'vunlimited','Orange'], 'norm': np.array([False, False, False, False, False])}
ds = demsys.DemandSystem(df, 
	['market', 'month', 'j_new'], 
	chars, 
	['spectreff3G2100', 'spectreff3G900', 'spectreff4G2600', 'spectreff4G800'], 
	['yc1', 'yc2', 'yc3', 'yc4', 'yc5', 'yc6', 'yc7', 'yc8', 'yc9'], 
	npaggshare, 
	s_notinclude, 
	products, 
	firms, 
	share_of_outside_option, 
	qname="q_ookla", 
	dbarname='dbar_new', 
	marketsharename='mktshare_new', 
	productname='j_new')


#----------------------------------------------------------------
#   Price Elasticity Imputation
#----------------------------------------------------------------

avg_price_elasts = np.array([-4., -2.5, -1.8])
#sigmas = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
sigmas = np.array([0., 0.2, 0.4, 0.6, 0.8, 0.9])
task_id = int(sys.argv[1])
elast_id = task_id // sigmas.shape[0]
nest_id = task_id % sigmas.shape[0]
avg_price_el = avg_price_elasts[elast_id]
sigma = sigmas[nest_id]

print('Average price elasticity: ' + str(avg_price_el))
print('Sigma: ' + str(sigma))


#----------------------------------------------------------------
#   Estimation
#----------------------------------------------------------------

K = moments.K

thetainit_p0 = -0.25 * (1. - nest_id / 6.) + -1.75 * (nest_id / 6.) - 0.25 * elast_id
thetainit_pz = -0.8 * (1. - nest_id / 6.) + -0.9 * (nest_id / 6.) - 0. * elast_id
thetainit_v = 1.7 * (1. - nest_id / 6.) + 0.35 * (nest_id / 6.) - 0.1 * elast_id
thetainit_O = 3.75 * (1. - nest_id / 6.) + 1.75 * (nest_id / 6.) - 0.25 * elast_id
thetainit_d0 = -1.3 * (1. - nest_id / 6.) + 0.5 * (nest_id / 6.) + 0.5 * elast_id
thetainit_dz = 0.3
thetainit_c = -6.3 * (1. - nest_id / 6.) + -8.0 * (nest_id / 6.) - 1.0 * elast_id
thetainit = np.array([thetainit_p0, thetainit_pz, thetainit_v, thetainit_O, thetainit_d0, thetainit_dz, thetainit_c])
Winit = np.identity(K)

thetahat, What = est.est(ds, thetainit, Winit, K, avg_price_el, sigma)


#------------------------------------------------------------------
#     Standard Errors
#------------------------------------------------------------------

N = ds.data.shape[0]
G_n = vm.G_n(ds, thetahat, ds.data, avg_price_el, sigma, N, K, 1.0e-7)
varmatrix_num = vm.V(G_n, What, np.linalg.inv(What))
print('Numerical variance matrix: ')
print(str(varmatrix_num))

stderrs_num = np.sqrt(np.diag(varmatrix_num) / float(N))
print('Numerical standard errors: ')
print(str(stderrs_num))


#------------------------------------------------------------------
#     Export
#------------------------------------------------------------------
np.save(paths.arrays_path + "thetahat_e" + str(elast_id) + "_n" + str(nest_id), thetahat)
np.save(paths.arrays_path + "What_e" + str(elast_id) + "_n" + str(nest_id), What)
np.save(paths.arrays_path + "Gn_e" + str(elast_id) + "_n" + str(nest_id), G_n)
np.save(paths.arrays_path + "stderrs_e" + str(elast_id) + "_n" + str(nest_id), stderrs_num)

df_est = pd.DataFrame(thetahat, index=['p0','pz','v','O','d0','dz','c'], columns=['estimate'])
df_est['std_errs'] = stderrs_num
df_est.to_csv(paths.res_path + "res_e" + str(elast_id) + "_n" + str(nest_id) + '.csv',index=True)

numdraws = 50
chol_decomp = np.linalg.cholesky(varmatrix_num / float(N))
nu_s = np.random.normal(size=(thetahat.shape[0],numdraws))
Nu_s = np.hstack((nu_s, -nu_s))
theta_s = thetahat[:,np.newaxis] + np.matmul(chol_decomp, Nu_s)
df_draws = pd.DataFrame(np.transpose(theta_s), columns=['p0','pz','v','O','d0','dz','c'])
df_draws.to_csv(paths.asym_draws_path + 'asym_e' + str(elast_id) + "_n" + str(nest_id) + '.csv', index=False)

theta = np.concatenate((thetahat, np.array([sigma])))
X = ds.data
xis = blp.xi(ds, theta, X, None)
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
actual_dbar = X[:,ds.Oproducts,dbaridx]
np.save(paths.dbar_path + 'predicted_e' + str(elast_id) + '_n' + str(nest_id) + '.npy', predicted_dbar)
np.save(paths.dbar_path + 'actual_e' + str(elast_id) + '_n' + str(nest_id) + '.npy', actual_dbar)
fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)
title = ['$\\bar{x} = 1000$', '$\\bar{x} = 4000$', '$\\bar{x} = 8000$']
for j in range(3):
    col = j % 3
    axs[col].scatter(actual_dbar[:,j+2], predicted_dbar[:,j+2] * 1000, alpha=0.6)
    axs[col].plot(np.arange(0,6000,50), np.arange(0,6000,50))
    axs[col].set_title(title[j])
    axs[col].set_xlabel("actual (MB)")
    axs[col].set_ylabel("predicted (MB)")
fig.tight_layout()
plt.savefig(graphs_path + "predict_vs_actual_dbar_e" + str(elast_id) + "_n" + str(nest_id) + ".pdf", bbox_inches="tight")
