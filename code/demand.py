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

import pickle

#----------------------------------------------------------------
#   Process Data
#----------------------------------------------------------------
        
df = pd.read_csv(f"{paths.data_path}demand_estimation_data.csv")
if not paths.include_ROF_in_moments: # if we aren't including "Rest of France" in the moments
    df['pdens_clean'] = np.nan_to_num(df['pdens_clean'].values, nan=0.001) # replace population density with a random non-zero number, doesn't end up enterring moments, so doesn't matter, but if they are 0 or NaN causes an issue with autograd gradient, resulting in all-NaN gradients
df_agg = pd.read_csv(f"{paths.data_path}agg_data.csv")

prodfirms = df[['j_new','opcode']].values
products, prod_idx = np.unique(prodfirms[:,0], return_index=True)
print(products)
firms = prodfirms[prod_idx,1]
products = products.astype(int)
firms = firms.astype(int)

# Aggregate shares
npagg = np.array(df_agg[df_agg['month'] == 24])[0][1:]
Oinagg = 1

# Adjust shares so weighted Orange market shares == Orange aggregate
org_mktshares = np.sum(np.reshape(df['mktshare_new'].values, (np.unique(df['market']).shape[0],np.unique(df['month']).shape[0],np.unique(df['j_new']).shape[0]))[:,-1,:5], axis=-1) # note: this relies on the dataset being sorted
msize = np.reshape(df['msize'].values, (np.unique(df['market']).shape[0],np.unique(df['month']).shape[0],np.unique(df['j_new']).shape[0]))[:,-1,0]
weighted_agg_org_mktshare = np.average(org_mktshares, weights=msize/np.sum(msize))
share_ratio = npagg[Oinagg] / weighted_agg_org_mktshare
df['mktshare_new_adjust'] = df['mktshare_new'] * share_ratio # adjust the Orange market shares proportionately so that they equal the aggregate number

# Reconstruct aggregate shares b/c we are using a different indexing and only some 
npaggshare = np.copy(npagg[1:]) # remove outside option
s_notinclude = npagg[0] # outside option, but DemandSystem puts it back using share_of_outside_option below (done this way due to old way we constructed the code - ultimately this doesn't adjust the outside options at all
share_of_outside_option = npagg[0]

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
                         popdensname='pdens_clean', 
                         qname="q_ookla", 
                         dbarname='dbar_new', 
                         marketsharename='mktshare_new_adjust', 
                         productname='j_new', 
                         include_ROF=paths.include_ROF_in_moments)


#----------------------------------------------------------------
#   Price Elasticity Imputation
#----------------------------------------------------------------

task_id = int(sys.argv[1])
elast_id = task_id // paths.sigmas.shape[0]
nest_id = task_id % paths.sigmas.shape[0]
avg_price_el = paths.avg_price_elasts[elast_id]
sigma = paths.sigmas[nest_id]

print('Average price elasticity: ' + str(avg_price_el), flush=True)
print('Sigma: ' + str(sigma), flush=True)

if task_id == 0: # only need to do this once
    with open(f"{paths.data_path}demandsystem.obj", "wb") as file_ds:
        pickle.dump(ds, file_ds)


#----------------------------------------------------------------
#   Estimation
#----------------------------------------------------------------

K = moments.K

thetainit_p0 = paths.thetainit_p0[elast_id,nest_id]
thetainit_pz = paths.thetainit_pz[elast_id,nest_id]
thetainit_v = paths.thetainit_v[elast_id,nest_id]
thetainit_O = paths.thetainit_O[elast_id,nest_id]
thetainit_d0 = paths.thetainit_d0[elast_id,nest_id]
thetainit_dz = paths.thetainit_dz[elast_id,nest_id]
thetainit_c = paths.thetainit_c[elast_id,nest_id]
thetainit = np.array([thetainit_p0, thetainit_pz, thetainit_v, thetainit_O, thetainit_d0, thetainit_dz, thetainit_c])
Winit = np.identity(K)

thetahat, What = est.est(ds, thetainit, Winit, K, avg_price_el, sigma)


#------------------------------------------------------------------
#     Standard Errors
#------------------------------------------------------------------

N = ds.num_markets_moms
G_n = vm.G_n(ds, thetahat, ds.data, avg_price_el, sigma, N, K, 1.0e-7)
varmatrix_num = vm.V(G_n, What, np.linalg.inv(What))
print('Numerical variance matrix: ', flush=True)
print(str(varmatrix_num), flush=True)

stderrs_num = np.sqrt(np.diag(varmatrix_num) / float(N))
print('Numerical standard errors: ', flush=True)
print(str(stderrs_num), flush=True)


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
    axs[col].set_title(title[j], fontsize=15.0)
    axs[col].set_xlabel("actual (MB)", fontsize=12.0)
    axs[col].set_ylabel("predicted (MB)", fontsize=12.0)
fig.tight_layout()
plt.savefig(paths.graphs_path + "predict_vs_actual_dbar_e" + str(elast_id) + "_n" + str(nest_id) + ".pdf", bbox_inches="tight", transparent=True)
