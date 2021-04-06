# %%
#*******************************************************
#* Counterfacturals                                    *
#*******************************************************

#----------------------------------------------------------------
#   Import
#----------------------------------------------------------------

#import autograd.numpy as np
import numpy as np
import pandas as pd

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
import copy

import paths

import counterfactuals.infrastructurefunctions as infr
import counterfactuals.priceequilibrium as pe
import counterfactuals.infrastructureequilibrium as ie
import counterfactuals.transmissionequilibrium as te
import counterfactuals.welfare as welfare

import demand.demandsystem as demsys
import demand.variancematrix as vm
import demand.dataexpressions as de
import demand.blpextension as blp
import demand.coefficients as coef

# %%
# Determine imputations
avg_price_elasts = np.array([-4., -2.5, -1.8])
sigmas = np.array([0., 0.2, 0.4, 0.6, 0.8, 0.9])
task_id = int(sys.argv[1])
elast_id = task_id // sigmas.shape[0]
nest_id = task_id % sigmas.shape[0]
avg_price_el = avg_price_elasts[elast_id]
sigma = sigmas[nest_id]

print(f"Orange price elasticity: {avg_price_el}")
print(f"sigma: {sigma}")

# %%
# Import infrastructure / quality data
df_inf = pd.read_csv(f"{paths.data_path}infrastructure_clean.csv", engine="python") # engine helps encoding, error with commune names, but doesn't matter b/c not used
df_q = pd.read_csv(f"{paths.data_path}quality_ookla.csv")

mno_codes = {
    'Orange': 1, 
    'SFR': 2,
    'Free': 3, 
    'Bouygues': 4, 
    'MVNO': 5
}

# Commune characteristics
population = df_inf['population'].values
area = df_inf['area_effective'].values # adjusted commune area
pop_dens = df_inf['pdens_clean'].values 

# Infrastructure
stations = df_inf[[f"stations{i}" for i in range(1,5)]].values # number of base stations
radius = np.sqrt(area[:,np.newaxis] / stations / (np.sqrt(3.) * 3. / 2.)) # cell radius assuming homogeneous hexagonal cells, in km
stations = np.nan_to_num(stations) # replace NaNs with 0
bw_3g = df_inf[[f"bw3g{i}" for i in range(1,5)]].values # in MHz
bw_4g = df_inf[[f"bw4g{i}" for i in range(1,5)]].values # in MHz

# Orange download speed in Mbps
q_avg = df_q[[f"q_ookla{i}" for i in range(1,5)]].values # in Mbps

# Compute 4G-equivalent channel capacities, in Mbps
cc_3g = np.zeros(bw_3g.shape)
cc_4g = np.zeros(bw_4g.shape)
for i in range(bw_3g.shape[0]): # slow, but only need to do once
    for j in range(bw_3g.shape[1]):
        cc_3g[i,j] = infr.rho_C_hex(bw_3g[i,j], radius[i,j], 1.) if not np.isnan(bw_3g[i,j]) and not np.isnan(radius[i,j]) else np.nan # channel capacity in Mbps
        cc_4g[i,j] = infr.rho_C_hex(bw_4g[i,j], radius[i,j], 1.) if not np.isnan(bw_4g[i,j]) and not np.isnan(radius[i,j]) else np.nan # channel capacity in Mbps
cc_tot = np.nan_to_num(cc_3g) * 2.5 / 4.08 + np.nan_to_num(cc_4g) # aggregate to 4G equivalent based on spectral efficiencies from https://en.wikipedia.org/wiki/Spectral_efficiency 

# Calculate efficiency factor using Orange data demanded
df_q_Orange = pd.read_csv(f"{paths.data_path}quality_clean.csv")
q_dem_3g = df_q_Orange['bitps3'].values / 10.0e6 / stations[:,mno_codes['Orange']-1] # in Mbps / station
q_dem_4g = df_q_Orange['bitps4'].values / 10.0e6 / stations[:,mno_codes['Orange']-1] # in Mbps / station
q_dem_tot = np.nan_to_num(q_dem_3g) + np.nan_to_num(q_dem_4g)
lamda = (q_avg[:,mno_codes['Orange']-1] + q_dem_tot) / cc_tot[:,mno_codes['Orange']-1] # efficiency factor

# Convert bandwidth to 4G-equivalent
bw_4g_equiv = np.nan_to_num(bw_3g) * 2.5 / 4.08 + np.nan_to_num(bw_4g) # aggregate to 4G equivalent based on spectral efficiencies from https://en.wikipedia.org/wiki/Spectral_efficiency 

# Adjust channel capacities by spectral efficiency factor
cc_tot = lamda[:,np.newaxis] * cc_tot

# Identify the markets in which Free has 0 bandwidth or stations, not going to use these markets
free_0_bw = cc_tot[:,mno_codes['Free']-1] == 0.
free_0_stations = stations[:,mno_codes['Free']-1] == 0.
markets_org_free_share = free_0_bw | free_0_stations

# %%
# NOTE: What follows is copied from  ../estimation/main.py - can just call this code once
# Import demand data
df_demand = pd.read_csv(f"{paths.data_path}demand_estimation_data_new.csv")
df_agg = pd.read_csv(f"{paths.data_path}agg_data.csv")

# Process products and firms
prodfirms = df_demand[['j_new','opcode']].values
products, prod_idx = np.unique(prodfirms[:,0], return_index=True)
firms = prodfirms[prod_idx,1]
products = products.astype(int)
firms = firms.astype(int)

# Select month for aggregate shares
npagg = np.array(df_agg[df_agg['month'] == 24])[0][1:]

# Adjust aggregate shares so Orange aggregate == weighted Orange market shares
s_O = np.sum(df_demand[(df_demand['month']==24) & (df_demand['j_new']<=5)]['customer']) / np.sum(df_demand[(df_demand['month']==24) & (df_demand['j']==1)]['msize'])
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
ds = demsys.DemandSystem(df_demand, 
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

# Indices
pidx = ds.chars.index(ds.pname)
qidx = ds.chars.index(ds.qname)
dlimidx = ds.chars.index(ds.dlimname)
vlimidx = ds.chars.index(ds.vunlimitedname)
Oidx = ds.chars.index(ds.Oname)
yc1idx = ds.dim3.index(ds.demolist[0])
yclastidx = ds.dim3.index(ds.demolist[-1])

# Drop markets in which Free shares infrastructure with Orange
ds.data = ds.data[~markets_org_free_share,:,:]
population = population[~markets_org_free_share]
stations = stations[~markets_org_free_share,:]
cc_tot = cc_tot[~markets_org_free_share,:]
radius = radius[~markets_org_free_share,:]
bw_4g_equiv = bw_4g_equiv[~markets_org_free_share,:]
lamda = lamda[~markets_org_free_share]
area = area[~markets_org_free_share]

# Get array of product prices
prices = ds.data[0,:,pidx] # 0 b/c market doesn't matter

# %%
# Import demand estimation results
N = np.unique(df_demand['market']).shape[0]
thetahat = np.load(f"{paths.arrays_path}thetahat_e{elast_id}_n{nest_id}.npy")
G_n = np.load(f"{paths.arrays_path}Gn_e{elast_id}_n{nest_id}.npy")
What = np.load(f"{paths.arrays_path}What_e{elast_id}_n{nest_id}.npy")
Sigma = vm.V(G_n, What, np.linalg.inv(What))

# %%
# Determine the size of epsilon to numerically approximate the gradient
compute_std_errs = True
eps_grad = 0.01
thetas_to_compute = np.vstack((thetahat[np.newaxis,:], thetahat[np.newaxis,:] + np.identity(thetahat.shape[0]) * eps_grad, thetahat[np.newaxis,:] - np.identity(thetahat.shape[0]) * eps_grad))

# Equilibria parameters
num_firms_to_simulate = 6
dlims = np.array([2000., 10000.])
vlims = np.array([1., 1.])
num_prods = dlims.shape[0]
num_firms_array = np.arange(num_firms_to_simulate, dtype=int) + 1
rep_market_size = 16.299135

# Welfare options
include_logit_shock = False
include_pop = False

def compute_equilibria(theta_n, p_starting_vals=None, R_starting_vals=None):
    """Compute the equilibria for a particular draw from the demand parameter's asymptotic distribution."""
    # Construct demand parameter
    theta_sigma = np.concatenate((thetas_to_compute[theta_n,:], np.array([sigma])))
    
    # Recover xis
    xis = blp.xi(ds, theta_sigma, ds.data, None)

    # %%
    # Estimate per-user costs

    # Calculate shares and Jacobian of shares with respect to prices
    shares = np.sum(blp.s_mj(ds, theta_sigma, ds.data, xis) * population[:,np.newaxis], axis=0)
    impute_MVNO = {
        'impute': True, 
        'firms_share': np.array([True, True, False, True]), # all firms share with MVNO, except Free
        'include': True
    }
    Jac = pe.s_jacobian_p(prices, cc_tot, ds, xis, theta_sigma, stations, population, impute_MVNO=impute_MVNO, q_0=None, eps=0.01)

    # Determine per-user costs
    c_u = np.zeros(ds.J)
    for f, firm in enumerate(np.unique(ds.firms)):
        firm_cond = ds.firms == firm
        inv_Jac_firm = np.linalg.inv(Jac[np.ix_(firm_cond, firm_cond)])
        shares_firm = shares[firm_cond]
        c_u[firm_cond] = prices[firm_cond] + np.matmul(inv_Jac_firm, shares_firm)

    # %%
    # Estimate per-base station costs

    # Calculate shares and Jacobian of shares with respect to prices
    impute_MVNO = {
        'impute': True, 
        'firms_share': np.array([True, True, False, True]), # all firms share with MVNO, except Free
        'include': True
    }
    MR = ie.pi_deriv_R(radius, bw_4g_equiv, lamda, ds, xis, theta_sigma, population, area, c_u, impute_MVNO=impute_MVNO, q_0=None, eps=0.01)

    # Determine per-base station costs
    c_R = MR / infr.num_stations_deriv(radius, area[:,np.newaxis])

    # %%
    # Compute counterfactual equilibrium

    per_user_costs = np.array([np.mean(c_u[(ds.data[0,:,dlimidx] >= 1000) & (ds.data[0,:,dlimidx] < 5000)]), np.mean(c_u[ds.data[0,:,dlimidx] >= 5000])])
    per_base_station_cost = np.mean(c_R[:,np.arange(4) != 2.])

    pop_cntrfctl = np.ones((1,)) * np.median(population)
    market_size_cntrfctl = np.ones((1,)) * np.median(area)
    market_bw = np.median(np.sum(bw_4g_equiv, axis=1))
    gamma_cntrfctl = np.ones((1,)) * np.median(lamda)

    p_stars = np.zeros((num_firms_array.shape[0], num_prods))
    R_stars = np.zeros(num_firms_array.shape)
    num_stations_stars = np.zeros(num_firms_array.shape)
    q_stars = np.zeros(num_firms_array.shape)
    cs = np.zeros(num_firms_array.shape)
    cs_by_type = np.zeros((num_firms_array.shape[0], yclastidx - yc1idx + 1))
    ps = np.zeros(num_firms_array.shape)
    ts = np.zeros(num_firms_array.shape)
    ccs = np.zeros(num_firms_array.shape)
    full_elasts = np.zeros((num_firms_array.shape[0], num_prods))
    partial_elasts = np.zeros((num_firms_array.shape[0], num_prods))

    partial_Pif_partial_bf = np.zeros(num_firms_array.shape)
    partial_Pif_partial_b = np.zeros(num_firms_array.shape)
    partial_CS_partial_b = np.zeros(num_firms_array.shape)
    
    # Determine the starting guesses for each
    if p_starting_vals is None:
        p_0s = np.zeros(p_stars.shape)
        p_0s[0,:] = 1.0 * per_user_costs
    else:
        p_0s = np.copy(p_starting_vals)
    if R_starting_vals is None:
        R_0s = np.zeros(R_stars.shape)[:,np.newaxis]
        R_0s[0,:] = 0.5
    else:
        R_0s = np.copy(R_starting_vals)

    for i, num_firms in enumerate(num_firms_array):
        print(f"num firms: {num_firms}")

        # Create ds with properties I want
        ds_cntrfctl = copy.deepcopy(ds)
        ds_cntrfctl.data = np.tile(ds.data[0,0,:][np.newaxis,np.newaxis,:], (1,num_firms * num_prods,1))
        ds_cntrfctl.data[:,:,pidx] = np.zeros(num_firms * num_prods) # doesn't matter
        ds_cntrfctl.data[:,:,qidx] = np.tile(np.ones(num_prods)[np.newaxis,:], (1,num_firms)) # doesn't matter
        ds_cntrfctl.data[:,:,dlimidx] = np.tile(dlims[np.newaxis,:], (1,num_firms))
        ds_cntrfctl.data[:,:,vlimidx] = np.tile(vlims[np.newaxis,:], (1,num_firms))
        ds_cntrfctl.data[:,:,Oidx] = 0.
        ds_cntrfctl.firms = np.repeat(np.arange(num_firms, dtype=int) + 1, num_prods)
        ds_cntrfctl.J = num_firms * num_prods

        # Create income distribution with properties I want
        ds_cntrfctl.data[:,:,yc1idx:yclastidx+1] = np.median(ds.data[:,0,yc1idx:yclastidx+1], axis=0)[np.newaxis,np.newaxis,:]

        # Create cost arrays
        c_u_cntrfctl = per_user_costs
        c_R_cntrfctl = np.ones((1,1)) * per_base_station_cost

        # Create market variables with properties I want
        bw_cntrfctl = np.ones((1,1)) * market_bw / float(num_firms)
        xis_cntrfctl = np.ones((1,num_prods)) * theta_sigma[coef.O]

        # Set starting values (if None, num_firms=1 can cause problems for convergence)
        R_0 = R_0s[i,:][:,np.newaxis]
        p_0 = p_0s[i,:]

        R_star, p_star, q_star = ie.infrastructure_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta_sigma, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, symmetric=True, print_msg=True, impute_MVNO={'impute': False}, q_0=None, eps_R=0.01, eps_p=0.01, factor=100.)

        p_stars[i,:] = p_star[:num_prods]
        R_stars[i] = R_star[0,0]
        num_stations_stars[i] = num_firms * infr.num_stations(R_stars[i], rep_market_size)
        q_stars[i] = q_star[0,0]

        # Update starting values for next time
        if i + 1 < num_firms_to_simulate:
            if p_starting_vals is None:
                p_0s[i+1,:] = 0.33 * p_stars[i,:] + 0.67 * per_user_costs
            if R_starting_vals is None:
                R_0s[i+1] = 0.9 * R_stars[i]

        # Update Demand System
        ds_cntrfctl.data[:,:,pidx] = np.copy(p_star)
        ds_cntrfctl.data[:,:,qidx] = np.tile(q_star, (num_prods,))

        # Calculate welfare impact
        cs_by_type[i,:] = welfare.consumer_surplus(ds_cntrfctl, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, include_logit_shock=include_logit_shock)
        cs[i] = welfare.agg_consumer_surplus(ds_cntrfctl, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, include_logit_shock=include_logit_shock, include_pop=include_pop)
        ps[i] = welfare.producer_surplus(ds_cntrfctl, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_pop=include_pop)
        ts[i] = welfare.total_surplus(ds_cntrfctl, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_logit_shock=include_logit_shock, include_pop=include_pop)

        # Calculate elasticities
        cc_cntrfctl = np.zeros((R_star.shape[0], 1))
        for m in range(R_star.shape[0]):
            cc_cntrfctl[m,0] = infr.rho_C_hex(bw_cntrfctl[m,0], R_star[m,0], gamma_cntrfctl[m])
        ccs[i] = cc_cntrfctl[0,0]
        num_stations_cntrfctl = infr.num_stations(np.array([[R_stars[i]]]), market_size_cntrfctl)
        full_elasts[i,:] = pe.price_elast(np.copy(p_stars[i,:]), cc_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta_sigma, num_stations_cntrfctl, pop_cntrfctl, symmetric=True, impute_MVNO={'impute': False}, q_0=None, eps=0.01, full=True)
        partial_elasts[i,:] = pe.price_elast(np.copy(p_stars[i,:]), cc_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta_sigma, num_stations_cntrfctl, pop_cntrfctl, symmetric=True, impute_MVNO={'impute': False}, q_0=None, eps=0.01, full=False)

        # Calculate bandwidth derivatives
        Pif_bf, Pif_b, CS_b = ie.bw_foc(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta_sigma, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, np.array([[R_stars[i]]]), p_star[:num_prods], symmetric=True, print_msg=False, impute_MVNO={'impute': False}, q_0=None, eps_R=0.01, eps_p=0.01, eps_bw=0.01, factor=100., include_logit_shock=include_logit_shock)
        partial_Pif_partial_bf[i] = Pif_bf[0,0]
        partial_Pif_partial_b[i] = Pif_b[0,0]
        partial_CS_partial_b[i] = CS_b[0]

        print("\n\n\n\n\n\n")
        
    # Calculate welfare relative to monopoly case
    cs = cs - cs[0]
    ps = ps - ps[0]
    ts = ts - ts[0]
    cs_by_type = cs_by_type - cs_by_type[0,:][np.newaxis,:]
        
    return p_stars, R_stars, num_stations_stars, q_stars, cs_by_type, cs, ps, ts, ccs, full_elasts, partial_elasts, partial_Pif_partial_bf, partial_Pif_partial_b, partial_CS_partial_b, c_u, c_R
    
# %%
# Compute the equilibria and perturbations

# Initialize variables
theta_N = thetas_to_compute.shape[0] if compute_std_errs else 1
p_stars = np.zeros((theta_N, num_firms_array.shape[0], num_prods))
R_stars = np.zeros((theta_N, num_firms_array.shape[0]))
num_stations_stars = np.zeros((theta_N, num_firms_array.shape[0]))
q_stars = np.zeros((theta_N, num_firms_array.shape[0]))
cs_by_type = np.zeros((theta_N, num_firms_array.shape[0], yclastidx - yc1idx + 1))
cs = np.zeros((theta_N, num_firms_array.shape[0]))
ps = np.zeros((theta_N, num_firms_array.shape[0]))
ts = np.zeros((theta_N, num_firms_array.shape[0]))
ccs = np.zeros((theta_N, num_firms_array.shape[0]))
full_elasts = np.zeros((theta_N, num_firms_array.shape[0], num_prods))
partial_elasts = np.zeros((theta_N, num_firms_array.shape[0], num_prods))
partial_Pif_partial_bf = np.zeros((theta_N, num_firms_array.shape[0]))
partial_Pif_partial_b = np.zeros((theta_N, num_firms_array.shape[0]))
partial_CS_partial_b = np.zeros((theta_N, num_firms_array.shape[0]))
c_u = np.zeros((theta_N, ds.J))
c_R = np.zeros((theta_N, radius.shape[0], radius.shape[1]))

# Compute point demand parameter estimate equilibria and store relevant variables
res = compute_equilibria(0)
p_stars[0,:,:] = res[0]
R_stars[0,:] = res[1]
num_stations_stars[0,:] = res[2]
q_stars[0,:] = res[3]
cs_by_type[0,:,:] = res[4]
cs[0,:] = res[5]
ps[0,:] = res[6]
ts[0,:] = res[7]
ccs[0,:] = res[8]
full_elasts[0,:,:] = res[9]
partial_elasts[0,:,:] = res[10]
partial_Pif_partial_bf[0,:] = res[11]
partial_Pif_partial_b[0,:] = res[12]
partial_CS_partial_b[0,:] = res[13]
c_u[0,:] = res[14]
c_R[0,:,:] = res[15]

def compute_equilibria_adjust(theta_n):
    """Provide starting values for computing equilibria based on results from the point estimate."""
    
    p_starting_vals = None
    R_starting_vals = None
    if elast_id == 0: # this elasticity has the most trouble computing equilibria when perturbed, so supply it the p*s and R*s in the point estimate
        p_starting_vals = np.copy(p_stars[0,:,:])
        R_starting_vals = np.copy(R_stars[0,:])[:,np.newaxis]
        
    return compute_equilibria(theta_n, p_starting_vals=p_starting_vals, R_starting_vals=R_starting_vals)

# Compute perturbed demand parameter estimate equilibria and store relevant variables
for idx in range(1, theta_N):
    res = compute_equilibria_adjust(idx)
    p_stars[idx,:,:] = res[0]
    R_stars[idx,:] = res[1]
    num_stations_stars[idx,:] = res[2]
    q_stars[idx,:] = res[3]
    cs_by_type[idx,:,:] = res[4]
    cs[idx,:] = res[5]
    ps[idx,:] = res[6]
    ts[idx,:] = res[7]
    ccs[idx,:] = res[8]
    full_elasts[idx,:,:] = res[9]
    partial_elasts[idx,:,:] = res[10]
    partial_Pif_partial_bf[idx,:] = res[11]
    partial_Pif_partial_b[idx,:] = res[12]
    partial_CS_partial_b[idx,:] = res[13]
    c_u[idx,:] = res[14]
    c_R[idx,:,:] = res[15]

# %%
# Determine point estimates and standard errors

def asym_distribution(var):
    """Determine the point estimate and standard errors given demand parameter using the Delta Method."""
    
    # Determine the point estimate
    hB = var[0,...]
    
    if compute_std_errs:
        # Determine the gradient
        grad_hB = (var[1:1 + thetahat.shape[0],...] - var[1 + thetahat.shape[0]:1 + 2 * thetahat.shape[0],...]) / (2. * eps_grad)
        grad_hB = np.moveaxis(grad_hB, 0, -1) # move the gradient axis to the end for matrix operations

        # Calculate (approximate) standard errors
        hB_normal_asymvar = (grad_hB[...,np.newaxis,:] @ Sigma @ grad_hB[...,np.newaxis])[...,0,0] # multivariate Delta Method
        hB_se = np.sqrt(hB_normal_asymvar / float(N)) # determine the standard errors

        return hB, hB_se
    
    else:
        return hB, 0
    
p_stars, p_stars_se = asym_distribution(p_stars)
R_stars, R_stars_se = asym_distribution(R_stars)
num_stations_stars, num_stations_stars_se = asym_distribution(num_stations_stars)
q_stars, q_stars_se = asym_distribution(q_stars)
cs_by_type, cs_by_type_se = asym_distribution(cs_by_type)
cs, cs_se = asym_distribution(cs)
ps, ps_se = asym_distribution(ps)
ts, ts_se = asym_distribution(ts)
ccs, ccs_se = asym_distribution(ccs)
full_elasts, full_elasts_se = asym_distribution(full_elasts)
partial_elasts, partial_elasts_se = asym_distribution(partial_elasts)
partial_Pif_partial_bf, partial_Pif_partial_bf_se = asym_distribution(partial_Pif_partial_bf)
partial_Pif_partial_b, partial_Pif_partial_b_se = asym_distribution(partial_Pif_partial_b)
partial_CS_partial_b, partial_CS_partial_b_se = asym_distribution(partial_CS_partial_b)
c_u, c_u_se = asym_distribution(c_u)
c_R, c_R_se = asym_distribution(c_R)
    
# %%
# Save variables

# Point estimates
np.save(f"{paths.arrays_path}p_stars_e{elast_id}_n{nest_id}.npy", p_stars)
np.save(f"{paths.arrays_path}R_stars_e{elast_id}_n{nest_id}.npy", R_stars)
np.save(f"{paths.arrays_path}num_stations_stars_e{elast_id}_n{nest_id}.npy", num_stations_stars)
np.save(f"{paths.arrays_path}q_stars_e{elast_id}_n{nest_id}.npy", q_stars)
np.save(f"{paths.arrays_path}cs_by_type_e{elast_id}_n{nest_id}.npy", cs_by_type)
np.save(f"{paths.arrays_path}cs_e{elast_id}_n{nest_id}.npy", cs)
np.save(f"{paths.arrays_path}ps_e{elast_id}_n{nest_id}.npy", ps)
np.save(f"{paths.arrays_path}ts_e{elast_id}_n{nest_id}.npy", ts)
np.save(f"{paths.arrays_path}ccs_e{elast_id}_n{nest_id}.npy", ccs)
np.save(f"{paths.arrays_path}full_elasts_e{elast_id}_n{nest_id}.npy", full_elasts)
np.save(f"{paths.arrays_path}partial_elasts_e{elast_id}_n{nest_id}.npy", partial_elasts)
np.save(f"{paths.arrays_path}partial_Pif_partial_bf_e{elast_id}_n{nest_id}.npy", partial_Pif_partial_bf)
np.save(f"{paths.arrays_path}partial_Pif_partial_b_e{elast_id}_n{nest_id}.npy", partial_Pif_partial_b)
np.save(f"{paths.arrays_path}partial_CS_partial_b_e{elast_id}_n{nest_id}.npy", partial_CS_partial_b)
np.save(f"{paths.arrays_path}c_u_e{elast_id}_n{nest_id}.npy", c_u)
np.save(f"{paths.arrays_path}c_R_e{elast_id}_n{nest_id}.npy", c_R)

# Standard errors
if compute_std_errs:
    np.save(f"{paths.arrays_path}p_stars_se_e{elast_id}_n{nest_id}.npy", p_stars_se)
    np.save(f"{paths.arrays_path}R_stars_se_e{elast_id}_n{nest_id}.npy", R_stars_se)
    np.save(f"{paths.arrays_path}num_stations_stars_se_e{elast_id}_n{nest_id}.npy", num_stations_stars_se)
    np.save(f"{paths.arrays_path}q_stars_se_e{elast_id}_n{nest_id}.npy", q_stars_se)
    np.save(f"{paths.arrays_path}cs_by_type_se_e{elast_id}_n{nest_id}.npy", cs_by_type_se)
    np.save(f"{paths.arrays_path}cs_se_e{elast_id}_n{nest_id}.npy", cs_se)
    np.save(f"{paths.arrays_path}ps_se_e{elast_id}_n{nest_id}.npy", ps_se)
    np.save(f"{paths.arrays_path}ts_se_e{elast_id}_n{nest_id}.npy", ts_se)
    np.save(f"{paths.arrays_path}ccs_se_e{elast_id}_n{nest_id}.npy", ccs_se)
    np.save(f"{paths.arrays_path}full_elasts_se_e{elast_id}_n{nest_id}.npy", full_elasts_se)
    np.save(f"{paths.arrays_path}partial_elasts_se_e{elast_id}_n{nest_id}.npy", partial_elasts_se)
    np.save(f"{paths.arrays_path}partial_Pif_partial_bf_se_e{elast_id}_n{nest_id}.npy", partial_Pif_partial_bf_se)
    np.save(f"{paths.arrays_path}partial_Pif_partial_b_se_e{elast_id}_n{nest_id}.npy", partial_Pif_partial_b_se)
    np.save(f"{paths.arrays_path}partial_CS_partial_b_se_e{elast_id}_n{nest_id}.npy", partial_CS_partial_b_se)
    np.save(f"{paths.arrays_path}c_u_se_e{elast_id}_n{nest_id}.npy", c_u_se)
    np.save(f"{paths.arrays_path}c_R_se_e{elast_id}_n{nest_id}.npy", c_R_se)
