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

import sys
import copy

from multiprocessing import Pool

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

import pickle

import time

# %%
# Determine imputations
task_id = int(sys.argv[1])
elast_id = task_id // paths.sigmas.shape[0]
nest_id = task_id % paths.sigmas.shape[0]
avg_price_el = paths.avg_price_elasts[elast_id]
sigma = paths.sigmas[nest_id]

print(f"Orange price elasticity: {avg_price_el}")
print(f"sigma: {sigma}")

# %%
# Determine parallel computation parameters
num_cpus = int(sys.argv[2])
print(f"number of CPUs running in parallel: {num_cpus}", flush=True)

# %%
# Determine whether to print detailed messages
print_msg = False
print_updates = True

# %%
# Determine whether to save arrays before processing standard errors
save_bf = True

# %%
# Import infrastructure / quality data
df_inf = pd.read_csv(f"{paths.data_path}infrastructure_clean.csv", engine="python") # engine helps encoding, error with commune names, but doesn't matter b/c not used
df_inf = df_inf[df_inf['market'] > 0] # don't include Rest-of-France market
df_q = pd.read_csv(f"{paths.data_path}quality_ookla.csv")
df_q = df_q[df_q['market'] > 0] # don't include Rest-of-France market

mno_codes = {
    'Orange': 1, 
    'SFR': 2,
    'Free': 3, 
    'Bouygues': 4, 
    'MVNO': 5
}

# Commune characteristics
population = df_inf['msize_24'].values
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
start = time.time()
cc_3g = np.zeros(bw_3g.shape)
cc_4g = np.zeros(bw_4g.shape)
for i in range(bw_3g.shape[0]): # slow, but only need to do once
    for j in range(bw_3g.shape[1]):
        cc_3g[i,j] = infr.rho_C_hex(bw_3g[i,j], radius[i,j], 1.) if not np.isnan(bw_3g[i,j]) and not np.isnan(radius[i,j]) else np.nan # channel capacity in Mbps
        cc_4g[i,j] = infr.rho_C_hex(bw_4g[i,j], radius[i,j], 1.) if not np.isnan(bw_4g[i,j]) and not np.isnan(radius[i,j]) else np.nan # channel capacity in Mbps
cc_tot = np.nan_to_num(cc_3g) * 2.5 / 4.08 + np.nan_to_num(cc_4g) # aggregate to 4G equivalent based on spectral efficiencies from https://en.wikipedia.org/wiki/Spectral_efficiency 
if print_updates:
    print(f"Finished calculating channel capacities in {np.round(time.time() - start, 1)} seconds.", flush=True)

# Calculate efficiency factor using Orange data demanded
df_q_Orange = pd.read_csv(f"{paths.data_path}quality_clean.csv")
df_q_Orange = df_q_Orange[df_q_Orange['market'] > 0] # don't include Rest-of-France market
q_dem_3g = df_q_Orange['bitps3'].values / 10.0**3.0 / stations[:,mno_codes['Orange']-1] # in Mbps / station
q_dem_4g = df_q_Orange['bitps4'].values / 10.0**3.0 / stations[:,mno_codes['Orange']-1] # in Mbps / station
q_dem_tot = np.nan_to_num(q_dem_3g) + np.nan_to_num(q_dem_4g)
lamda = (q_avg[:,mno_codes['Orange']-1] + q_dem_tot) / cc_tot[:,mno_codes['Orange']-1] # efficiency factor, NOTE: b/c power and JN noise are in / 5 MHz units, these parameters are going to absorb the adjustment of bandwidth to / 5 MHz since we do not do so explicitly in the infrastructure functions

# Convert bandwidth to 4G-equivalent
bw_4g_equiv = np.nan_to_num(bw_3g) * 2.5 / 4.08 + np.nan_to_num(bw_4g) # aggregate to 4G equivalent based on spectral efficiencies from https://en.wikipedia.org/wiki/Spectral_efficiency 

# Adjust channel capacities by spectral efficiency factor
cc_tot = lamda[:,np.newaxis] * cc_tot

# Identify the markets in which Free has 0 bandwidth or stations, not going to use these markets
free_0_bw = cc_tot[:,mno_codes['Free']-1] == 0.
free_0_stations = stations[:,mno_codes['Free']-1] == 0.
markets_org_free_share = free_0_bw | free_0_stations

# %%
# Load the DemandSystem created when estimating demand
with open(f"{paths.data_path}demandsystem.obj", "rb") as file_ds:
    ds = pickle.load(file_ds)

# Drop Rest-of-France market
market_idx = ds.dim3.index(ds.marketname)
market_numbers = np.max(ds.data[:,:,market_idx], axis=1)
ds.data = ds.data[market_numbers > 0,:,:] # drop "Rest of France"

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
pop_dens = pop_dens[~markets_org_free_share]

# Get array of product prices
prices = ds.data[0,:,pidx] # 0 b/c market doesn't matter

# %%
# Import demand estimation results
N = ds.M
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
num_firms_to_simulate_extend = 9
dlims = np.array([1000.0, 10000.0])
vlims = np.array([1.0, 1.0])
num_prods = dlims.shape[0]
num_firms_array = np.arange(num_firms_to_simulate, dtype=int) + 1
num_firms_array_extend = np.arange(num_firms_to_simulate_extend, dtype=int) + 1
rep_density = 2791.7 # contraharmonic mean for France # np.median(pop_dens)
rep_market_size = np.median(area)
rep_population = rep_density * rep_market_size

# Densities to test
vlow_dens = 43.1 # pop dens (people / km^2) of all of USA
low_dens = 123.9 # pop dens (people / km^2) of all of France
high_dens = 20588.2 # pop dens of Paris
densities = np.array([rep_density, vlow_dens, low_dens, high_dens]) # rep_density must be first (because its results just get copied over from the regular exercise)
np.save(f"{paths.arrays_path}cntrfctl_densities_e{elast_id}_n{nest_id}.npy", densities)
np.save(f"{paths.arrays_path}cntrfctl_densities_pop_e{elast_id}_n{nest_id}.npy", densities * rep_market_size)

# Bandwidth values to test
market_bw = np.average(np.sum(bw_4g_equiv, axis=1), weights=population)
low_bw_val = market_bw * 0.5
high_bw_val = market_bw * 1.5
bw_vals = np.array([market_bw, low_bw_val, high_bw_val]) # market_bw must be first (because its results just get copied over from the regular exercise)
np.save(f"{paths.arrays_path}cntrfctl_bw_vals_e{elast_id}_n{nest_id}.npy", bw_vals)

# Welfare options
include_logit_shock = False
include_pop = False

def compute_equilibria(theta_n, p_starting_vals=None, R_starting_vals=None):
    """Compute the equilibria for a particular draw from the demand parameter's asymptotic distribution."""
    
    # Construct demand parameter
    if print_updates:
        print(f"theta_n={theta_n} beginning computation...", flush=True)
    theta_sigma = np.concatenate((thetas_to_compute[theta_n,:], np.array([sigma])))
    
    # Recover xis
    start = time.time()
    xis = blp.xi(ds, theta_sigma, ds.data, None)
    if print_updates:
        print(f"theta_n={theta_n}: Finished calculating xis in {np.round(time.time() - start, 1)} seconds.", flush=True)

    # %%
    # Estimate per-user costs

    # Calculate shares and Jacobian of shares with respect to prices
    start = time.time()
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
    if print_updates:
        print(f"theta_n={theta_n}: Finished calculating per-user costs in {np.round(time.time() - start, 1)} seconds.", flush=True)

    # %%
    # Estimate per-base station costs

    # Calculate shares and Jacobian of shares with respect to prices
    start = time.time()
    impute_MVNO = {
        'impute': True, 
        'firms_share': np.array([True, True, False, True]), # all firms share with MVNO, except Free
        'include': True
    }
    MR = ie.pi_deriv_R(radius, bw_4g_equiv, lamda, ds, xis, theta_sigma, population, area, c_u, impute_MVNO=impute_MVNO, q_0=None, eps=0.01)

    # Determine per-base station costs
    c_R = MR / infr.num_stations_deriv(radius, area[:,np.newaxis])
    if print_updates:
        print(f"theta_n={theta_n}: Finished calculating per-base station costs in {np.round(time.time() - start, 1)} seconds.", flush=True)

    # %%
    # Compute counterfactual equilibria

    per_user_costs = np.array([np.mean(c_u[ds.data[0,:,dlimidx] < 5000.0]), np.mean(c_u[ds.data[0,:,dlimidx] >= 5000.0])])
    per_base_station_per_bw_cost = np.average(c_R[:,np.arange(4) != 2] / bw_4g_equiv[:,np.arange(4) != 2], weights=np.tile(population[:,np.newaxis], (1,3)))

    pop_cntrfctl = np.ones((1,)) * rep_population
    market_size_cntrfctl = np.ones((1,)) * rep_market_size
    gamma_cntrfctl = np.ones((1,)) * np.average(lamda, weights=population)

    p_stars = np.zeros((num_firms_array.shape[0], num_prods)) * np.nan
    R_stars = np.zeros(num_firms_array.shape) * np.nan
    num_stations_stars = np.zeros(num_firms_array.shape) * np.nan
    num_stations_per_firm_stars = np.zeros(num_firms_array.shape) * np.nan
    q_stars = np.zeros(num_firms_array.shape) * np.nan
    cs = np.zeros(num_firms_array_extend.shape) * np.nan
    cs_by_type = np.zeros((num_firms_array_extend.shape[0], yclastidx - yc1idx + 1)) * np.nan
    ps = np.zeros(num_firms_array_extend.shape) * np.nan
    ts = np.zeros(num_firms_array_extend.shape) * np.nan
    ccs = np.zeros(num_firms_array.shape) * np.nan
    ccs_per_bw = np.zeros(num_firms_array.shape) * np.nan
    avg_path_losses = np.zeros(num_firms_array.shape) * np.nan
    avg_SINR = np.zeros(num_firms_array.shape) * np.nan
    full_elasts = np.zeros((num_firms_array.shape[0], num_prods)) * np.nan
    partial_elasts = np.zeros((num_firms_array.shape[0], num_prods)) * np.nan

    partial_Pif_partial_bf_allfixed = np.zeros(num_firms_array.shape) * np.nan
    partial_Pif_partial_b_allfixed = np.zeros(num_firms_array.shape) * np.nan
    partial_CS_partial_b_allfixed = np.zeros(num_firms_array.shape) * np.nan
    partial_Pif_partial_bf_allbw = np.zeros(num_firms_array.shape) * np.nan
    partial_Pif_partial_b_allbw = np.zeros(num_firms_array.shape) * np.nan
    partial_CS_partial_b_allbw = np.zeros(num_firms_array.shape) * np.nan
    
    p_stars_shortrun = np.zeros((1, num_prods)) * np.nan
    R_stars_shortrun = np.zeros(1) * np.nan
    num_stations_stars_shortrun = np.zeros(1) * np.nan
    num_stations_per_firm_stars_shortrun = np.zeros(1) * np.nan
    q_stars_shortrun = np.zeros(1) * np.nan
    cs_shortrun = np.zeros(1) * np.nan
    cs_by_type_shortrun = np.zeros((1, yclastidx - yc1idx + 1)) * np.nan
    ps_shortrun = np.zeros(1) * np.nan
    ts_shortrun = np.zeros(1) * np.nan
    ccs_shortrun = np.zeros(1) * np.nan
    ccs_per_bw_shortrun = np.zeros(1) * np.nan
    avg_path_losses_shortrun = np.zeros(1) * np.nan
    
    p_stars_free_allfixed = np.zeros((2, num_prods)) * np.nan
    R_stars_free_allfixed = np.zeros(2) * np.nan
    num_stations_stars_free_allfixed = np.zeros(2) * np.nan
    num_stations_per_firm_stars_free_allfixed = np.zeros(2) * np.nan
    q_stars_free_allfixed = np.zeros(2) * np.nan
    cs_free_allfixed = np.zeros(2) * np.nan
    cs_by_type_free_allfixed = np.zeros((2, yclastidx - yc1idx + 1)) * np.nan
    ps_free_allfixed = np.zeros(2) * np.nan
    ts_free_allfixed = np.zeros(2) * np.nan
    ccs_free_allfixed = np.zeros(2) * np.nan
    ccs_per_bw_free_allfixed = np.zeros(2) * np.nan
    avg_path_losses_free_allfixed = np.zeros(2) * np.nan
    p_stars_free_allbw = np.zeros((2, num_prods)) * np.nan
    R_stars_free_allbw = np.zeros(2) * np.nan
    num_stations_stars_free_allbw = np.zeros(2) * np.nan
    num_stations_per_firm_stars_free_allbw = np.zeros(2) * np.nan
    q_stars_free_allbw = np.zeros(2) * np.nan
    cs_free_allbw = np.zeros(2) * np.nan
    cs_by_type_free_allbw = np.zeros((2, yclastidx - yc1idx + 1)) * np.nan
    ps_free_allbw = np.zeros(2) * np.nan
    ts_free_allbw = np.zeros(2) * np.nan
    ccs_free_allbw = np.zeros(2) * np.nan
    ccs_per_bw_free_allbw = np.zeros(2) * np.nan
    avg_path_losses_free_allbw = np.zeros(2) * np.nan

    p_stars_dens = np.zeros((num_firms_array.shape[0], densities.shape[0], num_prods)) * np.nan
    R_stars_dens = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    num_stations_stars_dens = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    num_stations_per_firm_stars_dens = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    q_stars_dens = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    cs_dens = np.zeros((num_firms_array_extend.shape[0], densities.shape[0])) * np.nan
    cs_by_type_dens = np.zeros((num_firms_array_extend.shape[0], densities.shape[0], yclastidx - yc1idx + 1)) * np.nan
    ps_dens = np.zeros((num_firms_array_extend.shape[0], densities.shape[0])) * np.nan
    ts_dens = np.zeros((num_firms_array_extend.shape[0], densities.shape[0])) * np.nan
    ccs_dens = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    ccs_per_bw_dens = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    avg_path_losses_dens = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    avg_SINR_dens = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan

    p_stars_bw = np.zeros((num_firms_array.shape[0], bw_vals.shape[0], num_prods)) * np.nan
    R_stars_bw = np.zeros((num_firms_array.shape[0], bw_vals.shape[0])) * np.nan
    num_stations_stars_bw = np.zeros((num_firms_array.shape[0], bw_vals.shape[0])) * np.nan
    num_stations_per_firm_stars_bw = np.zeros((num_firms_array.shape[0], bw_vals.shape[0])) * np.nan
    q_stars_bw = np.zeros((num_firms_array.shape[0], bw_vals.shape[0])) * np.nan
    cs_bw = np.zeros((num_firms_array_extend.shape[0], bw_vals.shape[0])) * np.nan
    cs_by_type_bw = np.zeros((num_firms_array_extend.shape[0], bw_vals.shape[0], yclastidx - yc1idx + 1)) * np.nan
    ps_bw = np.zeros((num_firms_array_extend.shape[0], bw_vals.shape[0])) * np.nan
    ts_bw = np.zeros((num_firms_array_extend.shape[0], bw_vals.shape[0])) * np.nan
    ccs_bw = np.zeros((num_firms_array.shape[0], bw_vals.shape[0])) * np.nan
    ccs_per_bw_bw = np.zeros((num_firms_array.shape[0], bw_vals.shape[0])) * np.nan
    avg_path_losses_bw = np.zeros((num_firms_array.shape[0], bw_vals.shape[0])) * np.nan
    avg_SINR_bw = np.zeros((num_firms_array.shape[0], bw_vals.shape[0])) * np.nan
    
    p_stars_dens_1p = np.zeros((num_firms_array.shape[0], densities.shape[0], 1)) * np.nan
    R_stars_dens_1p = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    num_stations_stars_dens_1p = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    num_stations_per_firm_stars_dens_1p = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    q_stars_dens_1p = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    cs_dens_1p = np.zeros((num_firms_array_extend.shape[0], densities.shape[0])) * np.nan
    cs_by_type_dens_1p = np.zeros((num_firms_array_extend.shape[0], densities.shape[0], yclastidx - yc1idx + 1)) * np.nan
    ps_dens_1p = np.zeros((num_firms_array_extend.shape[0], densities.shape[0])) * np.nan
    ts_dens_1p = np.zeros((num_firms_array_extend.shape[0], densities.shape[0])) * np.nan
    ccs_dens_1p = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    ccs_per_bw_dens_1p = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    avg_path_losses_dens_1p = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    avg_SINR_dens_1p = np.zeros((num_firms_array.shape[0], densities.shape[0])) * np.nan
    
    successful = np.ones(num_firms_array_extend.shape, dtype=bool)
    successful_bw_deriv_allfixed = np.ones(num_firms_array.shape, dtype=bool)
    successful_bw_deriv_allbw = np.ones(num_firms_array.shape, dtype=bool)
    successful_shortrun = np.ones(1, dtype=bool)
    successful_free_allfixed = np.ones(2, dtype=bool)
    successful_free_allbw = np.ones(2, dtype=bool)
    successful_dens = np.ones((num_firms_array_extend.shape[0], densities.shape[0]), dtype=bool)
    successful_bw = np.ones((num_firms_array_extend.shape[0], bw_vals.shape[0]), dtype=bool)
    successful_dens_1p = np.ones((num_firms_array_extend.shape[0], densities.shape[0]), dtype=bool)
        
    # Determine the starting guesses for each
    if p_starting_vals is None:
        p_0s = np.tile(per_user_costs[np.newaxis,:], (num_firms_array_extend.shape[0],1))
        p_0s[np.isin(num_firms_array_extend, np.array([1]))] = 2.0 * p_0s[np.isin(num_firms_array_extend, np.array([1]))] # these don't have good convergence properties if start at MC
    else:
        p_0s = np.copy(p_starting_vals)
    if R_starting_vals is None:
        R_0s = np.ones(num_firms_array_extend.shape)[:,np.newaxis] * 0.5
    else:
        R_0s = np.copy(R_starting_vals)

    num_firms_array_extend_idx = np.arange(num_firms_array_extend.shape[0])
    num_firms_array_extend_idx_3 = np.where(num_firms_array_extend == 3)[0]
    num_firms_array_extend_idx_4 = np.where(num_firms_array_extend == 4)[0]
    num_firms_array_extend_idx = np.concatenate((num_firms_array_extend_idx[:num_firms_array_extend_idx_3[0]], num_firms_array_extend_idx[num_firms_array_extend_idx_4], num_firms_array_extend_idx[num_firms_array_extend_idx_3], num_firms_array_extend_idx[num_firms_array_extend_idx_4[0]+1:])) # need to switch 3 and 4 for the short-run counterfactual, which uses the results from 4 in 3
    for i in num_firms_array_extend_idx:
        num_firms = num_firms_array_extend[i]
        if print_updates:
            print(f"theta_n={theta_n}, num_firms={num_firms} computing...", flush=True)

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
        ds_cntrfctl.data[:,:,yc1idx:yclastidx+1] = np.percentile(ds.data[:,0,yc1idx:yclastidx+1], np.linspace(10, 90, 9))[np.newaxis,np.newaxis,:]

        # Create ds for 1-product version
        ds_cntrfctl_1p = copy.deepcopy(ds_cntrfctl)
        ds_cntrfctl_1p.data = ds_cntrfctl_1p.data[:,np.arange(ds_cntrfctl_1p.data.shape[1]) % num_prods == np.argmax(dlims),:] # keep only the highest data limit plan
        ds_cntrfctl_1p.firms = np.repeat(np.arange(num_firms, dtype=int) + 1, 1)
        ds_cntrfctl_1p.J = num_firms * 1
        
        # Create market variables with properties I want
        bw_cntrfctl = np.ones((1,1)) * market_bw / float(num_firms)
        xis_cntrfctl = np.ones((1,num_prods)) * theta_sigma[coef.O]
        xis_cntrfctl_1p = np.ones((1,1)) * theta_sigma[coef.O]

        # Create cost arrays
        c_u_cntrfctl = per_user_costs
        select_1p = np.arange(num_prods) == np.argmax(dlims)
        c_u_cntrfctl_1p = c_u_cntrfctl[select_1p] # keep only the highest data limit plan
        c_R_cntrfctl = np.ones((1,1)) * per_base_station_per_bw_cost * bw_cntrfctl # per tower cost based on level of bandwidth for each firm

        # Set starting values (if None, num_firms=1 can cause problems for convergence)
        R_0 = R_0s[i,:][:,np.newaxis]
        p_0 = p_0s[i,:]
        p_0_1p = p_0[select_1p]
        
        def simple_symmetric_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta_sigma, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, num_firms, num_prods):
            """Compute the symmetric equilibrium."""
            
            ds_cntrfctl_ = copy.deepcopy(ds_cntrfctl)
            
            # Compute the equilibrium
            R_star, p_star, q_star, success = ie.infrastructure_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_, xis_cntrfctl, theta_sigma, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, symmetric=True, print_msg=print_msg, impute_MVNO={'impute': False}, q_0=None, eps_R=0.01, eps_p=0.01, factor=100.)
            
            # Update Demand System
            ds_cntrfctl_.data[:,:,pidx] = np.copy(p_star)
            ds_cntrfctl_.data[:,:,qidx] = np.tile(q_star, (num_prods,))
            
            # Calculate welfare impact
            cs_by_type_ = welfare.consumer_surplus(ds_cntrfctl_, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, include_logit_shock=include_logit_shock)
            cs_ = welfare.agg_consumer_surplus(ds_cntrfctl_, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, include_logit_shock=include_logit_shock, include_pop=include_pop)
            ps_ = welfare.producer_surplus(ds_cntrfctl_, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_pop=include_pop)
            ts_ = welfare.total_surplus(ds_cntrfctl_, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_logit_shock=include_logit_shock, include_pop=include_pop)
            
            p_stars_ = p_star[:num_prods]
            R_stars_ = R_star[0,0]
            num_stations_stars_ = num_firms * infr.num_stations(R_stars_, rep_market_size)
            num_stations_per_firm_stars_ = infr.num_stations(R_stars_, rep_market_size)
            q_stars_ = q_star[0,0]

            cc_cntrfctl = np.zeros((R_star.shape[0], 1))
            for m in range(R_star.shape[0]):
                cc_cntrfctl[m,0] = infr.rho_C_hex(bw_cntrfctl[m,0], R_star[m,0], gamma_cntrfctl[m])
            ccs_ = cc_cntrfctl[0,0]
            ccs_per_bw_ = (cc_cntrfctl / bw_cntrfctl)[0,0]
            avg_path_losses_ = infr.avg_path_loss(R_stars_)
            num_stations_cntrfctl = infr.num_stations(np.array([[R_stars_]]), market_size_cntrfctl)
            avg_SINR_ = infr.avg_SINR(R_stars_)
            
            return success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_

        # Simple symmetric equilibrium result for representative values
        start = time.time()
        success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = simple_symmetric_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta_sigma, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, num_firms, num_prods)
        ds_cntrfctl_baseline = copy.deepcopy(ds_cntrfctl_)
        if print_updates:
            print(f"theta_n={theta_n}, num_firms={num_firms}: Finished calculating symmetric equilibrium in {np.round(time.time() - start, 1)} seconds.", flush=True)
        successful[i], cs_by_type[i,:], cs[i], ps[i], ts[i] = success, cs_by_type_, cs_, ps_, ts_
        if np.isin(num_firms, num_firms_array): # don't need to record these if gone beyond num_firms_array
            p_stars[i,:], R_stars[i], num_stations_stars[i], num_stations_per_firm_stars[i], q_stars[i], ccs[i], ccs_per_bw[i], avg_path_losses[i], avg_SINR[i] = p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_

            # Calculate elasticities
            start = time.time()
            full_elasts[i,:] = pe.price_elast(np.copy(p_stars[i,:]), cc_cntrfctl, ds_cntrfctl_baseline, xis_cntrfctl, theta_sigma, num_stations_cntrfctl, pop_cntrfctl, symmetric=True, impute_MVNO={'impute': False}, q_0=None, eps=0.01, full=True)
            partial_elasts[i,:] = pe.price_elast(np.copy(p_stars[i,:]), cc_cntrfctl, ds_cntrfctl_baseline, xis_cntrfctl, theta_sigma, num_stations_cntrfctl, pop_cntrfctl, symmetric=True, impute_MVNO={'impute': False}, q_0=None, eps=0.01, full=False)
            if print_updates:
                print(f"theta_n={theta_n}, num_firms={num_firms}: Finished calculating elasticities in {np.round(time.time() - start, 1)} seconds.", flush=True)
        
        # Symmetric results by density
        for j in range(densities.shape[0]):
            start = time.time()
            pop_cntrfctl_dens = densities[j] * market_size_cntrfctl

            if j == 0: # this is just equivalent to the simple case
                successful_dens[i,j], cs_by_type_dens[i,j,:], cs_dens[i,j], ps_dens[i,j], ts_dens[i,j] = successful[i], cs_by_type[i,:], cs[i], ps[i], ts[i]
                if np.isin(num_firms, num_firms_array): # don't need to record these if gone beyond num_firms_array
                    p_stars_dens[i,j,:], R_stars_dens[i,j], num_stations_stars_dens[i,j], num_stations_per_firm_stars_dens[i,j], q_stars_dens[i,j], ccs_dens[i,j], ccs_per_bw_dens[i,j], avg_path_losses_dens[i,j], avg_SINR_dens[i,j] = p_stars[i,:], R_stars[i], num_stations_stars[i], num_stations_per_firm_stars[i], q_stars[i], ccs[i], ccs_per_bw[i], avg_path_losses[i], avg_SINR[i]

            elif not successful[i]: # representative case failed, no need to do the others
                successful_dens[i,:] = False

            else:
                R_0_dens = np.copy(R_0)
                p_0_dens = np.copy(p_0)
                if densities[j] / densities[0] > 1.5: # better starting guess
                    R_0_dens = (1.0 / 3.0) * R_0_dens
                if densities[j] / densities[0] < 0.5: # better starting guess
                    R_0_dens = 3.0 * R_0_dens
                success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = simple_symmetric_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta_sigma, pop_cntrfctl_dens, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0_dens, p_0_dens, num_firms, num_prods)
                successful_dens[i,j], cs_by_type_dens[i,j,:], cs_dens[i,j], ps_dens[i,j], ts_dens[i,j] = success, cs_by_type_, cs_, ps_, ts_
                if np.isin(num_firms, num_firms_array): # don't need to record these if gone beyond num_firms_array
                    p_stars_dens[i,j,:], R_stars_dens[i,j], num_stations_stars_dens[i,j], num_stations_per_firm_stars_dens[i,j], q_stars_dens[i,j], ccs_dens[i,j], ccs_per_bw_dens[i,j], avg_path_losses_dens[i,j], avg_SINR_dens[i,j] = p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_
            
            if print_updates:
                print(f"theta_n={theta_n}, num_firms={num_firms}, density={j}: Finished calculating symmetric density equilibrium in {np.round(time.time() - start, 1)} seconds.", flush=True)
                    
            # 1-product case
            start = time.time()
            R_0_dens_1p = np.copy(R_0)
            p_0_dens_1p = np.copy(p_0_1p)
            if densities[j] / densities[0] > 1.5: # better starting guess
                R_0_dens_1p = (1.0 / 3.0) * R_0_dens_1p
            if densities[j] / densities[0] < 0.5: # better starting guess
                R_0_dens_1p = 3.0 * R_0_dens_1p
            success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = simple_symmetric_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_1p, xis_cntrfctl_1p, theta_sigma, pop_cntrfctl_dens, market_size_cntrfctl, c_u_cntrfctl_1p, c_R_cntrfctl, R_0_dens_1p, p_0_dens_1p, num_firms, 1)
            if print_updates:
                print(f"theta_n={theta_n}, num_firms={num_firms}, density={j}: Finished calculating 1-product symmetric density equilibrium in {np.round(time.time() - start, 1)} seconds.", flush=True)
            successful_dens_1p[i,j], cs_by_type_dens_1p[i,j,:], cs_dens_1p[i,j], ps_dens_1p[i,j], ts_dens_1p[i,j] = success, cs_by_type_, cs_, ps_, ts_
            if np.isin(num_firms, num_firms_array): # don't need to record these if gone beyond num_firms_array
                p_stars_dens_1p[i,j,:], R_stars_dens_1p[i,j], num_stations_stars_dens_1p[i,j], num_stations_per_firm_stars_dens_1p[i,j], q_stars_dens_1p[i,j], ccs_dens_1p[i,j], ccs_per_bw_dens_1p[i,j], avg_path_losses_dens_1p[i,j], avg_SINR_dens_1p[i,j] = p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_

        # Symmetric results by bandwidth
        for j in range(bw_vals.shape[0]): # starting at 1 b/c the value for 0 is copied later
            start = time.time()
            bw_cntrfctl_bw = np.ones((1,1)) * bw_vals[j] / float(num_firms)
            c_R_cntrfctl_bw = np.ones((1,1)) * per_base_station_per_bw_cost * bw_cntrfctl_bw

            if j == 0: # this is just equivalent to the simple case
                successful_bw[i,j], cs_by_type_bw[i,j,:], cs_bw[i,j], ps_bw[i,j], ts_bw[i,j] = successful[i], cs_by_type[i,:], cs[i], ps[i], ts[i]
                if np.isin(num_firms, num_firms_array): # don't need to record these if gone beyond num_firms_array
                    p_stars_bw[i,j,:], R_stars_bw[i,j], num_stations_stars_bw[i,j], num_stations_per_firm_stars_bw[i,j], q_stars_bw[i,j], ccs_bw[i,j], ccs_per_bw_bw[i,j], avg_path_losses_bw[i,j], avg_SINR_bw[i,j] = p_stars[i,:], R_stars[i], num_stations_stars[i], num_stations_per_firm_stars[i], q_stars[i], ccs[i], ccs_per_bw[i], avg_path_losses[i], avg_SINR[i]

            elif not successful[i]: # representative case failed, no need to do the others
                successful_bw[i,:] = False

            else:
                success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = simple_symmetric_eqm(bw_cntrfctl_bw, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta_sigma, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl_bw, R_0, p_0, num_firms, num_prods)
                successful_bw[i,j], cs_by_type_bw[i,j,:], cs_bw[i,j], ps_bw[i,j], ts_bw[i,j] = success, cs_by_type_, cs_, ps_, ts_
                if np.isin(num_firms, num_firms_array): # don't need to record these if gone beyond num_firms_array
                    p_stars_bw[i,j,:], R_stars_bw[i,j], num_stations_stars_bw[i,j], num_stations_per_firm_stars_bw[i,j], q_stars_bw[i,j], ccs_bw[i,j], ccs_per_bw_bw[i,j], avg_path_losses_bw[i,j], avg_SINR_bw[i,j] = p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_

            if print_updates:
                print(f"theta_n={theta_n}, num_firms={num_firms}, bw={j}: Finished calculating symmetric bandwidth equilibrium in {np.round(time.time() - start, 1)} seconds.", flush=True)
                        
        # Only doing the extended search for the symmetric equilibria; if gone beyond num_firms_array, end there
        if not np.isin(num_firms, num_firms_array):
            continue

        # Calculate bandwidth derivatives

        # using all fixed costs
        start = time.time()
        Pif_bf, Pif_b, CS_b, success = ie.bw_foc(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_baseline, xis_cntrfctl, theta_sigma, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, np.array([[R_stars[i]]]), p_stars[i,:], symmetric=True, print_msg=print_msg, impute_MVNO={'impute': False}, q_0=None, eps_R=0.01, eps_p=0.01, eps_bw=0.01, factor=100., include_logit_shock=include_logit_shock, adjust_c_R=False)
        if print_updates:
            print(f"theta_n={theta_n}, num_firms={num_firms}: Finished calculating bandwidth derivatives (all fixed cost specification) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        successful_bw_deriv_allfixed[i] = success
        partial_Pif_partial_bf_allfixed[i] = Pif_bf[0,0]
        partial_Pif_partial_b_allfixed[i] = Pif_b[0,0]
        partial_CS_partial_b_allfixed[i] = CS_b[0]

        # using all scaled with bw
        start = time.time()
        Pif_bf, Pif_b, CS_b, success = ie.bw_foc(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_baseline, xis_cntrfctl, theta_sigma, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, np.array([[R_stars[i]]]), p_stars[i,:], symmetric=True, print_msg=print_msg, impute_MVNO={'impute': False}, q_0=None, eps_R=0.01, eps_p=0.01, eps_bw=0.01, factor=100., include_logit_shock=include_logit_shock, adjust_c_R=True)
        if print_updates:
            print(f"theta_n={theta_n}, num_firms={num_firms}: Finished calculating bandwidth derivatives (all bw cost specification) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        successful_bw_deriv_allbw[i] = success
        partial_Pif_partial_bf_allbw[i] = Pif_bf[0,0]
        partial_Pif_partial_b_allbw[i] = Pif_b[0,0]
        partial_CS_partial_b_allbw[i] = CS_b[0]
        
        # Calculate "short-run" equilibrium
        if np.isin(num_firms, np.array([3])):
            
            start = time.time()
            R_impute = R_stars[num_firms_array_extend_idx_4[0]] * np.ones((1,1)) # this is the R* from the 4-firm version
            ds_cntrfctl_shortrun = copy.deepcopy(ds_cntrfctl)
            R_star, p_star, q_star, success = ie.infrastructure_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_shortrun, xis_cntrfctl, theta_sigma, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_impute, p_0, symmetric=True, print_msg=print_msg, impute_MVNO={'impute': False}, q_0=None, eps_R=0.01, eps_p=0.01, factor=100., R_fixed=True)
            if print_updates:
                print(f"theta_n={theta_n}, num_firms={num_firms}: Finished calculating short-run equilibrium in {np.round(time.time() - start, 1)} seconds.", flush=True)

            shortrun_idx = 0 # just one
            successful_shortrun[shortrun_idx] = success
            p_stars_shortrun[shortrun_idx,:] = p_star[:num_prods]
            R_stars_shortrun[shortrun_idx] = R_star[0,0]
            num_stations_stars_shortrun[shortrun_idx] = num_firms * infr.num_stations(R_stars_shortrun[shortrun_idx], rep_market_size)
            num_stations_per_firm_stars_shortrun[shortrun_idx] = infr.num_stations(R_stars_shortrun[shortrun_idx], rep_market_size)
            q_stars_shortrun[shortrun_idx] = q_star[0,0]

            # Update Demand System
            ds_cntrfctl_shortrun.data[:,:,pidx] = np.copy(p_star)
            ds_cntrfctl_shortrun.data[:,:,qidx] = np.tile(q_star, (num_prods,))

            # Calculate welfare impact
            cs_by_type_shortrun[shortrun_idx,:] = welfare.consumer_surplus(ds_cntrfctl_shortrun, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, include_logit_shock=include_logit_shock)
            cs_shortrun[shortrun_idx] = welfare.agg_consumer_surplus(ds_cntrfctl_shortrun, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, include_logit_shock=include_logit_shock, include_pop=include_pop)
            ps_shortrun[shortrun_idx] = welfare.producer_surplus(ds_cntrfctl_shortrun, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_pop=include_pop)
            ts_shortrun[shortrun_idx] = welfare.total_surplus(ds_cntrfctl_shortrun, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_logit_shock=include_logit_shock, include_pop=include_pop)

            # Calculate channel capacities and average path loss
            ccs_shortrun[shortrun_idx] = infr.rho_C_hex(bw_cntrfctl[0,0], R_stars_shortrun[shortrun_idx], gamma_cntrfctl[0])
            ccs_per_bw_shortrun[shortrun_idx] = infr.rho_C_hex(bw_cntrfctl[0,0], R_stars_shortrun[shortrun_idx], gamma_cntrfctl[0]) / bw_cntrfctl[0,0]
            avg_path_losses_shortrun[shortrun_idx] = infr.avg_path_loss(R_stars_shortrun[shortrun_idx])

        # Calculate "add Free" equilibrium
        if np.isin(num_firms, np.array([3, 4])):
            # Update bandwidth
            bw_cntrfctl = np.ones((1,1)) * (4. / 3.) * market_bw / float(num_firms)

            # using all fixed costs
            start = time.time()
            ds_cntrfctl_free_allfixed = copy.deepcopy(ds_cntrfctl)
            R_star, p_star, q_star, success = ie.infrastructure_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_free_allfixed, xis_cntrfctl, theta_sigma, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, symmetric=True, print_msg=print_msg, impute_MVNO={'impute': False}, q_0=None, eps_R=0.01, eps_p=0.01, factor=100.)
            if print_updates:
                print(f"theta_n={theta_n}, num_firms={num_firms}: Finished \"add Free\" (all fixed cost specification) in {np.round(time.time() - start, 1)} seconds.", flush=True)

            free_idx = i - 2
            successful_free_allfixed[free_idx] = success
            p_stars_free_allfixed[free_idx,:] = p_star[:num_prods]
            R_stars_free_allfixed[free_idx] = R_star[0,0]
            num_stations_stars_free_allfixed[free_idx] = num_firms * infr.num_stations(R_stars_free_allfixed[free_idx], rep_market_size)
            num_stations_per_firm_stars_free_allfixed[free_idx] = infr.num_stations(R_stars_free_allfixed[free_idx], rep_market_size)
            q_stars_free_allfixed[free_idx] = q_star[0,0]

            # Update Demand System
            ds_cntrfctl_free_allfixed.data[:,:,pidx] = np.copy(p_star)
            ds_cntrfctl_free_allfixed.data[:,:,qidx] = np.tile(q_star, (num_prods,))

            # Calculate welfare impact
            cs_by_type_free_allfixed[free_idx,:] = welfare.consumer_surplus(ds_cntrfctl_free_allfixed, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, include_logit_shock=include_logit_shock)
            cs_free_allfixed[free_idx] = welfare.agg_consumer_surplus(ds_cntrfctl_free_allfixed, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, include_logit_shock=include_logit_shock, include_pop=include_pop)
            ps_free_allfixed[free_idx] = welfare.producer_surplus(ds_cntrfctl_free_allfixed, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_pop=include_pop)
            ts_free_allfixed[free_idx] = welfare.total_surplus(ds_cntrfctl_free_allfixed, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_logit_shock=include_logit_shock, include_pop=include_pop)

            # Calculate channel capacities and average path loss
            ccs_free_allfixed[free_idx] = infr.rho_C_hex(bw_cntrfctl[0,0], R_stars_free_allfixed[free_idx], gamma_cntrfctl[0])
            ccs_per_bw_free_allfixed[free_idx] = infr.rho_C_hex(bw_cntrfctl[0,0], R_stars_free_allfixed[free_idx], gamma_cntrfctl[0]) / bw_cntrfctl[0,0]
            avg_path_losses_free_allfixed[free_idx] = infr.avg_path_loss(R_stars_free_allfixed[free_idx])

            # using all scaled with bw
            start = time.time()
            ds_cntrfctl_free_allbw = copy.deepcopy(ds_cntrfctl)
            c_R_cntrfctl = np.ones((1,1)) * per_base_station_per_bw_cost * bw_cntrfctl
            R_star, p_star, q_star, success = ie.infrastructure_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_free_allbw, xis_cntrfctl, theta_sigma, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, symmetric=True, print_msg=print_msg, impute_MVNO={'impute': False}, q_0=None, eps_R=0.01, eps_p=0.01, factor=100.)
            if print_updates:
                print(f"theta_n={theta_n}, num_firms={num_firms}: Finished \"add Free\" (all bw cost specification) in {np.round(time.time() - start, 1)} seconds.", flush=True)

            free_idx = i - 2
            successful_free_allbw[free_idx] = success
            p_stars_free_allbw[free_idx,:] = p_star[:num_prods]
            R_stars_free_allbw[free_idx] = R_star[0,0]
            num_stations_stars_free_allbw[free_idx] = num_firms * infr.num_stations(R_stars_free_allbw[free_idx], rep_market_size)
            num_stations_per_firm_stars_free_allbw[free_idx] = infr.num_stations(R_stars_free_allbw[free_idx], rep_market_size)
            q_stars_free_allbw[free_idx] = q_star[0,0]

            # Update Demand System
            ds_cntrfctl_free_allbw.data[:,:,pidx] = np.copy(p_star)
            ds_cntrfctl_free_allbw.data[:,:,qidx] = np.tile(q_star, (num_prods,))

            # Calculate welfare impact
            cs_by_type_free_allbw[free_idx,:] = welfare.consumer_surplus(ds_cntrfctl_free_allbw, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, include_logit_shock=include_logit_shock)
            cs_free_allbw[free_idx] = welfare.agg_consumer_surplus(ds_cntrfctl_free_allbw, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, include_logit_shock=include_logit_shock, include_pop=include_pop)
            ps_free_allbw[free_idx] = welfare.producer_surplus(ds_cntrfctl_free_allbw, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_pop=include_pop)
            ts_free_allbw[free_idx] = welfare.total_surplus(ds_cntrfctl_free_allbw, np.tile(xis_cntrfctl, (1,num_firms)), theta_sigma, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_logit_shock=include_logit_shock, include_pop=include_pop)

            # Calculate channel capacities and average path loss
            ccs_free_allbw[free_idx] = infr.rho_C_hex(bw_cntrfctl[0,0], R_stars_free_allbw[free_idx], gamma_cntrfctl[0])
            ccs_per_bw_free_allbw[free_idx] = infr.rho_C_hex(bw_cntrfctl[0,0], R_stars_free_allbw[free_idx], gamma_cntrfctl[0]) / bw_cntrfctl[0,0]
            avg_path_losses_free_allbw[free_idx] = infr.avg_path_loss(R_stars_free_allbw[free_idx])

        if print_updates:
            print(f"theta_n={theta_n}, num_firms={num_firms} complete.", flush=True)
        
    # Put number of firms in per-person terms
    num_stations_stars = num_stations_stars / rep_population
    num_stations_per_firm_stars = num_stations_per_firm_stars / rep_population
    num_stations_stars_shortrun = num_stations_stars_shortrun / rep_population
    num_stations_per_firm_stars_shortrun = num_stations_per_firm_stars_shortrun / rep_population
    num_stations_stars_free_allfixed = num_stations_stars_free_allfixed / rep_population
    num_stations_per_firm_stars_free_allfixed = num_stations_per_firm_stars_free_allfixed / rep_population
    num_stations_stars_free_allbw = num_stations_stars_free_allbw / rep_population
    num_stations_per_firm_stars_free_allbw = num_stations_per_firm_stars_free_allbw / rep_population
    num_stations_stars_dens = num_stations_stars_dens / (densities[np.newaxis,:] * rep_market_size)
    num_stations_per_firm_stars_dens = num_stations_per_firm_stars_dens / (densities[np.newaxis,:] * rep_market_size)
    num_stations_stars_bw = num_stations_stars_bw / rep_population
    num_stations_per_firm_stars_bw = num_stations_per_firm_stars_bw / rep_population
    num_stations_stars_dens_1p = num_stations_stars_dens_1p / (densities[np.newaxis,:] * rep_market_size)
    num_stations_per_firm_stars_dens_1p = num_stations_per_firm_stars_dens_1p / (densities[np.newaxis,:] * rep_market_size)
    
    # Calculate "short-run" equilibria relative to 3- and 4-firm case
    transform_shortrun = lambda x_shortrun, x_longrun, firms_array: np.concatenate((x_shortrun - x_longrun[np.isin(firms_array, np.array([4]))], x_longrun[np.isin(firms_array, np.array([3]))] - x_longrun[np.isin(firms_array, np.array([4]))], x_shortrun - x_longrun[np.isin(firms_array, np.array([3]))]), axis=0)
    p_stars_shortrun = transform_shortrun(p_stars_shortrun, p_stars, num_firms_array)
    R_stars_shortrun = transform_shortrun(R_stars_shortrun, R_stars, num_firms_array)
    num_stations_stars_shortrun = transform_shortrun(num_stations_stars_shortrun, num_stations_stars, num_firms_array)
    num_stations_per_firm_stars_shortrun = transform_shortrun(num_stations_per_firm_stars_shortrun, num_stations_per_firm_stars, num_firms_array)
    q_stars_shortrun = transform_shortrun(q_stars_shortrun, q_stars, num_firms_array)
    cs_shortrun = transform_shortrun(cs_shortrun, cs, num_firms_array_extend)
    cs_by_type_shortrun = transform_shortrun(cs_by_type_shortrun, cs_by_type, num_firms_array_extend)
    ps_shortrun = transform_shortrun(ps_shortrun, ps, num_firms_array_extend)
    ts_shortrun = transform_shortrun(ts_shortrun, ts, num_firms_array_extend)
    ccs_shortrun = transform_shortrun(ccs_shortrun, ccs, num_firms_array)
    ccs_per_bw_shortrun = transform_shortrun(ccs_per_bw_shortrun, ccs_per_bw, num_firms_array)
    avg_path_losses_shortrun = transform_shortrun(avg_path_losses_shortrun, avg_path_losses, num_firms_array)
    successful_shortrun = np.concatenate((successful_shortrun & successful[np.isin(num_firms_array_extend, np.array([4]))], successful[np.isin(num_firms_array_extend, np.array([3]))] & successful[np.isin(num_firms_array_extend, np.array([4]))], successful_shortrun & successful[np.isin(num_firms_array_extend, np.array([3]))]))
    
    # Calculate "add Free" equilibria relative to 3-firm case
    p_stars_free_allfixed = p_stars_free_allfixed - p_stars[np.isin(num_firms_array, np.array([3])),:]
    R_stars_free_allfixed = R_stars_free_allfixed - R_stars[np.isin(num_firms_array, np.array([3]))]
    num_stations_stars_free_allfixed = num_stations_stars_free_allfixed - num_stations_stars[np.isin(num_firms_array, np.array([3]))]
    num_stations_per_firm_stars_free_allfixed = num_stations_per_firm_stars_free_allfixed - num_stations_per_firm_stars[np.isin(num_firms_array, np.array([3]))]
    q_stars_free_allfixed = q_stars_free_allfixed - q_stars[np.isin(num_firms_array, np.array([3]))]
    cs_free_allfixed = cs_free_allfixed - cs[np.isin(num_firms_array_extend, np.array([3]))]
    cs_by_type_free_allfixed = cs_by_type_free_allfixed - cs_by_type[np.isin(num_firms_array_extend, np.array([3])),:]
    ps_free_allfixed = ps_free_allfixed - ps[np.isin(num_firms_array_extend, np.array([3]))]
    ts_free_allfixed = ts_free_allfixed - ts[np.isin(num_firms_array_extend, np.array([3]))]
    ccs_free_allfixed = ccs_free_allfixed - ccs[np.isin(num_firms_array, np.array([3]))]
    ccs_per_bw_free_allfixed = ccs_per_bw_free_allfixed - ccs_per_bw[np.isin(num_firms_array, np.array([3]))]
    avg_path_losses_free_allfixed = avg_path_losses_free_allfixed - avg_path_losses[np.isin(num_firms_array, np.array([3]))]
    
    p_stars_free_allbw = p_stars_free_allbw - p_stars[np.isin(num_firms_array, np.array([3])),:]
    R_stars_free_allbw = R_stars_free_allbw - R_stars[np.isin(num_firms_array, np.array([3]))]
    num_stations_stars_free_allbw = num_stations_stars_free_allbw
    num_stations_per_firm_stars_free_allbw = num_stations_per_firm_stars_free_allbw
    q_stars_free_allbw = q_stars_free_allbw - q_stars[np.isin(num_firms_array, np.array([3]))]
    cs_free_allbw = cs_free_allbw - cs[np.isin(num_firms_array_extend, np.array([3]))]
    cs_by_type_free_allbw = cs_by_type_free_allbw - cs_by_type[np.isin(num_firms_array_extend, np.array([3])),:]
    ps_free_allbw = ps_free_allbw - ps[np.isin(num_firms_array_extend, np.array([3]))]
    ts_free_allbw = ts_free_allbw - ts[np.isin(num_firms_array_extend, np.array([3]))]
    ccs_free_allbw = ccs_free_allbw - ccs[np.isin(num_firms_array, np.array([3]))]
    ccs_per_bw_free_allbw = ccs_per_bw_free_allbw - ccs_per_bw[np.isin(num_firms_array, np.array([3]))]
    avg_path_losses_free_allbw = avg_path_losses_free_allbw - avg_path_losses[np.isin(num_firms_array, np.array([3]))]

    # Calculate welfare relative to monopoly case
    cs = cs - cs[0] #if elast_id > 0 else cs - cs[1]
    ps = ps - ps[0] #if elast_id > 0 else ps - ps[1]
    ts = ts - ts[0] #if elast_id > 0 else ts - ts[1]
    cs_by_type = cs_by_type - cs_by_type[0,:][np.newaxis,:] #if elast_id > 0 else cs_by_type - cs_by_type[1,:][np.newaxis,:]

    cs_dens = cs_dens - cs_dens[0,0] # note that [0,0] is the representative density's monopoly case
    ps_dens = ps_dens - ps_dens[0,0]
    ts_dens = ts_dens - ts_dens[0,0]
    cs_by_type_dens = cs_by_type_dens - cs_by_type_dens[0,0,:][np.newaxis,np.newaxis,:]

    cs_bw = cs_bw - cs_bw[0,0]
    ps_bw = ps_bw - ps_bw[0,0]
    ts_bw = ts_bw - ts_bw[0,0]
    cs_by_type_bw = cs_by_type_bw - cs_by_type_bw[0,0,:][np.newaxis,np.newaxis,:]
    
    cs_dens_1p = cs_dens_1p - cs_dens_1p[0,0]
    ps_dens_1p = ps_dens_1p - ps_dens_1p[0,0]
    ts_dens_1p = ts_dens_1p - ts_dens_1p[0,0]
    cs_by_type_dens_1p = cs_by_type_dens_1p - cs_by_type_dens_1p[0,0,:][np.newaxis,np.newaxis,:]
    
    if print_updates:
        print(f"theta_n={theta_n} finished computation.", flush=True)

    return p_stars, R_stars, num_stations_stars, num_stations_per_firm_stars, q_stars, cs_by_type, cs, ps, ts, ccs, ccs_per_bw, avg_path_losses, avg_SINR, full_elasts, partial_elasts, partial_Pif_partial_bf_allfixed, partial_Pif_partial_b_allfixed, partial_CS_partial_b_allfixed, partial_Pif_partial_bf_allbw, partial_Pif_partial_b_allbw, partial_CS_partial_b_allbw, c_u, c_R, p_stars_shortrun, R_stars_shortrun, num_stations_stars_shortrun, num_stations_per_firm_stars_shortrun, q_stars_shortrun, cs_by_type_shortrun, cs_shortrun, ps_shortrun, ts_shortrun, ccs_shortrun, ccs_per_bw_shortrun, avg_path_losses_shortrun, p_stars_free_allfixed, R_stars_free_allfixed, num_stations_stars_free_allfixed, num_stations_per_firm_stars_free_allfixed, q_stars_free_allfixed, cs_by_type_free_allfixed, cs_free_allfixed, ps_free_allfixed, ts_free_allfixed, ccs_free_allfixed, ccs_per_bw_free_allfixed, avg_path_losses_free_allfixed, p_stars_free_allbw, R_stars_free_allbw, num_stations_stars_free_allbw, num_stations_per_firm_stars_free_allbw, q_stars_free_allbw, cs_by_type_free_allbw, cs_free_allbw, ps_free_allbw, ts_free_allbw, ccs_free_allbw, ccs_per_bw_free_allbw, avg_path_losses_free_allbw, p_stars_dens, R_stars_dens, num_stations_stars_dens, num_stations_per_firm_stars_dens, q_stars_dens, cs_dens, cs_by_type_dens, ps_dens, ts_dens, ccs_dens, ccs_per_bw_dens, avg_path_losses_dens, avg_SINR_dens, p_stars_bw, R_stars_bw, num_stations_stars_bw, num_stations_per_firm_stars_bw, q_stars_bw, cs_bw, cs_by_type_bw, ps_bw, ts_bw, ccs_bw, ccs_per_bw_bw, avg_path_losses_bw, avg_SINR_bw, p_stars_dens_1p, R_stars_dens_1p, num_stations_stars_dens_1p, num_stations_per_firm_stars_dens_1p, q_stars_dens_1p, cs_dens_1p, cs_by_type_dens_1p, ps_dens_1p, ts_dens_1p, ccs_dens_1p, ccs_per_bw_dens_1p, avg_path_losses_dens_1p, avg_SINR_dens_1p, successful, successful_bw_deriv_allfixed, successful_bw_deriv_allbw, successful_shortrun, successful_free_allfixed, successful_free_allbw, successful_dens, successful_bw, successful_dens_1p, per_user_costs
    
# %%
# Compute the equilibria and perturbations

# Initialize variables
theta_N = thetas_to_compute.shape[0] if compute_std_errs else 1

p_stars = np.zeros((theta_N, num_firms_array.shape[0], num_prods))
R_stars = np.zeros((theta_N, num_firms_array.shape[0]))
num_stations_stars = np.zeros((theta_N, num_firms_array.shape[0]))
num_stations_per_firm_stars = np.zeros((theta_N, num_firms_array.shape[0]))
q_stars = np.zeros((theta_N, num_firms_array.shape[0]))

cs_by_type = np.zeros((theta_N, num_firms_array_extend.shape[0], yclastidx - yc1idx + 1))
cs = np.zeros((theta_N, num_firms_array_extend.shape[0]))
ps = np.zeros((theta_N, num_firms_array_extend.shape[0]))
ts = np.zeros((theta_N, num_firms_array_extend.shape[0]))

ccs = np.zeros((theta_N, num_firms_array.shape[0]))
ccs_per_bw = np.zeros((theta_N, num_firms_array.shape[0]))
avg_path_losses = np.zeros((theta_N, num_firms_array.shape[0]))
avg_SINR = np.zeros((theta_N, num_firms_array.shape[0]))

full_elasts = np.zeros((theta_N, num_firms_array.shape[0], num_prods))
partial_elasts = np.zeros((theta_N, num_firms_array.shape[0], num_prods))

partial_Pif_partial_bf_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
partial_Pif_partial_b_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
partial_CS_partial_b_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
partial_Pif_partial_bf_allbw = np.zeros((theta_N, num_firms_array.shape[0]))
partial_Pif_partial_b_allbw = np.zeros((theta_N, num_firms_array.shape[0]))
partial_CS_partial_b_allbw = np.zeros((theta_N, num_firms_array.shape[0]))

c_u = np.zeros((theta_N, ds.J))
c_R = np.zeros((theta_N, radius.shape[0], radius.shape[1]))

p_stars_shortrun = np.zeros((theta_N, 3, num_prods))
R_stars_shortrun = np.zeros((theta_N, 3))
num_stations_stars_shortrun = np.zeros((theta_N, 3))
num_stations_per_firm_stars_shortrun = np.zeros((theta_N, 3))
q_stars_shortrun = np.zeros((theta_N, 3))
cs_by_type_shortrun = np.zeros((theta_N, 3, yclastidx - yc1idx + 1))
cs_shortrun = np.zeros((theta_N, 3))
ps_shortrun = np.zeros((theta_N, 3))
ts_shortrun = np.zeros((theta_N, 3))
ccs_shortrun = np.zeros((theta_N, 3))
ccs_per_bw_shortrun = np.zeros((theta_N, 3))
avg_path_losses_shortrun = np.zeros((theta_N, 3))

p_stars_free_allfixed = np.zeros((theta_N, 2, num_prods))
R_stars_free_allfixed = np.zeros((theta_N, 2))
num_stations_stars_free_allfixed = np.zeros((theta_N, 2))
num_stations_per_firm_stars_free_allfixed = np.zeros((theta_N, 2))
q_stars_free_allfixed = np.zeros((theta_N, 2))
cs_by_type_free_allfixed = np.zeros((theta_N, 2, yclastidx - yc1idx + 1))
cs_free_allfixed = np.zeros((theta_N, 2))
ps_free_allfixed = np.zeros((theta_N, 2))
ts_free_allfixed = np.zeros((theta_N, 2))
ccs_free_allfixed = np.zeros((theta_N, 2))
ccs_per_bw_free_allfixed = np.zeros((theta_N, 2))
avg_path_losses_free_allfixed = np.zeros((theta_N, 2))
p_stars_free_allbw = np.zeros((theta_N, 2, num_prods))
R_stars_free_allbw = np.zeros((theta_N, 2))
num_stations_stars_free_allbw = np.zeros((theta_N, 2))
num_stations_per_firm_stars_free_allbw = np.zeros((theta_N, 2))
q_stars_free_allbw = np.zeros((theta_N, 2))
cs_by_type_free_allbw = np.zeros((theta_N, 2, yclastidx - yc1idx + 1))
cs_free_allbw = np.zeros((theta_N, 2))
ps_free_allbw = np.zeros((theta_N, 2))
ts_free_allbw = np.zeros((theta_N, 2))
ccs_free_allbw = np.zeros((theta_N, 2))
ccs_per_bw_free_allbw = np.zeros((theta_N, 2))
avg_path_losses_free_allbw = np.zeros((theta_N, 2))

p_stars_dens = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0], num_prods))
R_stars_dens = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
num_stations_stars_dens = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
num_stations_per_firm_stars_dens = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
q_stars_dens = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
cs_dens = np.zeros((theta_N, num_firms_array_extend.shape[0], densities.shape[0]))
cs_by_type_dens = np.zeros((theta_N, num_firms_array_extend.shape[0], densities.shape[0], yclastidx - yc1idx + 1))
ps_dens = np.zeros((theta_N, num_firms_array_extend.shape[0], densities.shape[0]))
ts_dens = np.zeros((theta_N, num_firms_array_extend.shape[0], densities.shape[0]))
ccs_dens = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
ccs_per_bw_dens = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
avg_path_losses_dens = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
avg_SINR_dens = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))

p_stars_bw = np.zeros((theta_N, num_firms_array.shape[0], bw_vals.shape[0], num_prods))
R_stars_bw = np.zeros((theta_N, num_firms_array.shape[0], bw_vals.shape[0]))
num_stations_stars_bw = np.zeros((theta_N, num_firms_array.shape[0], bw_vals.shape[0]))
num_stations_per_firm_stars_bw = np.zeros((theta_N, num_firms_array.shape[0], bw_vals.shape[0]))
q_stars_bw = np.zeros((theta_N, num_firms_array.shape[0], bw_vals.shape[0]))
cs_bw = np.zeros((theta_N, num_firms_array_extend.shape[0], bw_vals.shape[0]))
cs_by_type_bw = np.zeros((theta_N, num_firms_array_extend.shape[0], bw_vals.shape[0], yclastidx - yc1idx + 1))
ps_bw = np.zeros((theta_N, num_firms_array_extend.shape[0], bw_vals.shape[0]))
ts_bw = np.zeros((theta_N, num_firms_array_extend.shape[0], bw_vals.shape[0]))
ccs_bw = np.zeros((theta_N, num_firms_array.shape[0], bw_vals.shape[0]))
ccs_per_bw_bw = np.zeros((theta_N, num_firms_array.shape[0], bw_vals.shape[0]))
avg_path_losses_bw = np.zeros((theta_N, num_firms_array.shape[0], bw_vals.shape[0]))
avg_SINR_bw = np.zeros((theta_N, num_firms_array.shape[0], bw_vals.shape[0]))

p_stars_dens_1p = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0], 1))
R_stars_dens_1p = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
num_stations_stars_dens_1p = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
num_stations_per_firm_stars_dens_1p = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
q_stars_dens_1p = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
cs_dens_1p = np.zeros((theta_N, num_firms_array_extend.shape[0], densities.shape[0]))
cs_by_type_dens_1p = np.zeros((theta_N, num_firms_array_extend.shape[0], densities.shape[0], yclastidx - yc1idx + 1))
ps_dens_1p = np.zeros((theta_N, num_firms_array_extend.shape[0], densities.shape[0]))
ts_dens_1p = np.zeros((theta_N, num_firms_array_extend.shape[0], densities.shape[0]))
ccs_dens_1p = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
ccs_per_bw_dens_1p = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
avg_path_losses_dens_1p = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))
avg_SINR_dens_1p = np.zeros((theta_N, num_firms_array.shape[0], densities.shape[0]))

successful_extend = np.ones((theta_N, num_firms_array_extend.shape[0]), dtype=bool)
successful_bw_deriv_allfixed = np.ones((theta_N, num_firms_array.shape[0]), dtype=bool)
successful_bw_deriv_allbw = np.ones((theta_N, num_firms_array.shape[0]), dtype=bool)
successful_shortrun = np.ones((theta_N, 3), dtype=bool)
successful_free_allfixed = np.ones((theta_N, 2), dtype=bool)
successful_free_allbw = np.ones((theta_N, 2), dtype=bool)
successful_dens = np.ones((theta_N, num_firms_array_extend.shape[0], densities.shape[0]), dtype=bool)
successful_bw = np.ones((theta_N, num_firms_array_extend.shape[0], bw_vals.shape[0]), dtype=bool)
successful_dens_1p = np.ones((theta_N, num_firms_array_extend.shape[0], densities.shape[0]), dtype=bool)

per_user_costs = np.zeros((theta_N, num_prods))

def compute_equilibria_adjust(theta_n):
    """Provide starting values for computing equilibria based on results from the point estimate."""
    
    p_starting_vals = None
    R_starting_vals = None
        
    return compute_equilibria(theta_n, p_starting_vals=p_starting_vals, R_starting_vals=R_starting_vals)

# Initialize multiprocessing
pool = Pool(num_cpus)
chunksize = 1

# Compute perturbed demand parameter estimate equilibria and store relevant variables
for ind, res in enumerate(pool.imap(compute_equilibria_adjust, range(theta_N)), chunksize):
    idx = ind - chunksize
    p_stars[idx,:,:] = res[0]
    R_stars[idx,:] = res[1]
    num_stations_stars[idx,:] = res[2]
    num_stations_per_firm_stars[idx,:] = res[3]
    q_stars[idx,:] = res[4]
    cs_by_type[idx,:,:] = res[5]
    cs[idx,:] = res[6]
    ps[idx,:] = res[7]
    ts[idx,:] = res[8]
    ccs[idx,:] = res[9]
    ccs_per_bw[idx,:] = res[10]
    avg_path_losses[idx,:] = res[11]
    avg_SINR[idx,:] = res[12]
    full_elasts[idx,:,:] = res[13]
    partial_elasts[idx,:,:] = res[14]
    partial_Pif_partial_bf_allfixed[idx,:] = res[15]
    partial_Pif_partial_b_allfixed[idx,:] = res[16]
    partial_CS_partial_b_allfixed[idx,:] = res[17]
    partial_Pif_partial_bf_allbw[idx,:] = res[18]
    partial_Pif_partial_b_allbw[idx,:] = res[19]
    partial_CS_partial_b_allbw[idx,:] = res[20]
    c_u[idx,:] = res[21]
    c_R[idx,:,:] = res[22]
    p_stars_shortrun[idx,:,:] = res[23]
    R_stars_shortrun[idx,:] = res[24]
    num_stations_stars_shortrun[idx,:] = res[25]
    num_stations_per_firm_stars_shortrun[idx,:] = res[26]
    q_stars_shortrun[idx,:] = res[27]
    cs_by_type_shortrun[idx,:,:] = res[28]
    cs_shortrun[idx,:] = res[29]
    ps_shortrun[idx,:] = res[30]
    ts_shortrun[idx,:] = res[31]
    ccs_shortrun[idx,:] = res[32]
    ccs_per_bw_shortrun[idx,:] = res[33]
    avg_path_losses_shortrun[idx,:] = res[34]
    p_stars_free_allfixed[idx,:,:] = res[35]
    R_stars_free_allfixed[idx,:] = res[36]
    num_stations_stars_free_allfixed[idx,:] = res[37]
    num_stations_per_firm_stars_free_allfixed[idx,:] = res[38]
    q_stars_free_allfixed[idx,:] = res[39]
    cs_by_type_free_allfixed[idx,:,:] = res[40]
    cs_free_allfixed[idx,:] = res[41]
    ps_free_allfixed[idx,:] = res[42]
    ts_free_allfixed[idx,:] = res[43]
    ccs_free_allfixed[idx,:] = res[44]
    ccs_per_bw_free_allfixed[idx,:] = res[45]
    avg_path_losses_free_allfixed[idx,:] = res[46]
    p_stars_free_allbw[idx,:,:] = res[47]
    R_stars_free_allbw[idx,:] = res[48]
    num_stations_stars_free_allbw[idx,:] = res[49]
    num_stations_per_firm_stars_free_allbw[idx,:] = res[50]
    q_stars_free_allbw[idx,:] = res[51]
    cs_by_type_free_allbw[idx,:,:] = res[52]
    cs_free_allbw[idx,:] = res[53]
    ps_free_allbw[idx,:] = res[54]
    ts_free_allbw[idx,:] = res[55]
    ccs_free_allbw[idx,:] = res[56]
    ccs_per_bw_free_allbw[idx,:] = res[57]
    avg_path_losses_free_allbw[idx,:] = res[58]

    p_stars_dens[idx,:,:,:] = res[59]
    R_stars_dens[idx,:,:] = res[60]
    num_stations_stars_dens[idx,:,:] = res[61]
    num_stations_per_firm_stars_dens[idx,:,:] = res[62]
    q_stars_dens[idx,:,:] = res[63]
    cs_dens[idx,:,:] = res[64]
    cs_by_type_dens[idx,:,:,:] = res[65]
    ps_dens[idx,:,:] = res[66]
    ts_dens[idx,:,:] = res[67]
    ccs_dens[idx,:,:] = res[68]
    ccs_per_bw_dens[idx,:,:] = res[69]
    avg_path_losses_dens[idx,:,:] = res[70]
    avg_SINR_dens[idx,:,:] = res[71]

    p_stars_bw[idx,:,:,:] = res[72]
    R_stars_bw[idx,:,:] = res[73]
    num_stations_stars_bw[idx,:,:] = res[74]
    num_stations_per_firm_stars_bw[idx,:,:] = res[75]
    q_stars_bw[idx,:,:] = res[76]
    cs_bw[idx,:,:] = res[77]
    cs_by_type_bw[idx,:,:,:] = res[78]
    ps_bw[idx,:,:] = res[79]
    ts_bw[idx,:,:] = res[80]
    ccs_bw[idx,:,:] = res[81]
    ccs_per_bw_bw[idx,:,:] = res[82]
    avg_path_losses_bw[idx,:,:] = res[83]
    avg_SINR_bw[idx,:,:] = res[84]
    
    p_stars_dens_1p[idx,:,:,:] = res[85]
    R_stars_dens_1p[idx,:,:] = res[86]
    num_stations_stars_dens_1p[idx,:,:] = res[87]
    num_stations_per_firm_stars_dens_1p[idx,:,:] = res[88]
    q_stars_dens_1p[idx,:,:] = res[89]
    cs_dens_1p[idx,:,:] = res[90]
    cs_by_type_dens_1p[idx,:,:,:] = res[91]
    ps_dens_1p[idx,:,:] = res[92]
    ts_dens_1p[idx,:,:] = res[93]
    ccs_dens_1p[idx,:,:] = res[94]
    ccs_per_bw_dens_1p[idx,:,:] = res[95]
    avg_path_losses_dens_1p[idx,:,:] = res[96]
    avg_SINR_dens_1p[idx,:,:] = res[97]

    successful_extend[idx,:] = res[98]
    successful_bw_deriv_allfixed[idx,:] = res[99]
    successful_bw_deriv_allbw[idx,:] = res[100]
    successful_shortrun[idx,:] = res[101]
    successful_free_allfixed[idx,:] = res[102]
    successful_free_allbw[idx,:] = res[103]
    successful_dens[idx,:,:] = res[104]
    successful_bw[idx,:,:] = res[105]
    successful_dens_1p[idx,:,:] = res[106]
    
    per_user_costs[idx,:] = res[107]
    
pool.close()

# %%
# Save arrays before processing
if save_bf:
    np.savez_compressed(f"{paths.arrays_path}all_arrays_e{elast_id}_n{nest_id}.npz", p_stars, R_stars, num_stations_stars, num_stations_per_firm_stars, q_stars, cs_by_type, cs, ps, ts, ccs, avg_path_losses, avg_SINR, full_elasts, partial_elasts, partial_Pif_partial_bf_allfixed, partial_Pif_partial_b_allfixed, partial_CS_partial_b_allfixed, partial_Pif_partial_bf_allbw, partial_Pif_partial_b_allbw, partial_CS_partial_b_allbw, c_u, c_R, p_stars_shortrun, R_stars_shortrun, num_stations_stars_shortrun, num_stations_per_firm_stars_shortrun, q_stars_shortrun, cs_by_type_shortrun, cs_shortrun, ps_shortrun, ts_shortrun, ccs_shortrun, avg_path_losses_shortrun, p_stars_free_allfixed, R_stars_free_allfixed, num_stations_stars_free_allfixed, num_stations_per_firm_stars_free_allfixed, q_stars_free_allfixed, cs_by_type_free_allfixed, cs_free_allfixed, ps_free_allfixed, ts_free_allfixed, ccs_free_allfixed, avg_path_losses_free_allfixed, p_stars_free_allbw, R_stars_free_allbw, num_stations_stars_free_allbw, num_stations_per_firm_stars_free_allbw, q_stars_free_allbw, cs_by_type_free_allbw, cs_free_allbw, ps_free_allbw, ts_free_allbw, ccs_free_allbw, avg_path_losses_free_allbw, p_stars_dens, R_stars_dens, num_stations_stars_dens, num_stations_per_firm_stars_dens, q_stars_dens, cs_dens, cs_by_type_dens, ps_dens, ts_dens, ccs_dens, avg_path_losses_dens, avg_SINR_dens, p_stars_bw, R_stars_bw, num_stations_stars_bw, num_stations_per_firm_stars_bw, q_stars_bw, cs_bw, cs_by_type_bw, ps_bw, ts_bw, ccs_bw, avg_path_losses_bw, avg_SINR_bw, p_stars_dens_1p, R_stars_dens_1p, num_stations_stars_dens_1p, num_stations_per_firm_stars_dens_1p, q_stars_dens_1p, cs_dens_1p, cs_by_type_dens_1p, ps_dens_1p, ts_dens_1p, ccs_dens_1p, avg_path_losses_dens_1p, avg_SINR_dens_1p, successful_extend, successful_bw_deriv_allfixed, successful_bw_deriv_allbw, successful_shortrun, successful_free_allfixed, successful_free_allbw, successful_dens, successful_bw, successful_dens_1p, per_user_costs)

# %%
# Determine point estimates and standard errors

def asym_distribution(var, success):
    """Determine the point estimate and standard errors given demand parameter using the Delta Method."""
    
    # Reshape if var and success if success has more than two dimensions (b/c code below written for two dimensions)
    var_shape = var.shape
    success_shape = success.shape
    reshape_necessary = success.ndim > 2
    if reshape_necessary:
        success_combined_2nd_axis = np.prod(np.array(list(success_shape))[1:])
        success_reshape = (success_shape[0], success_combined_2nd_axis)
        var_reshape = tuple(list(success_reshape) + list(var_shape)[success.ndim:])
        var = np.reshape(var, var_reshape)
        success = np.reshape(success, success_reshape)
        
    # Determine the point estimate
    hB = var[0,...]
    
    # Compute standard errors
    if compute_std_errs:
        # Determine the gradient
        theta_size = int((var.shape[0] - 1) / 2)
        grad_hB = np.zeros(tuple([theta_size] + list(var.shape[1:])))
        for i in range(theta_size):
            for j in range(var.shape[1]):
                # Determine if there are any "bad" simulations (sometimes we get a crazy value, esp. for monopoly, but calc was "successful" - drop these)
                max_prop = 5.0 # max difference between the two that results in both being "accepted"
                if np.any(np.abs(var[1 + i,j,...] - var[0,j,...]) > max_prop * np.abs(var[1 + theta_size + i,j,...] - var[0,j])):
                    success[1 + i,j] = False
                if np.any(np.abs(var[1 + theta_size + i,j,...] - var[0,j]) > max_prop * np.abs(var[1 + i,j,...] - var[0,j,...])):
                    success[1 + theta_size + i,j] = False
                
                # Construct gradients
                if success[1 + i,j] and success[1 + theta_size + i,j]: # if both sides were successful
                    grad_hB[i,j,...] = (var[1 + i,j,...] - var[1 + theta_size + i,j,...]) / (2. * eps_grad)
                elif success[1 + i,j] and success[0,j]: # elif upper side and point estimate were successful
                    grad_hB[i,j,...] = (var[1 + i,j,...] - var[0,j,...]) / eps_grad
                elif success[1 + theta_size + i,j] and success[0,j]: # elif lower side and point estimate were successful
                    grad_hB[i,j,...] = (var[0,j,...] - var[1 + theta_size + i,j,...]) / eps_grad
                elif success[0,j]: # elif point estimate success but perturbations not
                    grad_hB[i,j,...] = np.nan
                else: # o/w nothing was successful
                    grad_hB[i,j,...] = np.nan
                    hB[j,...] = np.nan
                    
        # Reshape back if necessary
        if reshape_necessary:
            grad_hB = np.reshape(grad_hB, tuple([theta_size] + list(var_shape)[1:]))
            hB = np.reshape(hB, tuple(list(var_shape)[1:]))

        # Calculate (approximate) standard errors
        grad_hB = np.moveaxis(grad_hB, 0, -1) # move the gradient axis to the end for matrix operations
        hB_normal_asymvar = (grad_hB[...,np.newaxis,:] @ Sigma @ grad_hB[...,np.newaxis])[...,0,0] # multivariate Delta Method
        hB_se = np.sqrt(hB_normal_asymvar / float(N)) # determine the standard errors

        return hB, hB_se
    
    else:
        return hB, np.zeros(hB.shape)

successful = successful_extend[:,np.isin(num_firms_array_extend, num_firms_array)]
p_stars, p_stars_se = asym_distribution(p_stars, successful)
R_stars, R_stars_se = asym_distribution(R_stars, successful)
num_stations_stars, num_stations_stars_se = asym_distribution(num_stations_stars, successful)
num_stations_per_firm_stars, num_stations_per_firm_stars_se = asym_distribution(num_stations_per_firm_stars, successful)
q_stars, q_stars_se = asym_distribution(q_stars, successful)
cs_by_type, cs_by_type_se = asym_distribution(cs_by_type, successful_extend)
cs, cs_se = asym_distribution(cs, successful_extend)
ps, ps_se = asym_distribution(ps, successful_extend)
ts, ts_se = asym_distribution(ts, successful_extend)
ccs, ccs_se = asym_distribution(ccs, successful)
ccs_per_bw, ccs_per_bw_se = asym_distribution(ccs_per_bw, successful)
avg_path_losses, avg_path_losses_se = asym_distribution(avg_path_losses, successful)
avg_SINR, avg_SINR_se = asym_distribution(avg_SINR, successful)
full_elasts, full_elasts_se = asym_distribution(full_elasts, successful)
partial_elasts, partial_elasts_se = asym_distribution(partial_elasts, successful)
partial_Pif_partial_bf_allfixed, partial_Pif_partial_bf_allfixed_se = asym_distribution(partial_Pif_partial_bf_allfixed, successful_bw_deriv_allfixed)
partial_Pif_partial_b_allfixed, partial_Pif_partial_b_allfixed_se = asym_distribution(partial_Pif_partial_b_allfixed, successful_bw_deriv_allfixed)
partial_CS_partial_b_allfixed, partial_CS_partial_b_allfixed_se = asym_distribution(partial_CS_partial_b_allfixed, successful_bw_deriv_allfixed)
partial_Pif_partial_bf_allbw, partial_Pif_partial_bf_allbw_se = asym_distribution(partial_Pif_partial_bf_allbw, successful_bw_deriv_allbw)
partial_Pif_partial_b_allbw, partial_Pif_partial_b_allbw_se = asym_distribution(partial_Pif_partial_b_allbw, successful_bw_deriv_allbw)
partial_CS_partial_b_allbw, partial_CS_partial_b_allbw_se = asym_distribution(partial_CS_partial_b_allbw, successful_bw_deriv_allbw)
c_u, c_u_se = asym_distribution(c_u, np.ones((c_u.shape[0], c_u.shape[1]), dtype=bool)) # all should be successful
c_R, c_R_se = asym_distribution(c_R, np.ones((c_R.shape[0], c_R.shape[1]), dtype=bool)) # all should be successful
p_stars_shortrun, p_stars_shortrun_se = asym_distribution(p_stars_shortrun, successful_shortrun)
R_stars_shortrun, R_stars_shortrun_se = asym_distribution(R_stars_shortrun, successful_shortrun)
num_stations_stars_shortrun, num_stations_stars_shortrun_se = asym_distribution(num_stations_stars_shortrun, successful_shortrun)
num_stations_per_firm_stars_shortrun, num_stations_per_firm_stars_shortrun_se = asym_distribution(num_stations_per_firm_stars_shortrun, successful_shortrun)
q_stars_shortrun, q_stars_shortrun_se = asym_distribution(q_stars_shortrun, successful_shortrun)
cs_by_type_shortrun, cs_by_type_shortrun_se = asym_distribution(cs_by_type_shortrun, successful_shortrun)
cs_shortrun, cs_shortrun_se = asym_distribution(cs_shortrun, successful_shortrun)
ps_shortrun, ps_shortrun_se = asym_distribution(ps_shortrun, successful_shortrun)
ts_shortrun, ts_shortrun_se = asym_distribution(ts_shortrun, successful_shortrun)
ccs_shortrun, ccs_shortrun_se = asym_distribution(ccs_shortrun, successful_shortrun)
ccs_per_bw_shortrun, ccs_per_bw_shortrun_se = asym_distribution(ccs_per_bw_shortrun, successful_shortrun)
avg_path_losses_shortrun, avg_path_losses_shortrun_se = asym_distribution(avg_path_losses_shortrun, successful_shortrun)
p_stars_free_allfixed, p_stars_free_allfixed_se = asym_distribution(p_stars_free_allfixed, successful_free_allfixed)
R_stars_free_allfixed, R_stars_free_allfixed_se = asym_distribution(R_stars_free_allfixed, successful_free_allfixed)
num_stations_stars_free_allfixed, num_stations_stars_free_allfixed_se = asym_distribution(num_stations_stars_free_allfixed, successful_free_allfixed)
num_stations_per_firm_stars_free_allfixed, num_stations_per_firm_stars_free_allfixed_se = asym_distribution(num_stations_per_firm_stars_free_allfixed, successful_free_allfixed)
q_stars_free_allfixed, q_stars_free_allfixed_se = asym_distribution(q_stars_free_allfixed, successful_free_allfixed)
cs_by_type_free_allfixed, cs_by_type_free_allfixed_se = asym_distribution(cs_by_type_free_allfixed, successful_free_allfixed)
cs_free_allfixed, cs_free_allfixed_se = asym_distribution(cs_free_allfixed, successful_free_allfixed)
ps_free_allfixed, ps_free_allfixed_se = asym_distribution(ps_free_allfixed, successful_free_allfixed)
ts_free_allfixed, ts_free_allfixed_se = asym_distribution(ts_free_allfixed, successful_free_allfixed)
ccs_free_allfixed, ccs_free_allfixed_se = asym_distribution(ccs_free_allfixed, successful_free_allfixed)
ccs_per_bw_free_allfixed, ccs_per_bw_free_allfixed_se = asym_distribution(ccs_per_bw_free_allfixed, successful_free_allfixed)
avg_path_losses_free_allfixed, avg_path_losses_free_allfixed_se = asym_distribution(avg_path_losses_free_allfixed, successful_free_allfixed)
p_stars_free_allbw, p_stars_free_allbw_se = asym_distribution(p_stars_free_allbw, successful_free_allbw)
R_stars_free_allbw, R_stars_free_allbw_se = asym_distribution(R_stars_free_allbw, successful_free_allbw)
num_stations_stars_free_allbw, num_stations_stars_free_allbw_se = asym_distribution(num_stations_stars_free_allbw, successful_free_allbw)
num_stations_per_firm_stars_free_allbw, num_stations_per_firm_stars_free_allbw_se = asym_distribution(num_stations_per_firm_stars_free_allbw, successful_free_allbw)
q_stars_free_allbw, q_stars_free_allbw_se = asym_distribution(q_stars_free_allbw, successful_free_allbw)
cs_by_type_free_allbw, cs_by_type_free_allbw_se = asym_distribution(cs_by_type_free_allbw, successful_free_allbw)
cs_free_allbw, cs_free_allbw_se = asym_distribution(cs_free_allbw, successful_free_allbw)
ps_free_allbw, ps_free_allbw_se = asym_distribution(ps_free_allbw, successful_free_allbw)
ts_free_allbw, ts_free_allbw_se = asym_distribution(ts_free_allbw, successful_free_allbw)
ccs_free_allbw, ccs_free_allbw_se = asym_distribution(ccs_free_allbw, successful_free_allbw)
ccs_per_bw_free_allbw, ccs_per_bw_free_allbw_se = asym_distribution(ccs_per_bw_free_allbw, successful_free_allbw)
avg_path_losses_free_allbw, avg_path_losses_free_allbw_se = asym_distribution(avg_path_losses_free_allbw, successful_free_allbw)
successful_dens_ = successful_dens[:,np.isin(num_firms_array_extend, num_firms_array)]
p_stars_dens, p_stars_dens_se = asym_distribution(p_stars_dens, successful_dens_)
R_stars_dens, R_stars_dens_se = asym_distribution(R_stars_dens, successful_dens_)
num_stations_stars_dens, num_stations_stars_dens_se = asym_distribution(num_stations_stars_dens, successful_dens_)
num_stations_per_firm_stars_dens, num_stations_per_firm_stars_dens_se = asym_distribution(num_stations_per_firm_stars_dens, successful_dens_)
q_stars_dens, q_stars_dens_se = asym_distribution(q_stars_dens, successful_dens_)
cs_by_type_dens, cs_by_type_dens_se = asym_distribution(cs_by_type_dens, successful_dens)
cs_dens, cs_dens_se = asym_distribution(cs_dens, successful_dens)
ps_dens, ps_dens_se = asym_distribution(ps_dens, successful_dens)
ts_dens, ts_dens_se = asym_distribution(ts_dens, successful_dens)
ccs_dens, ccs_dens_se = asym_distribution(ccs_dens, successful_dens_)
ccs_per_bw_dens, ccs_per_bw_dens_se = asym_distribution(ccs_per_bw_dens, successful_dens_)
avg_path_losses_dens, avg_path_losses_dens_se = asym_distribution(avg_path_losses_dens, successful_dens_)
avg_SINR_dens, avg_SINR_dens_se = asym_distribution(avg_SINR_dens, successful_dens_)
successful_bw_ = successful_bw[:,np.isin(num_firms_array_extend, num_firms_array)]
p_stars_bw, p_stars_bw_se = asym_distribution(p_stars_bw, successful_bw_)
R_stars_bw, R_stars_bw_se = asym_distribution(R_stars_bw, successful_bw_)
num_stations_stars_bw, num_stations_stars_bw_se = asym_distribution(num_stations_stars_bw, successful_bw_)
num_stations_per_firm_stars_bw, num_stations_per_firm_stars_bw_se = asym_distribution(num_stations_per_firm_stars_bw, successful_bw_)
q_stars_bw, q_stars_bw_se = asym_distribution(q_stars_bw, successful_bw_)
cs_by_type_bw, cs_by_type_bw_se = asym_distribution(cs_by_type_bw, successful_bw)
cs_bw, cs_bw_se = asym_distribution(cs_bw, successful_bw)
ps_bw, ps_bw_se = asym_distribution(ps_bw, successful_bw)
ts_bw, ts_bw_se = asym_distribution(ts_bw, successful_bw)
ccs_bw, ccs_bw_se = asym_distribution(ccs_bw, successful_bw_)
ccs_per_bw_bw, ccs_per_bw_bw_se = asym_distribution(ccs_per_bw_bw, successful_bw_)
avg_path_losses_bw, avg_path_losses_bw_se = asym_distribution(avg_path_losses_bw, successful_bw_)
avg_SINR_bw, avg_SINR_bw_se = asym_distribution(avg_SINR_bw, successful_bw_)
successful_dens_1p_ = successful_dens_1p[:,np.isin(num_firms_array_extend, num_firms_array)]
p_stars_dens_1p, p_stars_dens_1p_se = asym_distribution(p_stars_dens_1p, successful_dens_1p_)
R_stars_dens_1p, R_stars_dens_1p_se = asym_distribution(R_stars_dens_1p, successful_dens_1p_)
num_stations_stars_dens_1p, num_stations_stars_dens_1p_se = asym_distribution(num_stations_stars_dens_1p, successful_dens_1p_)
num_stations_per_firm_stars_dens_1p, num_stations_per_firm_stars_dens_1p_se = asym_distribution(num_stations_per_firm_stars_dens_1p, successful_dens_1p_)
q_stars_dens_1p, q_stars_dens_1p_se = asym_distribution(q_stars_dens_1p, successful_dens_1p_)
cs_by_type_dens_1p, cs_by_type_dens_1p_se = asym_distribution(cs_by_type_dens_1p, successful_dens_1p)
cs_dens_1p, cs_dens_1p_se = asym_distribution(cs_dens_1p, successful_dens_1p)
ps_dens_1p, ps_dens_1p_se = asym_distribution(ps_dens_1p, successful_dens_1p)
ts_dens_1p, ts_dens_1p_se = asym_distribution(ts_dens_1p, successful_dens_1p)
ccs_dens_1p, ccs_dens_1p_se = asym_distribution(ccs_dens_1p, successful_dens_1p_)
ccs_per_bw_dens_1p, ccs_per_bw_dens_1p_se = asym_distribution(ccs_per_bw_dens_1p, successful_dens_1p_)
avg_path_losses_dens_1p, avg_path_losses_dens_1p_se = asym_distribution(avg_path_losses_dens_1p, successful_dens_1p_)
avg_SINR_dens_1p, avg_SINR_dens_1p_se = asym_distribution(avg_SINR_dens_1p, successful_dens_1p_)
per_user_costs, per_user_costs_se = asym_distribution(per_user_costs, np.ones((per_user_costs.shape[0], per_user_costs.shape[1]), dtype=bool)) # all should be successful
    
# %%
# Save variables

# Point estimates
np.save(f"{paths.arrays_path}p_stars_e{elast_id}_n{nest_id}.npy", p_stars)
np.save(f"{paths.arrays_path}R_stars_e{elast_id}_n{nest_id}.npy", R_stars)
np.save(f"{paths.arrays_path}num_stations_stars_e{elast_id}_n{nest_id}.npy", num_stations_stars)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars)
np.save(f"{paths.arrays_path}q_stars_e{elast_id}_n{nest_id}.npy", q_stars)
np.save(f"{paths.arrays_path}cs_by_type_e{elast_id}_n{nest_id}.npy", cs_by_type)
np.save(f"{paths.arrays_path}cs_e{elast_id}_n{nest_id}.npy", cs)
np.save(f"{paths.arrays_path}ps_e{elast_id}_n{nest_id}.npy", ps)
np.save(f"{paths.arrays_path}ts_e{elast_id}_n{nest_id}.npy", ts)
np.save(f"{paths.arrays_path}ccs_e{elast_id}_n{nest_id}.npy", ccs)
np.save(f"{paths.arrays_path}ccs_per_bw_e{elast_id}_n{nest_id}.npy", ccs_per_bw)
np.save(f"{paths.arrays_path}avg_path_losses_e{elast_id}_n{nest_id}.npy", avg_path_losses)
np.save(f"{paths.arrays_path}avg_SINR_e{elast_id}_n{nest_id}.npy", avg_SINR)
np.save(f"{paths.arrays_path}full_elasts_e{elast_id}_n{nest_id}.npy", full_elasts)
np.save(f"{paths.arrays_path}partial_elasts_e{elast_id}_n{nest_id}.npy", partial_elasts)
np.save(f"{paths.arrays_path}partial_Pif_partial_bf_allfixed_e{elast_id}_n{nest_id}.npy", partial_Pif_partial_bf_allfixed)
np.save(f"{paths.arrays_path}partial_Pif_partial_b_allfixed_e{elast_id}_n{nest_id}.npy", partial_Pif_partial_b_allfixed)
np.save(f"{paths.arrays_path}partial_CS_partial_b_allfixed_e{elast_id}_n{nest_id}.npy", partial_CS_partial_b_allfixed)
np.save(f"{paths.arrays_path}partial_Pif_partial_bf_allbw_e{elast_id}_n{nest_id}.npy", partial_Pif_partial_bf_allbw)
np.save(f"{paths.arrays_path}partial_Pif_partial_b_allbw_e{elast_id}_n{nest_id}.npy", partial_Pif_partial_b_allbw)
np.save(f"{paths.arrays_path}partial_CS_partial_b_allbw_e{elast_id}_n{nest_id}.npy", partial_CS_partial_b_allbw)
np.save(f"{paths.arrays_path}c_u_e{elast_id}_n{nest_id}.npy", c_u)
np.save(f"{paths.arrays_path}c_R_e{elast_id}_n{nest_id}.npy", c_R)
np.save(f"{paths.arrays_path}p_stars_shortrun_e{elast_id}_n{nest_id}.npy", p_stars_shortrun)
np.save(f"{paths.arrays_path}R_stars_shortrun_e{elast_id}_n{nest_id}.npy", R_stars_shortrun)
np.save(f"{paths.arrays_path}num_stations_stars_shortrun_e{elast_id}_n{nest_id}.npy", num_stations_stars_shortrun)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_shortrun_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_shortrun)
np.save(f"{paths.arrays_path}q_stars_shortrun_e{elast_id}_n{nest_id}.npy", q_stars_shortrun)
np.save(f"{paths.arrays_path}cs_by_type_shortrun_e{elast_id}_n{nest_id}.npy", cs_by_type_shortrun)
np.save(f"{paths.arrays_path}cs_shortrun_e{elast_id}_n{nest_id}.npy", cs_shortrun)
np.save(f"{paths.arrays_path}ps_shortrun_e{elast_id}_n{nest_id}.npy", ps_shortrun)
np.save(f"{paths.arrays_path}ts_shortrun_e{elast_id}_n{nest_id}.npy", ts_shortrun)
np.save(f"{paths.arrays_path}ccs_shortrun_e{elast_id}_n{nest_id}.npy", ccs_shortrun)
np.save(f"{paths.arrays_path}ccs_per_bw_shortrun_e{elast_id}_n{nest_id}.npy", ccs_per_bw_shortrun)
np.save(f"{paths.arrays_path}avg_path_losses_shortrun_e{elast_id}_n{nest_id}.npy", avg_path_losses_shortrun)
np.save(f"{paths.arrays_path}p_stars_free_allfixed_e{elast_id}_n{nest_id}.npy", p_stars_free_allfixed)
np.save(f"{paths.arrays_path}R_stars_free_allfixed_e{elast_id}_n{nest_id}.npy", R_stars_free_allfixed)
np.save(f"{paths.arrays_path}num_stations_stars_free_allfixed_e{elast_id}_n{nest_id}.npy", num_stations_stars_free_allfixed)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_free_allfixed_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_free_allfixed)
np.save(f"{paths.arrays_path}q_stars_free_allfixed_e{elast_id}_n{nest_id}.npy", q_stars_free_allfixed)
np.save(f"{paths.arrays_path}cs_by_type_free_allfixed_e{elast_id}_n{nest_id}.npy", cs_by_type_free_allfixed)
np.save(f"{paths.arrays_path}cs_free_allfixed_e{elast_id}_n{nest_id}.npy", cs_free_allfixed)
np.save(f"{paths.arrays_path}ps_free_allfixed_e{elast_id}_n{nest_id}.npy", ps_free_allfixed)
np.save(f"{paths.arrays_path}ts_free_allfixed_e{elast_id}_n{nest_id}.npy", ts_free_allfixed)
np.save(f"{paths.arrays_path}ccs_free_allfixed_e{elast_id}_n{nest_id}.npy", ccs_free_allfixed)
np.save(f"{paths.arrays_path}ccs_per_bw_free_allfixed_e{elast_id}_n{nest_id}.npy", ccs_per_bw_free_allfixed)
np.save(f"{paths.arrays_path}avg_path_losses_free_allfixed_e{elast_id}_n{nest_id}.npy", avg_path_losses_free_allfixed)
np.save(f"{paths.arrays_path}p_stars_free_allbw_e{elast_id}_n{nest_id}.npy", p_stars_free_allbw)
np.save(f"{paths.arrays_path}R_stars_free_allbw_e{elast_id}_n{nest_id}.npy", R_stars_free_allbw)
np.save(f"{paths.arrays_path}num_stations_stars_free_allbw_e{elast_id}_n{nest_id}.npy", num_stations_stars_free_allbw)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_free_allbw_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_free_allbw)
np.save(f"{paths.arrays_path}q_stars_free_allbw_e{elast_id}_n{nest_id}.npy", q_stars_free_allbw)
np.save(f"{paths.arrays_path}cs_by_type_free_allbw_e{elast_id}_n{nest_id}.npy", cs_by_type_free_allbw)
np.save(f"{paths.arrays_path}cs_free_allbw_e{elast_id}_n{nest_id}.npy", cs_free_allbw)
np.save(f"{paths.arrays_path}ps_free_allbw_e{elast_id}_n{nest_id}.npy", ps_free_allbw)
np.save(f"{paths.arrays_path}ts_free_allbw_e{elast_id}_n{nest_id}.npy", ts_free_allbw)
np.save(f"{paths.arrays_path}ccs_free_allbw_e{elast_id}_n{nest_id}.npy", ccs_free_allbw)
np.save(f"{paths.arrays_path}ccs_per_bw_free_allbw_e{elast_id}_n{nest_id}.npy", ccs_per_bw_free_allbw)
np.save(f"{paths.arrays_path}avg_path_losses_free_allbw_e{elast_id}_n{nest_id}.npy", avg_path_losses_free_allbw)
np.save(f"{paths.arrays_path}p_stars_dens_e{elast_id}_n{nest_id}.npy", p_stars_dens)
np.save(f"{paths.arrays_path}R_stars_dens_e{elast_id}_n{nest_id}.npy", R_stars_dens)
np.save(f"{paths.arrays_path}num_stations_stars_dens_e{elast_id}_n{nest_id}.npy", num_stations_stars_dens)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_dens_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_dens)
np.save(f"{paths.arrays_path}q_stars_dens_e{elast_id}_n{nest_id}.npy", q_stars_dens)
np.save(f"{paths.arrays_path}cs_by_type_dens_e{elast_id}_n{nest_id}.npy", cs_by_type_dens)
np.save(f"{paths.arrays_path}cs_dens_e{elast_id}_n{nest_id}.npy", cs_dens)
np.save(f"{paths.arrays_path}ps_dens_e{elast_id}_n{nest_id}.npy", ps_dens)
np.save(f"{paths.arrays_path}ts_dens_e{elast_id}_n{nest_id}.npy", ts_dens)
np.save(f"{paths.arrays_path}ccs_dens_e{elast_id}_n{nest_id}.npy", ccs_dens)
np.save(f"{paths.arrays_path}ccs_per_bw_dens_e{elast_id}_n{nest_id}.npy", ccs_per_bw_dens)
np.save(f"{paths.arrays_path}avg_path_losses_dens_e{elast_id}_n{nest_id}.npy", avg_path_losses_dens)
np.save(f"{paths.arrays_path}avg_SINR_dens_e{elast_id}_n{nest_id}.npy", avg_SINR_dens)
np.save(f"{paths.arrays_path}p_stars_bw_e{elast_id}_n{nest_id}.npy", p_stars_bw)
np.save(f"{paths.arrays_path}R_stars_bw_e{elast_id}_n{nest_id}.npy", R_stars_bw)
np.save(f"{paths.arrays_path}num_stations_stars_bw_e{elast_id}_n{nest_id}.npy", num_stations_stars_bw)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_bw_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_bw)
np.save(f"{paths.arrays_path}q_stars_bw_e{elast_id}_n{nest_id}.npy", q_stars_bw)
np.save(f"{paths.arrays_path}cs_by_type_bw_e{elast_id}_n{nest_id}.npy", cs_by_type_bw)
np.save(f"{paths.arrays_path}cs_bw_e{elast_id}_n{nest_id}.npy", cs_bw)
np.save(f"{paths.arrays_path}ps_bw_e{elast_id}_n{nest_id}.npy", ps_bw)
np.save(f"{paths.arrays_path}ts_bw_e{elast_id}_n{nest_id}.npy", ts_bw)
np.save(f"{paths.arrays_path}ccs_bw_e{elast_id}_n{nest_id}.npy", ccs_bw)
np.save(f"{paths.arrays_path}ccs_per_bw_bw_e{elast_id}_n{nest_id}.npy", ccs_per_bw_bw)
np.save(f"{paths.arrays_path}avg_path_losses_bw_e{elast_id}_n{nest_id}.npy", avg_path_losses_bw)
np.save(f"{paths.arrays_path}avg_SINR_bw_e{elast_id}_n{nest_id}.npy", avg_SINR_bw)
np.save(f"{paths.arrays_path}p_stars_dens_1p_e{elast_id}_n{nest_id}.npy", p_stars_dens_1p)
np.save(f"{paths.arrays_path}R_stars_dens_1p_e{elast_id}_n{nest_id}.npy", R_stars_dens_1p)
np.save(f"{paths.arrays_path}num_stations_stars_dens_1p_e{elast_id}_n{nest_id}.npy", num_stations_stars_dens_1p)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_dens_1p_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_dens_1p)
np.save(f"{paths.arrays_path}q_stars_dens_1p_e{elast_id}_n{nest_id}.npy", q_stars_dens_1p)
np.save(f"{paths.arrays_path}cs_by_type_dens_1p_e{elast_id}_n{nest_id}.npy", cs_by_type_dens_1p)
np.save(f"{paths.arrays_path}cs_dens_1p_e{elast_id}_n{nest_id}.npy", cs_dens_1p)
np.save(f"{paths.arrays_path}ps_dens_1p_e{elast_id}_n{nest_id}.npy", ps_dens_1p)
np.save(f"{paths.arrays_path}ts_dens_1p_e{elast_id}_n{nest_id}.npy", ts_dens_1p)
np.save(f"{paths.arrays_path}ccs_dens_1p_e{elast_id}_n{nest_id}.npy", ccs_dens_1p)
np.save(f"{paths.arrays_path}ccs_per_bw_dens_1p_e{elast_id}_n{nest_id}.npy", ccs_per_bw_dens_1p)
np.save(f"{paths.arrays_path}avg_path_losses_dens_1p_e{elast_id}_n{nest_id}.npy", avg_path_losses_dens_1p)
np.save(f"{paths.arrays_path}avg_SINR_dens_1p_e{elast_id}_n{nest_id}.npy", avg_SINR_dens_1p)
np.save(f"{paths.arrays_path}per_user_costs_e{elast_id}_n{nest_id}.npy", per_user_costs)

# Standard errors
if compute_std_errs:
    np.save(f"{paths.arrays_path}p_stars_se_e{elast_id}_n{nest_id}.npy", p_stars_se)
    np.save(f"{paths.arrays_path}R_stars_se_e{elast_id}_n{nest_id}.npy", R_stars_se)
    np.save(f"{paths.arrays_path}num_stations_stars_se_e{elast_id}_n{nest_id}.npy", num_stations_stars_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_se_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_se)
    np.save(f"{paths.arrays_path}q_stars_se_e{elast_id}_n{nest_id}.npy", q_stars_se)
    np.save(f"{paths.arrays_path}cs_by_type_se_e{elast_id}_n{nest_id}.npy", cs_by_type_se)
    np.save(f"{paths.arrays_path}cs_se_e{elast_id}_n{nest_id}.npy", cs_se)
    np.save(f"{paths.arrays_path}ps_se_e{elast_id}_n{nest_id}.npy", ps_se)
    np.save(f"{paths.arrays_path}ts_se_e{elast_id}_n{nest_id}.npy", ts_se)
    np.save(f"{paths.arrays_path}ccs_se_e{elast_id}_n{nest_id}.npy", ccs_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_se_e{elast_id}_n{nest_id}.npy", ccs_per_bw_se)
    np.save(f"{paths.arrays_path}avg_path_losses_se_e{elast_id}_n{nest_id}.npy", avg_path_losses_se)
    np.save(f"{paths.arrays_path}avg_SINR_se_e{elast_id}_n{nest_id}.npy", avg_SINR_se)
    np.save(f"{paths.arrays_path}full_elasts_se_e{elast_id}_n{nest_id}.npy", full_elasts_se)
    np.save(f"{paths.arrays_path}partial_elasts_se_e{elast_id}_n{nest_id}.npy", partial_elasts_se)
    np.save(f"{paths.arrays_path}partial_Pif_partial_bf_allfixed_se_e{elast_id}_n{nest_id}.npy", partial_Pif_partial_bf_allfixed_se)
    np.save(f"{paths.arrays_path}partial_Pif_partial_b_allfixed_se_e{elast_id}_n{nest_id}.npy", partial_Pif_partial_b_allfixed_se)
    np.save(f"{paths.arrays_path}partial_CS_partial_b_allbw_se_e{elast_id}_n{nest_id}.npy", partial_CS_partial_b_allbw_se)
    np.save(f"{paths.arrays_path}partial_Pif_partial_bf_allbw_se_e{elast_id}_n{nest_id}.npy", partial_Pif_partial_bf_allbw_se)
    np.save(f"{paths.arrays_path}partial_Pif_partial_b_allbw_se_e{elast_id}_n{nest_id}.npy", partial_Pif_partial_b_allbw_se)
    np.save(f"{paths.arrays_path}partial_CS_partial_b_allfixed_se_e{elast_id}_n{nest_id}.npy", partial_CS_partial_b_allfixed_se)
    np.save(f"{paths.arrays_path}c_u_se_e{elast_id}_n{nest_id}.npy", c_u_se)
    np.save(f"{paths.arrays_path}c_R_se_e{elast_id}_n{nest_id}.npy", c_R_se)
    np.save(f"{paths.arrays_path}p_stars_shortrun_se_e{elast_id}_n{nest_id}.npy", p_stars_shortrun_se)
    np.save(f"{paths.arrays_path}R_stars_shortrun_se_e{elast_id}_n{nest_id}.npy", R_stars_shortrun_se)
    np.save(f"{paths.arrays_path}num_stations_stars_shortrun_se_e{elast_id}_n{nest_id}.npy", num_stations_stars_shortrun_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_shortrun_se_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_shortrun_se)
    np.save(f"{paths.arrays_path}q_stars_shortrun_se_e{elast_id}_n{nest_id}.npy", q_stars_shortrun_se)
    np.save(f"{paths.arrays_path}cs_by_type_shortrun_se_e{elast_id}_n{nest_id}.npy", cs_by_type_shortrun_se)
    np.save(f"{paths.arrays_path}cs_shortrun_se_e{elast_id}_n{nest_id}.npy", cs_shortrun_se)
    np.save(f"{paths.arrays_path}ps_shortrun_se_e{elast_id}_n{nest_id}.npy", ps_shortrun_se)
    np.save(f"{paths.arrays_path}ts_shortrun_se_e{elast_id}_n{nest_id}.npy", ts_shortrun_se)
    np.save(f"{paths.arrays_path}ccs_shortrun_se_e{elast_id}_n{nest_id}.npy", ccs_shortrun_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_shortrun_se_e{elast_id}_n{nest_id}.npy", ccs_per_bw_shortrun_se)
    np.save(f"{paths.arrays_path}avg_path_losses_shortrun_se_e{elast_id}_n{nest_id}.npy", avg_path_losses_shortrun_se)
    np.save(f"{paths.arrays_path}p_stars_free_allfixed_se_e{elast_id}_n{nest_id}.npy", p_stars_free_allfixed_se)
    np.save(f"{paths.arrays_path}R_stars_free_allfixed_se_e{elast_id}_n{nest_id}.npy", R_stars_free_allfixed_se)
    np.save(f"{paths.arrays_path}num_stations_stars_free_allfixed_se_e{elast_id}_n{nest_id}.npy", num_stations_stars_free_allfixed_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_free_allfixed_se_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_free_allfixed_se)
    np.save(f"{paths.arrays_path}q_stars_free_allfixed_se_e{elast_id}_n{nest_id}.npy", q_stars_free_allfixed_se)
    np.save(f"{paths.arrays_path}cs_by_type_free_allfixed_se_e{elast_id}_n{nest_id}.npy", cs_by_type_free_allfixed_se)
    np.save(f"{paths.arrays_path}cs_free_se_allfixed_e{elast_id}_n{nest_id}.npy", cs_free_allfixed_se)
    np.save(f"{paths.arrays_path}ps_free_se_allfixed_e{elast_id}_n{nest_id}.npy", ps_free_allfixed_se)
    np.save(f"{paths.arrays_path}ts_free_se_allfixed_e{elast_id}_n{nest_id}.npy", ts_free_allfixed_se)
    np.save(f"{paths.arrays_path}ccs_free_allfixed_se_e{elast_id}_n{nest_id}.npy", ccs_free_allfixed_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_free_allfixed_se_e{elast_id}_n{nest_id}.npy", ccs_per_bw_free_allfixed_se)
    np.save(f"{paths.arrays_path}avg_path_losses_free_allfixed_se_e{elast_id}_n{nest_id}.npy", avg_path_losses_free_allfixed_se)
    np.save(f"{paths.arrays_path}p_stars_free_allbw_se_e{elast_id}_n{nest_id}.npy", p_stars_free_allbw_se)
    np.save(f"{paths.arrays_path}R_stars_free_allbw_se_e{elast_id}_n{nest_id}.npy", R_stars_free_allbw_se)
    np.save(f"{paths.arrays_path}num_stations_stars_free_allbw_se_e{elast_id}_n{nest_id}.npy", num_stations_stars_free_allbw_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_free_allbw_se_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_free_allbw_se)
    np.save(f"{paths.arrays_path}q_stars_free_allbw_se_e{elast_id}_n{nest_id}.npy", q_stars_free_allbw_se)
    np.save(f"{paths.arrays_path}cs_by_type_free_allbw_se_e{elast_id}_n{nest_id}.npy", cs_by_type_free_allbw_se)
    np.save(f"{paths.arrays_path}cs_free_se_allbw_e{elast_id}_n{nest_id}.npy", cs_free_allbw_se)
    np.save(f"{paths.arrays_path}ps_free_se_allbw_e{elast_id}_n{nest_id}.npy", ps_free_allbw_se)
    np.save(f"{paths.arrays_path}ts_free_se_allbw_e{elast_id}_n{nest_id}.npy", ts_free_allbw_se)
    np.save(f"{paths.arrays_path}ccs_free_allbw_se_e{elast_id}_n{nest_id}.npy", ccs_free_allbw_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_free_allbw_se_e{elast_id}_n{nest_id}.npy", ccs_per_bw_free_allbw_se)
    np.save(f"{paths.arrays_path}avg_path_losses_free_allbw_se_e{elast_id}_n{nest_id}.npy", avg_path_losses_free_allbw_se)
    
    np.save(f"{paths.arrays_path}p_stars_dens_se_e{elast_id}_n{nest_id}.npy", p_stars_dens_se)
    np.save(f"{paths.arrays_path}R_stars_dens_se_e{elast_id}_n{nest_id}.npy", R_stars_dens_se)
    np.save(f"{paths.arrays_path}num_stations_stars_dens_se_e{elast_id}_n{nest_id}.npy", num_stations_stars_dens_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_dens_se_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_dens_se)
    np.save(f"{paths.arrays_path}q_stars_dens_se_e{elast_id}_n{nest_id}.npy", q_stars_dens_se)
    np.save(f"{paths.arrays_path}cs_by_type_dens_se_e{elast_id}_n{nest_id}.npy", cs_by_type_dens_se)
    np.save(f"{paths.arrays_path}cs_dens_se_e{elast_id}_n{nest_id}.npy", cs_dens_se)
    np.save(f"{paths.arrays_path}ps_dens_se_e{elast_id}_n{nest_id}.npy", ps_dens_se)
    np.save(f"{paths.arrays_path}ts_dens_se_e{elast_id}_n{nest_id}.npy", ts_dens_se)
    np.save(f"{paths.arrays_path}ccs_dens_se_e{elast_id}_n{nest_id}.npy", ccs_dens_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_dens_se_e{elast_id}_n{nest_id}.npy", ccs_per_bw_dens_se)
    np.save(f"{paths.arrays_path}avg_path_losses_dens_se_e{elast_id}_n{nest_id}.npy", avg_path_losses_dens_se)
    np.save(f"{paths.arrays_path}avg_SINR_dens_se_e{elast_id}_n{nest_id}.npy", avg_SINR_dens_se)
    np.save(f"{paths.arrays_path}p_stars_bw_se_e{elast_id}_n{nest_id}.npy", p_stars_bw_se)
    np.save(f"{paths.arrays_path}R_stars_bw_se_e{elast_id}_n{nest_id}.npy", R_stars_bw_se)
    np.save(f"{paths.arrays_path}num_stations_stars_bw_se_e{elast_id}_n{nest_id}.npy", num_stations_stars_bw_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_bw_se_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_bw_se)
    np.save(f"{paths.arrays_path}q_stars_bw_se_e{elast_id}_n{nest_id}.npy", q_stars_bw_se)
    np.save(f"{paths.arrays_path}cs_by_type_bw_se_e{elast_id}_n{nest_id}.npy", cs_by_type_bw_se)
    np.save(f"{paths.arrays_path}cs_bw_se_e{elast_id}_n{nest_id}.npy", cs_bw_se)
    np.save(f"{paths.arrays_path}ps_bw_se_e{elast_id}_n{nest_id}.npy", ps_bw_se)
    np.save(f"{paths.arrays_path}ts_bw_se_e{elast_id}_n{nest_id}.npy", ts_bw_se)
    np.save(f"{paths.arrays_path}ccs_bw_se_e{elast_id}_n{nest_id}.npy", ccs_bw_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_bw_se_e{elast_id}_n{nest_id}.npy", ccs_per_bw_bw_se)
    np.save(f"{paths.arrays_path}avg_path_losses_bw_se_e{elast_id}_n{nest_id}.npy", avg_path_losses_bw_se)
    np.save(f"{paths.arrays_path}avg_SINR_bw_se_e{elast_id}_n{nest_id}.npy", avg_SINR_bw_se)
    np.save(f"{paths.arrays_path}p_stars_dens_1p_se_e{elast_id}_n{nest_id}.npy", p_stars_dens_1p_se)
    np.save(f"{paths.arrays_path}R_stars_dens_1p_se_e{elast_id}_n{nest_id}.npy", R_stars_dens_1p_se)
    np.save(f"{paths.arrays_path}num_stations_stars_dens_1p_se_e{elast_id}_n{nest_id}.npy", num_stations_stars_dens_1p_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_dens_1p_se_e{elast_id}_n{nest_id}.npy", num_stations_per_firm_stars_dens_1p_se)
    np.save(f"{paths.arrays_path}q_stars_dens_1p_se_e{elast_id}_n{nest_id}.npy", q_stars_dens_1p_se)
    np.save(f"{paths.arrays_path}cs_by_type_dens_1p_se_e{elast_id}_n{nest_id}.npy", cs_by_type_dens_1p_se)
    np.save(f"{paths.arrays_path}cs_dens_1p_se_e{elast_id}_n{nest_id}.npy", cs_dens_1p_se)
    np.save(f"{paths.arrays_path}ps_dens_1p_se_e{elast_id}_n{nest_id}.npy", ps_dens_1p_se)
    np.save(f"{paths.arrays_path}ts_dens_1p_se_e{elast_id}_n{nest_id}.npy", ts_dens_1p_se)
    np.save(f"{paths.arrays_path}ccs_dens_1p_se_e{elast_id}_n{nest_id}.npy", ccs_dens_1p_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_dens_1p_se_e{elast_id}_n{nest_id}.npy", ccs_per_bw_dens_1p_se)
    np.save(f"{paths.arrays_path}avg_path_losses_dens_1p_se_e{elast_id}_n{nest_id}.npy", avg_path_losses_dens_1p_se)
    np.save(f"{paths.arrays_path}avg_SINR_dens_1p_se_e{elast_id}_n{nest_id}.npy", avg_SINR_dens_1p_se)
    np.save(f"{paths.arrays_path}per_user_costs_se_e{elast_id}_n{nest_id}.npy", per_user_costs_se)
