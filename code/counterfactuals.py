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
import scipy.special as special

import sys
import copy

from itertools import combinations

from multiprocessing import Pool

import paths

import counterfactuals.infrastructurefunctions as infr
import counterfactuals.priceequilibrium as pe
import counterfactuals.infrastructureequilibrium as ie
import counterfactuals.transmissionequilibrium as te
import counterfactuals.welfare as welfare
import counterfactuals.costs as costs

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
elast_id = 0#task_id // paths.div_ratios.shape[0]
nest_id = 0#task_id % paths.div_ratios.shape[0]
avg_price_el = paths.avg_price_elasts[elast_id]
div_ratio = paths.div_ratios[nest_id]

print(f"Orange price elasticity: {avg_price_el}", flush=True)
print(f"Div ratio: {div_ratio}", flush=True)

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
# Determine whether compute long-run mergers (equilibrium calculation very sensitive)
long_run_merger_calc = False

# %%
# Determine whether to use different starting values for asymmetric equilibrium to check
check_for_multiplicity_asymmetric = True

# %%
# Import infrastructure / quality data
df_inf = pd.read_csv(f"{paths.data_path}infrastructure_clean.csv", engine="python") # engine helps encoding, error with commune names, but doesn't matter b/c not used
df_inf = df_inf[df_inf['market'] > 0] # don't include Rest-of-France market
df_q = pd.read_csv(f"{paths.data_path}quality_ookla.csv")
df_q = df_q[df_q['market'] > 0] # don't include Rest-of-France market

mno_codes = {
    'Orange': 1, 
    'Bouygues': 2, 
    'Free': 3, 
    'SFR': 4,
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

# Merged base station numbers and implied radii
mnos = ['Orange', 'SFR', 'Bouygues', 'Free']
mnos = [mno[0] for mno in mnos] # only want the first letter
merger_codes = [f"{combo[0]}{combo[1]}" for combo in combinations(mnos, 2)]
merged_base_stations = df_inf[[f"stations{merger_code}" for merger_code in merger_codes]].values
radius_mergers = np.sqrt(area[:,np.newaxis] / merged_base_stations / (np.sqrt(3.) * 3. / 2.)) # cell radius assuming homogeneous hexagonal cells, in km
radius_mergers_all = np.tile(np.copy(radius)[:,:,np.newaxis], (1,1,len(merger_codes) + 1))
list_MNOwMVNO = np.array(list(mno_codes.keys()))
list_MNOwMVNOnums = np.array(list(mno_codes.values()))
list_MNOwoMVNOnums = list_MNOwMVNOnums[list_MNOwMVNO != "MVNO"]
merger_combos = [list(combo) for combo in combinations(list_MNOwoMVNOnums, 2)]
for i, merger in enumerate(merger_combos):
    acquirer = np.min(np.array(merger))
    acquiree = np.max(np.array(merger))
    idx_acquirer = acquirer - 1
    idx_acquiree = acquiree - 1
    radius_mergers_all[:,idx_acquiree,i+1] = np.nan # don't need to track acquire
    merging_mnos = np.array(list(mno_codes.keys()))[np.isin(np.array(list(mno_codes.values())), np.array(merger))]
    select_merger = (np.array(merger_codes) == f"{merging_mnos[0][0]}{merging_mnos[1][0]}") | (np.array(merger_codes) == f"{merging_mnos[1][0]}{merging_mnos[0][0]}")
    radius_mergers_idx = np.arange(len(merger_codes))[select_merger][0]
    radius_mergers_all[:,idx_acquirer,i+1] = radius_mergers[:,radius_mergers_idx]

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
with open(f"{paths.data_path}demandsystem_{task_id}.obj", "rb") as file_ds:
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
q_dem_tot = q_dem_tot[~markets_org_free_share]
q_avg = q_avg[~markets_org_free_share,:]
radius = radius[~markets_org_free_share,:]
bw_4g_equiv = bw_4g_equiv[~markets_org_free_share,:]
lamda = lamda[~markets_org_free_share]
area = area[~markets_org_free_share]
pop_dens = pop_dens[~markets_org_free_share]
radius_mergers_all = radius_mergers_all[~markets_org_free_share,:,:]
if task_id == 0:
    np.save(f"{paths.arrays_path}spectral_efficiencies.npy", lamda)
    np.save(f"{paths.arrays_path}populations.npy", population)
    np.save(f"{paths.arrays_path}cc_tot.npy", cc_tot)
    np.save(f"{paths.arrays_path}q_dem_tot.npy", q_dem_tot)
    np.save(f"{paths.arrays_path}q_avg.npy", q_avg)
    np.save(f"{paths.arrays_path}bw_4g_equiv.npy", bw_4g_equiv)
    np.save(f"{paths.arrays_path}radius_mergers_all.npy", radius_mergers_all)

# Get array of product prices
prices = ds.data[0,:,pidx] # 0 b/c market doesn't matter

# %%
# Import demand estimation results
N = ds.M
thetahat = np.load(f"{paths.arrays_path}thetahat_{task_id}.npy")
G_n = np.load(f"{paths.arrays_path}Gn_{task_id}.npy")
What = np.load(f"{paths.arrays_path}What_{task_id}.npy")
Sigma = vm.V(G_n, What, np.linalg.inv(What))

# %%
# Determine the size of epsilon to numerically approximate the gradient
compute_std_errs = True
eps_grad = 0.025
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
densities_areatype = np.array(["urban", "urban", "urban", "urban"]) # np.array(["urban", "rural", "suburban", "urban"]) # area types used for Hata path loss model
np.save(f"{paths.arrays_path}cntrfctl_densities_{task_id}.npy", densities)
np.save(f"{paths.arrays_path}cntrfctl_densities_pop_{task_id}.npy", densities * rep_market_size)

# Bandwidth values to test
market_bw = np.average(np.sum(bw_4g_equiv, axis=1), weights=population)
low_bw_val = market_bw * 0.5
high_bw_val = market_bw * 1.5
bw_vals = np.array([market_bw, low_bw_val, high_bw_val]) # market_bw must be first (because its results just get copied over from the regular exercise)
np.save(f"{paths.arrays_path}cntrfctl_bw_vals_{task_id}.npy", bw_vals)

# Bandwidth values and merged base stations to use for short-run counterfactual
radius_mergers_use = np.average(radius_mergers_all, weights=np.tile(population[:,np.newaxis,np.newaxis], (1,radius_mergers_all.shape[1],radius_mergers_all.shape[2])), axis=0)
bw_4g_equiv_weightedaverage = np.average(bw_4g_equiv, weights=np.tile(population[:,np.newaxis], (1,bw_4g_equiv.shape[1])), axis=0)
if task_id == 0:
    np.save(f"{paths.arrays_path}cntrfctl_bw_vals_by_firm.npy", bw_4g_equiv_weightedaverage)
    np.save(f"{paths.arrays_path}cntrfctl_firms.npy", list_MNOwMVNO[list_MNOwMVNO != "MVNO"])

# Save spectral efficiency used in counterfactuals
if task_id == 0:
    np.save(f"{paths.arrays_path}cntrfctl_gamma.npy", np.array([np.average(lamda, weights=population)]))

# Welfare options
include_logit_shock = False
include_pop = False

# Dictionary for how to treat MVNO in computing transmission equilibrium
impute_MVNO = {
    'impute': True, 
    'firms_share': np.array([True, True, False, True]), # all firms share with MVNO, except Free
    'include': True
}

# Determine which markets to use to mimic all of France
start = time.time()
theta_0 = thetas_to_compute[0,:]
xis_0 = blp.xi(ds, theta_0, ds.data, None)
c_u_0 = costs.per_user_costs(theta_0, xis_0, ds, population, prices, cc_tot, stations, impute_MVNO=impute_MVNO)
c_R_0 = costs.per_base_station_costs(theta_0, xis_0, c_u_0, radius, bw_4g_equiv, lamda, ds, population, area, impute_MVNO=impute_MVNO)
if print_updates:
    print(f"Finished calculating xis, per-user costs, per-base station costs for estimated theta in {np.round(time.time() - start, 1)} seconds.", flush=True)
population_categories_upper_lim = np.array([35000.0, 100000.0, np.inf]) # these define the population categories
population_categories_lower_lim = np.concatenate((np.array([0.0]), population_categories_upper_lim[:-1]))
avg_c_R = np.zeros((population_categories_upper_lim.shape[0], c_R_0.shape[1]))
frac_population = np.zeros(population_categories_upper_lim.shape)
constructed_markets_idx = np.zeros(population_categories_upper_lim.shape, dtype=int)
markets_idx_arange = np.arange(population.shape[0])
df_pop_dens = pd.read_csv(f"{paths.data_path}effective_pop_dens.csv", encoding="ISO-8859-1")
df_pop_dens.sort_values(by="market", inplace=True)
df_pop_dens = df_pop_dens[np.isin(df_pop_dens['market'].values, market_numbers[market_numbers > 0])]
market_names = df_pop_dens['com_label'].values
market_names = market_names[~markets_org_free_share]
mimic_market_names = []
for i in range(population_categories_upper_lim.shape[0]):
    select_markets = (population >= population_categories_lower_lim[i]) & (population < population_categories_upper_lim[i])
    avg_c_R[i,:] = np.average(c_R_0[select_markets,:], weights=population[select_markets], axis=0)
    frac_population[i] = np.sum(population[select_markets]) / np.sum(population)
#     if i + 1 == population_categories_upper_lim.shape[0]: # final market (largest one)
#         allowed_markets = np.array(["Nantes", "Marseille", "Rennes"])#np.array(["Nantes", "Bordeaux", "Rennes", "Tours", "Toulouse"])
#         select_markets = select_markets & np.isin(market_names, allowed_markets)
    constructed_markets_idx[i] = markets_idx_arange[select_markets][np.argmin(np.sum((c_R_0[select_markets,:] - avg_c_R[i,:])**2.0, axis=1))]
    market_name = market_names[constructed_markets_idx[i]]
    mimic_market_names = mimic_market_names + [market_name]
    print(f"Mimic market category {i + 1} [{population_categories_lower_lim[i]}, {population_categories_upper_lim[i]}) making up {np.round(frac_population[i] * 100.0, 2)}% of population: {np.sum(select_markets)} markets with average c_R of {np.round(avg_c_R[i,:], 3)}. Chose market {constructed_markets_idx[i]} ({market_name}) with c_R {np.round(c_R_0[constructed_markets_idx[i],:], 3)}.", flush=True)
frac_population_orig = np.copy(frac_population)
frac_population = frac_population * np.sum(population[constructed_markets_idx]) / population[constructed_markets_idx] # adjust for the population associated with the markets so that market weighting is correct
frac_population = frac_population / np.sum(frac_population)
np.save(f"{paths.arrays_path}mimic_market_names.npy", np.array(mimic_market_names))
np.save(f"{paths.arrays_path}mimic_market_weights.npy", frac_population_orig)
np.save(f"{paths.arrays_path}mimic_market_population_categories.npy", population_categories_upper_lim)
np.save(f"{paths.arrays_path}constructed_markets_idx.npy", constructed_markets_idx)

def compute_equilibria(theta_n, p_starting_vals=None, R_starting_vals=None):
    """Compute the equilibria for a particular draw from the demand parameter's asymptotic distribution."""
    
    # Construct demand parameter
    if print_updates:
        print(f"theta_n={theta_n} beginning computation...", flush=True)
    theta = thetas_to_compute[theta_n,:]
    
    # Recover xis
    start = time.time()
    if theta_n == 0:
        xis = xis_0
        np.save(f"{paths.arrays_path}xis_{task_id}.npy", xis)
    else:
        xis = blp.xi(ds, theta, ds.data, None)
    if print_updates:
        print(f"theta_n={theta_n}: Finished calculating xis in {np.round(time.time() - start, 1)} seconds.", flush=True)

    # %%
    # Estimate per-user costs
    start = time.time()
    if theta_n == 0:
        c_u = c_u_0
    else:
        c_u = costs.per_user_costs(theta, xis, ds, population, prices, cc_tot, stations, impute_MVNO=impute_MVNO)
    if print_updates:
        print(f"theta_n={theta_n}: Finished calculating per-user costs in {np.round(time.time() - start, 1)} seconds.", flush=True)

    # %%
    # Estimate per-base station costs
    start = time.time()
    if theta_n == 0:
        c_R = c_R_0
    else:
        c_R = costs.per_base_station_costs(theta, xis, c_u, radius, bw_4g_equiv, lamda, ds, population, area, impute_MVNO=impute_MVNO)
    if print_updates:
        print(f"theta_n={theta_n}: Finished calculating per-base station costs in {np.round(time.time() - start, 1)} seconds.", flush=True)

    # %%
    # Compute counterfactual equilibria

    per_user_costs = np.array([np.mean(c_u[ds.data[0,:,dlimidx] < 5000.0]), np.mean(c_u[ds.data[0,:,dlimidx] >= 5000.0])])
    per_base_station_per_bw_cost = np.average(c_R / bw_4g_equiv, weights=np.tile(population[:,np.newaxis], (1,4)))
    per_base_station_per_bw_cost_by_firm = np.average(c_R / bw_4g_equiv, weights=np.tile(population[:,np.newaxis], (1,4)), axis=0)[np.newaxis,:]

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
    
    p_stars_allfixed = np.zeros((num_firms_array.shape[0], num_prods)) * np.nan
    R_stars_allfixed = np.zeros(num_firms_array.shape) * np.nan
    num_stations_stars_allfixed = np.zeros(num_firms_array.shape) * np.nan
    num_stations_per_firm_stars_allfixed = np.zeros(num_firms_array.shape) * np.nan
    q_stars_allfixed = np.zeros(num_firms_array.shape) * np.nan
    cs_allfixed = np.zeros(num_firms_array_extend.shape) * np.nan
    cs_by_type_allfixed = np.zeros((num_firms_array_extend.shape[0], yclastidx - yc1idx + 1)) * np.nan
    ps_allfixed = np.zeros(num_firms_array_extend.shape) * np.nan
    ts_allfixed = np.zeros(num_firms_array_extend.shape) * np.nan
    ccs_allfixed = np.zeros(num_firms_array.shape) * np.nan
    ccs_per_bw_allfixed = np.zeros(num_firms_array.shape) * np.nan
    avg_path_losses_allfixed = np.zeros(num_firms_array.shape) * np.nan
    avg_SINR_allfixed = np.zeros(num_firms_array.shape) * np.nan

    partial_Pif_partial_bf_allfixed = np.zeros(num_firms_array.shape) * np.nan
    partial_Piotherf_partial_bf_allfixed = np.zeros(num_firms_array.shape) * np.nan
    partial_diffPif_partial_bf_allfixed = np.zeros(num_firms_array.shape) * np.nan
    partial_Pif_partial_b_allfixed = np.zeros(num_firms_array.shape) * np.nan
    partial_CS_partial_b_allfixed = np.zeros(num_firms_array.shape) * np.nan
    partial_Pif_partial_bf_allbw = np.zeros(num_firms_array.shape) * np.nan
    partial_Piotherf_partial_bf_allbw = np.zeros(num_firms_array.shape) * np.nan
    partial_diffPif_partial_bf_allbw = np.zeros(num_firms_array.shape) * np.nan
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
    
    p_stars_asymmetric_allbw = np.zeros((3, 2, num_prods)) * np.nan
    R_stars_asymmetric_allbw = np.zeros((3, 2)) * np.nan
    num_stations_stars_asymmetric_allbw = np.zeros((3,)) * np.nan
    num_stations_per_firm_stars_asymmetric_allbw = np.zeros((3, 2)) * np.nan
    q_stars_asymmetric_allbw = np.zeros((3, 2)) * np.nan
    cs_asymmetric_allbw = np.zeros(3) * np.nan
    cs_by_type_asymmetric_allbw = np.zeros((3, yclastidx - yc1idx + 1)) * np.nan
    ps_asymmetric_allbw = np.zeros(3) * np.nan
    ts_asymmetric_allbw = np.zeros(3) * np.nan
    ccs_asymmetric_allbw = np.zeros((3, 2)) * np.nan
    ccs_per_bw_asymmetric_allbw = np.zeros((3, 2)) * np.nan
    avg_path_losses_asymmetric_allbw = np.zeros((3, 2)) * np.nan
    avg_SINR_asymmetric_allbw = np.zeros((3, 2)) * np.nan
    
    num_prods_all = prices.shape[0] # number of products in data
    num_firms_all = len(mno_codes.keys()) - 1 # number of firms in data (not including MVNO, which is an entry in mno_codes)
    num_firms_allwMVNO = num_firms_all + 1 # number of firms in data (including MVNO)
    num_combos_merge = 1 + int(special.comb(num_firms_all, 2)) # original (no mergers) + every possible combo of merger of two
    p_stars_shortrunall = np.zeros((num_combos_merge, num_prods_all)) * np.nan
    R_stars_shortrunall = np.zeros((num_combos_merge, num_firms_all)) * np.nan
    num_stations_stars_shortrunall = np.zeros((num_combos_merge,)) * np.nan
    num_stations_per_firm_stars_shortrunall = np.zeros((num_combos_merge, num_firms_all)) * np.nan
    q_stars_shortrunall = np.zeros((num_combos_merge, num_firms_allwMVNO)) * np.nan
    cs_shortrunall = np.zeros(num_combos_merge) * np.nan
    cs_by_type_shortrunall = np.zeros((num_combos_merge, yclastidx - yc1idx + 1)) * np.nan
    ps_shortrunall = np.zeros(num_combos_merge) * np.nan
    ts_shortrunall = np.zeros(num_combos_merge) * np.nan
    ccs_shortrunall = np.zeros((num_combos_merge, num_firms_all)) * np.nan
    ccs_per_bw_shortrunall = np.zeros((num_combos_merge, num_firms_all)) * np.nan
    avg_path_losses_shortrunall = np.zeros((num_combos_merge, num_firms_all)) * np.nan
    
    p_stars_longrunall = np.zeros((num_combos_merge, num_prods_all)) * np.nan
    R_stars_longrunall = np.zeros((num_combos_merge, num_firms_all)) * np.nan
    num_stations_stars_longrunall = np.zeros((num_combos_merge,)) * np.nan
    num_stations_per_firm_stars_longrunall = np.zeros((num_combos_merge, num_firms_all)) * np.nan
    q_stars_longrunall = np.zeros((num_combos_merge, num_firms_allwMVNO)) * np.nan
    cs_longrunall = np.zeros(num_combos_merge) * np.nan
    cs_by_type_longrunall = np.zeros((num_combos_merge, yclastidx - yc1idx + 1)) * np.nan
    ps_longrunall = np.zeros(num_combos_merge) * np.nan
    ts_longrunall = np.zeros(num_combos_merge) * np.nan
    ccs_longrunall = np.zeros((num_combos_merge, num_firms_all)) * np.nan
    ccs_per_bw_longrunall = np.zeros((num_combos_merge, num_firms_all)) * np.nan
    avg_path_losses_longrunall = np.zeros((num_combos_merge, num_firms_all)) * np.nan
    
    successful = np.ones(num_firms_array_extend.shape, dtype=bool)
    successful_allfixed = np.ones(num_firms_array_extend.shape, dtype=bool)
    successful_bw_deriv_allfixed = np.ones(num_firms_array.shape, dtype=bool)
    successful_bw_deriv_allbw = np.ones(num_firms_array.shape, dtype=bool)
    successful_shortrun = np.ones(1, dtype=bool)
    successful_free_allfixed = np.ones(2, dtype=bool)
    successful_free_allbw = np.ones(2, dtype=bool)
    successful_dens = np.ones((num_firms_array_extend.shape[0], densities.shape[0]), dtype=bool)
    successful_bw = np.ones((num_firms_array_extend.shape[0], bw_vals.shape[0]), dtype=bool)
    successful_dens_1p = np.ones((num_firms_array_extend.shape[0], densities.shape[0]), dtype=bool)
    successful_asymmetric_allbw = np.ones(3, dtype=bool)
    successful_shortrunall = np.ones(num_combos_merge, dtype=bool)
    successful_longrunall = np.ones(num_combos_merge, dtype=bool)
        
    # Determine the starting guesses for each
    if p_starting_vals is None:
        p_0s = np.tile(per_user_costs[np.newaxis,:], (num_firms_array_extend.shape[0],1))
        #p_0s[np.isin(num_firms_array_extend, np.array([1]))] = 2.0 * p_0s[np.isin(num_firms_array_extend, np.array([1]))] # these don't have good convergence properties if start at MC
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
        xis_cntrfctl = np.ones((1,num_prods)) * theta[coef.O]
        xis_cntrfctl_1p = np.ones((1,1)) * theta[coef.O]

        # Create cost arrays
        c_u_cntrfctl = per_user_costs
        select_1p = np.arange(num_prods) == np.argmax(dlims)
        c_u_cntrfctl_1p = c_u_cntrfctl[select_1p] # keep only the highest data limit plan
        c_R_cntrfctl = np.ones((1,1)) * per_base_station_per_bw_cost * bw_cntrfctl # per tower cost based on level of bandwidth for each firm

        # Set starting values (if None, num_firms=1 can cause problems for convergence)
        R_0 = R_0s[i,:][:,np.newaxis]
        p_0 = p_0s[i,:]
        p_0_1p = p_0[select_1p]
        
        def simple_symmetric_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, num_firms, num_prods, areatype="urban"):
            """Compute the symmetric equilibrium."""
            
            ds_cntrfctl_ = copy.deepcopy(ds_cntrfctl)
            
            # Compute the equilibrium
            R_star, p_star, q_star, success = ie.infrastructure_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, symmetric=True, print_msg=print_msg, impute_MVNO={'impute': False}, q_0=None, eps_R=0.001, eps_p=0.001, factor=100., areatype=areatype)
            
            # Update Demand System
            ds_cntrfctl_.data[:,:,pidx] = np.copy(p_star)
            ds_cntrfctl_.data[:,:,qidx] = np.tile(q_star, (num_prods,))
            
            # Calculate welfare impact
            cs_by_type_ = welfare.consumer_surplus(ds_cntrfctl_, np.tile(xis_cntrfctl, (1,num_firms)), theta, include_logit_shock=include_logit_shock)
            cs_ = welfare.agg_consumer_surplus(ds_cntrfctl_, np.tile(xis_cntrfctl, (1,num_firms)), theta, pop_cntrfctl, include_logit_shock=include_logit_shock, include_pop=include_pop)
            ps_ = welfare.producer_surplus(ds_cntrfctl_, np.tile(xis_cntrfctl, (1,num_firms)), theta, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_pop=include_pop)
            ts_ = welfare.total_surplus(ds_cntrfctl_, np.tile(xis_cntrfctl, (1,num_firms)), theta, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_logit_shock=include_logit_shock, include_pop=include_pop)
            
            p_stars_ = p_star[:num_prods]
            R_stars_ = R_star[0,0]
            num_stations_stars_ = num_firms * infr.num_stations(R_stars_, rep_market_size)
            num_stations_per_firm_stars_ = infr.num_stations(R_stars_, rep_market_size)
            q_stars_ = q_star[0,0]

            cc_cntrfctl = np.zeros((R_star.shape[0], 1))
            for m in range(R_star.shape[0]):
                cc_cntrfctl[m,0] = infr.rho_C_hex(bw_cntrfctl[m,0], R_star[m,0], gamma_cntrfctl[m], areatype=areatype)
            ccs_ = cc_cntrfctl[0,0]
            ccs_per_bw_ = (cc_cntrfctl / bw_cntrfctl)[0,0]
            avg_path_losses_ = infr.avg_path_loss(R_stars_, areatype=areatype)
            num_stations_cntrfctl = infr.num_stations(np.array([[R_stars_]]), market_size_cntrfctl)
            avg_SINR_ = infr.avg_SINR(R_stars_, areatype=areatype)
            
            return success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_

        # Simple symmetric equilibrium result for representative values
        start = time.time()
        success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = simple_symmetric_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, num_firms, num_prods)
        ds_cntrfctl_baseline = copy.deepcopy(ds_cntrfctl_)
        if print_updates:
            print(f"theta_n={theta_n}, num_firms={num_firms}: Finished calculating symmetric equilibrium in {np.round(time.time() - start, 1)} seconds.", flush=True)
        successful[i], cs_by_type[i,:], cs[i], ps[i], ts[i] = success, cs_by_type_, cs_, ps_, ts_
        if np.isin(num_firms, num_firms_array): # don't need to record these if gone beyond num_firms_array
            p_stars[i,:], R_stars[i], num_stations_stars[i], num_stations_per_firm_stars[i], q_stars[i], ccs[i], ccs_per_bw[i], avg_path_losses[i], avg_SINR[i] = p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_

            # Calculate elasticities
            start = time.time()
            full_elasts[i,:] = pe.price_elast(np.copy(p_stars[i,:]), cc_cntrfctl, ds_cntrfctl_baseline, xis_cntrfctl, theta, num_stations_cntrfctl, pop_cntrfctl, symmetric=True, impute_MVNO={'impute': False}, q_0=None, eps=0.001, full=True)
            partial_elasts[i,:] = pe.price_elast(np.copy(p_stars[i,:]), cc_cntrfctl, ds_cntrfctl_baseline, xis_cntrfctl, theta, num_stations_cntrfctl, pop_cntrfctl, symmetric=True, impute_MVNO={'impute': False}, q_0=None, eps=0.001, full=False)
            if print_updates:
                print(f"theta_n={theta_n}, num_firms={num_firms}: Finished calculating elasticities in {np.round(time.time() - start, 1)} seconds.", flush=True)
                
        # Simple symmetric equilibrium result for representative values where base station costs are all fixed
        start = time.time()
        c_R_cntrfctl_allfixed = np.ones((1,1)) * per_base_station_per_bw_cost * market_bw / 4.0
        p_0_allfixed = np.copy(p_0)
        if num_firms == 1:
            p_0_allfixed = 2.0 * p_0 # this is better for convergence for monopoly case
        success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = simple_symmetric_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl_allfixed, R_0, p_0_allfixed, num_firms, num_prods)
        ds_cntrfctl_baseline = copy.deepcopy(ds_cntrfctl_)
        if print_updates:
            print(f"theta_n={theta_n}, num_firms={num_firms}: Finished calculating symmetric equilibrium (all fixed costs) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        successful_allfixed[i], cs_by_type_allfixed[i,:], cs_allfixed[i], ps_allfixed[i], ts_allfixed[i] = success, cs_by_type_, cs_, ps_, ts_
        if np.isin(num_firms, num_firms_array): # don't need to record these if gone beyond num_firms_array
            p_stars_allfixed[i,:], R_stars_allfixed[i], num_stations_stars_allfixed[i], num_stations_per_firm_stars_allfixed[i], q_stars_allfixed[i], ccs_allfixed[i], ccs_per_bw_allfixed[i], avg_path_losses_allfixed[i], avg_SINR_allfixed[i] = p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_
        
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
                success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = simple_symmetric_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta, pop_cntrfctl_dens, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0_dens, p_0_dens, num_firms, num_prods, areatype=densities_areatype[j])
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
            success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = simple_symmetric_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_1p, xis_cntrfctl_1p, theta, pop_cntrfctl_dens, market_size_cntrfctl, c_u_cntrfctl_1p, c_R_cntrfctl, R_0_dens_1p, p_0_dens_1p, num_firms, 1, areatype=densities_areatype[j])
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
                p_0_bw = np.copy(p_0)
                if (num_firms == 1) and (j == 1):
                    p_0_bw = 2.0 * p_0 # this is better for convergence for monopoly case in the case of bw j == 1
                success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = simple_symmetric_eqm(bw_cntrfctl_bw, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl_bw, R_0, p_0_bw, num_firms, num_prods)
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
        Pif_bf, Piotherf_bf, Pif_b, CS_b, success = ie.bw_foc(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_baseline, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, np.array([[R_stars[i]]]), p_stars[i,:], symmetric=True, print_msg=print_msg, impute_MVNO={'impute': False}, q_0=None, eps_R=0.001, eps_p=0.001, eps_bw=0.01, factor=100., include_logit_shock=include_logit_shock, adjust_c_R=False)
        if print_updates:
            print(f"theta_n={theta_n}, num_firms={num_firms}: Finished calculating bandwidth derivatives (all fixed cost specification) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        successful_bw_deriv_allfixed[i] = success
        partial_Pif_partial_bf_allfixed[i] = Pif_bf[0,0]
        partial_Piotherf_partial_bf_allfixed[i] = Piotherf_bf[0,0]
        partial_diffPif_partial_bf_allfixed[i] = Pif_bf[0,0] - Piotherf_bf[0,0]
        partial_Pif_partial_b_allfixed[i] = Pif_b[0,0]
        partial_CS_partial_b_allfixed[i] = CS_b[0]

        # using all scaled with bw
        start = time.time()
        Pif_bf, Piotherf_bf, Pif_b, CS_b, success = ie.bw_foc(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_baseline, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, np.array([[R_stars[i]]]), p_stars[i,:], symmetric=True, print_msg=print_msg, impute_MVNO={'impute': False}, q_0=None, eps_R=0.001, eps_p=0.001, eps_bw=0.01, factor=100., include_logit_shock=include_logit_shock, adjust_c_R=True)
        if print_updates:
            print(f"theta_n={theta_n}, num_firms={num_firms}: Finished calculating bandwidth derivatives (all bw cost specification) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        successful_bw_deriv_allbw[i] = success
        partial_Pif_partial_bf_allbw[i] = Pif_bf[0,0]
        partial_Piotherf_partial_bf_allbw[i] = Piotherf_bf[0,0]
        partial_diffPif_partial_bf_allbw[i] = Pif_bf[0,0] - Piotherf_bf[0,0]
        partial_Pif_partial_b_allbw[i] = Pif_b[0,0]
        partial_CS_partial_b_allbw[i] = CS_b[0]
        
        # Calculate "short-run" equilibrium
        if np.isin(num_firms, np.array([3])):
            
            start = time.time()
            R_impute = R_stars[num_firms_array_extend_idx_4[0]] * np.ones((1,1)) # this is the R* from the 4-firm version
            ds_cntrfctl_shortrun = copy.deepcopy(ds_cntrfctl)
            R_star, p_star, q_star, success = ie.infrastructure_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_shortrun, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_impute, p_0, symmetric=True, print_msg=print_msg, impute_MVNO={'impute': False}, q_0=None, eps_R=0.001, eps_p=0.001, factor=100., R_fixed=True)
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
            cs_by_type_shortrun[shortrun_idx,:] = welfare.consumer_surplus(ds_cntrfctl_shortrun, np.tile(xis_cntrfctl, (1,num_firms)), theta, include_logit_shock=include_logit_shock)
            cs_shortrun[shortrun_idx] = welfare.agg_consumer_surplus(ds_cntrfctl_shortrun, np.tile(xis_cntrfctl, (1,num_firms)), theta, pop_cntrfctl, include_logit_shock=include_logit_shock, include_pop=include_pop)
            ps_shortrun[shortrun_idx] = welfare.producer_surplus(ds_cntrfctl_shortrun, np.tile(xis_cntrfctl, (1,num_firms)), theta, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_pop=include_pop)
            ts_shortrun[shortrun_idx] = welfare.total_surplus(ds_cntrfctl_shortrun, np.tile(xis_cntrfctl, (1,num_firms)), theta, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_logit_shock=include_logit_shock, include_pop=include_pop)

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
            R_star, p_star, q_star, success = ie.infrastructure_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_free_allfixed, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, symmetric=True, print_msg=print_msg, impute_MVNO={'impute': False}, q_0=None, eps_R=0.001, eps_p=0.001, factor=100.)
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
            cs_by_type_free_allfixed[free_idx,:] = welfare.consumer_surplus(ds_cntrfctl_free_allfixed, np.tile(xis_cntrfctl, (1,num_firms)), theta, include_logit_shock=include_logit_shock)
            cs_free_allfixed[free_idx] = welfare.agg_consumer_surplus(ds_cntrfctl_free_allfixed, np.tile(xis_cntrfctl, (1,num_firms)), theta, pop_cntrfctl, include_logit_shock=include_logit_shock, include_pop=include_pop)
            ps_free_allfixed[free_idx] = welfare.producer_surplus(ds_cntrfctl_free_allfixed, np.tile(xis_cntrfctl, (1,num_firms)), theta, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_pop=include_pop)
            ts_free_allfixed[free_idx] = welfare.total_surplus(ds_cntrfctl_free_allfixed, np.tile(xis_cntrfctl, (1,num_firms)), theta, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_logit_shock=include_logit_shock, include_pop=include_pop)

            # Calculate channel capacities and average path loss
            ccs_free_allfixed[free_idx] = infr.rho_C_hex(bw_cntrfctl[0,0], R_stars_free_allfixed[free_idx], gamma_cntrfctl[0])
            ccs_per_bw_free_allfixed[free_idx] = infr.rho_C_hex(bw_cntrfctl[0,0], R_stars_free_allfixed[free_idx], gamma_cntrfctl[0]) / bw_cntrfctl[0,0]
            avg_path_losses_free_allfixed[free_idx] = infr.avg_path_loss(R_stars_free_allfixed[free_idx])

            # using all scaled with bw
            start = time.time()
            ds_cntrfctl_free_allbw = copy.deepcopy(ds_cntrfctl)
            c_R_cntrfctl = np.ones((1,1)) * per_base_station_per_bw_cost * bw_cntrfctl
            R_star, p_star, q_star, success = ie.infrastructure_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_free_allbw, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, symmetric=True, print_msg=print_msg, impute_MVNO={'impute': False}, q_0=None, eps_R=0.001, eps_p=0.001, factor=100.)
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
            cs_by_type_free_allbw[free_idx,:] = welfare.consumer_surplus(ds_cntrfctl_free_allbw, np.tile(xis_cntrfctl, (1,num_firms)), theta, include_logit_shock=include_logit_shock)
            cs_free_allbw[free_idx] = welfare.agg_consumer_surplus(ds_cntrfctl_free_allbw, np.tile(xis_cntrfctl, (1,num_firms)), theta, pop_cntrfctl, include_logit_shock=include_logit_shock, include_pop=include_pop)
            ps_free_allbw[free_idx] = welfare.producer_surplus(ds_cntrfctl_free_allbw, np.tile(xis_cntrfctl, (1,num_firms)), theta, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_pop=include_pop)
            ts_free_allbw[free_idx] = welfare.total_surplus(ds_cntrfctl_free_allbw, np.tile(xis_cntrfctl, (1,num_firms)), theta, pop_cntrfctl, market_size_cntrfctl, R_star, np.tile(c_u_cntrfctl, (num_firms,)), np.tile(c_R_cntrfctl, (1,num_firms)), include_logit_shock=include_logit_shock, include_pop=include_pop)

            # Calculate channel capacities and average path loss
            ccs_free_allbw[free_idx] = infr.rho_C_hex(bw_cntrfctl[0,0], R_stars_free_allbw[free_idx], gamma_cntrfctl[0])
            ccs_per_bw_free_allbw[free_idx] = infr.rho_C_hex(bw_cntrfctl[0,0], R_stars_free_allbw[free_idx], gamma_cntrfctl[0]) / bw_cntrfctl[0,0]
            avg_path_losses_free_allbw[free_idx] = infr.avg_path_loss(R_stars_free_allbw[free_idx])
            
        def asymmetric_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, product_firm_correspondence, areatype="urban", impute_MVNO={'impute': False}, R_fixed=False, market_weights=None, method=None, factor=100.0, calc_q_carefully=False, print_addl_msg=False):
            """Compute an asymmetric equilibrium."""
            
            ds_cntrfctl_ = copy.deepcopy(ds_cntrfctl)
            
            # Compute the equilibrium
            R_star, p_star, q_star, success = ie.infrastructure_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, symmetric=False, impute_MVNO=impute_MVNO, q_0=None, eps_R=0.001, eps_p=0.001, factor=factor, areatype=areatype, R_fixed=R_fixed, market_weights=market_weights, method=method, calc_q_carefully=calc_q_carefully, print_msg=print_addl_msg)
            
            # Update Demand System
            ds_cntrfctl_.data[:,:,pidx] = p_star[np.newaxis,:]
            ds_cntrfctl_.data[:,:,qidx] = np.take_along_axis(q_star, product_firm_correspondence[np.newaxis,:], 1)
            
            # Calculate welfare impact
            if market_weights is None:
                cs_market_weights = None
            else:
                cs_market_weights = market_weights * pop_cntrfctl / np.sum(pop_cntrfctl * market_weights)
            cs_by_type_ = np.average(welfare.consumer_surplus(ds_cntrfctl_, xis_cntrfctl, theta, include_logit_shock=include_logit_shock), weights=cs_market_weights, axis=0)
            cs_ = welfare.agg_consumer_surplus(ds_cntrfctl_, xis_cntrfctl, theta, pop_cntrfctl, include_logit_shock=include_logit_shock, include_pop=include_pop, market_weights=cs_market_weights)
            ps_ = welfare.producer_surplus(ds_cntrfctl_, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, R_star, c_u_cntrfctl, c_R_cntrfctl, include_pop=include_pop, market_weights=market_weights)
            ts_ = cs_ + ps_
            
            p_stars_ = np.copy(p_star)
            R_stars_ = R_star[0,:]
            num_stations_stars_ = np.sum(infr.num_stations(R_stars_, rep_market_size))
            num_stations_per_firm_stars_ = infr.num_stations(R_stars_, rep_market_size)
            q_stars_ = q_star[0,:]

            cc_cntrfctl = np.zeros(R_star.shape)
            for m in range(R_star.shape[0]):
                for f in range(R_star.shape[1]):
                    cc_cntrfctl[m,f] = infr.rho_C_hex(bw_cntrfctl[m,f], R_star[m,f], gamma_cntrfctl[m], areatype=areatype)
            ccs_ = cc_cntrfctl[0,:]
            ccs_per_bw_ = (cc_cntrfctl / bw_cntrfctl[0,:])[0,:]
            avg_path_losses_ = np.zeros(R_star.shape[1])
            for f in range(R_star.shape[1]):
                avg_path_losses_[f] = infr.avg_path_loss(R_stars_[f], areatype=areatype)
            num_stations_cntrfctl = infr.num_stations(R_stars_, market_size_cntrfctl)
            avg_SINR_ = np.zeros(R_star.shape[1])
            for f in range(R_star.shape[1]):
                avg_SINR_[f] = infr.avg_SINR(R_stars_[f], areatype=areatype)
            
            return success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_
        
        # Calculate equilibrium with asymmetric firms
        if np.isin(num_firms, np.array([3])):
            start = time.time()
            bw_cntrfctl = market_bw * np.array([[0.5, 0.25, 0.25]])
            c_R_cntrfctl_asymmetric = per_base_station_per_bw_cost * bw_cntrfctl
            expand_to_num_firms = lambda x: np.tile(x, 3)
            product_firm_correspondence = np.repeat(np.arange(num_firms), num_prods)
            success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = asymmetric_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, expand_to_num_firms(xis_cntrfctl), theta, pop_cntrfctl, market_size_cntrfctl, expand_to_num_firms(c_u_cntrfctl), c_R_cntrfctl_asymmetric, expand_to_num_firms(R_0), expand_to_num_firms(p_0), product_firm_correspondence)
            if print_updates:
                print(f"theta_n={theta_n}: Finished calculating asymmetric equilibrium in {np.round(time.time() - start, 1)} seconds.", flush=True)
            successful_asymmetric_allbw[1], cs_by_type_asymmetric_allbw[1,:], cs_asymmetric_allbw[1], ps_asymmetric_allbw[1], ts_asymmetric_allbw[1] = success, cs_by_type_, cs_, ps_, ts_
            select_idx = np.array([0,1]) # select one of the big bw firms and one of the small bw firms
            select_idx_prod = np.concatenate((np.arange(num_prods) + 0 * num_prods, np.arange(num_prods) + 2 * num_prods)) # select the products of one of the big bw firms and products of the small bw firms
            p_stars_asymmetric_allbw[1,:,:], R_stars_asymmetric_allbw[1,:], num_stations_stars_asymmetric_allbw[1], num_stations_per_firm_stars_asymmetric_allbw[1,:], q_stars_asymmetric_allbw[1,:], ccs_asymmetric_allbw[1,:], ccs_per_bw_asymmetric_allbw[1,:], avg_path_losses_asymmetric_allbw[1,:], avg_SINR_asymmetric_allbw[1,:] = np.reshape(p_stars_[select_idx_prod], (2, num_prods)), R_stars_[select_idx], num_stations_stars_, num_stations_per_firm_stars_[select_idx], q_stars_[select_idx], ccs_[select_idx], ccs_per_bw_[select_idx], avg_path_losses_[select_idx], avg_SINR_[select_idx]
            
            # Add idx 0, which is just copied over from symmetric equilibrium
            successful_asymmetric_allbw[0], cs_by_type_asymmetric_allbw[0,:], cs_asymmetric_allbw[0], ps_asymmetric_allbw[0], ts_asymmetric_allbw[0] = successful[i], cs_by_type[i,:], cs[i], ps[i], ts[i]
            p_stars_asymmetric_allbw[0,:,:], R_stars_asymmetric_allbw[0,:], num_stations_stars_asymmetric_allbw[0], num_stations_per_firm_stars_asymmetric_allbw[0,:], q_stars_asymmetric_allbw[0,:], ccs_asymmetric_allbw[0,:], ccs_per_bw_asymmetric_allbw[0,:], avg_path_losses_asymmetric_allbw[0,:], avg_SINR_asymmetric_allbw[0,:] = p_stars[i,:][np.newaxis,:], R_stars[i], num_stations_stars[i], num_stations_per_firm_stars[i], q_stars[i], ccs[i], ccs_per_bw[i], avg_path_losses[i], avg_SINR[i]
            
            if check_for_multiplicity_asymmetric and (task_id == 0) and (theta_n == 0):
                np.random.seed(123456)
                num_tests = 30
                success_test = np.zeros((num_tests,), dtype=bool)
                p_stars_test = np.ones((6, num_tests))
                R_stars_test = np.ones((3, num_tests))
                for test_i in range(num_tests):
                    start = time.time()
                    R_0_test = np.array([0.5, 0.5, 0.5]) + np.random.uniform(-0.1, 0.35, size=3)
                    p_0_test = np.array([10.0, 20.0, 10.0, 20.0, 10.0, 20.0]) + np.random.uniform(-1.0, 10.0, size=6)
                    asymmetric_eqm_test_res = asymmetric_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, expand_to_num_firms(xis_cntrfctl), theta, pop_cntrfctl, market_size_cntrfctl, expand_to_num_firms(c_u_cntrfctl), c_R_cntrfctl_asymmetric, R_0_test, p_0_test, product_firm_correspondence)
                    success_test[test_i], p_stars_test[:,test_i], R_stars_test[:,test_i] = asymmetric_eqm_test_res[0], asymmetric_eqm_test_res[5], asymmetric_eqm_test_res[6]
                np.save(f"{paths.arrays_path}success_test.npy", success_test)
                np.save(f"{paths.arrays_path}p_stars_test.npy", p_stars_test)
                np.save(f"{paths.arrays_path}R_stars_test.npy", R_stars_test)
        
        # Calculate equilibrium of different mergers
        if num_firms == 1: # has nothing to do with monopoly case, just doesn't depend on number of firms so choosing first case
            
            start_asym_all = time.time()
            
            # Create arrays that describe this market
            bw_cntrfctl = np.copy(bw_4g_equiv[constructed_markets_idx,:])
            xis_cntrfctl = np.copy(xis[constructed_markets_idx,:])
            c_R_cntrfctl = np.copy(c_R[constructed_markets_idx,:])
            R_0 = np.copy(radius[constructed_markets_idx,:])
            R_0_longrun = np.ones(R_0.shape) * 0.5 # this is better behaved when we allow all to adjust
            p_0 = np.copy(c_u)
            p_0_longrun = np.copy(c_u) # this is better behaved when we allow all to adjust
            product_firm_correspondence = np.unique(ds.firms, return_inverse=True)[1]
            
            # Create ds with properties I want
            ds_cntrfctl_income_distribution = np.copy(ds.data[constructed_markets_idx,0,yc1idx:yclastidx+1])
            ds_cntrfctl = copy.deepcopy(ds) # can rename it ds_cntrfctl b/c don't need to use old ds_cntrfctl again in this iteration of num_firms
            ds_cntrfctl.data = np.copy(ds.data[constructed_markets_idx,:,:]) # just copy over from the true ds, no need to adjust pidx, dlimidx, vlimdx, Oidx
            ds_cntrfctl.data[:,:,qidx] = np.zeros((constructed_markets_idx.shape[0], ds.J)) # doesn't matter, so just initialize
            ds_cntrfctl.data[:,:,yc1idx:yclastidx+1] = ds_cntrfctl_income_distribution[:,np.newaxis,:] # same income distribution for the other counterfactuals
            
            # Create other market description variables with properties I want
            pop_cntrfctl_merge = population[constructed_markets_idx]
            market_size_cntrfctl_merge = area[constructed_markets_idx,np.newaxis]
            market_weights_merge = np.copy(frac_population)
            gamma_cntrfctl_merge = lamda[constructed_markets_idx]
            
            for j in range(num_combos_merge):# go through each possible merger
                start = time.time()
                
                # Determine the merger
                merger_ = j > 0 # j == 0 is just the case with no merger
                merger_indices = np.array(merger_combos)[j-1,:] - 1 # j-1 b/c we're starting index with no merger case, -1 b/c firm numbering starts at 1
                num_firms_woMVNO = bw_cntrfctl.shape[1] - 1 if merger_ else bw_cntrfctl.shape[1]
                bw_cntrfctl_merge = np.zeros((bw_cntrfctl.shape[0], num_firms_woMVNO))
                if merger_:
                    bw_cntrfctl_merge[:,np.min(merger_indices)] = np.sum(bw_cntrfctl[:,merger_indices], axis=1) # for merged firm, bw is the sum of the two firms' bw
                    bw_cntrfctl_merge[:,np.arange(bw_cntrfctl_merge.shape[1]) != np.min(merger_indices)] = bw_cntrfctl[:,~np.isin(np.arange(bw_cntrfctl.shape[1]), merger_indices)] # for all other firms, their bw is the same
                else:
                    bw_cntrfctl_merge[:,:] = np.copy(bw_cntrfctl)
                ds_cntrfctl_merge = copy.deepcopy(ds_cntrfctl)
                if merger_:
                    smaller_idx_merge, larger_idx_merge = np.min(merger_indices + 1), np.max(merger_indices + 1) # +1 b/c want the indexing used in ds.firms
                    ds_cntrfctl_merge.firms[ds_cntrfctl_merge.firms == larger_idx_merge] = smaller_idx_merge # replace firm index
                xis_cntrfctl_merge = np.copy(xis_cntrfctl) # nothing changes about them in the merger
                c_R_cntrfctl_merge = np.zeros((c_R_cntrfctl.shape[0], num_firms_woMVNO))
                if merger_:
                    c_R_cntrfctl_merge[:,np.min(merger_indices)] = np.min((c_R / bw_4g_equiv)[np.ix_(constructed_markets_idx,merger_indices)], axis=1) # for merged firm, c_R is based on the minimum of the two firms' per-bw base station cost
                    c_R_cntrfctl_merge[:,np.arange(bw_cntrfctl_merge.shape[1]) != np.min(merger_indices)] = (c_R / bw_4g_equiv)[np.ix_(constructed_markets_idx,~np.isin(np.arange(bw_cntrfctl.shape[1]), merger_indices))] # for all other firms, their c_R is the same
                    c_R_cntrfctl_merge = c_R_cntrfctl_merge * bw_cntrfctl_merge # account for how much bandwidth each firm has since this is a specification that scales with bandwidth
                else:
                    c_R_cntrfctl_merge[:,:] = np.copy(c_R_cntrfctl)
                R_0_merge = np.zeros((R_0.shape[0], num_firms_woMVNO)) # this is the exact radius, we are not changing it
                if merger_:
                    R_0_merge[:,np.min(merger_indices)] = radius_mergers_all[constructed_markets_idx,:,:][:,np.min(merger_indices),j] # for merged firm, it's whatever the merged firm's radius is in the data (held fixed in short run and starting point in long run)
                    R_0_merge[:,np.arange(R_0_merge.shape[1]) != np.min(merger_indices)] = radius_mergers_all[constructed_markets_idx,:,:][:,~np.isin(np.arange(R_0.shape[1]), merger_indices),j] # for all other firms, their radii are the same
                else:
                    R_0_merge[:,:] = np.copy(radius_mergers_all[constructed_markets_idx,:,:][:,:,j])
                p_0_merge = np.copy(p_0) # nothing changes here for the merger
                product_firm_correspondence_merge = np.unique(ds_cntrfctl_merge.firms, return_inverse=True)[1]
                if merger_ and impute_MVNO['impute']:
                    impute_MVNO_merge_firms_share = np.ones(num_firms_woMVNO)
                    impute_MVNO_merge_firms_share[np.min(merger_indices)] = np.mean(impute_MVNO['firms_share'][merger_indices])
                    impute_MVNO_merge_firms_share[np.arange(impute_MVNO_merge_firms_share.shape[0]) != np.min(merger_indices)] = np.copy(impute_MVNO['firms_share'])[~np.isin(np.arange(num_firms_woMVNO + 1), merger_indices)]
                    impute_MVNO_merge = {
                        'impute': True, 
                        'firms_share': impute_MVNO_merge_firms_share, 
                        'include': True
                    }
                else:
                    impute_MVNO_merge = {
                        'impute': impute_MVNO['impute'], 
                        'firms_share': impute_MVNO['firms_share'], 
                        'include': impute_MVNO['include']
                    }
                
                # Calculate the short-run equilibrium
                success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = asymmetric_eqm(bw_cntrfctl_merge, gamma_cntrfctl_merge, ds_cntrfctl_merge, xis_cntrfctl_merge, theta, pop_cntrfctl_merge, market_size_cntrfctl_merge, c_u, c_R_cntrfctl_merge, R_0_merge, p_0_merge, product_firm_correspondence_merge, impute_MVNO=impute_MVNO_merge, R_fixed=True, market_weights=market_weights_merge)
                if not merger_: # replace R_0_longrun with the optimal non-merger case
                    p_0[:] = np.copy(p_stars_)
                if print_updates:
                    print(f"theta_n={theta_n}: Finished calculating short-run asymmetric equilibrium for merger {j} in {np.round(time.time() - start, 1)} seconds.", flush=True)
                    if not success:
                        print(f"\ttheta_n={theta_n}: short-run asymmetric equilibrium for merger {j} in not successful", flush=True)
                
                # Add to arrays
                expand_w_nan = lambda var, idx: np.concatenate((var[:idx], np.array([np.nan]), var[idx:])) if merger_ else var # add a NaN to the location of the merged (now non-existent) firm, assuming var is 1-d
                idx_acquired_firm = np.max(merger_indices)
                successful_shortrunall[j], cs_by_type_shortrunall[j,:], cs_shortrunall[j], ps_shortrunall[j], ts_shortrunall[j] = success, cs_by_type_, cs_, ps_, ts_
                p_stars_shortrunall[j,:], R_stars_shortrunall[j,:], num_stations_stars_shortrunall[j], num_stations_per_firm_stars_shortrunall[j,:], q_stars_shortrunall[j,:], ccs_shortrunall[j,:], ccs_per_bw_shortrunall[j,:], avg_path_losses_shortrunall[j,:] = p_stars_, expand_w_nan(R_stars_, idx_acquired_firm), num_stations_stars_, expand_w_nan(num_stations_per_firm_stars_, idx_acquired_firm), expand_w_nan(q_stars_, idx_acquired_firm), expand_w_nan(ccs_, idx_acquired_firm), expand_w_nan(ccs_per_bw_, idx_acquired_firm), expand_w_nan(avg_path_losses_, idx_acquired_firm)
                
                # Calculate the long-run equilibrium
                if long_run_merger_calc:
                    p_0_merge = np.copy(p_0_longrun)
                    if merger_:
                        select_indices = ~np.isin(np.arange(R_0.shape[1]), merger_indices) | (np.arange(R_0.shape[1]) == np.min(merger_indices))
                        R_0_merge[:,:] = R_0_longrun[:,select_indices]
                        p_0_merge[-1] = 2.0 * p_0_merge[-1] # o/w poorly behaved
                    else:
                        R_0_merge[:,:] = np.copy(R_0_longrun)
                    success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = asymmetric_eqm(bw_cntrfctl_merge, gamma_cntrfctl_merge, ds_cntrfctl_merge, xis_cntrfctl_merge, theta, pop_cntrfctl_merge, market_size_cntrfctl_merge, c_u, c_R_cntrfctl_merge, R_0_merge, p_0_merge, product_firm_correspondence_merge, impute_MVNO=impute_MVNO_merge, R_fixed=False, market_weights=market_weights_merge, factor=0.1, calc_q_carefully=True, print_addl_msg=True if (theta_n == 0) and print_updates else False)
                    if not merger_: # replace R_0_longrun with the optimal non-merger case
                        R_0_longrun[:,:] = np.copy(R_stars_)
                    if print_updates:
                        print(f"theta_n={theta_n}: Finished calculating long-run asymmetric equilibrium for merger {j} in {np.round(time.time() - start, 1)} seconds.", flush=True)
                        if not success:
                            print(f"\ttheta_n={theta_n}: long-run asymmetric equilibrium for merger {j} in not successful", flush=True)

                    # Add to arrays
                    successful_longrunall[j], cs_by_type_longrunall[j,:], cs_longrunall[j], ps_longrunall[j], ts_longrunall[j] = success, cs_by_type_, cs_, ps_, ts_
                    p_stars_longrunall[j,:], R_stars_longrunall[j,:], num_stations_stars_longrunall[j], num_stations_per_firm_stars_longrunall[j,:], q_stars_longrunall[j,:], ccs_longrunall[j,:], ccs_per_bw_longrunall[j,:], avg_path_losses_longrunall[j,:] = p_stars_, expand_w_nan(R_stars_, idx_acquired_firm), num_stations_stars_, expand_w_nan(num_stations_per_firm_stars_, idx_acquired_firm), expand_w_nan(q_stars_, idx_acquired_firm), expand_w_nan(ccs_, idx_acquired_firm), expand_w_nan(ccs_per_bw_, idx_acquired_firm), expand_w_nan(avg_path_losses_, idx_acquired_firm)
                
            if print_updates:
                if long_run_merger_calc:
                    print(f"theta_n={theta_n}: Finished calculating short- and long-run mergers in {np.round(time.time() - start_asym_all, 1)} seconds.", flush=True)
                else:
                    print(f"theta_n={theta_n}: Finished calculating short--run mergers in {np.round(time.time() - start_asym_all, 1)} seconds.", flush=True)
        
        if print_updates:
            print(f"theta_n={theta_n}, num_firms={num_firms} complete.", flush=True)
        
    # Put number of firms in per-person terms
    num_stations_stars = num_stations_stars / rep_population
    num_stations_per_firm_stars = num_stations_per_firm_stars / rep_population
    num_stations_stars_allfixed = num_stations_stars_allfixed / rep_population
    num_stations_per_firm_stars_allfixed = num_stations_per_firm_stars_allfixed / rep_population
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
    num_stations_stars_asymmetric_allbw = num_stations_stars_asymmetric_allbw / rep_population
    num_stations_per_firm_stars_asymmetric_allbw = num_stations_per_firm_stars_asymmetric_allbw / rep_population
    num_stations_stars_shortrunall = num_stations_stars_shortrunall / rep_population
    num_stations_per_firm_stars_shortrunall = num_stations_per_firm_stars_shortrunall / rep_population
    num_stations_stars_longrunall = num_stations_stars_longrunall / rep_population
    num_stations_per_firm_stars_longrunall = num_stations_per_firm_stars_longrunall / rep_population
    
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
    num_stations_per_firm_stars_free_allbw = num_stations_per_firm_stars_free_allbw - num_stations_per_firm_stars[np.isin(num_firms_array, np.array([3]))]
    q_stars_free_allbw = q_stars_free_allbw - q_stars[np.isin(num_firms_array, np.array([3]))]
    cs_free_allbw = cs_free_allbw - cs[np.isin(num_firms_array_extend, np.array([3]))]
    cs_by_type_free_allbw = cs_by_type_free_allbw - cs_by_type[np.isin(num_firms_array_extend, np.array([3])),:]
    ps_free_allbw = ps_free_allbw - ps[np.isin(num_firms_array_extend, np.array([3]))]
    ts_free_allbw = ts_free_allbw - ts[np.isin(num_firms_array_extend, np.array([3]))]
    ccs_free_allbw = ccs_free_allbw - ccs[np.isin(num_firms_array, np.array([3]))]
    ccs_per_bw_free_allbw = ccs_per_bw_free_allbw - ccs_per_bw[np.isin(num_firms_array, np.array([3]))]
    avg_path_losses_free_allbw = avg_path_losses_free_allbw - avg_path_losses[np.isin(num_firms_array, np.array([3]))]
    
    p_stars_asymmetric_allbw[2,:,:] = p_stars_asymmetric_allbw[0,:,:] - p_stars_asymmetric_allbw[1,:,:]
    R_stars_asymmetric_allbw[2,:] = R_stars_asymmetric_allbw[0,:] - R_stars_asymmetric_allbw[1,:]
    num_stations_stars_asymmetric_allbw[2] = num_stations_stars_asymmetric_allbw[0] - num_stations_stars_asymmetric_allbw[1]
    num_stations_per_firm_stars_asymmetric_allbw[2,:] = num_stations_per_firm_stars_asymmetric_allbw[0,:] - num_stations_per_firm_stars_asymmetric_allbw[1,:]
    q_stars_asymmetric_allbw[2,:] = q_stars_asymmetric_allbw[0,:] - q_stars_asymmetric_allbw[1,:]
    cs_asymmetric_allbw[2] = cs_asymmetric_allbw[0] - cs_asymmetric_allbw[1]
    ps_asymmetric_allbw[2] = ps_asymmetric_allbw[0] - ps_asymmetric_allbw[1]
    ts_asymmetric_allbw[2] = ts_asymmetric_allbw[0] - ts_asymmetric_allbw[1]
    cs_by_type_asymmetric_allbw[2,:] = cs_by_type_asymmetric_allbw[0,:] - cs_by_type_asymmetric_allbw[1,:]
    ccs_asymmetric_allbw[2,:] = ccs_asymmetric_allbw[0,:] - ccs_asymmetric_allbw[1,:]
    ccs_per_bw_asymmetric_allbw[2,:] = ccs_per_bw_asymmetric_allbw[0,:] - ccs_per_bw_asymmetric_allbw[1,:]
    avg_path_losses_asymmetric_allbw[2,:] = avg_path_losses_asymmetric_allbw[0,:] - avg_path_losses_asymmetric_allbw[1,:]
    avg_SINR_asymmetric_allbw[2,:] = avg_SINR_asymmetric_allbw[0,:] - avg_SINR_asymmetric_allbw[1,:]
    successful_asymmetric_allbw[2] = successful_asymmetric_allbw[0] & successful_asymmetric_allbw[1]
    
    cs_asymmetric_allbw[:2] = cs_asymmetric_allbw[:2] - cs[3] # compare to the 4-firm symmetric case
    ps_asymmetric_allbw[:2] = ps_asymmetric_allbw[:2] - ps[3]
    ts_asymmetric_allbw[:2] = ts_asymmetric_allbw[:2] - ts[3]
    cs_by_type_asymmetric_allbw[:2,:] = cs_by_type_asymmetric_allbw[:2,:] - cs_by_type[3,:][np.newaxis,:]

    # Calculate welfare relative to monopoly case
    cs = cs - cs[0] #if elast_id > 0 else cs - cs[1]
    ps = ps - ps[0] #if elast_id > 0 else ps - ps[1]
    ts = ts - ts[0] #if elast_id > 0 else ts - ts[1]
    cs_by_type = cs_by_type - cs_by_type[0,:][np.newaxis,:] #if elast_id > 0 else cs_by_type - cs_by_type[1,:][np.newaxis,:]
    
    cs_allfixed = cs_allfixed - cs_allfixed[0]
    ps_allfixed = ps_allfixed - ps_allfixed[0]
    ts_allfixed = ts_allfixed - ts_allfixed[0]
    cs_by_type_allfixed = cs_by_type_allfixed - cs_by_type_allfixed[0,:][np.newaxis,:]

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
    
    cs_by_type_shortrunall = cs_by_type_shortrunall - cs_by_type_shortrunall[0]
    cs_shortrunall = cs_shortrunall - cs_shortrunall[0]
    ps_shortrunall = ps_shortrunall - ps_shortrunall[0]
    ts_shortrunall = ts_shortrunall - ts_shortrunall[0]
    
    cs_by_type_longrunall = cs_by_type_longrunall - cs_by_type_longrunall[0]
    cs_longrunall = cs_longrunall - cs_longrunall[0]
    ps_longrunall = ps_longrunall - ps_longrunall[0]
    ts_longrunall = ts_longrunall - ts_longrunall[0]
    
    if print_updates:
        print(f"theta_n={theta_n} finished computation.", flush=True)
        
    # Return per unit of bandwidth estimates of c_R
    c_R_per_unit_bw = c_R / bw_4g_equiv

    return p_stars, R_stars, num_stations_stars, num_stations_per_firm_stars, q_stars, cs_by_type, cs, ps, ts, ccs, ccs_per_bw, avg_path_losses, avg_SINR, full_elasts, partial_elasts, p_stars_allfixed, R_stars_allfixed, num_stations_stars_allfixed, num_stations_per_firm_stars_allfixed, q_stars_allfixed, cs_by_type_allfixed, cs_allfixed, ps_allfixed, ts_allfixed, ccs_allfixed, ccs_per_bw_allfixed, avg_path_losses_allfixed, avg_SINR_allfixed, partial_Pif_partial_bf_allfixed, partial_Piotherf_partial_bf_allfixed, partial_diffPif_partial_bf_allfixed, partial_Pif_partial_b_allfixed, partial_CS_partial_b_allfixed, partial_Pif_partial_bf_allbw, partial_Piotherf_partial_bf_allbw, partial_diffPif_partial_bf_allbw, partial_Pif_partial_b_allbw, partial_CS_partial_b_allbw, c_u, c_R_per_unit_bw, p_stars_shortrun, R_stars_shortrun, num_stations_stars_shortrun, num_stations_per_firm_stars_shortrun, q_stars_shortrun, cs_by_type_shortrun, cs_shortrun, ps_shortrun, ts_shortrun, ccs_shortrun, ccs_per_bw_shortrun, avg_path_losses_shortrun, p_stars_free_allfixed, R_stars_free_allfixed, num_stations_stars_free_allfixed, num_stations_per_firm_stars_free_allfixed, q_stars_free_allfixed, cs_by_type_free_allfixed, cs_free_allfixed, ps_free_allfixed, ts_free_allfixed, ccs_free_allfixed, ccs_per_bw_free_allfixed, avg_path_losses_free_allfixed, p_stars_free_allbw, R_stars_free_allbw, num_stations_stars_free_allbw, num_stations_per_firm_stars_free_allbw, q_stars_free_allbw, cs_by_type_free_allbw, cs_free_allbw, ps_free_allbw, ts_free_allbw, ccs_free_allbw, ccs_per_bw_free_allbw, avg_path_losses_free_allbw, p_stars_dens, R_stars_dens, num_stations_stars_dens, num_stations_per_firm_stars_dens, q_stars_dens, cs_dens, cs_by_type_dens, ps_dens, ts_dens, ccs_dens, ccs_per_bw_dens, avg_path_losses_dens, avg_SINR_dens, p_stars_bw, R_stars_bw, num_stations_stars_bw, num_stations_per_firm_stars_bw, q_stars_bw, cs_bw, cs_by_type_bw, ps_bw, ts_bw, ccs_bw, ccs_per_bw_bw, avg_path_losses_bw, avg_SINR_bw, p_stars_dens_1p, R_stars_dens_1p, num_stations_stars_dens_1p, num_stations_per_firm_stars_dens_1p, q_stars_dens_1p, cs_dens_1p, cs_by_type_dens_1p, ps_dens_1p, ts_dens_1p, ccs_dens_1p, ccs_per_bw_dens_1p, avg_path_losses_dens_1p, avg_SINR_dens_1p, p_stars_asymmetric_allbw, R_stars_asymmetric_allbw, num_stations_stars_asymmetric_allbw, num_stations_per_firm_stars_asymmetric_allbw, q_stars_asymmetric_allbw, cs_asymmetric_allbw, cs_by_type_asymmetric_allbw, ps_asymmetric_allbw, ts_asymmetric_allbw, ccs_asymmetric_allbw, ccs_per_bw_asymmetric_allbw, avg_path_losses_asymmetric_allbw, avg_SINR_asymmetric_allbw, p_stars_shortrunall, R_stars_shortrunall, num_stations_stars_shortrunall, num_stations_per_firm_stars_shortrunall, q_stars_shortrunall, cs_shortrunall, cs_by_type_shortrunall, ps_shortrunall, ts_shortrunall, ccs_shortrunall, ccs_per_bw_shortrunall, avg_path_losses_shortrunall, p_stars_longrunall, R_stars_longrunall, num_stations_stars_longrunall, num_stations_per_firm_stars_longrunall, q_stars_longrunall, cs_longrunall, cs_by_type_longrunall, ps_longrunall, ts_longrunall, ccs_longrunall, ccs_per_bw_longrunall, avg_path_losses_longrunall, successful, successful_allfixed, successful_bw_deriv_allfixed, successful_bw_deriv_allbw, successful_shortrun, successful_free_allfixed, successful_free_allbw, successful_dens, successful_bw, successful_dens_1p, successful_asymmetric_allbw, successful_shortrunall, successful_longrunall, per_user_costs
    
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

p_stars_allfixed = np.zeros((theta_N, num_firms_array.shape[0], num_prods))
R_stars_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
num_stations_stars_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
num_stations_per_firm_stars_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
q_stars_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))

cs_by_type_allfixed = np.zeros((theta_N, num_firms_array_extend.shape[0], yclastidx - yc1idx + 1))
cs_allfixed = np.zeros((theta_N, num_firms_array_extend.shape[0]))
ps_allfixed = np.zeros((theta_N, num_firms_array_extend.shape[0]))
ts_allfixed = np.zeros((theta_N, num_firms_array_extend.shape[0]))

ccs_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
ccs_per_bw_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
avg_path_losses_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
avg_SINR_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))

partial_Pif_partial_bf_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
partial_Piotherf_partial_bf_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
partial_diffPif_partial_bf_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
partial_Pif_partial_b_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
partial_CS_partial_b_allfixed = np.zeros((theta_N, num_firms_array.shape[0]))
partial_Pif_partial_bf_allbw = np.zeros((theta_N, num_firms_array.shape[0]))
partial_Piotherf_partial_bf_allbw = np.zeros((theta_N, num_firms_array.shape[0]))
partial_diffPif_partial_bf_allbw = np.zeros((theta_N, num_firms_array.shape[0]))
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

p_stars_asymmetric_allbw = np.zeros((theta_N, 3, 2, num_prods))
R_stars_asymmetric_allbw = np.zeros((theta_N, 3, 2))
num_stations_stars_asymmetric_allbw = np.zeros((theta_N, 3))
num_stations_per_firm_stars_asymmetric_allbw = np.zeros((theta_N, 3, 2))
q_stars_asymmetric_allbw = np.zeros((theta_N, 3, 2))
cs_asymmetric_allbw = np.zeros((theta_N, 3))
cs_by_type_asymmetric_allbw = np.zeros((theta_N, 3, yclastidx - yc1idx + 1))
ps_asymmetric_allbw = np.zeros((theta_N, 3))
ts_asymmetric_allbw = np.zeros((theta_N, 3))
ccs_asymmetric_allbw = np.zeros((theta_N, 3, 2))
ccs_per_bw_asymmetric_allbw = np.zeros((theta_N, 3, 2))
avg_path_losses_asymmetric_allbw = np.zeros((theta_N, 3, 2))
avg_SINR_asymmetric_allbw = np.zeros((theta_N, 3, 2))

num_prods_all = prices.shape[0] # number of products in data
num_firms_all = len(mno_codes.keys()) - 1 # number of firms in data (not including MVNO, which is an entry in mno_codes)
num_firms_allwMVNO = num_firms_all + 1 # number of firms in data (including MVNO)
num_combos_merge = 1 + int(special.comb(num_firms_all, 2)) # original (no mergers) + every possible combo of merger of two
p_stars_shortrunall = np.zeros((theta_N, num_combos_merge, num_prods_all))
R_stars_shortrunall = np.zeros((theta_N, num_combos_merge, num_firms_all))
num_stations_stars_shortrunall = np.zeros((theta_N, num_combos_merge))
num_stations_per_firm_stars_shortrunall = np.zeros((theta_N, num_combos_merge, num_firms_all))
q_stars_shortrunall = np.zeros((theta_N, num_combos_merge, num_firms_allwMVNO))
cs_shortrunall = np.zeros((theta_N, num_combos_merge))
cs_by_type_shortrunall = np.zeros((theta_N, num_combos_merge, yclastidx - yc1idx + 1))
ps_shortrunall = np.zeros((theta_N, num_combos_merge))
ts_shortrunall = np.zeros((theta_N, num_combos_merge))
ccs_shortrunall = np.zeros((theta_N, num_combos_merge, num_firms_all))
ccs_per_bw_shortrunall = np.zeros((theta_N, num_combos_merge, num_firms_all))
avg_path_losses_shortrunall = np.zeros((theta_N, num_combos_merge, num_firms_all))

p_stars_longrunall = np.zeros((theta_N, num_combos_merge, num_prods_all))
R_stars_longrunall = np.zeros((theta_N, num_combos_merge, num_firms_all))
num_stations_stars_longrunall = np.zeros((theta_N, num_combos_merge))
num_stations_per_firm_stars_longrunall = np.zeros((theta_N, num_combos_merge, num_firms_all))
q_stars_longrunall = np.zeros((theta_N, num_combos_merge, num_firms_allwMVNO))
cs_longrunall = np.zeros((theta_N, num_combos_merge))
cs_by_type_longrunall = np.zeros((theta_N, num_combos_merge, yclastidx - yc1idx + 1))
ps_longrunall = np.zeros((theta_N, num_combos_merge))
ts_longrunall = np.zeros((theta_N, num_combos_merge))
ccs_longrunall = np.zeros((theta_N, num_combos_merge, num_firms_all))
ccs_per_bw_longrunall = np.zeros((theta_N, num_combos_merge, num_firms_all))
avg_path_losses_longrunall = np.zeros((theta_N, num_combos_merge, num_firms_all))

successful_extend = np.ones((theta_N, num_firms_array_extend.shape[0]), dtype=bool)
successful_extend_allfixed = np.ones((theta_N, num_firms_array_extend.shape[0]), dtype=bool)
successful_bw_deriv_allfixed = np.ones((theta_N, num_firms_array.shape[0]), dtype=bool)
successful_bw_deriv_allbw = np.ones((theta_N, num_firms_array.shape[0]), dtype=bool)
successful_shortrun = np.ones((theta_N, 3), dtype=bool)
successful_free_allfixed = np.ones((theta_N, 2), dtype=bool)
successful_free_allbw = np.ones((theta_N, 2), dtype=bool)
successful_dens = np.ones((theta_N, num_firms_array_extend.shape[0], densities.shape[0]), dtype=bool)
successful_bw = np.ones((theta_N, num_firms_array_extend.shape[0], bw_vals.shape[0]), dtype=bool)
successful_dens_1p = np.ones((theta_N, num_firms_array_extend.shape[0], densities.shape[0]), dtype=bool)
successful_asymmetric_allbw = np.ones((theta_N, 3), dtype=bool)
successful_shortrunall = np.ones((theta_N, num_combos_merge), dtype=bool)
successful_longrunall = np.ones((theta_N, num_combos_merge), dtype=bool)

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
    p_stars_allfixed[idx,:,:] = res[15]
    R_stars_allfixed[idx,:] = res[16]
    num_stations_stars_allfixed[idx,:] = res[17]
    num_stations_per_firm_stars_allfixed[idx,:] = res[18]
    q_stars_allfixed[idx,:] = res[19]
    cs_by_type_allfixed[idx,:,:] = res[20]
    cs_allfixed[idx,:] = res[21]
    ps_allfixed[idx,:] = res[22]
    ts_allfixed[idx,:] = res[23]
    ccs_allfixed[idx,:] = res[24]
    ccs_per_bw_allfixed[idx,:] = res[25]
    avg_path_losses_allfixed[idx,:] = res[26]
    avg_SINR_allfixed[idx,:] = res[27]
    partial_Pif_partial_bf_allfixed[idx,:] = res[28]
    partial_Piotherf_partial_bf_allfixed[idx,:] = res[29]
    partial_diffPif_partial_bf_allfixed[idx,:] = res[30]
    partial_Pif_partial_b_allfixed[idx,:] = res[31]
    partial_CS_partial_b_allfixed[idx,:] = res[32]
    partial_Pif_partial_bf_allbw[idx,:] = res[33]
    partial_Piotherf_partial_bf_allbw[idx,:] = res[34]
    partial_diffPif_partial_bf_allbw[idx,:] = res[35]
    partial_Pif_partial_b_allbw[idx,:] = res[36]
    partial_CS_partial_b_allbw[idx,:] = res[37]
    c_u[idx,:] = res[38]
    c_R[idx,:,:] = res[39]
    p_stars_shortrun[idx,:,:] = res[40]
    R_stars_shortrun[idx,:] = res[41]
    num_stations_stars_shortrun[idx,:] = res[42]
    num_stations_per_firm_stars_shortrun[idx,:] = res[43]
    q_stars_shortrun[idx,:] = res[44]
    cs_by_type_shortrun[idx,:,:] = res[45]
    cs_shortrun[idx,:] = res[46]
    ps_shortrun[idx,:] = res[47]
    ts_shortrun[idx,:] = res[48]
    ccs_shortrun[idx,:] = res[49]
    ccs_per_bw_shortrun[idx,:] = res[50]
    avg_path_losses_shortrun[idx,:] = res[51]
    p_stars_free_allfixed[idx,:,:] = res[52]
    R_stars_free_allfixed[idx,:] = res[53]
    num_stations_stars_free_allfixed[idx,:] = res[54]
    num_stations_per_firm_stars_free_allfixed[idx,:] = res[55]
    q_stars_free_allfixed[idx,:] = res[56]
    cs_by_type_free_allfixed[idx,:,:] = res[57]
    cs_free_allfixed[idx,:] = res[58]
    ps_free_allfixed[idx,:] = res[59]
    ts_free_allfixed[idx,:] = res[60]
    ccs_free_allfixed[idx,:] = res[61]
    ccs_per_bw_free_allfixed[idx,:] = res[62]
    avg_path_losses_free_allfixed[idx,:] = res[63]
    p_stars_free_allbw[idx,:,:] = res[64]
    R_stars_free_allbw[idx,:] = res[65]
    num_stations_stars_free_allbw[idx,:] = res[66]
    num_stations_per_firm_stars_free_allbw[idx,:] = res[67]
    q_stars_free_allbw[idx,:] = res[68]
    cs_by_type_free_allbw[idx,:,:] = res[69]
    cs_free_allbw[idx,:] = res[70]
    ps_free_allbw[idx,:] = res[71]
    ts_free_allbw[idx,:] = res[72]
    ccs_free_allbw[idx,:] = res[73]
    ccs_per_bw_free_allbw[idx,:] = res[74]
    avg_path_losses_free_allbw[idx,:] = res[75]

    p_stars_dens[idx,:,:,:] = res[76]
    R_stars_dens[idx,:,:] = res[77]
    num_stations_stars_dens[idx,:,:] = res[78]
    num_stations_per_firm_stars_dens[idx,:,:] = res[79]
    q_stars_dens[idx,:,:] = res[80]
    cs_dens[idx,:,:] = res[81]
    cs_by_type_dens[idx,:,:,:] = res[82]
    ps_dens[idx,:,:] = res[83]
    ts_dens[idx,:,:] = res[84]
    ccs_dens[idx,:,:] = res[85]
    ccs_per_bw_dens[idx,:,:] = res[86]
    avg_path_losses_dens[idx,:,:] = res[87]
    avg_SINR_dens[idx,:,:] = res[88]

    p_stars_bw[idx,:,:,:] = res[89]
    R_stars_bw[idx,:,:] = res[90]
    num_stations_stars_bw[idx,:,:] = res[91]
    num_stations_per_firm_stars_bw[idx,:,:] = res[92]
    q_stars_bw[idx,:,:] = res[93]
    cs_bw[idx,:,:] = res[94]
    cs_by_type_bw[idx,:,:,:] = res[95]
    ps_bw[idx,:,:] = res[96]
    ts_bw[idx,:,:] = res[97]
    ccs_bw[idx,:,:] = res[98]
    ccs_per_bw_bw[idx,:,:] = res[99]
    avg_path_losses_bw[idx,:,:] = res[100]
    avg_SINR_bw[idx,:,:] = res[101]
    
    p_stars_dens_1p[idx,:,:,:] = res[102]
    R_stars_dens_1p[idx,:,:] = res[103]
    num_stations_stars_dens_1p[idx,:,:] = res[104]
    num_stations_per_firm_stars_dens_1p[idx,:,:] = res[105]
    q_stars_dens_1p[idx,:,:] = res[106]
    cs_dens_1p[idx,:,:] = res[107]
    cs_by_type_dens_1p[idx,:,:,:] = res[108]
    ps_dens_1p[idx,:,:] = res[109]
    ts_dens_1p[idx,:,:] = res[110]
    ccs_dens_1p[idx,:,:] = res[111]
    ccs_per_bw_dens_1p[idx,:,:] = res[112]
    avg_path_losses_dens_1p[idx,:,:] = res[113]
    avg_SINR_dens_1p[idx,:,:] = res[114]
    
    p_stars_asymmetric_allbw[idx,:,:,:] = res[115]
    R_stars_asymmetric_allbw[idx,:,:] = res[116]
    num_stations_stars_asymmetric_allbw[idx,:] = res[117]
    num_stations_per_firm_stars_asymmetric_allbw[idx,:,:] = res[118]
    q_stars_asymmetric_allbw[idx,:,:] = res[119]
    cs_asymmetric_allbw[idx,:] = res[120]
    cs_by_type_asymmetric_allbw[idx,:] = res[121]
    ps_asymmetric_allbw[idx,:] = res[122]
    ts_asymmetric_allbw[idx,:] = res[123]
    ccs_asymmetric_allbw[idx,:,:] = res[124]
    ccs_per_bw_asymmetric_allbw[idx,:,:] = res[125]
    avg_path_losses_asymmetric_allbw[idx,:,:] = res[126]
    avg_SINR_asymmetric_allbw[idx,:,:] = res[127]

    p_stars_shortrunall[idx,:,:] = res[128]
    R_stars_shortrunall[idx,:,:] = res[129]
    num_stations_stars_shortrunall[idx,:] = res[130]
    num_stations_per_firm_stars_shortrunall[idx,:,:] = res[131]
    q_stars_shortrunall[idx,:,:] = res[132]
    cs_shortrunall[idx,:] = res[133]
    cs_by_type_shortrunall[idx,:] = res[134]
    ps_shortrunall[idx,:] = res[135]
    ts_shortrunall[idx,:] = res[136]
    ccs_shortrunall[idx,:,:] = res[137]
    ccs_per_bw_shortrunall[idx,:,:] = res[138]
    avg_path_losses_shortrunall[idx,:,:] = res[139]
    
    p_stars_longrunall[idx,:,:] = res[140]
    R_stars_longrunall[idx,:,:] = res[141]
    num_stations_stars_longrunall[idx,:] = res[142]
    num_stations_per_firm_stars_longrunall[idx,:,:] = res[143]
    q_stars_longrunall[idx,:,:] = res[144]
    cs_longrunall[idx,:] = res[145]
    cs_by_type_longrunall[idx,:] = res[146]
    ps_longrunall[idx,:] = res[147]
    ts_longrunall[idx,:] = res[148]
    ccs_longrunall[idx,:,:] = res[149]
    ccs_per_bw_longrunall[idx,:,:] = res[150]
    avg_path_losses_longrunall[idx,:,:] = res[151]

    successful_extend[idx,:] = res[152]
    successful_extend_allfixed[idx,:] = res[153]
    successful_bw_deriv_allfixed[idx,:] = res[154]
    successful_bw_deriv_allbw[idx,:] = res[155]
    successful_shortrun[idx,:] = res[156]
    successful_free_allfixed[idx,:] = res[157]
    successful_free_allbw[idx,:] = res[158]
    successful_dens[idx,:,:] = res[159]
    successful_bw[idx,:,:] = res[160]
    successful_dens_1p[idx,:,:] = res[161]
    successful_asymmetric_allbw[idx,:] = res[162]
    successful_shortrunall[idx,:] = res[163]
    successful_longrunall[idx,:] = res[164]
    
    per_user_costs[idx,:] = res[165]
    
pool.close()

# %%
# Save arrays before processing
if save_bf:
    np.savez_compressed(f"{paths.arrays_path}all_arrays_{task_id}.npz", p_stars, R_stars, num_stations_stars, num_stations_per_firm_stars, q_stars, cs_by_type, cs, ps, ts, ccs, ccs_per_bw, avg_path_losses, avg_SINR, full_elasts, partial_elasts, p_stars_allfixed, R_stars_allfixed, num_stations_stars_allfixed, num_stations_per_firm_stars_allfixed, q_stars_allfixed, cs_by_type_allfixed, cs_allfixed, ps_allfixed, ts_allfixed, ccs_allfixed, ccs_per_bw_allfixed, avg_path_losses_allfixed, avg_SINR_allfixed, partial_Pif_partial_bf_allfixed, partial_Piotherf_partial_bf_allfixed, partial_diffPif_partial_bf_allfixed, partial_Pif_partial_b_allfixed, partial_CS_partial_b_allfixed, partial_Pif_partial_bf_allbw, partial_Piotherf_partial_bf_allbw, partial_diffPif_partial_bf_allbw, partial_Pif_partial_b_allbw, partial_CS_partial_b_allbw, c_u, c_R, p_stars_shortrun, R_stars_shortrun, num_stations_stars_shortrun, num_stations_per_firm_stars_shortrun, q_stars_shortrun, cs_by_type_shortrun, cs_shortrun, ps_shortrun, ts_shortrun, ccs_shortrun, ccs_per_bw_shortrun, avg_path_losses_shortrun, p_stars_free_allfixed, R_stars_free_allfixed, num_stations_stars_free_allfixed, num_stations_per_firm_stars_free_allfixed, q_stars_free_allfixed, cs_by_type_free_allfixed, cs_free_allfixed, ps_free_allfixed, ts_free_allfixed, ccs_free_allfixed, ccs_per_bw_free_allfixed, avg_path_losses_free_allfixed, p_stars_free_allbw, R_stars_free_allbw, num_stations_stars_free_allbw, num_stations_per_firm_stars_free_allbw, q_stars_free_allbw, cs_by_type_free_allbw, cs_free_allbw, ps_free_allbw, ts_free_allbw, ccs_free_allbw, ccs_per_bw_free_allbw, avg_path_losses_free_allbw, p_stars_dens, R_stars_dens, num_stations_stars_dens, num_stations_per_firm_stars_dens, q_stars_dens, cs_dens, cs_by_type_dens, ps_dens, ts_dens, ccs_dens, ccs_per_bw_dens, avg_path_losses_dens, avg_SINR_dens, p_stars_bw, R_stars_bw, num_stations_stars_bw, num_stations_per_firm_stars_bw, q_stars_bw, cs_bw, cs_by_type_bw, ps_bw, ts_bw, ccs_bw, ccs_per_bw_bw, avg_path_losses_bw, avg_SINR_bw, p_stars_dens_1p, R_stars_dens_1p, num_stations_stars_dens_1p, num_stations_per_firm_stars_dens_1p, q_stars_dens_1p, cs_dens_1p, cs_by_type_dens_1p, ps_dens_1p, ts_dens_1p, ccs_dens_1p, ccs_per_bw_dens_1p, avg_path_losses_dens_1p, avg_SINR_dens_1p, p_stars_asymmetric_allbw, R_stars_asymmetric_allbw, num_stations_stars_asymmetric_allbw, num_stations_per_firm_stars_asymmetric_allbw, q_stars_asymmetric_allbw, cs_asymmetric_allbw, cs_by_type_asymmetric_allbw, ps_asymmetric_allbw, ts_asymmetric_allbw, ccs_asymmetric_allbw, ccs_per_bw_asymmetric_allbw, avg_path_losses_asymmetric_allbw, avg_SINR_asymmetric_allbw, p_stars_shortrunall, R_stars_shortrunall, num_stations_stars_shortrunall, num_stations_per_firm_stars_shortrunall, q_stars_shortrunall, cs_shortrunall, cs_by_type_shortrunall, ps_shortrunall, ts_shortrunall, ccs_shortrunall, ccs_per_bw_shortrunall, avg_path_losses_shortrunall, p_stars_longrunall, R_stars_longrunall, num_stations_stars_longrunall, num_stations_per_firm_stars_longrunall, q_stars_longrunall, cs_longrunall, cs_by_type_longrunall, ps_longrunall, ts_longrunall, ccs_longrunall, ccs_per_bw_longrunall, avg_path_losses_longrunall, successful_extend, successful_extend_allfixed, successful_bw_deriv_allfixed, successful_bw_deriv_allbw, successful_shortrun, successful_free_allfixed, successful_free_allbw, successful_dens, successful_bw, successful_dens_1p, successful_asymmetric_allbw, successful_shortrunall, successful_longrunall, per_user_costs)

# %%
# Determine point estimates and standard errors

def asym_distribution(var_array, success_array, override_hBnan=False):
    """Determine the point estimate and standard errors given demand parameter using the Delta Method."""
    
    # Copy arrays
    var = np.copy(var_array)
    success = np.copy(success_array)
    
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
        
    # if 0 didn't work, just use one of the deviations, they're going to be approximately equal
    for j in range(var.shape[1]):
        if not success[0,j]:
            j_successful = np.where(success[:,j])[0]
            if j_successful.shape[0] > 0:
                idx_use = j_successful[0]
                var[0,...] = var[idx_use,...]
                success[0,j] = success[idx_use,j]
                print(f"index 0 of variable entry {j} not successful, therefore using index {idx_use}, which was", flush=True)
    
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
                    if override_hBnan and np.any(success[:,j]): # if some perturbations were successful and we're allowed to use perturbation values for point estimate (works if perturbations small)
                        i_acceptable = np.where(successful_dens[:,0,1])[0]
                        i_use = i_acceptable[1] if i_acceptable.shape[0] > 1 else i_acceptable[0]
                        hB[j,...] = var[i_use,j,...]
                    else:
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
successful_allfixed = successful_extend_allfixed[:,np.isin(num_firms_array_extend, num_firms_array)]
p_stars_allfixed, p_stars_allfixed_se = asym_distribution(p_stars_allfixed, successful_allfixed)
R_stars_allfixed, R_stars_allfixed_se = asym_distribution(R_stars_allfixed, successful_allfixed)
num_stations_stars_allfixed, num_stations_stars_allfixed_se = asym_distribution(num_stations_stars_allfixed, successful_allfixed)
num_stations_per_firm_stars_allfixed, num_stations_per_firm_stars_allfixed_se = asym_distribution(num_stations_per_firm_stars_allfixed, successful_allfixed)
q_stars_allfixed, q_stars_allfixed_se = asym_distribution(q_stars_allfixed, successful_allfixed)
cs_by_type_allfixed, cs_by_type_allfixed_se = asym_distribution(cs_by_type_allfixed, successful_extend_allfixed)
cs_allfixed, cs_allfixed_se = asym_distribution(cs_allfixed, successful_extend_allfixed)
ps_allfixed, ps_allfixed_se = asym_distribution(ps_allfixed, successful_extend_allfixed)
ts_allfixed, ts_allfixed_se = asym_distribution(ts_allfixed, successful_extend_allfixed)
ccs_allfixed, ccs_allfixed_se = asym_distribution(ccs_allfixed, successful_allfixed)
ccs_per_bw_allfixed, ccs_per_bw_allfixed_se = asym_distribution(ccs_per_bw_allfixed, successful_allfixed)
avg_path_losses_allfixed, avg_path_losses_allfixed_se = asym_distribution(avg_path_losses_allfixed, successful_allfixed)
avg_SINR_allfixed, avg_SINR_allfixed_se = asym_distribution(avg_SINR_allfixed, successful_allfixed)
full_elasts, full_elasts_se = asym_distribution(full_elasts, successful)
partial_elasts, partial_elasts_se = asym_distribution(partial_elasts, successful)
partial_Pif_partial_bf_allfixed, partial_Pif_partial_bf_allfixed_se = asym_distribution(partial_Pif_partial_bf_allfixed, successful_bw_deriv_allfixed)
partial_Piotherf_partial_bf_allfixed, partial_Piotherf_partial_bf_allfixed_se = asym_distribution(partial_Piotherf_partial_bf_allfixed, successful_bw_deriv_allfixed)
partial_diffPif_partial_bf_allfixed, partial_diffPif_partial_bf_allfixed_se = asym_distribution(partial_diffPif_partial_bf_allfixed, successful_bw_deriv_allfixed)
partial_Pif_partial_b_allfixed, partial_Pif_partial_b_allfixed_se = asym_distribution(partial_Pif_partial_b_allfixed, successful_bw_deriv_allfixed)
partial_CS_partial_b_allfixed, partial_CS_partial_b_allfixed_se = asym_distribution(partial_CS_partial_b_allfixed, successful_bw_deriv_allfixed)
partial_Pif_partial_bf_allbw, partial_Pif_partial_bf_allbw_se = asym_distribution(partial_Pif_partial_bf_allbw, successful_bw_deriv_allbw)
partial_Piotherf_partial_bf_allbw, partial_Piotherf_partial_bf_allbw_se = asym_distribution(partial_Piotherf_partial_bf_allbw, successful_bw_deriv_allbw)
partial_diffPif_partial_bf_allbw, partial_diffPif_partial_bf_allbw_se = asym_distribution(partial_diffPif_partial_bf_allbw, successful_bw_deriv_allbw)
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
p_stars_dens, p_stars_dens_se = asym_distribution(p_stars_dens, successful_dens_, override_hBnan=True)
R_stars_dens, R_stars_dens_se = asym_distribution(R_stars_dens, successful_dens_, override_hBnan=True)
num_stations_stars_dens, num_stations_stars_dens_se = asym_distribution(num_stations_stars_dens, successful_dens_, override_hBnan=True)
num_stations_per_firm_stars_dens, num_stations_per_firm_stars_dens_se = asym_distribution(num_stations_per_firm_stars_dens, successful_dens_, override_hBnan=True)
q_stars_dens, q_stars_dens_se = asym_distribution(q_stars_dens, successful_dens_, override_hBnan=True)
cs_by_type_dens, cs_by_type_dens_se = asym_distribution(cs_by_type_dens, successful_dens)
cs_dens, cs_dens_se = asym_distribution(cs_dens, successful_dens)
ps_dens, ps_dens_se = asym_distribution(ps_dens, successful_dens)
ts_dens, ts_dens_se = asym_distribution(ts_dens, successful_dens)
ccs_dens, ccs_dens_se = asym_distribution(ccs_dens, successful_dens_, override_hBnan=True)
ccs_per_bw_dens, ccs_per_bw_dens_se = asym_distribution(ccs_per_bw_dens, successful_dens_, override_hBnan=True)
avg_path_losses_dens, avg_path_losses_dens_se = asym_distribution(avg_path_losses_dens, successful_dens_, override_hBnan=True)
avg_SINR_dens, avg_SINR_dens_se = asym_distribution(avg_SINR_dens, successful_dens_, override_hBnan=True)
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
p_stars_asymmetric_allbw, p_stars_asymmetric_allbw_se = asym_distribution(p_stars_asymmetric_allbw, successful_asymmetric_allbw)
R_stars_asymmetric_allbw, R_stars_asymmetric_allbw_se = asym_distribution(R_stars_asymmetric_allbw, successful_asymmetric_allbw)
num_stations_stars_asymmetric_allbw, num_stations_stars_asymmetric_allbw_se = asym_distribution(num_stations_stars_asymmetric_allbw, successful_asymmetric_allbw)
num_stations_per_firm_stars_asymmetric_allbw, num_stations_per_firm_stars_asymmetric_allbw_se = asym_distribution(num_stations_per_firm_stars_asymmetric_allbw, successful_asymmetric_allbw)
q_stars_asymmetric_allbw, q_stars_asymmetric_allbw_se = asym_distribution(q_stars_asymmetric_allbw, successful_asymmetric_allbw)
cs_by_type_asymmetric_allbw, cs_by_type_asymmetric_allbw_se = asym_distribution(cs_by_type_asymmetric_allbw, successful_asymmetric_allbw)
cs_asymmetric_allbw, cs_asymmetric_allbw_se = asym_distribution(cs_asymmetric_allbw, successful_asymmetric_allbw)
ps_asymmetric_allbw, ps_asymmetric_allbw_se = asym_distribution(ps_asymmetric_allbw, successful_asymmetric_allbw)
ts_asymmetric_allbw, ts_asymmetric_allbw_se = asym_distribution(ts_asymmetric_allbw, successful_asymmetric_allbw)
ccs_asymmetric_allbw, ccs_asymmetric_allbw_se = asym_distribution(ccs_asymmetric_allbw, successful_asymmetric_allbw)
ccs_per_bw_asymmetric_allbw, ccs_per_bw_asymmetric_allbw_se = asym_distribution(ccs_per_bw_asymmetric_allbw, successful_asymmetric_allbw)
avg_path_losses_asymmetric_allbw, avg_path_losses_asymmetric_allbw_se = asym_distribution(avg_path_losses_asymmetric_allbw, successful_asymmetric_allbw)
avg_SINR_asymmetric_allbw, avg_SINR_asymmetric_allbw_se = asym_distribution(avg_SINR_asymmetric_allbw, successful_asymmetric_allbw)
p_stars_shortrunall, p_stars_shortrunall_se = asym_distribution(p_stars_shortrunall, successful_shortrunall)
R_stars_shortrunall, R_stars_shortrunall_se = asym_distribution(R_stars_shortrunall, successful_shortrunall)
num_stations_stars_shortrunall, num_stations_stars_shortrunall_se = asym_distribution(num_stations_stars_shortrunall, successful_shortrunall)
num_stations_per_firm_stars_shortrunall, num_stations_per_firm_stars_shortrunall_se = asym_distribution(num_stations_per_firm_stars_shortrunall, successful_shortrunall)
q_stars_shortrunall, q_stars_shortrunall_se = asym_distribution(q_stars_shortrunall, successful_shortrunall)
cs_by_type_shortrunall, cs_by_type_shortrunall_se = asym_distribution(cs_by_type_shortrunall, successful_shortrunall)
cs_shortrunall, cs_shortrunall_se = asym_distribution(cs_shortrunall, successful_shortrunall)
ps_shortrunall, ps_shortrunall_se = asym_distribution(ps_shortrunall, successful_shortrunall)
ts_shortrunall, ts_shortrunall_se = asym_distribution(ts_shortrunall, successful_shortrunall)
ccs_shortrunall, ccs_shortrunall_se = asym_distribution(ccs_shortrunall, successful_shortrunall)
ccs_per_bw_shortrunall, ccs_per_bw_shortrunall_se = asym_distribution(ccs_per_bw_shortrunall, successful_shortrunall)
avg_path_losses_shortrunall, avg_path_losses_shortrunall_se = asym_distribution(avg_path_losses_shortrunall, successful_shortrunall)
p_stars_longrunall, p_stars_longrunall_se = asym_distribution(p_stars_longrunall, successful_longrunall)
R_stars_longrunall, R_stars_longrunall_se = asym_distribution(R_stars_longrunall, successful_longrunall)
num_stations_stars_longrunall, num_stations_stars_longrunall_se = asym_distribution(num_stations_stars_longrunall, successful_longrunall)
num_stations_per_firm_stars_longrunall, num_stations_per_firm_stars_longrunall_se = asym_distribution(num_stations_per_firm_stars_longrunall, successful_longrunall)
q_stars_longrunall, q_stars_longrunall_se = asym_distribution(q_stars_longrunall, successful_longrunall)
cs_by_type_longrunall, cs_by_type_longrunall_se = asym_distribution(cs_by_type_longrunall, successful_longrunall)
cs_longrunall, cs_longrunall_se = asym_distribution(cs_longrunall, successful_longrunall)
ps_longrunall, ps_longrunall_se = asym_distribution(ps_longrunall, successful_longrunall)
ts_longrunall, ts_longrunall_se = asym_distribution(ts_longrunall, successful_longrunall)
ccs_longrunall, ccs_longrunall_se = asym_distribution(ccs_longrunall, successful_longrunall)
ccs_per_bw_longrunall, ccs_per_bw_longrunall_se = asym_distribution(ccs_per_bw_longrunall, successful_longrunall)
avg_path_losses_longrunall, avg_path_losses_longrunall_se = asym_distribution(avg_path_losses_longrunall, successful_longrunall)
per_user_costs, per_user_costs_se = asym_distribution(per_user_costs, np.ones((per_user_costs.shape[0], per_user_costs.shape[1]), dtype=bool)) # all should be successful
    
# %%
# Save variables

# Point estimates
np.save(f"{paths.arrays_path}p_stars_{task_id}.npy", p_stars)
np.save(f"{paths.arrays_path}R_stars_{task_id}.npy", R_stars)
np.save(f"{paths.arrays_path}num_stations_stars_{task_id}.npy", num_stations_stars)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_{task_id}.npy", num_stations_per_firm_stars)
np.save(f"{paths.arrays_path}q_stars_{task_id}.npy", q_stars)
np.save(f"{paths.arrays_path}cs_by_type_{task_id}.npy", cs_by_type)
np.save(f"{paths.arrays_path}cs_{task_id}.npy", cs)
np.save(f"{paths.arrays_path}ps_{task_id}.npy", ps)
np.save(f"{paths.arrays_path}ts_{task_id}.npy", ts)
np.save(f"{paths.arrays_path}ccs_{task_id}.npy", ccs)
np.save(f"{paths.arrays_path}ccs_per_bw_{task_id}.npy", ccs_per_bw)
np.save(f"{paths.arrays_path}avg_path_losses_{task_id}.npy", avg_path_losses)
np.save(f"{paths.arrays_path}avg_SINR_{task_id}.npy", avg_SINR)
np.save(f"{paths.arrays_path}p_stars_allfixed_{task_id}.npy", p_stars_allfixed)
np.save(f"{paths.arrays_path}R_stars_allfixed_{task_id}.npy", R_stars_allfixed)
np.save(f"{paths.arrays_path}num_stations_stars_allfixed_{task_id}.npy", num_stations_stars_allfixed)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_allfixed_{task_id}.npy", num_stations_per_firm_stars_allfixed)
np.save(f"{paths.arrays_path}q_stars_allfixed_{task_id}.npy", q_stars_allfixed)
np.save(f"{paths.arrays_path}cs_by_type_allfixed_{task_id}.npy", cs_by_type_allfixed)
np.save(f"{paths.arrays_path}cs_allfixed_{task_id}.npy", cs_allfixed)
np.save(f"{paths.arrays_path}ps_allfixed_{task_id}.npy", ps_allfixed)
np.save(f"{paths.arrays_path}ts_allfixed_{task_id}.npy", ts_allfixed)
np.save(f"{paths.arrays_path}ccs_allfixed_{task_id}.npy", ccs_allfixed)
np.save(f"{paths.arrays_path}ccs_per_bw_allfixed_{task_id}.npy", ccs_per_bw_allfixed)
np.save(f"{paths.arrays_path}avg_path_losses_allfixed_{task_id}.npy", avg_path_losses_allfixed)
np.save(f"{paths.arrays_path}avg_SINR_allfixed_{task_id}.npy", avg_SINR_allfixed)
np.save(f"{paths.arrays_path}full_elasts_{task_id}.npy", full_elasts)
np.save(f"{paths.arrays_path}partial_elasts_{task_id}.npy", partial_elasts)
np.save(f"{paths.arrays_path}partial_Pif_partial_bf_allfixed_{task_id}.npy", partial_Pif_partial_bf_allfixed)
np.save(f"{paths.arrays_path}partial_Piotherf_partial_bf_allfixed_{task_id}.npy", partial_Piotherf_partial_bf_allfixed)
np.save(f"{paths.arrays_path}partial_diffPif_partial_bf_allfixed_{task_id}.npy", partial_diffPif_partial_bf_allfixed)
np.save(f"{paths.arrays_path}partial_Pif_partial_b_allfixed_{task_id}.npy", partial_Pif_partial_b_allfixed)
np.save(f"{paths.arrays_path}partial_CS_partial_b_allfixed_{task_id}.npy", partial_CS_partial_b_allfixed)
np.save(f"{paths.arrays_path}partial_Pif_partial_bf_allbw_{task_id}.npy", partial_Pif_partial_bf_allbw)
np.save(f"{paths.arrays_path}partial_Piotherf_partial_bf_allbw_{task_id}.npy", partial_Piotherf_partial_bf_allbw)
np.save(f"{paths.arrays_path}partial_diffPif_partial_bf_allbw_{task_id}.npy", partial_diffPif_partial_bf_allbw)
np.save(f"{paths.arrays_path}partial_Pif_partial_b_allbw_{task_id}.npy", partial_Pif_partial_b_allbw)
np.save(f"{paths.arrays_path}partial_CS_partial_b_allbw_{task_id}.npy", partial_CS_partial_b_allbw)
np.save(f"{paths.arrays_path}c_u_{task_id}.npy", c_u)
np.save(f"{paths.arrays_path}c_R_{task_id}.npy", c_R)
np.save(f"{paths.arrays_path}p_stars_shortrun_{task_id}.npy", p_stars_shortrun)
np.save(f"{paths.arrays_path}R_stars_shortrun_{task_id}.npy", R_stars_shortrun)
np.save(f"{paths.arrays_path}num_stations_stars_shortrun_{task_id}.npy", num_stations_stars_shortrun)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_shortrun_{task_id}.npy", num_stations_per_firm_stars_shortrun)
np.save(f"{paths.arrays_path}q_stars_shortrun_{task_id}.npy", q_stars_shortrun)
np.save(f"{paths.arrays_path}cs_by_type_shortrun_{task_id}.npy", cs_by_type_shortrun)
np.save(f"{paths.arrays_path}cs_shortrun_{task_id}.npy", cs_shortrun)
np.save(f"{paths.arrays_path}ps_shortrun_{task_id}.npy", ps_shortrun)
np.save(f"{paths.arrays_path}ts_shortrun_{task_id}.npy", ts_shortrun)
np.save(f"{paths.arrays_path}ccs_shortrun_{task_id}.npy", ccs_shortrun)
np.save(f"{paths.arrays_path}ccs_per_bw_shortrun_{task_id}.npy", ccs_per_bw_shortrun)
np.save(f"{paths.arrays_path}avg_path_losses_shortrun_{task_id}.npy", avg_path_losses_shortrun)
np.save(f"{paths.arrays_path}p_stars_free_allfixed_{task_id}.npy", p_stars_free_allfixed)
np.save(f"{paths.arrays_path}R_stars_free_allfixed_{task_id}.npy", R_stars_free_allfixed)
np.save(f"{paths.arrays_path}num_stations_stars_free_allfixed_{task_id}.npy", num_stations_stars_free_allfixed)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_free_allfixed_{task_id}.npy", num_stations_per_firm_stars_free_allfixed)
np.save(f"{paths.arrays_path}q_stars_free_allfixed_{task_id}.npy", q_stars_free_allfixed)
np.save(f"{paths.arrays_path}cs_by_type_free_allfixed_{task_id}.npy", cs_by_type_free_allfixed)
np.save(f"{paths.arrays_path}cs_free_allfixed_{task_id}.npy", cs_free_allfixed)
np.save(f"{paths.arrays_path}ps_free_allfixed_{task_id}.npy", ps_free_allfixed)
np.save(f"{paths.arrays_path}ts_free_allfixed_{task_id}.npy", ts_free_allfixed)
np.save(f"{paths.arrays_path}ccs_free_allfixed_{task_id}.npy", ccs_free_allfixed)
np.save(f"{paths.arrays_path}ccs_per_bw_free_allfixed_{task_id}.npy", ccs_per_bw_free_allfixed)
np.save(f"{paths.arrays_path}avg_path_losses_free_allfixed_{task_id}.npy", avg_path_losses_free_allfixed)
np.save(f"{paths.arrays_path}p_stars_free_allbw_{task_id}.npy", p_stars_free_allbw)
np.save(f"{paths.arrays_path}R_stars_free_allbw_{task_id}.npy", R_stars_free_allbw)
np.save(f"{paths.arrays_path}num_stations_stars_free_allbw_{task_id}.npy", num_stations_stars_free_allbw)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_free_allbw_{task_id}.npy", num_stations_per_firm_stars_free_allbw)
np.save(f"{paths.arrays_path}q_stars_free_allbw_{task_id}.npy", q_stars_free_allbw)
np.save(f"{paths.arrays_path}cs_by_type_free_allbw_{task_id}.npy", cs_by_type_free_allbw)
np.save(f"{paths.arrays_path}cs_free_allbw_{task_id}.npy", cs_free_allbw)
np.save(f"{paths.arrays_path}ps_free_allbw_{task_id}.npy", ps_free_allbw)
np.save(f"{paths.arrays_path}ts_free_allbw_{task_id}.npy", ts_free_allbw)
np.save(f"{paths.arrays_path}ccs_free_allbw_{task_id}.npy", ccs_free_allbw)
np.save(f"{paths.arrays_path}ccs_per_bw_free_allbw_{task_id}.npy", ccs_per_bw_free_allbw)
np.save(f"{paths.arrays_path}avg_path_losses_free_allbw_{task_id}.npy", avg_path_losses_free_allbw)
np.save(f"{paths.arrays_path}p_stars_dens_{task_id}.npy", p_stars_dens)
np.save(f"{paths.arrays_path}R_stars_dens_{task_id}.npy", R_stars_dens)
np.save(f"{paths.arrays_path}num_stations_stars_dens_{task_id}.npy", num_stations_stars_dens)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_dens_{task_id}.npy", num_stations_per_firm_stars_dens)
np.save(f"{paths.arrays_path}q_stars_dens_{task_id}.npy", q_stars_dens)
np.save(f"{paths.arrays_path}cs_by_type_dens_{task_id}.npy", cs_by_type_dens)
np.save(f"{paths.arrays_path}cs_dens_{task_id}.npy", cs_dens)
np.save(f"{paths.arrays_path}ps_dens_{task_id}.npy", ps_dens)
np.save(f"{paths.arrays_path}ts_dens_{task_id}.npy", ts_dens)
np.save(f"{paths.arrays_path}ccs_dens_{task_id}.npy", ccs_dens)
np.save(f"{paths.arrays_path}ccs_per_bw_dens_{task_id}.npy", ccs_per_bw_dens)
np.save(f"{paths.arrays_path}avg_path_losses_dens_{task_id}.npy", avg_path_losses_dens)
np.save(f"{paths.arrays_path}avg_SINR_dens_{task_id}.npy", avg_SINR_dens)
np.save(f"{paths.arrays_path}p_stars_bw_{task_id}.npy", p_stars_bw)
np.save(f"{paths.arrays_path}R_stars_bw_{task_id}.npy", R_stars_bw)
np.save(f"{paths.arrays_path}num_stations_stars_bw_{task_id}.npy", num_stations_stars_bw)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_bw_{task_id}.npy", num_stations_per_firm_stars_bw)
np.save(f"{paths.arrays_path}q_stars_bw_{task_id}.npy", q_stars_bw)
np.save(f"{paths.arrays_path}cs_by_type_bw_{task_id}.npy", cs_by_type_bw)
np.save(f"{paths.arrays_path}cs_bw_{task_id}.npy", cs_bw)
np.save(f"{paths.arrays_path}ps_bw_{task_id}.npy", ps_bw)
np.save(f"{paths.arrays_path}ts_bw_{task_id}.npy", ts_bw)
np.save(f"{paths.arrays_path}ccs_bw_{task_id}.npy", ccs_bw)
np.save(f"{paths.arrays_path}ccs_per_bw_bw_{task_id}.npy", ccs_per_bw_bw)
np.save(f"{paths.arrays_path}avg_path_losses_bw_{task_id}.npy", avg_path_losses_bw)
np.save(f"{paths.arrays_path}avg_SINR_bw_{task_id}.npy", avg_SINR_bw)
np.save(f"{paths.arrays_path}p_stars_dens_1p_{task_id}.npy", p_stars_dens_1p)
np.save(f"{paths.arrays_path}R_stars_dens_1p_{task_id}.npy", R_stars_dens_1p)
np.save(f"{paths.arrays_path}num_stations_stars_dens_1p_{task_id}.npy", num_stations_stars_dens_1p)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_dens_1p_{task_id}.npy", num_stations_per_firm_stars_dens_1p)
np.save(f"{paths.arrays_path}q_stars_dens_1p_{task_id}.npy", q_stars_dens_1p)
np.save(f"{paths.arrays_path}cs_by_type_dens_1p_{task_id}.npy", cs_by_type_dens_1p)
np.save(f"{paths.arrays_path}cs_dens_1p_{task_id}.npy", cs_dens_1p)
np.save(f"{paths.arrays_path}ps_dens_1p_{task_id}.npy", ps_dens_1p)
np.save(f"{paths.arrays_path}ts_dens_1p_{task_id}.npy", ts_dens_1p)
np.save(f"{paths.arrays_path}ccs_dens_1p_{task_id}.npy", ccs_dens_1p)
np.save(f"{paths.arrays_path}ccs_per_bw_dens_1p_{task_id}.npy", ccs_per_bw_dens_1p)
np.save(f"{paths.arrays_path}avg_path_losses_dens_1p_{task_id}.npy", avg_path_losses_dens_1p)
np.save(f"{paths.arrays_path}avg_SINR_dens_1p_{task_id}.npy", avg_SINR_dens_1p)
np.save(f"{paths.arrays_path}p_stars_asymmetric_allbw_{task_id}.npy", p_stars_asymmetric_allbw)
np.save(f"{paths.arrays_path}R_stars_asymmetric_allbw_{task_id}.npy", R_stars_asymmetric_allbw)
np.save(f"{paths.arrays_path}num_stations_stars_asymmetric_allbw_{task_id}.npy", num_stations_stars_asymmetric_allbw)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_asymmetric_allbw_{task_id}.npy", num_stations_per_firm_stars_asymmetric_allbw)
np.save(f"{paths.arrays_path}q_stars_asymmetric_allbw_{task_id}.npy", q_stars_asymmetric_allbw)
np.save(f"{paths.arrays_path}cs_by_type_asymmetric_allbw_{task_id}.npy", cs_by_type_asymmetric_allbw)
np.save(f"{paths.arrays_path}cs_asymmetric_allbw_{task_id}.npy", cs_asymmetric_allbw)
np.save(f"{paths.arrays_path}ps_asymmetric_allbw_{task_id}.npy", ps_asymmetric_allbw)
np.save(f"{paths.arrays_path}ts_asymmetric_allbw_{task_id}.npy", ts_asymmetric_allbw)
np.save(f"{paths.arrays_path}ccs_asymmetric_allbw_{task_id}.npy", ccs_asymmetric_allbw)
np.save(f"{paths.arrays_path}ccs_per_bw_asymmetric_allbw_{task_id}.npy", ccs_per_bw_asymmetric_allbw)
np.save(f"{paths.arrays_path}avg_path_losses_asymmetric_allbw_{task_id}.npy", avg_path_losses_asymmetric_allbw)
np.save(f"{paths.arrays_path}avg_SINR_asymmetric_allbw_{task_id}.npy", avg_SINR_asymmetric_allbw)
np.save(f"{paths.arrays_path}p_stars_shortrunall_{task_id}.npy", p_stars_shortrunall)
np.save(f"{paths.arrays_path}R_stars_shortrunall_{task_id}.npy", R_stars_shortrunall)
np.save(f"{paths.arrays_path}num_stations_stars_shortrunall_{task_id}.npy", num_stations_stars_shortrunall)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_shortrunall_{task_id}.npy", num_stations_per_firm_stars_shortrunall)
np.save(f"{paths.arrays_path}q_stars_shortrunall_{task_id}.npy", q_stars_shortrunall)
np.save(f"{paths.arrays_path}cs_by_type_shortrunall_{task_id}.npy", cs_by_type_shortrunall)
np.save(f"{paths.arrays_path}cs_shortrunall_{task_id}.npy", cs_shortrunall)
np.save(f"{paths.arrays_path}ps_shortrunall_{task_id}.npy", ps_shortrunall)
np.save(f"{paths.arrays_path}ts_shortrunall_{task_id}.npy", ts_shortrunall)
np.save(f"{paths.arrays_path}ccs_shortrunall_{task_id}.npy", ccs_shortrunall)
np.save(f"{paths.arrays_path}ccs_per_bw_shortrunall_{task_id}.npy", ccs_per_bw_shortrunall)
np.save(f"{paths.arrays_path}avg_path_losses_shortrunall_{task_id}.npy", avg_path_losses_shortrunall)
np.save(f"{paths.arrays_path}p_stars_longrunall_{task_id}.npy", p_stars_longrunall)
np.save(f"{paths.arrays_path}R_stars_longrunall_{task_id}.npy", R_stars_longrunall)
np.save(f"{paths.arrays_path}num_stations_stars_longrunall_{task_id}.npy", num_stations_stars_longrunall)
np.save(f"{paths.arrays_path}num_stations_per_firm_stars_longrunall_{task_id}.npy", num_stations_per_firm_stars_longrunall)
np.save(f"{paths.arrays_path}q_stars_longrunall_{task_id}.npy", q_stars_longrunall)
np.save(f"{paths.arrays_path}cs_by_type_longrunall_{task_id}.npy", cs_by_type_longrunall)
np.save(f"{paths.arrays_path}cs_longrunall_{task_id}.npy", cs_longrunall)
np.save(f"{paths.arrays_path}ps_longrunall_{task_id}.npy", ps_longrunall)
np.save(f"{paths.arrays_path}ts_longrunall_{task_id}.npy", ts_longrunall)
np.save(f"{paths.arrays_path}ccs_longrunall_{task_id}.npy", ccs_longrunall)
np.save(f"{paths.arrays_path}ccs_per_bw_longrunall_{task_id}.npy", ccs_per_bw_longrunall)
np.save(f"{paths.arrays_path}avg_path_losses_longrunall_{task_id}.npy", avg_path_losses_longrunall)
np.save(f"{paths.arrays_path}per_user_costs_{task_id}.npy", per_user_costs)

# Standard errors
if compute_std_errs:
    np.save(f"{paths.arrays_path}p_stars_se_{task_id}.npy", p_stars_se)
    np.save(f"{paths.arrays_path}R_stars_se_{task_id}.npy", R_stars_se)
    np.save(f"{paths.arrays_path}num_stations_stars_se_{task_id}.npy", num_stations_stars_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_se_{task_id}.npy", num_stations_per_firm_stars_se)
    np.save(f"{paths.arrays_path}q_stars_se_{task_id}.npy", q_stars_se)
    np.save(f"{paths.arrays_path}cs_by_type_se_{task_id}.npy", cs_by_type_se)
    np.save(f"{paths.arrays_path}cs_se_{task_id}.npy", cs_se)
    np.save(f"{paths.arrays_path}ps_se_{task_id}.npy", ps_se)
    np.save(f"{paths.arrays_path}ts_se_{task_id}.npy", ts_se)
    np.save(f"{paths.arrays_path}ccs_se_{task_id}.npy", ccs_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_se_{task_id}.npy", ccs_per_bw_se)
    np.save(f"{paths.arrays_path}avg_path_losses_se_{task_id}.npy", avg_path_losses_se)
    np.save(f"{paths.arrays_path}avg_SINR_se_{task_id}.npy", avg_SINR_se)
    np.save(f"{paths.arrays_path}full_elasts_se_{task_id}.npy", full_elasts_se)
    np.save(f"{paths.arrays_path}partial_elasts_se_{task_id}.npy", partial_elasts_se)
    np.save(f"{paths.arrays_path}p_stars_allfixed_se_{task_id}.npy", p_stars_allfixed_se)
    np.save(f"{paths.arrays_path}R_stars_allfixed_se_{task_id}.npy", R_stars_allfixed_se)
    np.save(f"{paths.arrays_path}num_stations_stars_allfixed_se_{task_id}.npy", num_stations_stars_allfixed_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_allfixed_se_{task_id}.npy", num_stations_per_firm_stars_allfixed_se)
    np.save(f"{paths.arrays_path}q_stars_allfixed_se_{task_id}.npy", q_stars_allfixed_se)
    np.save(f"{paths.arrays_path}cs_by_type_allfixed_se_{task_id}.npy", cs_by_type_allfixed_se)
    np.save(f"{paths.arrays_path}cs_allfixed_se_{task_id}.npy", cs_allfixed_se)
    np.save(f"{paths.arrays_path}ps_allfixed_se_{task_id}.npy", ps_allfixed_se)
    np.save(f"{paths.arrays_path}ts_allfixed_se_{task_id}.npy", ts_allfixed_se)
    np.save(f"{paths.arrays_path}ccs_allfixed_se_{task_id}.npy", ccs_allfixed_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_allfixed_se_{task_id}.npy", ccs_per_bw_allfixed_se)
    np.save(f"{paths.arrays_path}avg_path_losses_allfixed_se_{task_id}.npy", avg_path_losses_allfixed_se)
    np.save(f"{paths.arrays_path}avg_SINR_allfixed_se_{task_id}.npy", avg_SINR_allfixed_se)
    np.save(f"{paths.arrays_path}partial_Pif_partial_bf_allfixed_se_{task_id}.npy", partial_Pif_partial_bf_allfixed_se)
    np.save(f"{paths.arrays_path}partial_Piotherf_partial_bf_allfixed_se_{task_id}.npy", partial_Piotherf_partial_bf_allfixed_se)
    np.save(f"{paths.arrays_path}partial_diffPif_partial_bf_allfixed_se_{task_id}.npy", partial_diffPif_partial_bf_allfixed_se)
    np.save(f"{paths.arrays_path}partial_Pif_partial_b_allfixed_se_{task_id}.npy", partial_Pif_partial_b_allfixed_se)
    np.save(f"{paths.arrays_path}partial_CS_partial_b_allfixed_se_{task_id}.npy", partial_CS_partial_b_allfixed_se)
    np.save(f"{paths.arrays_path}partial_Pif_partial_bf_allbw_se_{task_id}.npy", partial_Pif_partial_bf_allbw_se)
    np.save(f"{paths.arrays_path}partial_Piotherf_partial_bf_allbw_se_{task_id}.npy", partial_Piotherf_partial_bf_allbw_se)
    np.save(f"{paths.arrays_path}partial_diffPif_partial_bf_allbw_se_{task_id}.npy", partial_diffPif_partial_bf_allbw_se)
    np.save(f"{paths.arrays_path}partial_Pif_partial_b_allbw_se_{task_id}.npy", partial_Pif_partial_b_allbw_se)
    np.save(f"{paths.arrays_path}partial_CS_partial_b_allbw_se_{task_id}.npy", partial_CS_partial_b_allbw_se)
    np.save(f"{paths.arrays_path}c_u_se_{task_id}.npy", c_u_se)
    np.save(f"{paths.arrays_path}c_R_se_{task_id}.npy", c_R_se)
    np.save(f"{paths.arrays_path}p_stars_shortrun_se_{task_id}.npy", p_stars_shortrun_se)
    np.save(f"{paths.arrays_path}R_stars_shortrun_se_{task_id}.npy", R_stars_shortrun_se)
    np.save(f"{paths.arrays_path}num_stations_stars_shortrun_se_{task_id}.npy", num_stations_stars_shortrun_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_shortrun_se_{task_id}.npy", num_stations_per_firm_stars_shortrun_se)
    np.save(f"{paths.arrays_path}q_stars_shortrun_se_{task_id}.npy", q_stars_shortrun_se)
    np.save(f"{paths.arrays_path}cs_by_type_shortrun_se_{task_id}.npy", cs_by_type_shortrun_se)
    np.save(f"{paths.arrays_path}cs_shortrun_se_{task_id}.npy", cs_shortrun_se)
    np.save(f"{paths.arrays_path}ps_shortrun_se_{task_id}.npy", ps_shortrun_se)
    np.save(f"{paths.arrays_path}ts_shortrun_se_{task_id}.npy", ts_shortrun_se)
    np.save(f"{paths.arrays_path}ccs_shortrun_se_{task_id}.npy", ccs_shortrun_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_shortrun_se_{task_id}.npy", ccs_per_bw_shortrun_se)
    np.save(f"{paths.arrays_path}avg_path_losses_shortrun_se_{task_id}.npy", avg_path_losses_shortrun_se)
    np.save(f"{paths.arrays_path}p_stars_free_allfixed_se_{task_id}.npy", p_stars_free_allfixed_se)
    np.save(f"{paths.arrays_path}R_stars_free_allfixed_se_{task_id}.npy", R_stars_free_allfixed_se)
    np.save(f"{paths.arrays_path}num_stations_stars_free_allfixed_se_{task_id}.npy", num_stations_stars_free_allfixed_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_free_allfixed_se_{task_id}.npy", num_stations_per_firm_stars_free_allfixed_se)
    np.save(f"{paths.arrays_path}q_stars_free_allfixed_se_{task_id}.npy", q_stars_free_allfixed_se)
    np.save(f"{paths.arrays_path}cs_by_type_free_allfixed_se_{task_id}.npy", cs_by_type_free_allfixed_se)
    np.save(f"{paths.arrays_path}cs_free_se_allfixed_{task_id}.npy", cs_free_allfixed_se)
    np.save(f"{paths.arrays_path}ps_free_se_allfixed_{task_id}.npy", ps_free_allfixed_se)
    np.save(f"{paths.arrays_path}ts_free_se_allfixed_{task_id}.npy", ts_free_allfixed_se)
    np.save(f"{paths.arrays_path}ccs_free_allfixed_se_{task_id}.npy", ccs_free_allfixed_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_free_allfixed_se_{task_id}.npy", ccs_per_bw_free_allfixed_se)
    np.save(f"{paths.arrays_path}avg_path_losses_free_allfixed_se_{task_id}.npy", avg_path_losses_free_allfixed_se)
    np.save(f"{paths.arrays_path}p_stars_free_allbw_se_{task_id}.npy", p_stars_free_allbw_se)
    np.save(f"{paths.arrays_path}R_stars_free_allbw_se_{task_id}.npy", R_stars_free_allbw_se)
    np.save(f"{paths.arrays_path}num_stations_stars_free_allbw_se_{task_id}.npy", num_stations_stars_free_allbw_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_free_allbw_se_{task_id}.npy", num_stations_per_firm_stars_free_allbw_se)
    np.save(f"{paths.arrays_path}q_stars_free_allbw_se_{task_id}.npy", q_stars_free_allbw_se)
    np.save(f"{paths.arrays_path}cs_by_type_free_allbw_se_{task_id}.npy", cs_by_type_free_allbw_se)
    np.save(f"{paths.arrays_path}cs_free_se_allbw_{task_id}.npy", cs_free_allbw_se)
    np.save(f"{paths.arrays_path}ps_free_se_allbw_{task_id}.npy", ps_free_allbw_se)
    np.save(f"{paths.arrays_path}ts_free_se_allbw_{task_id}.npy", ts_free_allbw_se)
    np.save(f"{paths.arrays_path}ccs_free_allbw_se_{task_id}.npy", ccs_free_allbw_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_free_allbw_se_{task_id}.npy", ccs_per_bw_free_allbw_se)
    np.save(f"{paths.arrays_path}avg_path_losses_free_allbw_se_{task_id}.npy", avg_path_losses_free_allbw_se)
    
    np.save(f"{paths.arrays_path}p_stars_dens_se_{task_id}.npy", p_stars_dens_se)
    np.save(f"{paths.arrays_path}R_stars_dens_se_{task_id}.npy", R_stars_dens_se)
    np.save(f"{paths.arrays_path}num_stations_stars_dens_se_{task_id}.npy", num_stations_stars_dens_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_dens_se_{task_id}.npy", num_stations_per_firm_stars_dens_se)
    np.save(f"{paths.arrays_path}q_stars_dens_se_{task_id}.npy", q_stars_dens_se)
    np.save(f"{paths.arrays_path}cs_by_type_dens_se_{task_id}.npy", cs_by_type_dens_se)
    np.save(f"{paths.arrays_path}cs_dens_se_{task_id}.npy", cs_dens_se)
    np.save(f"{paths.arrays_path}ps_dens_se_{task_id}.npy", ps_dens_se)
    np.save(f"{paths.arrays_path}ts_dens_se_{task_id}.npy", ts_dens_se)
    np.save(f"{paths.arrays_path}ccs_dens_se_{task_id}.npy", ccs_dens_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_dens_se_{task_id}.npy", ccs_per_bw_dens_se)
    np.save(f"{paths.arrays_path}avg_path_losses_dens_se_{task_id}.npy", avg_path_losses_dens_se)
    np.save(f"{paths.arrays_path}avg_SINR_dens_se_{task_id}.npy", avg_SINR_dens_se)
    np.save(f"{paths.arrays_path}p_stars_bw_se_{task_id}.npy", p_stars_bw_se)
    np.save(f"{paths.arrays_path}R_stars_bw_se_{task_id}.npy", R_stars_bw_se)
    np.save(f"{paths.arrays_path}num_stations_stars_bw_se_{task_id}.npy", num_stations_stars_bw_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_bw_se_{task_id}.npy", num_stations_per_firm_stars_bw_se)
    np.save(f"{paths.arrays_path}q_stars_bw_se_{task_id}.npy", q_stars_bw_se)
    np.save(f"{paths.arrays_path}cs_by_type_bw_se_{task_id}.npy", cs_by_type_bw_se)
    np.save(f"{paths.arrays_path}cs_bw_se_{task_id}.npy", cs_bw_se)
    np.save(f"{paths.arrays_path}ps_bw_se_{task_id}.npy", ps_bw_se)
    np.save(f"{paths.arrays_path}ts_bw_se_{task_id}.npy", ts_bw_se)
    np.save(f"{paths.arrays_path}ccs_bw_se_{task_id}.npy", ccs_bw_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_bw_se_{task_id}.npy", ccs_per_bw_bw_se)
    np.save(f"{paths.arrays_path}avg_path_losses_bw_se_{task_id}.npy", avg_path_losses_bw_se)
    np.save(f"{paths.arrays_path}avg_SINR_bw_se_{task_id}.npy", avg_SINR_bw_se)
    np.save(f"{paths.arrays_path}p_stars_dens_1p_se_{task_id}.npy", p_stars_dens_1p_se)
    np.save(f"{paths.arrays_path}R_stars_dens_1p_se_{task_id}.npy", R_stars_dens_1p_se)
    np.save(f"{paths.arrays_path}num_stations_stars_dens_1p_se_{task_id}.npy", num_stations_stars_dens_1p_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_dens_1p_se_{task_id}.npy", num_stations_per_firm_stars_dens_1p_se)
    np.save(f"{paths.arrays_path}q_stars_dens_1p_se_{task_id}.npy", q_stars_dens_1p_se)
    np.save(f"{paths.arrays_path}cs_by_type_dens_1p_se_{task_id}.npy", cs_by_type_dens_1p_se)
    np.save(f"{paths.arrays_path}cs_dens_1p_se_{task_id}.npy", cs_dens_1p_se)
    np.save(f"{paths.arrays_path}ps_dens_1p_se_{task_id}.npy", ps_dens_1p_se)
    np.save(f"{paths.arrays_path}ts_dens_1p_se_{task_id}.npy", ts_dens_1p_se)
    np.save(f"{paths.arrays_path}ccs_dens_1p_se_{task_id}.npy", ccs_dens_1p_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_dens_1p_se_{task_id}.npy", ccs_per_bw_dens_1p_se)
    np.save(f"{paths.arrays_path}avg_path_losses_dens_1p_se_{task_id}.npy", avg_path_losses_dens_1p_se)
    np.save(f"{paths.arrays_path}avg_SINR_dens_1p_se_{task_id}.npy", avg_SINR_dens_1p_se)
    np.save(f"{paths.arrays_path}p_stars_asymmetric_allbw_se_{task_id}.npy", p_stars_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}R_stars_asymmetric_allbw_se_{task_id}.npy", R_stars_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}num_stations_stars_asymmetric_allbw_se_{task_id}.npy", num_stations_stars_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_asymmetric_allbw_se_{task_id}.npy", num_stations_per_firm_stars_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}q_stars_asymmetric_allbw_se_{task_id}.npy", q_stars_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}cs_by_type_asymmetric_allbw_se_{task_id}.npy", cs_by_type_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}cs_asymmetric_allbw_se_{task_id}.npy", cs_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}ps_asymmetric_allbw_se_{task_id}.npy", ps_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}ts_asymmetric_allbw_se_{task_id}.npy", ts_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}ccs_asymmetric_allbw_se_{task_id}.npy", ccs_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_asymmetric_allbw_se_{task_id}.npy", ccs_per_bw_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}avg_path_losses_asymmetric_allbw_se_{task_id}.npy", avg_path_losses_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}avg_SINR_asymmetric_allbw_se_{task_id}.npy", avg_SINR_asymmetric_allbw_se)
    np.save(f"{paths.arrays_path}p_stars_shortrunall_se_{task_id}.npy", p_stars_shortrunall_se)
    np.save(f"{paths.arrays_path}R_stars_shortrunall_se_{task_id}.npy", R_stars_shortrunall_se)
    np.save(f"{paths.arrays_path}num_stations_stars_shortrunall_se_{task_id}.npy", num_stations_stars_shortrunall_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_shortrunall_se_{task_id}.npy", num_stations_per_firm_stars_shortrunall_se)
    np.save(f"{paths.arrays_path}q_stars_shortrunall_se_{task_id}.npy", q_stars_shortrunall_se)
    np.save(f"{paths.arrays_path}cs_by_type_shortrunall_se_{task_id}.npy", cs_by_type_shortrunall_se)
    np.save(f"{paths.arrays_path}cs_shortrunall_se_{task_id}.npy", cs_shortrunall_se)
    np.save(f"{paths.arrays_path}ps_shortrunall_se_{task_id}.npy", ps_shortrunall_se)
    np.save(f"{paths.arrays_path}ts_shortrunall_se_{task_id}.npy", ts_shortrunall_se)
    np.save(f"{paths.arrays_path}ccs_shortrunall_se_{task_id}.npy", ccs_shortrunall_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_shortrunall_se_{task_id}.npy", ccs_per_bw_shortrunall_se)
    np.save(f"{paths.arrays_path}avg_path_losses_shortrunall_se_{task_id}.npy", avg_path_losses_shortrunall_se)
    np.save(f"{paths.arrays_path}p_stars_longrunall_se_{task_id}.npy", p_stars_longrunall_se)
    np.save(f"{paths.arrays_path}R_stars_longrunall_se_{task_id}.npy", R_stars_longrunall_se)
    np.save(f"{paths.arrays_path}num_stations_stars_longrunall_se_{task_id}.npy", num_stations_stars_longrunall_se)
    np.save(f"{paths.arrays_path}num_stations_per_firm_stars_longrunall_se_{task_id}.npy", num_stations_per_firm_stars_longrunall_se)
    np.save(f"{paths.arrays_path}q_stars_longrunall_se_{task_id}.npy", q_stars_longrunall_se)
    np.save(f"{paths.arrays_path}cs_by_type_longrunall_se_{task_id}.npy", cs_by_type_longrunall_se)
    np.save(f"{paths.arrays_path}cs_longrunall_se_{task_id}.npy", cs_longrunall_se)
    np.save(f"{paths.arrays_path}ps_longrunall_se_{task_id}.npy", ps_longrunall_se)
    np.save(f"{paths.arrays_path}ts_longrunall_se_{task_id}.npy", ts_longrunall_se)
    np.save(f"{paths.arrays_path}ccs_longrunall_se_{task_id}.npy", ccs_longrunall_se)
    np.save(f"{paths.arrays_path}ccs_per_bw_longrunall_se_{task_id}.npy", ccs_per_bw_longrunall_se)
    np.save(f"{paths.arrays_path}avg_path_losses_longrunall_se_{task_id}.npy", avg_path_losses_longrunall_se)
    np.save(f"{paths.arrays_path}per_user_costs_se_{task_id}.npy", per_user_costs_se)
