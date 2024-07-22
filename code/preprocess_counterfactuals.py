# %%
# Import packages

import numpy as np
import pandas as pd

import sys

from multiprocessing import Pool

from itertools import combinations

import paths

import supply.infrastructurefunctions as infr
import supply.costs as costs

import demand.demandsystem as demsys
import demand.demandfunctions as blp

import pickle

import time

# %%
# Determine which specification of demand to use
task_id = int(sys.argv[1]) # default is 0, one set of MVNOs unrepeated for each MNO

# %%
# Number of CPUs
num_cpus = int(sys.argv[2])

# %%
# Determine whether to print detailed messages
print_updates = True

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
if task_id == 0:
    np.save(f"{paths.arrays_path}avg_radius.npy", np.array([np.mean(radius[np.isfinite(radius)])]))

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
if task_id == 0:
    np.save(f"{paths.arrays_path}list_MNOwMVNO.npy", list_MNOwMVNO)
    np.save(f"{paths.arrays_path}list_MNOwMVNOnums.npy", list_MNOwMVNOnums)
    np.save(f"{paths.arrays_path}merger_combos.npy", merger_combos)
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
thetahat = np.load(f"{paths.arrays_path}thetahat_{task_id}.npy")
num_obs = ds.M
num_prods = ds.J
np.save(f"{paths.arrays_path}num_obs.npy", np.array([num_obs]))
np.save(f"{paths.arrays_path}num_prods_{task_id}.npy", np.array([num_prods]))

# %%
# Determine the size of epsilon to numerically approximate the gradient
eps_grad = 0.025
thetas_to_compute = np.vstack((thetahat[np.newaxis,:], thetahat[np.newaxis,:] + np.identity(thetahat.shape[0]) * eps_grad, thetahat[np.newaxis,:] - np.identity(thetahat.shape[0]) * eps_grad))
np.save(f"{paths.arrays_path}thetas_to_compute_{task_id}.npy", thetas_to_compute)
if task_id == 0:
    np.save(f"{paths.arrays_path}eps_grad.npy", np.array([eps_grad]))

# Equilibria parameters
num_firms_to_simulate = 6
num_firms_to_simulate_extend = 9
num_firms_array = np.arange(num_firms_to_simulate, dtype=int) + 1
num_firms_array_extend = np.arange(num_firms_to_simulate_extend, dtype=int) + 1
rep_density = 2791.7 # contraharmonic mean for France
rep_market_size = np.median(area)
rep_population = rep_density * rep_market_size
rep_income_distribution = np.percentile(ds.data[:,0,yc1idx:yclastidx+1], np.linspace(10, 90, 9))
if task_id == 0:
    np.save(f"{paths.arrays_path}rep_income_distribution.npy", rep_income_distribution)
    np.save(f"{paths.arrays_path}rep_market_size.npy", np.array([rep_market_size]))
    np.save(f"{paths.arrays_path}rep_population.npy", np.array([rep_population]))
    np.save(f"{paths.arrays_path}num_firms_array.npy", num_firms_array)
    np.save(f"{paths.arrays_path}num_firms_array_extend.npy", num_firms_array_extend)

# Densities to test
vlow_dens = 43.1 # pop dens (people / km^2) of all of USA
low_dens = 123.9 # pop dens (people / km^2) of all of France
high_dens = 20588.2 # pop dens of Paris
densities = np.array([rep_density, vlow_dens, low_dens, high_dens]) # rep_density must be first (because its results just get copied over from the regular exercise)
densities_areatype = np.array(["urban", "urban", "urban", "urban"]) # np.array(["urban", "rural", "suburban", "urban"]) # area types used for Hata path loss model
if task_id == 0:
    np.save(f"{paths.arrays_path}cntrfctl_densities.npy", densities)
    np.save(f"{paths.arrays_path}cntrfctl_densities_areatype.npy", densities_areatype)
    np.save(f"{paths.arrays_path}cntrfctl_densities_pop.npy", densities * rep_market_size)

# Bandwidth values to test
market_bw = np.average(np.sum(bw_4g_equiv, axis=1), weights=population)
low_bw_val = market_bw * 0.5
high_bw_val = market_bw * 1.5
bw_vals = np.array([market_bw, low_bw_val, high_bw_val]) # market_bw must be first (because its results just get copied over from the regular exercise)
if task_id == 0:
    np.save(f"{paths.arrays_path}cntrfctl_bw_vals.npy", bw_vals)

# Bandwidth values and merged base stations to use for short-run counterfactual
radius_mergers_use = np.average(radius_mergers_all, weights=np.tile(population[:,np.newaxis,np.newaxis], (1,radius_mergers_all.shape[1],radius_mergers_all.shape[2])), axis=0)
bw_4g_equiv_weightedaverage = np.average(bw_4g_equiv, weights=np.tile(population[:,np.newaxis], (1,bw_4g_equiv.shape[1])), axis=0)
if task_id == 0:
    np.save(f"{paths.arrays_path}cntrfctl_bw_vals_by_firm.npy", bw_4g_equiv_weightedaverage)
    np.save(f"{paths.arrays_path}cntrfctl_firms.npy", list_MNOwMVNO[list_MNOwMVNO != "MVNO"])

# Save spectral efficiency used in counterfactuals
if task_id == 0:
    np.save(f"{paths.arrays_path}cntrfctl_gamma.npy", np.array([np.average(lamda, weights=population)]))

# Dictionary for how to treat MVNO in computing transmission equilibrium
impute_MVNO = {
    'impute': True, 
    'firms_share': np.array([True, True, False, True]), # all firms share with MVNO, except Free
    'include': True
}

# Determine xis, per-user costs, and per-base station costs
start = time.time()

def compute_combined(theta):
    xis = blp.xi(ds, theta, ds.data, None)
    c_u = costs.per_user_costs(theta, xis, ds, population, prices, cc_tot, stations, impute_MVNO=impute_MVNO)
    c_R = costs.per_base_station_costs(theta, xis, c_u, radius, bw_4g_equiv, lamda, ds, population, area, impute_MVNO=impute_MVNO)
    return xis, c_u, c_R
def compute_combined_n(theta_n):
    return compute_combined(thetas_to_compute[theta_n,:])

xis = np.zeros((thetas_to_compute.shape[0], ds.data.shape[0], num_prods))
c_u = np.zeros((thetas_to_compute.shape[0], num_prods))
c_R = np.zeros((thetas_to_compute.shape[0], ds.data.shape[0], stations.shape[1]))
pool = Pool(num_cpus)
chunksize = 1
for ind, res in enumerate(pool.imap(compute_combined_n, range(thetas_to_compute.shape[0])), chunksize):
    idx = ind - chunksize
    xis[idx,:,:] = res[0]
    c_u[idx,:] = res[1]
    c_R[idx,:] = res[2]
pool.close()

np.save(f"{paths.arrays_path}xis_{task_id}.npy", xis)
np.save(f"{paths.arrays_path}c_u_{task_id}.npy", c_u)
np.save(f"{paths.arrays_path}c_R_{task_id}.npy", c_R)
if print_updates:
    print(f"Finished calculating xis, per-user costs, per-base station costs for estimated theta in {np.round(time.time() - start, 1)} seconds.", flush=True)
    
# Determine representative values for the simplified counterfactual environment
c_u_simplified = np.concatenate((np.mean(c_u[:,ds.data[0,:,dlimidx] < 5000.0], axis=1)[:,np.newaxis], np.mean(c_u[:,ds.data[0,:,dlimidx] >= 5000.0], axis=1)[:,np.newaxis]), axis=1)
per_base_station_per_bw_cost_simplified = np.average(c_R / bw_4g_equiv[np.newaxis,:,:], weights=np.tile(population[np.newaxis,:,np.newaxis], (thetas_to_compute.shape[0],1,4)), axis=(1,2))
np.save(f"{paths.arrays_path}c_u_simplified_{task_id}.npy", c_u_simplified)
np.save(f"{paths.arrays_path}per_base_station_per_bw_cost_simplified_{task_id}.npy", per_base_station_per_bw_cost_simplified)

# Determine which markets to use to mimic all of France
xis_0 = xis[0,:,:] # match based on our estimate rather than on perturbations
c_u_0 = c_u[0,:]
c_R_0 = c_R[0,:,:]
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
    constructed_markets_idx[i] = markets_idx_arange[select_markets][np.argmin(np.sum((c_R_0[select_markets,:] - avg_c_R[i,:])**2.0, axis=1))]
    market_name = market_names[constructed_markets_idx[i]]
    mimic_market_names = mimic_market_names + [market_name]
    print(f"Mimic market category {i + 1} [{population_categories_lower_lim[i]}, {population_categories_upper_lim[i]}) making up {np.round(frac_population[i] * 100.0, 2)}% of population: {np.sum(select_markets)} markets with average c_R of {np.round(avg_c_R[i,:], 3)}. Chose market {constructed_markets_idx[i]} ({market_name}) with c_R {np.round(c_R_0[constructed_markets_idx[i],:], 3)}.", flush=True)
frac_population_orig = np.copy(frac_population)
frac_population = frac_population * np.sum(population[constructed_markets_idx]) / population[constructed_markets_idx] # adjust for the population associated with the markets so that market weighting is correct
frac_population = frac_population / np.sum(frac_population)
if task_id == 0:
    np.save(f"{paths.arrays_path}mimic_market_names.npy", np.array(mimic_market_names))
    np.save(f"{paths.arrays_path}mimic_market_weights.npy", frac_population_orig)
    np.save(f"{paths.arrays_path}mimic_market_population_categories.npy", population_categories_upper_lim)
    np.save(f"{paths.arrays_path}constructed_markets_idx.npy", constructed_markets_idx)
    np.save(f"{paths.arrays_path}mimic_market_bw.npy", bw_4g_equiv[constructed_markets_idx,:])
    np.save(f"{paths.arrays_path}mimic_market_radii.npy", radius[constructed_markets_idx,:])
    np.save(f"{paths.arrays_path}mimic_market_radii_mergers.npy", radius_mergers_all[constructed_markets_idx,:,:])
    np.save(f"{paths.arrays_path}mimic_market_income_distribution.npy", ds.data[constructed_markets_idx,0,yc1idx:yclastidx+1])
    np.save(f"{paths.arrays_path}mimic_market_population.npy", population[constructed_markets_idx])
    np.save(f"{paths.arrays_path}mimic_market_market_size.npy", area[constructed_markets_idx,np.newaxis])
    np.save(f"{paths.arrays_path}mimic_market_market_weights.npy", frac_population)
    np.save(f"{paths.arrays_path}mimic_market_gamma.npy", lamda[constructed_markets_idx])
np.save(f"{paths.arrays_path}mimic_market_product_firm_correspondence_{task_id}.npy", np.unique(ds.firms, return_inverse=True)[1])
np.save(f"{paths.arrays_path}mimic_market_prices_{task_id}.npy", ds.data[constructed_markets_idx,:,pidx])
np.save(f"{paths.arrays_path}mimic_market_dlim_{task_id}.npy", ds.data[constructed_markets_idx,:,dlimidx])
np.save(f"{paths.arrays_path}mimic_market_vlim_{task_id}.npy", ds.data[constructed_markets_idx,:,vlimidx])
np.save(f"{paths.arrays_path}mimic_market_O_{task_id}.npy", ds.data[constructed_markets_idx,:,Oidx])
np.save(f"{paths.arrays_path}mimic_market_firms_{task_id}.npy", ds.firms)
np.save(f"{paths.arrays_path}mimic_market_xis_{task_id}.npy", xis[:,constructed_markets_idx,:])
np.save(f"{paths.arrays_path}mimic_market_c_R_{task_id}.npy", c_R[:,constructed_markets_idx,:])
np.save(f"{paths.arrays_path}mimic_market_c_u_{task_id}.npy", c_u)

# %%
# Save some characteristics of ds that are used later while processing counterfactual results
np.save(f"{paths.arrays_path}ds_firms_{task_id}.npy", ds.firms)
np.save(f"{paths.arrays_path}ds_J_{task_id}.npy", np.array([ds.J]))
np.save(f"{paths.arrays_path}ds_vlims_{task_id}.npy", ds.data[0,:,vlimidx])
np.save(f"{paths.arrays_path}ds_dlims_{task_id}.npy", ds.data[0,:,dlimidx])

# %%
# Difference in diversion ratios when average particular ways
div_ratios = blp.div_ratio(ds, thetahat, ds.data, xis=xis_0)
div_ratios_alt = blp.div_ratio_numdenom(ds, thetahat, ds.data, xis=xis_0)
diff_div_ratios = np.abs(np.mean(div_ratios[1:]) + np.mean(div_ratios_alt[0][1:]) / np.mean(div_ratios_alt[1][1:]))
if save_:
    create_file(f"{paths.stats_path}diff_div_ratios_averaging.tex", f"{diff_div_ratios:.5f}")
    