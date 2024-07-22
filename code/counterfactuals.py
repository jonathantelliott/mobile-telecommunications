# %%
# Import packages

import numpy as np
import pandas as pd
import scipy.special as special

import sys
import copy

from multiprocessing import Pool

import paths

import supply.infrastructurefunctions as infr
import supply.priceequilibrium as pe
import supply.infrastructureequilibrium as ie

import demand.demandsystem as demsys
import demand.coefficients as coef

import welfare.welfare as welfare

import time

# %%
# Determine parameters for running counterfactuals
task_id = int(sys.argv[1]) # demand specification, default is 0
num_cpus = int(sys.argv[2]) # number of CPUs (for parallel computing)
print_msg = False # whether to print very detailed messages about every iteration of solving for equilibria
print_updates = True # whether to print updates as compute counterfactuals
include_logit_shock = False # don't include logit shocks in consumer surplus
include_pop = False # normalize in per person terms
run_example = False
compute_std_errs = True
if task_id == 0:
    np.save(f"{paths.arrays_path}compute_std_errs.npy", np.array([compute_std_errs]))

# %%
# Function to compute counterfactuals and solve for important variables given market structure

def compute_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, num_firms=None, num_prods=None, product_firm_correspondence=None, areatype="urban", impute_MVNO={'impute': False}, symmetric=False, R_fixed=False, market_weights=None, method=None, factor=100.0, calc_q_carefully=False, print_addl_msg=False, pidx=0, qidx=1):
    """Compute an equilibrium given description of market structure."""

    # Deep copy DemandSystem so can make changes to it and not change original
    ds_cntrfctl_ = copy.deepcopy(ds_cntrfctl)

    # Compute the equilibrium
    R_star, p_star, q_star, success = ie.infrastructure_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, symmetric=symmetric, impute_MVNO=impute_MVNO, q_0=None, eps_R=0.001, eps_p=0.001, factor=factor, areatype=areatype, R_fixed=R_fixed, market_weights=market_weights, method=method, calc_q_carefully=calc_q_carefully, print_msg=print_addl_msg)

    # Update Demand System
    if symmetric:
        ds_cntrfctl_.data[:,:,pidx] = np.copy(p_star)
        ds_cntrfctl_.data[:,:,qidx] = np.tile(q_star, (num_prods,))
    else:
        ds_cntrfctl_.data[:,:,pidx] = p_star[np.newaxis,:]
        ds_cntrfctl_.data[:,:,qidx] = np.take_along_axis(q_star, product_firm_correspondence[np.newaxis,:], 1)

    # Calculate welfare impact
    if market_weights is None:
        cs_market_weights = None
    else:
        cs_market_weights = market_weights * pop_cntrfctl / np.sum(pop_cntrfctl * market_weights)
    if symmetric:
        xis_cntrfctl = np.tile(xis_cntrfctl, (1,num_firms))
        c_u_cntrfctl = np.tile(c_u_cntrfctl, (num_firms,))
        c_R_cntrfctl = np.tile(c_R_cntrfctl, (1,num_firms))
    cs_by_type_ = np.average(welfare.consumer_surplus(ds_cntrfctl_, xis_cntrfctl, theta, include_logit_shock=include_logit_shock), weights=cs_market_weights, axis=0)
    cs_ = welfare.agg_consumer_surplus(ds_cntrfctl_, xis_cntrfctl, theta, pop_cntrfctl, include_logit_shock=include_logit_shock, include_pop=include_pop, market_weights=cs_market_weights)
    ps_ = welfare.producer_surplus(ds_cntrfctl_, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, R_star, c_u_cntrfctl, c_R_cntrfctl, include_pop=include_pop, market_weights=market_weights)
    ts_ = cs_ + ps_

    # Determine equilibrium values to return
    # NOTE: only returns eqm radius and download speed of first market, in theory may be interested in more and could change below code, but never needed for our counterfactuals
    p_stars_ = p_star[:num_prods] if symmetric else np.copy(p_star)
    R_stars_ = R_star[0,0] if symmetric else R_star[0,:]
    num_stations_stars_ = num_firms * infr.num_stations(R_stars_, market_size_cntrfctl[0]) if symmetric else np.sum(infr.num_stations(R_stars_, market_size_cntrfctl[0]))
    num_stations_per_firm_stars_ = infr.num_stations(R_stars_, market_size_cntrfctl[0])
    q_stars_ = q_star[0,0] if symmetric else q_star[0,:]

    # Determine variables that describe infrastructure investment
    # NOTE: once again, only calculating these values for first market, but could change that
    cc_cntrfctl = np.zeros((R_star.shape[0], 1)) if symmetric else np.zeros(R_star.shape)
    for m in range(R_star.shape[0]):
        num_firms_iterate = 1 if symmetric else R_star.shape[1]
        for f in range(num_firms_iterate):
            cc_cntrfctl[m,f] = infr.rho_C_hex(bw_cntrfctl[m,f], R_star[m,f], gamma_cntrfctl[m], areatype=areatype)
    ccs_ = cc_cntrfctl[0,0] if symmetric else cc_cntrfctl[0,:]
    ccs_per_bw_ = (cc_cntrfctl / bw_cntrfctl)[0,0] if symmetric else (cc_cntrfctl / bw_cntrfctl[0,:])[0,:]
    if symmetric:
        avg_path_losses_ = infr.avg_path_loss(R_stars_, areatype=areatype)
    else:
        avg_path_losses_ = np.zeros(R_star.shape[1])
        for f in range(R_star.shape[1]):
            avg_path_losses_[f] = infr.avg_path_loss(R_stars_[f], areatype=areatype)
    num_stations_cntrfctl = infr.num_stations(np.array([[R_stars_]]), market_size_cntrfctl) if symmetric else infr.num_stations(R_stars_, market_size_cntrfctl)
    if symmetric:
        avg_SINR_ = infr.avg_SINR(R_stars_, areatype=areatype)
    else:
        avg_SINR_ = np.zeros(R_star.shape[1])
        for f in range(R_star.shape[1]):
            avg_SINR_[f] = infr.avg_SINR(R_stars_[f], areatype=areatype)
            
    return success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_

# %%
# Parameters that define simplified environment counterfactuals

# Demand parameters
thetas_to_compute = np.load(f"{paths.arrays_path}thetas_to_compute_{task_id}.npy") # demand parameters (axis 0 is number of versions of demand parameters to compute, axis 1 is each of the parameters)

# Description of plans
dlims_simplified = np.array([1000.0, 10000.0])
vlims_simplified = np.array([1.0, 1.0])
num_prods = dlims_simplified.shape[0]
xis_simplified = np.ones((1,num_prods))[np.newaxis,:,:] * np.load(f"{paths.arrays_path}thetas_to_compute_{task_id}.npy")[:,coef.O][:,np.newaxis,np.newaxis] # demand shocks (axis 0 is number of version of demand parameters to compute, axis 1 is each market, axis 2 is each plan)
c_u_simplified = np.load(f"{paths.arrays_path}c_u_simplified_{task_id}.npy") # per-plan MCs (axis 0 is number of versions of demand parameters to compute, axis 1 is each plan)

# Description of market(s)
gamma_simplified = np.load(f"{paths.arrays_path}cntrfctl_gamma.npy") # spectral efficiency parameters (axis 0 is each market)
market_bw = np.ones((1,)) * np.load(f"{paths.arrays_path}cntrfctl_bw_vals.npy")[0] # amount of bandwidth in market (axis 0 is each market)
pop_simplified = np.ones((1,)) * np.load(f"{paths.arrays_path}rep_population.npy") # number of people in each market (axis 0 is each market)
market_size_simplified = np.ones((1,)) * np.load(f"{paths.arrays_path}rep_market_size.npy") # size of each market (axis 0 is each market)
income_distribution_simplified = np.load(f"{paths.arrays_path}rep_income_distribution.npy")[np.newaxis,:] # income distribution in each market (axis 0 is each market, axis 1 is each decile)

# Description of infrastructure
per_base_station_per_bw_cost_simplified = np.load(f"{paths.arrays_path}per_base_station_per_bw_cost_simplified_{task_id}.npy")[:,np.newaxis] # per unit bw per tower infrastructure MC (axis 0 is number of versions of demand parameters to compute, axis 1 is each MNO)

# Construct empty DemandSystem
# NOTE: This just sets up the basic structure, the data is filled in later, so there is no need to change the below code unless changing characteristics of plans.
chars = {'names': ['p', 'q', 'dlim', 'vunlimited', 'Orange'], 'norm': np.array([False, False, False, False, False])} # product characteristics
elist = [] # spectral efficiencies, not used, so can leave blank
demolist = ['yc1', 'yc2', 'yc3', 'yc4', 'yc5', 'yc6', 'yc7', 'yc8', 'yc9'] # income deciles names
col_names = chars['names'] + ['dbar', 'pop_dens'] + elist + demolist + ['mktshare', 'msize', 'market', 'j']
df_ds = pd.DataFrame({col: np.arange(dlims_simplified.shape[0]) + 1 for col in col_names}) # values don't matter except for 'j' (which should be arange (# of products)), all replaced later
ds = demsys.DemandSystem(df_ds, chars, elist, demolist, np.zeros((1,)), 0.0, np.arange(dlims_simplified.shape[0]), np.arange(dlims_simplified.shape[0]), 0.0)
pidx = ds.chars.index(ds.pname)
qidx = ds.chars.index(ds.qname)
dlimidx = ds.chars.index(ds.dlimname)
vlimidx = ds.chars.index(ds.vunlimitedname)
Oidx = ds.chars.index(ds.Oname)
yc1idx = ds.dim3.index(ds.demolist[0])
yclastidx = ds.dim3.index(ds.demolist[-1])

# %%
# Example of how to use above values to run symmetric equilibrium with four firms
if run_example:
    # Initialize variables, we're going to use the symmetric option in the compute_eqm function, so we only need to specify these variables for a single firm
    num_firms = 4
    bw_cntrfctl = np.ones((1,1)) * market_bw / float(num_firms) # bandwidth allocations
    gamma_cntrfctl = gamma_simplified # spectral efficiencies
    ds_cntrfctl = copy.deepcopy(ds) # create DemandSystem by copying the empty one with the basic structure, will change its contents later
    xis_cntrfctl = xis_simplified[0,:,:] # demand shocks for estimate of demand parameters
    theta = thetas_to_compute[0,:] # estimate of demand parameters
    pop_cntrfctl = pop_simplified # population in representative market
    market_size_cntrfctl = market_size_simplified # market size in representative market
    c_u_cntrfctl = c_u_simplified[0,:] # plan MCs based on estimate of demand parameters
    c_R_cntrfctl = np.ones((1,1)) * per_base_station_per_bw_cost_simplified[0] * bw_cntrfctl # base station costs given bandwidth allocation based on estimate of demand parameters
    R_0 = 0.5 * np.ones(c_R_cntrfctl.shape) # initial guess of equilibrium radii
    p_0 = np.copy(c_u_cntrfctl) # initial guess of equilibrium prices
    num_prods = dlims_simplified.shape[0] # number of products each firm has
    
    # Update DemandSystem
    ds_cntrfctl.data = np.zeros((pop_cntrfctl.shape[0],num_firms * num_prods,ds.data.shape[2])) 
    ds_cntrfctl.data[:,:,pidx] = np.zeros((pop_cntrfctl.shape[0], num_firms * num_prods)) # prices, doesn't matter b/c will be replaced in compute_eqm function
    ds_cntrfctl.data[:,:,qidx] = np.zeros((pop_cntrfctl.shape[0], num_firms * num_prods)) # download speeds, doesn't matter b/c will be replaced in compute_eqm function
    ds_cntrfctl.data[:,:,dlimidx] = np.tile(dlims_simplified[np.newaxis,:], (pop_cntrfctl.shape[0],num_firms)) # data limits
    ds_cntrfctl.data[:,:,vlimidx] = np.tile(vlims_simplified[np.newaxis,:], (pop_cntrfctl.shape[0],num_firms)) # voice limits
    ds_cntrfctl.data[:,:,Oidx] = 0. # Orange dummy, zero b/c captured by the \xi's
    ds_cntrfctl.firms = np.repeat(np.arange(num_firms, dtype=int) + 1, num_prods)
    ds_cntrfctl.J = num_firms * num_prods
    ds_cntrfctl.data[:,:,yc1idx:yclastidx+1] = income_distribution_simplified[:,np.newaxis,:]
    
    success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = compute_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, num_firms=num_firms, num_prods=num_prods, symmetric=True)
    
    if success:
        print(f"Successful computation. Equilibrium values:", flush=True)
        print(f"\tprices: {p_stars_}", flush=True)
        print(f"\tradii: {R_stars_}", flush=True)
        print(f"\tnumber of stations: {num_stations_stars_}", flush=True)
        print(f"\tnumber of stations per firm: {num_stations_per_firm_stars_}", flush=True)
        print(f"\tdownload speeds: {q_stars_}", flush=True)
        print(f"\tchannel capacities: {ccs_}", flush=True)
        print(f"\tchannel capacities per MHz: {ccs_per_bw_}", flush=True)
        print(f"\taverage path loss: {avg_path_losses_}", flush=True)
        print(f"\taverage signal-to-noise+inference ratio: {avg_SINR_}", flush=True)
    else:
        print(f"Equilibrium computation failed.", flush=True)

# %%
# Parameters used for richer environment for merger counterfactuals

constructed_markets_idx = np.load(f"{paths.arrays_path}constructed_markets_idx.npy")
bw_richer = np.load(f"{paths.arrays_path}mimic_market_bw.npy")
radii_richer = np.load(f"{paths.arrays_path}mimic_market_radii.npy")
product_firm_correspondence_richer = np.load(f"{paths.arrays_path}mimic_market_product_firm_correspondence_{task_id}.npy")
income_distribution_richer = np.load(f"{paths.arrays_path}mimic_market_income_distribution.npy")
prices_richer = np.load(f"{paths.arrays_path}mimic_market_prices_{task_id}.npy")
dlim_richer = np.load(f"{paths.arrays_path}mimic_market_dlim_{task_id}.npy")
vlim_richer = np.load(f"{paths.arrays_path}mimic_market_vlim_{task_id}.npy")
O_richer = np.load(f"{paths.arrays_path}mimic_market_O_{task_id}.npy")
firms_richer = np.load(f"{paths.arrays_path}mimic_market_firms_{task_id}.npy")
population_richer = np.load(f"{paths.arrays_path}mimic_market_population.npy")
market_size_richer = np.load(f"{paths.arrays_path}mimic_market_market_size.npy")
market_weights_richer = np.load(f"{paths.arrays_path}mimic_market_market_weights.npy")
gamma_richer = np.load(f"{paths.arrays_path}mimic_market_gamma.npy")
xis_richer = np.load(f"{paths.arrays_path}mimic_market_xis_{task_id}.npy")
c_R_richer = np.load(f"{paths.arrays_path}mimic_market_c_R_{task_id}.npy")
c_u_richer = np.load(f"{paths.arrays_path}mimic_market_c_u_{task_id}.npy")
bw_4g_equiv_richer = np.load(f"{paths.arrays_path}mimic_market_bw.npy")
radius_mergers_all = np.load(f"{paths.arrays_path}mimic_market_radii_mergers.npy")
impute_MVNO = {
    'impute': True, 
    'firms_share': np.array([True, True, False, True]), # all firms share with MVNO, except Free
    'include': True
}
list_MNOwMVNO = np.load(f"{paths.arrays_path}list_MNOwMVNO.npy")
list_MNOwMVNOnums = np.load(f"{paths.arrays_path}list_MNOwMVNOnums.npy")
mno_codes = {}
for mno_abbrev, mno_code in zip(list_MNOwMVNO, list_MNOwMVNOnums):
    mno_codes[mno_abbrev] = mno_code
merger_combos = np.load(f"{paths.arrays_path}merger_combos.npy")
num_prods_orig = np.load(f"{paths.arrays_path}num_prods_{task_id}.npy")[0]

# %%
# Parameters for all markets
c_u_all = np.load(f"{paths.arrays_path}c_u_{task_id}.npy")
c_R_all = np.load(f"{paths.arrays_path}c_R_{task_id}.npy")
bw_4g_equiv_all = np.load(f"{paths.arrays_path}bw_4g_equiv.npy")

# %%
# Parameters for symmetric counterfactuals
num_firms_array = np.load(f"{paths.arrays_path}num_firms_array.npy")
num_firms_array_extend = np.load(f"{paths.arrays_path}num_firms_array_extend.npy")
rep_market_size = np.load(f"{paths.arrays_path}rep_market_size.npy")[0]
rep_population = np.load(f"{paths.arrays_path}rep_population.npy")[0]
densities = np.load(f"{paths.arrays_path}cntrfctl_densities.npy")
densities_areatype = np.load(f"{paths.arrays_path}cntrfctl_densities_areatype.npy")
bw_vals = np.load(f"{paths.arrays_path}cntrfctl_bw_vals.npy")

# %%
# Function to run all counterfactuals given a particular vector of demand parameters to use
def compute_counterfactuals(theta_n):
    """Compute counterfactuals for a particular vector of demand parameters."""
    
    if print_updates:
        print(f"theta_n={theta_n} beginning computation...", flush=True)
    
    # Construct demand parameter and associated variables
    theta = thetas_to_compute[theta_n,:]
    
    # Copy variables we already have
    c_u = c_u_all[theta_n,:]
    c_R = c_R_all[theta_n,:,:]
    c_R_per_unit_bw = c_R / bw_4g_equiv_all

    # Compute counterfactual equilibria

    per_user_costs = c_u_simplified[theta_n,:]
    per_base_station_per_bw_cost = per_base_station_per_bw_cost_simplified[theta_n]

    pop_cntrfctl = pop_simplified
    market_size_cntrfctl = market_size_simplified
    gamma_cntrfctl = gamma_simplified
    
    num_prods = dlims_simplified.shape[0]

    # Create empty arrays for all of the counterfactual results that we will fill in
    
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
    
    num_prods_all = num_prods_orig # number of products in data
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
    p_0s = np.tile(per_user_costs[np.newaxis,:], (num_firms_array_extend.shape[0],1))
    R_0s = np.ones(num_firms_array_extend.shape)[:,np.newaxis] * 0.5

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
        ds_cntrfctl.data[:,:,dlimidx] = np.tile(dlims_simplified[np.newaxis,:], (1,num_firms))
        ds_cntrfctl.data[:,:,vlimidx] = np.tile(vlims_simplified[np.newaxis,:], (1,num_firms))
        ds_cntrfctl.data[:,:,Oidx] = 0.
        ds_cntrfctl.firms = np.repeat(np.arange(num_firms, dtype=int) + 1, num_prods)
        ds_cntrfctl.J = num_firms * num_prods

        # Create income distribution with properties I want
        ds_cntrfctl.data[:,:,yc1idx:yclastidx+1] = income_distribution_simplified[:,np.newaxis,:]

        # Create ds for 1-product version
        ds_cntrfctl_1p = copy.deepcopy(ds_cntrfctl)
        ds_cntrfctl_1p.data = ds_cntrfctl_1p.data[:,np.arange(ds_cntrfctl_1p.data.shape[1]) % num_prods == np.argmax(dlims_simplified),:] # keep only the highest data limit plan
        ds_cntrfctl_1p.firms = np.repeat(np.arange(num_firms, dtype=int) + 1, 1)
        ds_cntrfctl_1p.J = num_firms * 1
        
        # Create market variables with properties I want
        bw_cntrfctl = np.ones((1,1)) * market_bw / float(num_firms)
        xis_cntrfctl = xis_simplified[theta_n,:,:]
        xis_cntrfctl_1p = np.ones((1,1)) * theta[coef.O]

        # Create cost arrays
        c_u_cntrfctl = per_user_costs
        select_1p = np.arange(num_prods) == np.argmax(dlims_simplified)
        c_u_cntrfctl_1p = c_u_cntrfctl[select_1p] # keep only the highest data limit plan
        c_R_cntrfctl = np.ones((1,1)) * per_base_station_per_bw_cost * bw_cntrfctl # per tower cost based on level of bandwidth for each firm

        # Set starting values (if None, num_firms=1 can cause problems for convergence)
        R_0 = R_0s[i,:][:,np.newaxis]
        p_0 = p_0s[i,:]
        p_0_1p = p_0[select_1p]

        # Simple symmetric equilibrium result for representative values
        start = time.time()
        success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = compute_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0, p_0, num_firms=num_firms, num_prods=num_prods, symmetric=True)
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
        success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = compute_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl_allfixed, R_0, p_0_allfixed, num_firms=num_firms, num_prods=num_prods, symmetric=True)
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
                success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = compute_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta, pop_cntrfctl_dens, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl, R_0_dens, p_0_dens, num_firms=num_firms, num_prods=num_prods, symmetric=True, areatype=densities_areatype[j])
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
            success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = compute_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl_1p, xis_cntrfctl_1p, theta, pop_cntrfctl_dens, market_size_cntrfctl, c_u_cntrfctl_1p, c_R_cntrfctl, R_0_dens_1p, p_0_dens_1p, num_firms, 1, symmetric=True, areatype=densities_areatype[j])
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
                success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = compute_eqm(bw_cntrfctl_bw, gamma_cntrfctl, ds_cntrfctl, xis_cntrfctl, theta, pop_cntrfctl, market_size_cntrfctl, c_u_cntrfctl, c_R_cntrfctl_bw, R_0, p_0_bw, num_firms=num_firms, num_prods=num_prods, symmetric=True)
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
        
        # Calculate equilibrium with asymmetric firms
        if np.isin(num_firms, np.array([3])):
            start = time.time()
            bw_cntrfctl = market_bw * np.array([[0.5, 0.25, 0.25]])
            c_R_cntrfctl_asymmetric = per_base_station_per_bw_cost * bw_cntrfctl
            expand_to_num_firms = lambda x: np.tile(x, 3)
            product_firm_correspondence = np.repeat(np.arange(num_firms), num_prods)
            success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = compute_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, expand_to_num_firms(xis_cntrfctl), theta, pop_cntrfctl, market_size_cntrfctl, expand_to_num_firms(c_u_cntrfctl), c_R_cntrfctl_asymmetric, expand_to_num_firms(R_0), expand_to_num_firms(p_0), product_firm_correspondence=product_firm_correspondence)
            if print_updates:
                print(f"theta_n={theta_n}: Finished calculating asymmetric equilibrium in {np.round(time.time() - start, 1)} seconds.", flush=True)
            successful_asymmetric_allbw[1], cs_by_type_asymmetric_allbw[1,:], cs_asymmetric_allbw[1], ps_asymmetric_allbw[1], ts_asymmetric_allbw[1] = success, cs_by_type_, cs_, ps_, ts_
            select_idx = np.array([0,1]) # select one of the big bw firms and one of the small bw firms
            select_idx_prod = np.concatenate((np.arange(num_prods) + 0 * num_prods, np.arange(num_prods) + 2 * num_prods)) # select the products of one of the big bw firms and products of the small bw firms
            p_stars_asymmetric_allbw[1,:,:], R_stars_asymmetric_allbw[1,:], num_stations_stars_asymmetric_allbw[1], num_stations_per_firm_stars_asymmetric_allbw[1,:], q_stars_asymmetric_allbw[1,:], ccs_asymmetric_allbw[1,:], ccs_per_bw_asymmetric_allbw[1,:], avg_path_losses_asymmetric_allbw[1,:], avg_SINR_asymmetric_allbw[1,:] = np.reshape(p_stars_[select_idx_prod], (2, num_prods)), R_stars_[select_idx], num_stations_stars_, num_stations_per_firm_stars_[select_idx], q_stars_[select_idx], ccs_[select_idx], ccs_per_bw_[select_idx], avg_path_losses_[select_idx], avg_SINR_[select_idx]
            
            # Add idx 0, which is just copied over from symmetric equilibrium
            successful_asymmetric_allbw[0], cs_by_type_asymmetric_allbw[0,:], cs_asymmetric_allbw[0], ps_asymmetric_allbw[0], ts_asymmetric_allbw[0] = successful[i], cs_by_type[i,:], cs[i], ps[i], ts[i]
            p_stars_asymmetric_allbw[0,:,:], R_stars_asymmetric_allbw[0,:], num_stations_stars_asymmetric_allbw[0], num_stations_per_firm_stars_asymmetric_allbw[0,:], q_stars_asymmetric_allbw[0,:], ccs_asymmetric_allbw[0,:], ccs_per_bw_asymmetric_allbw[0,:], avg_path_losses_asymmetric_allbw[0,:], avg_SINR_asymmetric_allbw[0,:] = p_stars[i,:][np.newaxis,:], R_stars[i], num_stations_stars[i], num_stations_per_firm_stars[i], q_stars[i], ccs[i], ccs_per_bw[i], avg_path_losses[i], avg_SINR[i]
            
            if (task_id == 0) and (theta_n == 0):
                np.random.seed(123456)
                num_tests = 30
                success_test = np.zeros((num_tests,), dtype=bool)
                p_stars_test = np.ones((6, num_tests))
                R_stars_test = np.ones((3, num_tests))
                for test_i in range(num_tests):
                    start = time.time()
                    R_0_test = np.array([0.5, 0.5, 0.5]) + np.random.uniform(-0.1, 0.35, size=3)
                    p_0_test = np.array([10.0, 20.0, 10.0, 20.0, 10.0, 20.0]) + np.random.uniform(-1.0, 10.0, size=6)
                    asymmetric_eqm_test_res = compute_eqm(bw_cntrfctl, gamma_cntrfctl, ds_cntrfctl, expand_to_num_firms(xis_cntrfctl), theta, pop_cntrfctl, market_size_cntrfctl, expand_to_num_firms(c_u_cntrfctl), c_R_cntrfctl_asymmetric, R_0_test, p_0_test, product_firm_correspondence=product_firm_correspondence)
                    success_test[test_i], p_stars_test[:,test_i], R_stars_test[:,test_i] = asymmetric_eqm_test_res[0], asymmetric_eqm_test_res[5], asymmetric_eqm_test_res[6]
                np.save(f"{paths.arrays_path}success_test.npy", success_test)
                np.save(f"{paths.arrays_path}p_stars_test.npy", p_stars_test)
                np.save(f"{paths.arrays_path}R_stars_test.npy", R_stars_test)
        
        # Calculate equilibrium of different mergers
        if num_firms == 1: # has nothing to do with monopoly case, just doesn't depend on number of firms so choosing first case
            
            start_asym_all = time.time()
            
            # Create arrays that describe this market
            bw_cntrfctl = bw_richer
            xis_cntrfctl = xis_richer[theta_n,:,:]
            c_R_cntrfctl = c_R_richer[theta_n,:,:]
            R_0 = np.copy(radii_richer)
            R_0_longrun = np.ones(R_0.shape) * 0.5 # this is better behaved when we allow all to adjust
            p_0 = np.copy(c_u_richer[theta_n,:])
            p_0_longrun = np.copy(c_u_richer[theta_n,:]) # this is better behaved when we allow all to adjust
            product_firm_correspondence = product_firm_correspondence_richer
            
            # Create ds with properties I want
            ds_cntrfctl_income_distribution = income_distribution_richer
            ds_cntrfctl = copy.deepcopy(ds) # can rename it ds_cntrfctl b/c don't need to use old ds_cntrfctl again in this iteration of num_firms
            ds_cntrfctl.data = np.zeros((constructed_markets_idx.shape[0], p_0.shape[0], len(ds.dim3))) # initialize
            ds_cntrfctl.data[:,:,pidx] = prices_richer
            ds_cntrfctl.data[:,:,qidx] = 0.0 # doesn't matter, so just initialize with zero
            ds_cntrfctl.data[:,:,dlimidx] = dlim_richer
            ds_cntrfctl.data[:,:,vlimidx] = vlim_richer
            ds_cntrfctl.data[:,:,Oidx] = O_richer
            ds_cntrfctl.J = ds_cntrfctl.data.shape[1]
            ds_cntrfctl.M = ds_cntrfctl.data.shape[0]
            ds_cntrfctl.firms = firms_richer
            ds_cntrfctl.data[:,:,yc1idx:yclastidx+1] = ds_cntrfctl_income_distribution[:,np.newaxis,:] # same income distribution for the other counterfactuals
            
            # Create other market description variables with properties I want
            pop_cntrfctl_merge = population_richer
            market_size_cntrfctl_merge = market_size_richer
            market_weights_merge = market_weights_richer
            gamma_cntrfctl_merge = gamma_richer
            
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
                    c_R_cntrfctl_merge[:,np.min(merger_indices)] = np.min((c_R_cntrfctl / bw_4g_equiv_richer)[:,merger_indices], axis=1) # for merged firm, c_R is based on the minimum of the two firms' per-bw base station cost
                    c_R_cntrfctl_merge[:,np.arange(bw_cntrfctl_merge.shape[1]) != np.min(merger_indices)] = (c_R_cntrfctl / bw_4g_equiv_richer)[:,~np.isin(np.arange(bw_cntrfctl.shape[1]), merger_indices)] # for all other firms, their c_R is the same
                    c_R_cntrfctl_merge = c_R_cntrfctl_merge * bw_cntrfctl_merge # account for how much bandwidth each firm has since this is a specification that scales with bandwidth
                else:
                    c_R_cntrfctl_merge[:,:] = np.copy(c_R_cntrfctl)
                R_0_merge = np.zeros((R_0.shape[0], num_firms_woMVNO)) # this is the exact radius, we are not changing it
                if merger_:
                    R_0_merge[:,np.min(merger_indices)] = radius_mergers_all[:,np.min(merger_indices),j] # for merged firm, it's whatever the merged firm's radius is in the data (held fixed in short run and starting point in long run)
                    R_0_merge[:,np.arange(R_0_merge.shape[1]) != np.min(merger_indices)] = radius_mergers_all[:,~np.isin(np.arange(R_0.shape[1]), merger_indices),j] # for all other firms, their radii are the same
                else:
                    R_0_merge[:,:] = np.copy(radius_mergers_all[:,:,j])
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
                success, cs_by_type_, cs_, ps_, ts_, p_stars_, R_stars_, num_stations_stars_, num_stations_per_firm_stars_, q_stars_, ccs_, ccs_per_bw_, avg_path_losses_, avg_SINR_, cc_cntrfctl, num_stations_cntrfctl, ds_cntrfctl_ = compute_eqm(bw_cntrfctl_merge, gamma_cntrfctl_merge, ds_cntrfctl_merge, xis_cntrfctl_merge, theta, pop_cntrfctl_merge, market_size_cntrfctl_merge, c_u, c_R_cntrfctl_merge, R_0_merge, p_0_merge, product_firm_correspondence=product_firm_correspondence_merge, impute_MVNO=impute_MVNO_merge, R_fixed=True, market_weights=market_weights_merge)
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
                
            if print_updates:
                print(f"theta_n={theta_n}: Finished calculating mergers in {np.round(time.time() - start_asym_all, 1)} seconds.", flush=True)
        
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
    cs = cs - cs[0]
    ps = ps - ps[0]
    ts = ts - ts[0]
    cs_by_type = cs_by_type - cs_by_type[0,:][np.newaxis,:]
    
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

c_u = np.zeros(np.load(f"{paths.arrays_path}c_u_{task_id}.npy").shape)
c_R = np.zeros(np.load(f"{paths.arrays_path}c_R_{task_id}.npy").shape)

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

num_prods_all = num_prods_orig # number of products in data
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

# Initialize multiprocessing
pool = Pool(num_cpus)
chunksize = 1

# Compute perturbed demand parameter estimate equilibria and store relevant variables
for ind, res in enumerate(pool.imap(compute_counterfactuals, range(theta_N)), chunksize):
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
# Save arrays
np.savez_compressed(f"{paths.arrays_path}all_arrays_{task_id}.npz", p_stars, R_stars, num_stations_stars, num_stations_per_firm_stars, q_stars, cs_by_type, cs, ps, ts, ccs, ccs_per_bw, avg_path_losses, avg_SINR, full_elasts, partial_elasts, p_stars_allfixed, R_stars_allfixed, num_stations_stars_allfixed, num_stations_per_firm_stars_allfixed, q_stars_allfixed, cs_by_type_allfixed, cs_allfixed, ps_allfixed, ts_allfixed, ccs_allfixed, ccs_per_bw_allfixed, avg_path_losses_allfixed, avg_SINR_allfixed, partial_Pif_partial_bf_allfixed, partial_Piotherf_partial_bf_allfixed, partial_diffPif_partial_bf_allfixed, partial_Pif_partial_b_allfixed, partial_CS_partial_b_allfixed, partial_Pif_partial_bf_allbw, partial_Piotherf_partial_bf_allbw, partial_diffPif_partial_bf_allbw, partial_Pif_partial_b_allbw, partial_CS_partial_b_allbw, c_u, c_R, p_stars_shortrun, R_stars_shortrun, num_stations_stars_shortrun, num_stations_per_firm_stars_shortrun, q_stars_shortrun, cs_by_type_shortrun, cs_shortrun, ps_shortrun, ts_shortrun, ccs_shortrun, ccs_per_bw_shortrun, avg_path_losses_shortrun, p_stars_free_allfixed, R_stars_free_allfixed, num_stations_stars_free_allfixed, num_stations_per_firm_stars_free_allfixed, q_stars_free_allfixed, cs_by_type_free_allfixed, cs_free_allfixed, ps_free_allfixed, ts_free_allfixed, ccs_free_allfixed, ccs_per_bw_free_allfixed, avg_path_losses_free_allfixed, p_stars_free_allbw, R_stars_free_allbw, num_stations_stars_free_allbw, num_stations_per_firm_stars_free_allbw, q_stars_free_allbw, cs_by_type_free_allbw, cs_free_allbw, ps_free_allbw, ts_free_allbw, ccs_free_allbw, ccs_per_bw_free_allbw, avg_path_losses_free_allbw, p_stars_dens, R_stars_dens, num_stations_stars_dens, num_stations_per_firm_stars_dens, q_stars_dens, cs_dens, cs_by_type_dens, ps_dens, ts_dens, ccs_dens, ccs_per_bw_dens, avg_path_losses_dens, avg_SINR_dens, p_stars_bw, R_stars_bw, num_stations_stars_bw, num_stations_per_firm_stars_bw, q_stars_bw, cs_bw, cs_by_type_bw, ps_bw, ts_bw, ccs_bw, ccs_per_bw_bw, avg_path_losses_bw, avg_SINR_bw, p_stars_dens_1p, R_stars_dens_1p, num_stations_stars_dens_1p, num_stations_per_firm_stars_dens_1p, q_stars_dens_1p, cs_dens_1p, cs_by_type_dens_1p, ps_dens_1p, ts_dens_1p, ccs_dens_1p, ccs_per_bw_dens_1p, avg_path_losses_dens_1p, avg_SINR_dens_1p, p_stars_asymmetric_allbw, R_stars_asymmetric_allbw, num_stations_stars_asymmetric_allbw, num_stations_per_firm_stars_asymmetric_allbw, q_stars_asymmetric_allbw, cs_asymmetric_allbw, cs_by_type_asymmetric_allbw, ps_asymmetric_allbw, ts_asymmetric_allbw, ccs_asymmetric_allbw, ccs_per_bw_asymmetric_allbw, avg_path_losses_asymmetric_allbw, avg_SINR_asymmetric_allbw, p_stars_shortrunall, R_stars_shortrunall, num_stations_stars_shortrunall, num_stations_per_firm_stars_shortrunall, q_stars_shortrunall, cs_shortrunall, cs_by_type_shortrunall, ps_shortrunall, ts_shortrunall, ccs_shortrunall, ccs_per_bw_shortrunall, avg_path_losses_shortrunall, p_stars_longrunall, R_stars_longrunall, num_stations_stars_longrunall, num_stations_per_firm_stars_longrunall, q_stars_longrunall, cs_longrunall, cs_by_type_longrunall, ps_longrunall, ts_longrunall, ccs_longrunall, ccs_per_bw_longrunall, avg_path_losses_longrunall, successful_extend, successful_extend_allfixed, successful_bw_deriv_allfixed, successful_bw_deriv_allbw, successful_shortrun, successful_free_allfixed, successful_free_allbw, successful_dens, successful_bw, successful_dens_1p, successful_asymmetric_allbw, successful_shortrunall, successful_longrunall, per_user_costs)
