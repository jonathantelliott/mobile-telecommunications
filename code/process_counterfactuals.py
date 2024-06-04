import numpy as np
import pandas as pd

import matplotlib as mpl
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
plt.rcParams['font.family'] = "serif"
plt.rcParams['mathtext.fontset'] = "dejavuserif"

from itertools import combinations

import paths

import demand.demandsystem as demsys

import counterfactuals.infrastructurefunctions as infr
import counterfactuals.infrastructureequilibrium as ie
import demand.coefficients as coef
import demand.blpextension as blp

import pickle

# %%
print_ = False
save_ = True

# %%
avg_price_elasts = paths.avg_price_elasts
div_ratios = paths.div_ratios

# %%
# Import infrastructure / quality data
df_inf = pd.read_csv(f"{paths.data_path}infrastructure_clean.csv", engine="python") # engine helps encoding, error with commune names, but doesn't matter b/c not used
df_inf = df_inf[df_inf['market'] > 0] # don't include Rest-of-France market
df_q = pd.read_csv(f"{paths.data_path}quality_ookla.csv")
df_q = df_q[df_q['market'] > 0] # don't include Rest-of-France market

# %%
# Load the DemandSystem created when estimating demand
with open(f"{paths.data_path}demandsystem_0.obj", "rb") as file_ds:
    ds = pickle.load(file_ds)
    
# Drop Rest-of-France market
market_idx = ds.dim3.index(ds.marketname)
market_numbers = np.max(ds.data[:,:,market_idx], axis=1)
ds.data = ds.data[market_numbers > 0,:,:] # drop "Rest of France"# %%

# %%
# Define functions to load results
c_u = lambda task_id: np.load(f"{paths.arrays_path}c_u_{task_id}.npy")
c_R = lambda task_id: np.load(f"{paths.arrays_path}c_R_{task_id}.npy")

p_stars = lambda task_id: np.load(f"{paths.arrays_path}p_stars_{task_id}.npy")
R_stars = lambda task_id: np.load(f"{paths.arrays_path}R_stars_{task_id}.npy")
num_stations_stars = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_{task_id}.npy")
num_stations_per_firm_stars = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_{task_id}.npy")
q_stars = lambda task_id: np.load(f"{paths.arrays_path}q_stars_{task_id}.npy")
cs_by_type = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_{task_id}.npy")
cs = lambda task_id: np.load(f"{paths.arrays_path}cs_{task_id}.npy")
ps = lambda task_id: np.load(f"{paths.arrays_path}ps_{task_id}.npy")
ts = lambda task_id: np.load(f"{paths.arrays_path}ts_{task_id}.npy")
ccs = lambda task_id: np.load(f"{paths.arrays_path}ccs_{task_id}.npy")
ccs_per_bw = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_{task_id}.npy")
avg_path_losses = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_{task_id}.npy")
avg_SINR = lambda task_id: np.load(f"{paths.arrays_path}avg_SINR_{task_id}.npy")
partial_elasts = lambda task_id: np.load(f"{paths.arrays_path}partial_elasts_{task_id}.npy")
full_elasts = lambda task_id: np.load(f"{paths.arrays_path}full_elasts_{task_id}.npy")
p_stars_allfixed = lambda task_id: np.load(f"{paths.arrays_path}p_stars_allfixed_{task_id}.npy")
R_stars_allfixed = lambda task_id: np.load(f"{paths.arrays_path}R_stars_allfixed_{task_id}.npy")
num_stations_stars_allfixed = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_allfixed_{task_id}.npy")
num_stations_per_firm_stars_allfixed = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_allfixed_{task_id}.npy")
q_stars_allfixed = lambda task_id: np.load(f"{paths.arrays_path}q_stars_allfixed_{task_id}.npy")
cs_by_type_allfixed = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_allfixed_{task_id}.npy")
cs_allfixed = lambda task_id: np.load(f"{paths.arrays_path}cs_allfixed_{task_id}.npy")
ps_allfixed = lambda task_id: np.load(f"{paths.arrays_path}ps_allfixed_{task_id}.npy")
ts_allfixed = lambda task_id: np.load(f"{paths.arrays_path}ts_allfixed_{task_id}.npy")
ccs_allfixed = lambda task_id: np.load(f"{paths.arrays_path}ccs_allfixed_{task_id}.npy")
ccs_per_bw_allfixed = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_allfixed_{task_id}.npy")
avg_path_losses_allfixed = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_allfixed_{task_id}.npy")
avg_SINR_allfixed = lambda task_id: np.load(f"{paths.arrays_path}avg_SINR_allfixed_{task_id}.npy")
partial_Pif_partial_bf_allfixed = lambda task_id: np.load(f"{paths.arrays_path}partial_Pif_partial_bf_allfixed_{task_id}.npy")
partial_Piotherf_partial_bf_allfixed = lambda task_id: np.load(f"{paths.arrays_path}partial_Piotherf_partial_bf_allfixed_{task_id}.npy")
partial_diffPif_partial_bf_allfixed = lambda task_id: np.load(f"{paths.arrays_path}partial_diffPif_partial_bf_allfixed_{task_id}.npy")
partial_Pif_partial_b_allfixed = lambda task_id: np.load(f"{paths.arrays_path}partial_Pif_partial_b_allfixed_{task_id}.npy")
partial_CS_partial_b_allfixed = lambda task_id: np.load(f"{paths.arrays_path}partial_CS_partial_b_allfixed_{task_id}.npy")
p_stars_shortrun = lambda task_id: np.load(f"{paths.arrays_path}p_stars_shortrun_{task_id}.npy")
R_stars_shortrun = lambda task_id: np.load(f"{paths.arrays_path}R_stars_shortrun_{task_id}.npy")
num_stations_stars_shortrun = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_shortrun_{task_id}.npy")
num_stations_per_firm_stars_shortrun = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_shortrun_{task_id}.npy")
q_stars_shortrun = lambda task_id: np.load(f"{paths.arrays_path}q_stars_shortrun_{task_id}.npy")
cs_by_type_shortrun = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_shortrun_{task_id}.npy")
cs_shortrun = lambda task_id: np.load(f"{paths.arrays_path}cs_shortrun_{task_id}.npy")
ps_shortrun = lambda task_id: np.load(f"{paths.arrays_path}ps_shortrun_{task_id}.npy")
ts_shortrun = lambda task_id: np.load(f"{paths.arrays_path}ts_shortrun_{task_id}.npy")
ccs_shortrun = lambda task_id: np.load(f"{paths.arrays_path}ccs_shortrun_{task_id}.npy")
ccs_per_bw_shortrun = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_shortrun_{task_id}.npy")
avg_path_losses_shortrun = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_shortrun_{task_id}.npy")
p_stars_free_allfixed = lambda task_id: np.load(f"{paths.arrays_path}p_stars_free_allfixed_{task_id}.npy")
R_stars_free_allfixed = lambda task_id: np.load(f"{paths.arrays_path}R_stars_free_allfixed_{task_id}.npy")
num_stations_stars_free_allfixed = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_free_allfixed_{task_id}.npy")
num_stations_per_firm_stars_free_allfixed = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_free_allfixed_{task_id}.npy")
q_stars_free_allfixed = lambda task_id: np.load(f"{paths.arrays_path}q_stars_free_allfixed_{task_id}.npy")
cs_by_type_free_allfixed = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_free_allfixed_{task_id}.npy")
cs_free_allfixed = lambda task_id: np.load(f"{paths.arrays_path}cs_free_allfixed_{task_id}.npy")
ps_free_allfixed = lambda task_id: np.load(f"{paths.arrays_path}ps_free_allfixed_{task_id}.npy")
ts_free_allfixed = lambda task_id: np.load(f"{paths.arrays_path}ts_free_allfixed_{task_id}.npy")
ccs_free_allfixed = lambda task_id: np.load(f"{paths.arrays_path}ccs_free_allfixed_{task_id}.npy")
ccs_per_bw_free_allfixed = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_free_allfixed_{task_id}.npy")
avg_path_losses_free_allfixed = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_free_allfixed_{task_id}.npy")
partial_Pif_partial_bf_allbw = lambda task_id: np.load(f"{paths.arrays_path}partial_Pif_partial_bf_allbw_{task_id}.npy")
partial_Piotherf_partial_bf_allbw = lambda task_id: np.load(f"{paths.arrays_path}partial_Piotherf_partial_bf_allbw_{task_id}.npy")
partial_diffPif_partial_bf_allbw = lambda task_id: np.load(f"{paths.arrays_path}partial_diffPif_partial_bf_allbw_{task_id}.npy")
partial_Pif_partial_b_allbw = lambda task_id: np.load(f"{paths.arrays_path}partial_Pif_partial_b_allbw_{task_id}.npy")
partial_CS_partial_b_allbw = lambda task_id: np.load(f"{paths.arrays_path}partial_CS_partial_b_allbw_{task_id}.npy")
p_stars_free_allbw = lambda task_id: np.load(f"{paths.arrays_path}p_stars_free_allbw_{task_id}.npy")
R_stars_free_allbw = lambda task_id: np.load(f"{paths.arrays_path}R_stars_free_allbw_{task_id}.npy")
num_stations_stars_free_allbw = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_free_allbw_{task_id}.npy")
num_stations_per_firm_stars_free_allbw = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_free_allbw_{task_id}.npy")
q_stars_free_allbw = lambda task_id: np.load(f"{paths.arrays_path}q_stars_free_allbw_{task_id}.npy")
cs_by_type_free_allbw = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_free_allbw_{task_id}.npy")
cs_free_allbw = lambda task_id: np.load(f"{paths.arrays_path}cs_free_allbw_{task_id}.npy")
ps_free_allbw = lambda task_id: np.load(f"{paths.arrays_path}ps_free_allbw_{task_id}.npy")
ts_free_allbw = lambda task_id: np.load(f"{paths.arrays_path}ts_free_allbw_{task_id}.npy")
ccs_free_allbw = lambda task_id: np.load(f"{paths.arrays_path}ccs_free_allbw_{task_id}.npy")
ccs_per_bw_free_allbw = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_free_allbw_{task_id}.npy")
avg_path_losses_free_allbw = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_free_allbw_{task_id}.npy")
p_stars_dens = lambda task_id: np.load(f"{paths.arrays_path}p_stars_dens_{task_id}.npy")
R_stars_dens = lambda task_id: np.load(f"{paths.arrays_path}R_stars_dens_{task_id}.npy")
num_stations_stars_dens = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_dens_{task_id}.npy")
num_stations_per_firm_stars_dens = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_dens_{task_id}.npy")
q_stars_dens = lambda task_id: np.load(f"{paths.arrays_path}q_stars_dens_{task_id}.npy")
cs_by_type_dens = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_dens_{task_id}.npy")
cs_dens = lambda task_id: np.load(f"{paths.arrays_path}cs_dens_{task_id}.npy")
ps_dens = lambda task_id: np.load(f"{paths.arrays_path}ps_dens_{task_id}.npy")
ts_dens = lambda task_id: np.load(f"{paths.arrays_path}ts_dens_{task_id}.npy")
ccs_dens = lambda task_id: np.load(f"{paths.arrays_path}ccs_dens_{task_id}.npy")
ccs_per_bw_dens = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_dens_{task_id}.npy")
avg_path_losses_dens = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_dens_{task_id}.npy")
avg_SINR_dens = lambda task_id: np.load(f"{paths.arrays_path}avg_SINR_dens_{task_id}.npy")
p_stars_bw = lambda task_id: np.load(f"{paths.arrays_path}p_stars_bw_{task_id}.npy")
R_stars_bw = lambda task_id: np.load(f"{paths.arrays_path}R_stars_bw_{task_id}.npy")
num_stations_stars_bw = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_bw_{task_id}.npy")
num_stations_per_firm_stars_bw = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_bw_{task_id}.npy")
q_stars_bw = lambda task_id: np.load(f"{paths.arrays_path}q_stars_bw_{task_id}.npy")
cs_by_type_bw = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_bw_{task_id}.npy")
cs_bw = lambda task_id: np.load(f"{paths.arrays_path}cs_bw_{task_id}.npy")
ps_bw = lambda task_id: np.load(f"{paths.arrays_path}ps_bw_{task_id}.npy")
ts_bw = lambda task_id: np.load(f"{paths.arrays_path}ts_bw_{task_id}.npy")
ccs_bw = lambda task_id: np.load(f"{paths.arrays_path}ccs_bw_{task_id}.npy")
ccs_per_bw_bw = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_bw_{task_id}.npy")
avg_path_losses_bw = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_bw_{task_id}.npy")
avg_SINR_bw = lambda task_id: np.load(f"{paths.arrays_path}avg_SINR_bw_{task_id}.npy")
p_stars_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}p_stars_dens_1p_{task_id}.npy")
R_stars_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}R_stars_dens_1p_{task_id}.npy")
num_stations_stars_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_dens_1p_{task_id}.npy")
num_stations_per_firm_stars_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_dens_1p_{task_id}.npy")
q_stars_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}q_stars_dens_1p_{task_id}.npy")
cs_by_type_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_dens_1p_{task_id}.npy")
cs_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}cs_dens_1p_{task_id}.npy")
ps_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}ps_dens_1p_{task_id}.npy")
ts_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}ts_dens_1p_{task_id}.npy")
ccs_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}ccs_dens_1p_{task_id}.npy")
ccs_per_bw_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_dens_1p_{task_id}.npy")
avg_path_losses_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_dens_1p_{task_id}.npy")
avg_SINR_dens_1p = lambda task_id: np.load(f"{paths.arrays_path}avg_SINR_dens_1p_{task_id}.npy")
p_stars_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}p_stars_asymmetric_allbw_{task_id}.npy")
R_stars_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}R_stars_asymmetric_allbw_{task_id}.npy")
num_stations_stars_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_asymmetric_allbw_{task_id}.npy")
num_stations_per_firm_stars_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_asymmetric_allbw_{task_id}.npy")
q_stars_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}q_stars_asymmetric_allbw_{task_id}.npy")
cs_by_type_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_asymmetric_allbw_{task_id}.npy")
cs_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}cs_asymmetric_allbw_{task_id}.npy")
ps_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}ps_asymmetric_allbw_{task_id}.npy")
ts_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}ts_asymmetric_allbw_{task_id}.npy")
ccs_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}ccs_asymmetric_allbw_{task_id}.npy")
ccs_per_bw_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_asymmetric_allbw_{task_id}.npy")
avg_path_losses_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_asymmetric_allbw_{task_id}.npy")
avg_SINR_asymmetric_allbw = lambda task_id: np.load(f"{paths.arrays_path}avg_SINR_asymmetric_allbw_{task_id}.npy")
p_stars_shortrunall = lambda task_id: np.load(f"{paths.arrays_path}p_stars_shortrunall_{task_id}.npy")
R_stars_shortrunall = lambda task_id: np.load(f"{paths.arrays_path}R_stars_shortrunall_{task_id}.npy")
num_stations_stars_shortrunall = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_shortrunall_{task_id}.npy")
num_stations_per_firm_stars_shortrunall = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_shortrunall_{task_id}.npy")
q_stars_shortrunall = lambda task_id: np.load(f"{paths.arrays_path}q_stars_shortrunall_{task_id}.npy")
cs_by_type_shortrunall = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_shortrunall_{task_id}.npy")
cs_shortrunall = lambda task_id: np.load(f"{paths.arrays_path}cs_shortrunall_{task_id}.npy")
ps_shortrunall = lambda task_id: np.load(f"{paths.arrays_path}ps_shortrunall_{task_id}.npy")
ts_shortrunall = lambda task_id: np.load(f"{paths.arrays_path}ts_shortrunall_{task_id}.npy")
ccs_shortrunall = lambda task_id: np.load(f"{paths.arrays_path}ccs_shortrunall_{task_id}.npy")
ccs_per_bw_shortrunall = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_shortrunall_{task_id}.npy")
avg_path_losses_shortrunall = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_shortrunall_{task_id}.npy")
p_stars_longrunall = lambda task_id: np.load(f"{paths.arrays_path}p_stars_longrunall_{task_id}.npy")
R_stars_longrunall = lambda task_id: np.load(f"{paths.arrays_path}R_stars_longrunall_{task_id}.npy")
num_stations_stars_longrunall = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_longrunall_{task_id}.npy")
num_stations_per_firm_stars_longrunall = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_longrunall_{task_id}.npy")
q_stars_longrunall = lambda task_id: np.load(f"{paths.arrays_path}q_stars_longrunall_{task_id}.npy")
cs_by_type_longrunall = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_longrunall_{task_id}.npy")
cs_longrunall = lambda task_id: np.load(f"{paths.arrays_path}cs_longrunall_{task_id}.npy")
ps_longrunall = lambda task_id: np.load(f"{paths.arrays_path}ps_longrunall_{task_id}.npy")
ts_longrunall = lambda task_id: np.load(f"{paths.arrays_path}ts_longrunall_{task_id}.npy")
ccs_longrunall = lambda task_id: np.load(f"{paths.arrays_path}ccs_longrunall_{task_id}.npy")
ccs_per_bw_longrunall = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_longrunall_{task_id}.npy")
avg_path_losses_longrunall = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_longrunall_{task_id}.npy")
per_user_costs = lambda task_id: np.load(f"{paths.arrays_path}per_user_costs_{task_id}.npy")

c_u_se = lambda task_id: np.load(f"{paths.arrays_path}c_u_se_{task_id}.npy")
c_R_se = lambda task_id: np.load(f"{paths.arrays_path}c_R_se_{task_id}.npy")

p_stars_se = lambda task_id: np.load(f"{paths.arrays_path}p_stars_se_{task_id}.npy")
R_stars_se = lambda task_id: np.load(f"{paths.arrays_path}R_stars_se_{task_id}.npy")
num_stations_stars_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_se_{task_id}.npy")
num_stations_per_firm_stars_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_se_{task_id}.npy")
q_stars_se = lambda task_id: np.load(f"{paths.arrays_path}q_stars_se_{task_id}.npy")
cs_by_type_se = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_se_{task_id}.npy")
cs_se = lambda task_id: np.load(f"{paths.arrays_path}cs_se_{task_id}.npy")
ps_se = lambda task_id: np.load(f"{paths.arrays_path}ps_se_{task_id}.npy")
ts_se = lambda task_id: np.load(f"{paths.arrays_path}ts_se_{task_id}.npy")
ccs_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_se_{task_id}.npy")
ccs_per_bw_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_se_{task_id}.npy")
avg_path_losses_se = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_se_{task_id}.npy")
partial_elasts_se = lambda task_id: np.load(f"{paths.arrays_path}partial_elasts_se_{task_id}.npy")
full_elasts_se = lambda task_id: np.load(f"{paths.arrays_path}full_elasts_se_{task_id}.npy")
p_stars_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}p_stars_allfixed_se_{task_id}.npy")
R_stars_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}R_stars_allfixed_se_{task_id}.npy")
num_stations_stars_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_allfixed_se_{task_id}.npy")
num_stations_per_firm_stars_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_allfixed_se_{task_id}.npy")
q_stars_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}q_stars_allfixed_se_{task_id}.npy")
cs_by_type_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_allfixed_se_{task_id}.npy")
cs_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}cs_allfixed_se_{task_id}.npy")
ps_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}ps_allfixed_se_{task_id}.npy")
ts_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}ts_allfixed_se_{task_id}.npy")
ccs_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_allfixed_se_{task_id}.npy")
ccs_per_bw_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_allfixed_se_{task_id}.npy")
avg_path_losses_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_allfixed_se_{task_id}.npy")
partial_Pif_partial_bf_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}partial_Pif_partial_bf_allfixed_se_{task_id}.npy")
partial_Piotherf_partial_bf_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}partial_Piotherf_partial_bf_allfixed_se_{task_id}.npy")
partial_diffPif_partial_bf_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}partial_diffPif_partial_bf_allfixed_se_{task_id}.npy")
partial_Pif_partial_b_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}partial_Pif_partial_b_allfixed_se_{task_id}.npy")
partial_CS_partial_b_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}partial_CS_partial_b_allfixed_se_{task_id}.npy")
p_stars_shortrun_se = lambda task_id: np.load(f"{paths.arrays_path}p_stars_shortrun_se_{task_id}.npy")
R_stars_shortrun_se = lambda task_id: np.load(f"{paths.arrays_path}R_stars_shortrun_se_{task_id}.npy")
num_stations_stars_shortrun_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_shortrun_se_{task_id}.npy")
num_stations_per_firm_stars_shortrun_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_shortrun_se_{task_id}.npy")
q_stars_shortrun_se = lambda task_id: np.load(f"{paths.arrays_path}q_stars_shortrun_se_{task_id}.npy")
cs_by_type_shortrun_se = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_shortrun_se_{task_id}.npy")
cs_shortrun_se = lambda task_id: np.load(f"{paths.arrays_path}cs_shortrun_se_{task_id}.npy")
ps_shortrun_se = lambda task_id: np.load(f"{paths.arrays_path}ps_shortrun_se_{task_id}.npy")
ts_shortrun_se = lambda task_id: np.load(f"{paths.arrays_path}ts_shortrun_se_{task_id}.npy")
ccs_shortrun_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_shortrun_se_{task_id}.npy")
ccs_per_bw_shortrun_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_shortrun_se_{task_id}.npy")
avg_path_losses_shortrun_se = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_shortrun_se_{task_id}.npy")
p_stars_free_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}p_stars_free_allfixed_se_{task_id}.npy")
R_stars_free_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}R_stars_free_allfixed_se_{task_id}.npy")
num_stations_stars_free_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_free_allfixed_se_{task_id}.npy")
num_stations_per_firm_stars_free_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_free_allfixed_se_{task_id}.npy")
q_stars_free_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}q_stars_free_allfixed_se_{task_id}.npy")
cs_by_type_free_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_free_allfixed_se_{task_id}.npy")
cs_free_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}cs_free_se_allfixed_{task_id}.npy")
ps_free_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}ps_free_se_allfixed_{task_id}.npy")
ts_free_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}ts_free_se_allfixed_{task_id}.npy")
ccs_free_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_free_allfixed_se_{task_id}.npy")
ccs_per_bw_free_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_free_allfixed_se_{task_id}.npy")
avg_path_losses_free_allfixed_se = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_free_allfixed_se_{task_id}.npy")
partial_Pif_partial_bf_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}partial_Pif_partial_bf_allbw_se_{task_id}.npy")
partial_Piotherf_partial_bf_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}partial_Piotherf_partial_bf_allbw_se_{task_id}.npy")
partial_diffPif_partial_bf_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}partial_diffPif_partial_bf_allbw_se_{task_id}.npy")
partial_Pif_partial_b_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}partial_Pif_partial_b_allbw_se_{task_id}.npy")
partial_CS_partial_b_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}partial_CS_partial_b_allbw_se_{task_id}.npy")
p_stars_free_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}p_stars_free_allbw_se_{task_id}.npy")
R_stars_free_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}R_stars_free_allbw_se_{task_id}.npy")
num_stations_stars_free_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_free_allbw_se_{task_id}.npy")
num_stations_per_firm_stars_free_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_free_allbw_se_{task_id}.npy")
q_stars_free_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}q_stars_free_allbw_se_{task_id}.npy")
cs_by_type_free_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_free_allbw_se_{task_id}.npy")
cs_free_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}cs_free_se_allbw_{task_id}.npy")
ps_free_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}ps_free_se_allbw_{task_id}.npy")
ts_free_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}ts_free_se_allbw_{task_id}.npy")
ccs_free_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_free_allbw_se_{task_id}.npy")
ccs_per_bw_free_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_free_allbw_se_{task_id}.npy")
avg_path_losses_free_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_free_allbw_se_{task_id}.npy")
p_stars_dens_se = lambda task_id: np.load(f"{paths.arrays_path}p_stars_dens_se_{task_id}.npy")
R_stars_dens_se = lambda task_id: np.load(f"{paths.arrays_path}R_stars_dens_se_{task_id}.npy")
num_stations_stars_dens_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_dens_se_{task_id}.npy")
num_stations_per_firm_stars_dens_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_dens_se_{task_id}.npy")
q_stars_dens_se = lambda task_id: np.load(f"{paths.arrays_path}q_stars_dens_se_{task_id}.npy")
cs_by_type_dens_se = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_dens_se_{task_id}.npy")
cs_dens_se = lambda task_id: np.load(f"{paths.arrays_path}cs_dens_se_{task_id}.npy")
ps_dens_se = lambda task_id: np.load(f"{paths.arrays_path}ps_dens_se_{task_id}.npy")
ts_dens_se = lambda task_id: np.load(f"{paths.arrays_path}ts_dens_se_{task_id}.npy")
ccs_dens_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_dens_se_{task_id}.npy")
ccs_per_bw_dens_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_dens_se_{task_id}.npy")
avg_path_losses_dens_se = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_dens_se_{task_id}.npy")
p_stars_bw_se = lambda task_id: np.load(f"{paths.arrays_path}p_stars_bw_se_{task_id}.npy")
R_stars_bw_se = lambda task_id: np.load(f"{paths.arrays_path}R_stars_bw_se_{task_id}.npy")
num_stations_stars_bw_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_bw_se_{task_id}.npy")
num_stations_per_firm_stars_bw_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_bw_se_{task_id}.npy")
q_stars_bw_se = lambda task_id: np.load(f"{paths.arrays_path}q_stars_bw_se_{task_id}.npy")
cs_by_type_bw_se = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_bw_se_{task_id}.npy")
cs_bw_se = lambda task_id: np.load(f"{paths.arrays_path}cs_bw_se_{task_id}.npy")
ps_bw_se = lambda task_id: np.load(f"{paths.arrays_path}ps_bw_se_{task_id}.npy")
ts_bw_se = lambda task_id: np.load(f"{paths.arrays_path}ts_bw_se_{task_id}.npy")
ccs_bw_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_bw_se_{task_id}.npy")
ccs_per_bw_bw_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_bw_se_{task_id}.npy")
avg_path_losses_bw_se = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_bw_se_{task_id}.npy")
p_stars_dens_1p_se = lambda task_id: np.load(f"{paths.arrays_path}p_stars_dens_1p_se_{task_id}.npy")
R_stars_dens_1p_se = lambda task_id: np.load(f"{paths.arrays_path}R_stars_dens_1p_se_{task_id}.npy")
num_stations_stars_dens_1p_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_dens_1p_se_{task_id}.npy")
num_stations_per_firm_stars_dens_1p_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_dens_1p_se_{task_id}.npy")
q_stars_dens_1p_se = lambda task_id: np.load(f"{paths.arrays_path}q_stars_dens_1p_se_{task_id}.npy")
cs_by_type_dens_1p_se = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_dens_1p_se_{task_id}.npy")
cs_dens_1p_se = lambda task_id: np.load(f"{paths.arrays_path}cs_dens_1p_se_{task_id}.npy")
ps_dens_1p_se = lambda task_id: np.load(f"{paths.arrays_path}ps_dens_1p_se_{task_id}.npy")
ts_dens_1p_se = lambda task_id: np.load(f"{paths.arrays_path}ts_dens_1p_se_{task_id}.npy")
ccs_dens_1p_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_dens_1p_se_{task_id}.npy")
ccs_per_bw_dens_1p_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_dens_1p_se_{task_id}.npy")
avg_path_losses_dens_1p_se = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_dens_1p_se_{task_id}.npy")
p_stars_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}p_stars_asymmetric_allbw_se_{task_id}.npy")
R_stars_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}R_stars_asymmetric_allbw_se_{task_id}.npy")
num_stations_stars_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_asymmetric_allbw_se_{task_id}.npy")
num_stations_per_firm_stars_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_asymmetric_allbw_se_{task_id}.npy")
q_stars_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}q_stars_asymmetric_allbw_se_{task_id}.npy")
cs_by_type_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_asymmetric_allbw_se_{task_id}.npy")
cs_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}cs_asymmetric_allbw_se_{task_id}.npy")
ps_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}ps_asymmetric_allbw_se_{task_id}.npy")
ts_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}ts_asymmetric_allbw_se_{task_id}.npy")
ccs_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_asymmetric_allbw_se_{task_id}.npy")
ccs_per_bw_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_asymmetric_allbw_se_{task_id}.npy")
avg_path_losses_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_asymmetric_allbw_se_{task_id}.npy")
avg_SINR_asymmetric_allbw_se = lambda task_id: np.load(f"{paths.arrays_path}avg_SINR_asymmetric_allbw_se_{task_id}.npy")
p_stars_shortrunall_se = lambda task_id: np.load(f"{paths.arrays_path}p_stars_shortrunall_se_{task_id}.npy")
R_stars_shortrunall_se = lambda task_id: np.load(f"{paths.arrays_path}R_stars_shortrunall_se_{task_id}.npy")
num_stations_stars_shortrunall_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_shortrunall_se_{task_id}.npy")
num_stations_per_firm_stars_shortrunall_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_shortrunall_se_{task_id}.npy")
q_stars_shortrunall_se = lambda task_id: np.load(f"{paths.arrays_path}q_stars_shortrunall_se_{task_id}.npy")
cs_by_type_shortrunall_se = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_shortrunall_se_{task_id}.npy")
cs_shortrunall_se = lambda task_id: np.load(f"{paths.arrays_path}cs_shortrunall_se_{task_id}.npy")
ps_shortrunall_se = lambda task_id: np.load(f"{paths.arrays_path}ps_shortrunall_se_{task_id}.npy")
ts_shortrunall_se = lambda task_id: np.load(f"{paths.arrays_path}ts_shortrunall_se_{task_id}.npy")
ccs_shortrunall_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_shortrunall_se_{task_id}.npy")
ccs_per_bw_shortrunall_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_shortrunall_se_{task_id}.npy")
avg_path_losses_shortrunall_se = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_shortrunall_se_{task_id}.npy")
p_stars_longrunall_se = lambda task_id: np.load(f"{paths.arrays_path}p_stars_longrunall_se_{task_id}.npy")
R_stars_longrunall_se = lambda task_id: np.load(f"{paths.arrays_path}R_stars_longrunall_se_{task_id}.npy")
num_stations_stars_longrunall_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_stars_longrunall_se_{task_id}.npy")
num_stations_per_firm_stars_longrunall_se = lambda task_id: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_longrunall_se_{task_id}.npy")
q_stars_longrunall_se = lambda task_id: np.load(f"{paths.arrays_path}q_stars_longrunall_se_{task_id}.npy")
cs_by_type_longrunall_se = lambda task_id: np.load(f"{paths.arrays_path}cs_by_type_longrunall_se_{task_id}.npy")
cs_longrunall_se = lambda task_id: np.load(f"{paths.arrays_path}cs_longrunall_se_{task_id}.npy")
ps_longrunall_se = lambda task_id: np.load(f"{paths.arrays_path}ps_longrunall_se_{task_id}.npy")
ts_longrunall_se = lambda task_id: np.load(f"{paths.arrays_path}ts_longrunall_se_{task_id}.npy")
ccs_longrunall_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_longrunall_se_{task_id}.npy")
ccs_per_bw_longrunall_se = lambda task_id: np.load(f"{paths.arrays_path}ccs_per_bw_longrunall_se_{task_id}.npy")
avg_path_losses_longrunall_se = lambda task_id: np.load(f"{paths.arrays_path}avg_path_losses_longrunall_se_{task_id}.npy")
per_user_costs_se = lambda task_id: np.load(f"{paths.arrays_path}per_user_costs_se_{task_id}.npy")

densities = lambda task_id: np.load(f"{paths.arrays_path}cntrfctl_densities_{task_id}.npy")
densities_pops = lambda task_id: np.load(f"{paths.arrays_path}cntrfctl_densities_pop_{task_id}.npy")
bw_vals = lambda task_id: np.load(f"{paths.arrays_path}cntrfctl_bw_vals_{task_id}.npy")

bw_by_firm = np.load(f"{paths.arrays_path}cntrfctl_bw_vals_by_firm.npy")
list_MNOwoMVNO = np.load(f"{paths.arrays_path}cntrfctl_firms.npy")

# %%
# Define common graph features
num_firms_to_simulate = 6
num_firms_to_simulate_extend = 9
num_firms_array = np.arange(num_firms_to_simulate, dtype=int) + 1
num_firms_array_extend = np.arange(num_firms_to_simulate_extend, dtype=int) + 1
elast_ids = np.arange(avg_price_elasts.shape[0])
nest_ids = np.arange(div_ratios.shape[0])
elast_ids_sparse = np.copy(elast_ids)
nest_ids_sparse = np.copy(nest_ids)
default_task_id = 0
default_num_firm_idx = 3
alpha = 0.6
lw = 3.

invalid_return = " "
    
num_digits_round = 3
def round_var(var, round_dig, stderrs=False):
    if np.isnan(var):
        return invalid_return
    rounded_var = round(var, round_dig)
    if rounded_var == 0.0:
        format_var = "<0." + "0" * (round_dig - 1) + "1"
    else:
        format_var = "{0:,.3f}".format(rounded_var).replace(",", "\\,")
    if stderrs:
        format_var = f"({format_var})"
    return format_var

def create_file(file_name, file_contents):
    """Create file with name file_name and content file_contents"""
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()
    
# %%
# Spectral efficiencies
spectral_efficiencies_params = np.load(f"{paths.arrays_path}spectral_efficiencies.npy")
populations = np.load(f"{paths.arrays_path}populations.npy")

if save_:
    create_file(f"{paths.stats_path}mean_spectral_efficiency.tex", f"{np.mean(spectral_efficiencies_params):.3f}")
    create_file(f"{paths.stats_path}std_spectral_efficiency.tex", f"{np.std(spectral_efficiencies_params):.3f}")
    create_file(f"{paths.stats_path}weighted_avg_spectral_efficiency.tex", f"{np.average(spectral_efficiencies_params, weights=populations):.3f}")
    
# %%
# Per-user estimated costs

fig, ax = plt.subplots(1, 1, figsize=(13,5))

x_pos = [i for i, _ in enumerate(ds.firms)]
barlist = ax.bar(x_pos, c_u(default_task_id), yerr=1.96 * c_u_se(default_task_id), capsize=4.0)
ax.set_ylabel("$\hat{c}_{u}$ (in \u20ac)", fontsize=15)
ax.set_xticks(x_pos)

dlimidx = ds.chars.index(ds.dlimname)
vlimidx = ds.chars.index(ds.vunlimitedname)
ax.set_xticklabels(["$\\bar{d} = " + str(int(dlim)) + "$, $v = " + str(int(ds.data[0,i,vlimidx])) + "$" for i, dlim in enumerate(ds.data[0,:,dlimidx])], rotation=60, ha="right")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

mnos = ["Orange", "SFR", "Free", "Bouygues", "MVNO"]

for i, firm in enumerate(np.unique(ds.firms)):
    mno_name = mnos[i]
    x_loc = np.mean(np.arange(ds.J)[ds.firms == firm])
    y_loc = 0.85 * np.max(c_u(default_task_id))
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
    ax.text(x_loc, y_loc, mno_name, ha="center", fontsize=15, color=color)

for i in range(ds.J):
    barlist[i].set_color(plt.rcParams['axes.prop_cycle'].by_key()['color'][ds.firms[i]-1])
    barlist[i].set_alpha(0.7)
    
plt.tight_layout()
    
if save_:
    plt.savefig(f"{paths.graphs_path}c_u_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Per-base station estimated costs

fig, axs = plt.subplots(2, 2, figsize=(13,8), sharex=True)

for i, mno in enumerate(mnos[:-1]):
    row = i // 2
    col = i % 2
    axs[row,col].hist(c_R(default_task_id)[:,i] * 75.0, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], alpha=0.7)
    axs[row,col].set_xlabel("$\hat{c}_{fm}$ (in \u20ac)" if row == 1 else "", fontsize=12)
    axs[row,col].set_title(mno, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], fontsize=20, y=0.85)
    axs[row,col].axvline(x=np.mean(c_R(default_task_id)[:,i]) * 75.0, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], linestyle="--", linewidth=2.)
    axs[row,col].spines['top'].set_visible(False)
    axs[row,col].spines['right'].set_visible(False)
    axs[row,col].spines['left'].set_visible(False)
    axs[row,col].get_yaxis().set_visible(False)
    
plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}c_R_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Costs table

dlimidx = ds.chars.index(ds.dlimname)
dlims = ds.data[0,:,dlimidx]

# Per-user costs

def process_c_u(c_u):
    c_u_small = np.mean(c_u[dlims < 1000.0])
    c_u_medium = np.mean(c_u[(dlims >= 1000.0) & (dlims < 5000.0)])
    c_u_big = np.mean(c_u[dlims >= 5000.0])
    return np.array([c_u_small, c_u_medium, c_u_big])

to_tex = "\\begin{tabular}{l c c c c c c c c} \\hline \n"
to_tex += "\\textit{Per-user costs} & & & & $\\bar{d} < 1\\,000$ & & $1\\,000 \\leq \\bar{d} < 5\\,000$ & & $\\bar{d} \\geq 5\\,000$ \\\\ \n"
to_tex += "$\\qquad \\hat{c}_{j}^{u}$ & & & & (in \\euro{}) & & (in \\euro{}) & & (in \\euro{}) \\\\ \n "
to_tex += "\\cline{5-5} \\cline{7-7} \\cline{9-9}"
to_tex +=  " & & & & " + " & & ".join(f"${c_u:.2f}$ " for c_u in process_c_u(c_u(default_task_id)))
to_tex += " \\\\ \n"
to_tex += " & & & & (" + ") & & (".join(f"${c_u:.2f}$" for c_u in process_c_u(c_u_se(default_task_id))) + ") \\\\ \n"
to_tex += " & & & & & & & & \\\\ \n"
    
if save_:
    c_u_prefspec = process_c_u(c_u(default_task_id))
    create_file(f"{paths.stats_path}c_u_small.tex", f"{c_u_prefspec[0]:.2f}")
    create_file(f"{paths.stats_path}c_u_med.tex", f"{c_u_prefspec[1]:.2f}")
    create_file(f"{paths.stats_path}c_u_large.tex", f"{c_u_prefspec[2]:.2f}")
    
# Per-base station costs
    
def process_c_R(c_R):
    return 201.0 * np.mean(c_R * 75.0, axis=0) # 201 to convert from monthly to perpetuity, 75 MHz

def process_c_R_sd(c_R):
    return np.std(201.0 * c_R * 75.0, axis=0) # 201 to convert from monthly to perpetuity, 75 MHz

to_tex += "\\textit{Per-base station costs} & & Orange & & SFR & & Free & & Bouygues \\\\ \n"
to_tex += "$\\qquad \\hat{C}_{f}$ & & (in \\euro{}) & & (in \\euro{}) & & (in \\euro{}) & & (in \\euro{}) \\\\ \n"
to_tex += "\\cline{3-3} \\cline{5-5} \\cline{7-7} \\cline{9-9}"
to_tex += " & & " + " & & ".join(f"${c_R:,.0f}$ ".replace(",","\\,") for c_R in process_c_R(c_R(default_task_id))) + " \\\\ \n"
to_tex += " & & (" + ") & & (".join(f"${c_R:,.0f}$".replace(",","\\,") for c_R in process_c_R_sd(c_R(default_task_id))) + ") \\\\ \n"
to_tex += "\\hline \n"
to_tex += "\\end{tabular} \n"
if save_:
    create_file(f"{paths.tables_path}costs_estimates_table.tex", to_tex)
if print_:
    print(to_tex)
    
if save_:
    c_R_prefspec = process_c_R(c_R(default_task_id))
    create_file(f"{paths.stats_path}c_R_ORG.tex", f"{np.round(c_R_prefspec[0], -3):,.0f}".replace(",","\\,"))
    create_file(f"{paths.stats_path}c_R_SFR.tex", f"{np.round(c_R_prefspec[1], -3):,.0f}".replace(",","\\,"))
    create_file(f"{paths.stats_path}c_R_FREE.tex", f"{np.round(c_R_prefspec[2], -3):,.0f}".replace(",","\\,"))
    create_file(f"{paths.stats_path}c_R_BYG.tex", f"{np.round(c_R_prefspec[3], -3):,.0f}".replace(",","\\,"))

    c_R_prefspec_std = process_c_R_sd(c_R(default_task_id))
    create_file(f"{paths.stats_path}c_R_std_ORG.tex", f"{np.round(c_R_prefspec_std[0], -3):,.0f}".replace(",","\\,"))
    create_file(f"{paths.stats_path}c_R_std_SFR.tex", f"{np.round(c_R_prefspec_std[1], -3):,.0f}".replace(",","\\,"))
    create_file(f"{paths.stats_path}c_R_std_FREE.tex", f"{np.round(c_R_prefspec_std[2], -3):,.0f}".replace(",","\\,"))
    create_file(f"{paths.stats_path}c_R_std_BYG.tex", f"{np.round(c_R_prefspec_std[3], -3):,.0f}".replace(",","\\,"))
    
# %%
# Values used in counterfactuals

print("Values used in counterfactuals")
print(f"c_R (per unit of bw): {np.mean(c_R(default_task_id)[:,np.array([True,True,False,True])]) * 201.0}")
print(f"1 GB c_u: {np.mean(c_u(default_task_id)[ds.data[0,:,dlimidx] < 5000.0])}")
print(f"10 GB c_u: {np.mean(c_u(default_task_id)[ds.data[0,:,dlimidx] >= 5000.0])}")

if save_:
    create_file(f"{paths.stats_path}per_user_cost_lowdlim.tex", f"{per_user_costs(default_task_id)[0]:.2f}")
    create_file(f"{paths.stats_path}per_user_cost_highdlim.tex", f"{per_user_costs(default_task_id)[1]:.2f}")
    
# %%
# Endogenous variables - number of firms

fig, axs = plt.subplots(2, 3, figsize=(12.0, 6.5), squeeze=False)

x_fontsize = "x-large"
y_fontsize = "x-large"
title_fontsize = "xx-large"

# dlim = 1,000 prices
axs[0,0].plot(num_firms_array, p_stars(default_task_id)[:,0], color="black", lw=lw, alpha=alpha)
axs[0,0].plot(num_firms_array, p_stars(default_task_id)[:,0] + 1.96 * p_stars_se(default_task_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].plot(num_firms_array, p_stars(default_task_id)[:,0] - 1.96 * p_stars_se(default_task_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
axs[0,1].plot(num_firms_array, p_stars(default_task_id)[:,1], color="black", lw=lw, alpha=alpha)
axs[0,1].plot(num_firms_array, p_stars(default_task_id)[:,1] + 1.96 * p_stars_se(default_task_id)[:,1], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].plot(num_firms_array, p_stars(default_task_id)[:,1] - 1.96 * p_stars_se(default_task_id)[:,1], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

# radius
axs[0,2].plot(num_firms_array, num_stations_per_firm_stars(default_task_id) * 1000.0, color="black", lw=lw, alpha=alpha)
axs[0,2].plot(num_firms_array, num_stations_per_firm_stars(default_task_id) * 1000.0 + 1.96 * num_stations_per_firm_stars_se(default_task_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].plot(num_firms_array, num_stations_per_firm_stars(default_task_id) * 1000.0 - 1.96 * num_stations_per_firm_stars_se(default_task_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[0,2].set_title("number of stations / firm", fontsize=title_fontsize)

# total number of stations
axs[1,0].plot(num_firms_array, num_stations_stars(default_task_id) * 1000.0, color="black", lw=lw, alpha=alpha)
axs[1,0].plot(num_firms_array, num_stations_stars(default_task_id) * 1000.0 + 1.96 * num_stations_stars_se(default_task_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,0].plot(num_firms_array, num_stations_stars(default_task_id) * 1000.0 - 1.96 * num_stations_stars_se(default_task_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,0].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[1,0].set_title("total number of stations", fontsize=title_fontsize)

# path loss
# axs[1,1].plot(num_firms_array, avg_path_losses(default_task_id), color="black", lw=lw, alpha=alpha)
# axs[1,1].plot(num_firms_array, avg_path_losses(default_task_id) + 1.96 * avg_path_losses_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
# axs[1,1].plot(num_firms_array, avg_path_losses(default_task_id) - 1.96 * avg_path_losses_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
# axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
# axs[1,1].set_ylabel("dB", fontsize=y_fontsize)
# axs[1,1].set_title("average path loss", fontsize=title_fontsize)
axs[1,1].plot(num_firms_array, ccs_per_bw(default_task_id), color="black", lw=lw, alpha=alpha)
axs[1,1].plot(num_firms_array, ccs_per_bw(default_task_id) + 1.96 * ccs_per_bw_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,1].plot(num_firms_array, ccs_per_bw(default_task_id) - 1.96 * ccs_per_bw_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,1].set_ylabel("Mbps / MHz", fontsize=y_fontsize)
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)
axs[1,1].ticklabel_format(useOffset=False)

# download speeds
axs[1,2].plot(num_firms_array, q_stars(default_task_id), color="black", lw=lw, alpha=alpha, label="download speed")
axs[1,2].plot(num_firms_array, q_stars(default_task_id) + 1.96 * q_stars_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,2].plot(num_firms_array, q_stars(default_task_id) - 1.96 * q_stars_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,2].plot(num_firms_array, ccs(default_task_id), color="black", lw=lw, alpha=0.9, ls=(0, (3, 1, 1, 1)), label="channel capacity")
# axs[1,2].plot(num_firms_array, ccs(default_task_id) + 1.96 * ccs_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls=(0, (3, 1, 1, 1)))
# axs[1,2].plot(num_firms_array, ccs(default_task_id) - 1.96 * ccs_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls=(0, (3, 1, 1, 1)))
axs[1,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,2].set_ylabel("$Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.nanmin(p_stars(default_task_id)[1:,:]) - 2.0
max_y_p = np.nanmax(p_stars(default_task_id)[1:,:]) + 5.0
min_y_num_stations_per_firm = np.nanmin(num_stations_per_firm_stars(default_task_id) * 1000.0)
max_y_num_stations_per_firm = np.nanmax(num_stations_per_firm_stars(default_task_id) * 1000.0)
min_y_num_stations = np.nanmin(num_stations_stars(default_task_id)[:] * 1000.0)
max_y_num_stations = np.nanmax(num_stations_stars(default_task_id)[:] * 1000.0)
# min_y_pl = np.nanmin(avg_path_losses(default_task_id)[:]) - 2.
# max_y_pl = np.nanmax(avg_path_losses(default_task_id)[:]) + 2.
min_y_q = np.minimum(np.nanmin(q_stars(default_task_id)[:]), np.nanmin(ccs(default_task_id)[1:]))
max_y_q = np.maximum(np.nanmax(q_stars(default_task_id)[:]), np.nanmax(ccs(default_task_id)[1:]))
diff_p = max_y_p - min_y_p
diff_num_stations_per_firm = max_y_num_stations_per_firm - min_y_num_stations_per_firm
diff_num_stations = max_y_num_stations - min_y_num_stations
# diff_pl = max_y_pl - min_y_pl
diff_q = max_y_q - min_y_q
margin = 0.1
for i in range(2):
    axs[0,i].set_ylim((min_y_p - margin * diff_p, max_y_p + margin * diff_p))
# axs[0,2].set_ylim((min_y_num_stations_per_firm - margin * diff_num_stations_per_firm, max_y_num_stations_per_firm + margin * diff_num_stations_per_firm))
# axs[1,0].set_ylim((min_y_num_stations - margin * diff_num_stations, max_y_num_stations + margin * diff_num_stations))
# axs[1,1].set_ylim((min_y_pl - margin * diff_pl, max_y_pl + margin * diff_pl))
axs[1,2].set_ylim((min_y_q - margin * diff_q, max_y_q + margin * diff_q))
for i in range(2):
    for j in range(3):
        axs[i,j].set_xticks(num_firms_array)
        
axs[1,2].legend(loc="best")

plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_variables_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Endogenous variables - number of firms - all fixed cost

fig, axs = plt.subplots(2, 3, figsize=(12.0, 6.5), squeeze=False)

x_fontsize = "x-large"
y_fontsize = "x-large"
title_fontsize = "xx-large"

# dlim = 1,000 prices
axs[0,0].plot(num_firms_array, p_stars_allfixed(default_task_id)[:,0], color="black", lw=lw, alpha=alpha)
axs[0,0].plot(num_firms_array, p_stars_allfixed(default_task_id)[:,0] + 1.96 * p_stars_allfixed_se(default_task_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].plot(num_firms_array, p_stars_allfixed(default_task_id)[:,0] - 1.96 * p_stars_allfixed_se(default_task_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
axs[0,1].plot(num_firms_array, p_stars_allfixed(default_task_id)[:,1], color="black", lw=lw, alpha=alpha)
axs[0,1].plot(num_firms_array, p_stars_allfixed(default_task_id)[:,1] + 1.96 * p_stars_allfixed_se(default_task_id)[:,1], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].plot(num_firms_array, p_stars_allfixed(default_task_id)[:,1] - 1.96 * p_stars_allfixed_se(default_task_id)[:,1], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

# radius
axs[0,2].plot(num_firms_array, num_stations_per_firm_stars_allfixed(default_task_id) * 1000.0, color="black", lw=lw, alpha=alpha)
axs[0,2].plot(num_firms_array, num_stations_per_firm_stars_allfixed(default_task_id) * 1000.0 + 1.96 * num_stations_per_firm_stars_allfixed_se(default_task_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].plot(num_firms_array, num_stations_per_firm_stars_allfixed(default_task_id) * 1000.0 - 1.96 * num_stations_per_firm_stars_allfixed_se(default_task_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[0,2].set_title("number of stations / firm", fontsize=title_fontsize)

# total number of stations
axs[1,0].plot(num_firms_array, num_stations_stars_allfixed(default_task_id) * 1000.0, color="black", lw=lw, alpha=alpha)
axs[1,0].plot(num_firms_array, num_stations_stars_allfixed(default_task_id) * 1000.0 + 1.96 * num_stations_stars_allfixed_se(default_task_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,0].plot(num_firms_array, num_stations_stars_allfixed(default_task_id) * 1000.0 - 1.96 * num_stations_stars_allfixed_se(default_task_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,0].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[1,0].set_title("total number of stations", fontsize=title_fontsize)

# path loss
# axs[1,1].plot(num_firms_array, avg_path_losses_allfixed(default_task_id), color="black", lw=lw, alpha=alpha)
# axs[1,1].plot(num_firms_array, avg_path_losses_allfixed(default_task_id) + 1.96 * avg_path_losses_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
# axs[1,1].plot(num_firms_array, avg_path_losses_allfixed(default_task_id) - 1.96 * avg_path_losses_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
# axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
# axs[1,1].set_ylabel("dB", fontsize=y_fontsize)
# axs[1,1].set_title("average path loss", fontsize=title_fontsize)
axs[1,1].plot(num_firms_array, ccs_per_bw_allfixed(default_task_id), color="black", lw=lw, alpha=alpha)
axs[1,1].plot(num_firms_array, ccs_per_bw_allfixed(default_task_id) + 1.96 * ccs_per_bw_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,1].plot(num_firms_array, ccs_per_bw_allfixed(default_task_id) - 1.96 * ccs_per_bw_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,1].set_ylabel("Mbps / MHz", fontsize=y_fontsize)
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)
axs[1,1].ticklabel_format(useOffset=False)

# download speeds
axs[1,2].plot(num_firms_array, q_stars_allfixed(default_task_id), color="black", lw=lw, alpha=alpha, label="download speed")
axs[1,2].plot(num_firms_array, q_stars_allfixed(default_task_id) + 1.96 * q_stars_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,2].plot(num_firms_array, q_stars_allfixed(default_task_id) - 1.96 * q_stars_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,2].plot(num_firms_array, ccs_allfixed(default_task_id), color="black", lw=lw, alpha=0.9, ls=(0, (3, 1, 1, 1)), label="channel capacity")
# axs[1,2].plot(num_firms_array, ccs_allfixed(default_task_id) + 1.96 * ccs_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls=(0, (3, 1, 1, 1)))
# axs[1,2].plot(num_firms_array, ccs_allfixed(default_task_id) - 1.96 * ccs_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls=(0, (3, 1, 1, 1)))
axs[1,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,2].set_ylabel("$Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.nanmin(p_stars_allfixed(default_task_id)[1:,:]) - 2.0
max_y_p = np.nanmax(p_stars_allfixed(default_task_id)[1:,:]) + 5.0
min_y_num_stations_per_firm = np.nanmin(num_stations_per_firm_stars_allfixed(default_task_id) * 1000.0)
max_y_num_stations_per_firm = np.nanmax(num_stations_per_firm_stars_allfixed(default_task_id) * 1000.0)
min_y_num_stations = np.nanmin(num_stations_stars_allfixed(default_task_id)[:] * 1000.0)
max_y_num_stations = np.nanmax(num_stations_stars_allfixed(default_task_id)[:] * 1000.0)
# min_y_pl = np.nanmin(avg_path_losses(default_task_id)[:]) - 2.
# max_y_pl = np.nanmax(avg_path_losses(default_task_id)[:]) + 2.
min_y_q = np.minimum(np.nanmin(q_stars_allfixed(default_task_id)[:]), np.nanmin(ccs_allfixed(default_task_id)[1:]))
max_y_q = np.maximum(np.nanmax(q_stars_allfixed(default_task_id)[:]), np.nanmax(ccs_allfixed(default_task_id)[1:]))
diff_p = max_y_p - min_y_p
diff_num_stations_per_firm = max_y_num_stations_per_firm - min_y_num_stations_per_firm
diff_num_stations = max_y_num_stations - min_y_num_stations
# diff_pl = max_y_pl - min_y_pl
diff_q = max_y_q - min_y_q
margin = 0.1
for i in range(2):
    axs[0,i].set_ylim((min_y_p - margin * diff_p, max_y_p + margin * diff_p))
# axs[0,2].set_ylim((min_y_num_stations_per_firm - margin * diff_num_stations_per_firm, max_y_num_stations_per_firm + margin * diff_num_stations_per_firm))
# axs[1,0].set_ylim((min_y_num_stations - margin * diff_num_stations, max_y_num_stations + margin * diff_num_stations))
# axs[1,1].set_ylim((min_y_pl - margin * diff_pl, max_y_pl + margin * diff_pl))
axs[1,2].set_ylim((min_y_q - margin * diff_q, max_y_q + margin * diff_q))
for i in range(2):
    for j in range(3):
        axs[i,j].set_xticks(num_firms_array)
        
axs[1,2].legend(loc="best")

plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_variables_1gb10gb_allfixed.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Elasticities

fig, axs = plt.subplots(1, 2, figsize=(7.0, 4.0), sharex=True, squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"
    
# dlim = 1,000 elasticities
axs[0,0].plot(num_firms_array, partial_elasts(default_task_id)[:,0], lw=lw, alpha=alpha, color="black", ls="--", label="partial")
axs[0,0].plot(num_firms_array, full_elasts(default_task_id)[:,0], lw=lw, alpha=alpha, color="black", label="full")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].legend(loc="upper right")
axs[0,0].set_title("1$\,$000 MB plan", fontsize=title_fontsize)

# dlim = 10,000 elasticities
axs[0,1].plot(num_firms_array, partial_elasts(default_task_id)[:,1], lw=lw, alpha=alpha, color="black", ls="--", label="partial")
axs[0,1].plot(num_firms_array, full_elasts(default_task_id)[:,1], lw=lw, alpha=alpha, color="black", label="full")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].legend(loc="upper right")
axs[0,1].set_title("10$\,$000 MB plan", fontsize=title_fontsize)

# Set axis limits
min_y = np.nanmin(np.concatenate(tuple([full_elasts(default_task_id)] + [partial_elasts(default_task_id)]))) - 0.3
max_y = np.nanmax(np.concatenate(tuple([full_elasts(default_task_id) for elast_id in elast_ids] + [partial_elasts(default_task_id)]))) + 0.3
for i in range(2): # all columns
    axs[0,i].set_ylim((min_y, max_y))
    axs[0,i].set_xticks(num_firms_array)

plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_elasticities_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Bandwidth derivatives (all fixed)

fig, axs = plt.subplots(1, 3, figsize=(9.0,3.5), sharex=True, squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "xx-large"
title_pad = 15.0

# partial_Pif_partial_bf
axs[0,0].plot(num_firms_array, partial_diffPif_partial_bf_allfixed(default_task_id), color="black", lw=lw, alpha=alpha)
axs[0,0].plot(num_firms_array, partial_diffPif_partial_bf_allfixed(default_task_id) + 1.96 * partial_diffPif_partial_bf_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].plot(num_firms_array, partial_diffPif_partial_bf_allfixed(default_task_id) - 1.96 * partial_diffPif_partial_bf_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("\u20ac / person / MHz", fontsize=y_fontsize)
axs[0,0].set_title("$d \\Pi_{f} / d B_{f} - d \\Pi_{f} / d B_{f^{\\prime}}$", fontsize=title_fontsize, pad=title_pad)

# partial_Pif_partial_b
axs[0,1].plot(num_firms_array, partial_Pif_partial_b_allfixed(default_task_id), color="black", lw=lw, alpha=alpha)
axs[0,1].plot(num_firms_array, partial_Pif_partial_b_allfixed(default_task_id) + 1.96 * partial_Pif_partial_b_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].plot(num_firms_array, partial_Pif_partial_b_allfixed(default_task_id) - 1.96 * partial_Pif_partial_b_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("\u20ac / person / MHz", fontsize=y_fontsize)
axs[0,1].set_title("$d \\Pi_{f} / d B$", fontsize=title_fontsize, pad=title_pad)

# partial_CS_partial_b
axs[0,2].plot(num_firms_array, partial_CS_partial_b_allfixed(default_task_id), color="black", lw=lw, alpha=alpha)
axs[0,2].plot(num_firms_array, partial_CS_partial_b_allfixed(default_task_id) + 1.96 * partial_CS_partial_b_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].plot(num_firms_array, partial_CS_partial_b_allfixed(default_task_id) - 1.96 * partial_CS_partial_b_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("\u20ac / person / MHz", fontsize=y_fontsize)
axs[0,2].set_title("$d CS / d B$", fontsize=title_fontsize, pad=title_pad)

# Set axis limits
min_y_Pif_bf = np.nanmin(partial_diffPif_partial_bf_allfixed(default_task_id)) - 0.006
max_y_Pif_bf = np.nanmax(partial_diffPif_partial_bf_allfixed(default_task_id)) + 0.006
min_y_Pif_b = np.nanmin(partial_Pif_partial_b_allfixed(default_task_id)) - 0.001
max_y_Pif_b = np.nanmax(partial_Pif_partial_b_allfixed(default_task_id)) + 0.0005
min_y_CS_b = np.nanmin(partial_CS_partial_b_allfixed(default_task_id)) - 0.01
max_y_CS_b = np.nanmax(partial_CS_partial_b_allfixed(default_task_id)) + 0.02
axs[0,0].set_ylim((min_y_Pif_bf, max_y_Pif_bf))
axs[0,1].set_ylim((min_y_Pif_b, max_y_Pif_b))
axs[0,2].set_ylim((min_y_CS_b, max_y_CS_b))
for i in range(3):
    axs[0,i].set_xticks(num_firms_array)
        
plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_bw_deriv_allfixed_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
if save_:
    create_file(f"{paths.stats_path}auction_val_allfixed.tex", f"{partial_diffPif_partial_bf_allfixed(default_task_id)[3] * 201.0:.2f}")
    
# %%
# Bandwidth derivatives (all bw)

fig, axs = plt.subplots(1, 3, figsize=(9.0,3.5), sharex=True, squeeze=False)

# partial_Pif_partial_bf
axs[0,0].plot(num_firms_array, partial_diffPif_partial_bf_allbw(default_task_id), color="black", lw=lw, alpha=alpha)
axs[0,0].plot(num_firms_array, partial_diffPif_partial_bf_allbw(default_task_id) + 1.96 * partial_diffPif_partial_bf_allbw_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].plot(num_firms_array, partial_diffPif_partial_bf_allbw(default_task_id) - 1.96 * partial_diffPif_partial_bf_allbw_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("\u20ac / person / MHz", fontsize=y_fontsize)
axs[0,0].set_title("$d \\Pi_{f} / d B_{f} - d \\Pi_{f} / d B_{f^{\\prime}}$", fontsize=title_fontsize, pad=title_pad)

# partial_Pif_partial_b
axs[0,1].plot(num_firms_array, partial_Pif_partial_b_allbw(default_task_id), color="black", lw=lw, alpha=alpha)
axs[0,1].plot(num_firms_array, partial_Pif_partial_b_allbw(default_task_id) + 1.96 * partial_Pif_partial_b_allbw_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].plot(num_firms_array, partial_Pif_partial_b_allbw(default_task_id) - 1.96 * partial_Pif_partial_b_allbw_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("\u20ac / person / MHz", fontsize=y_fontsize)
axs[0,1].set_title("$d \\Pi_{f} / d B$", fontsize=title_fontsize, pad=title_pad)

# partial_CS_partial_b
axs[0,2].plot(num_firms_array, partial_CS_partial_b_allbw(default_task_id), color="black", lw=lw, alpha=alpha)
axs[0,2].plot(num_firms_array, partial_CS_partial_b_allbw(default_task_id) + 1.96 * partial_CS_partial_b_allbw_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].plot(num_firms_array, partial_CS_partial_b_allbw(default_task_id) - 1.96 * partial_CS_partial_b_allbw_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("\u20ac / person / MHz", fontsize=y_fontsize)
axs[0,2].set_title("$d CS / d B$", fontsize=title_fontsize, pad=title_pad)

# Set axis limits
min_y_Pif_bf = np.nanmin(partial_diffPif_partial_bf_allbw(default_task_id)) - 0.002
max_y_Pif_bf = np.nanmax(partial_diffPif_partial_bf_allbw(default_task_id)) + 0.003
min_y_Pif_b = np.nanmin(partial_Pif_partial_b_allbw(default_task_id)) - 0.001
max_y_Pif_b = np.nanmax(partial_Pif_partial_b_allbw(default_task_id)) + 0.001
min_y_CS_b = np.nanmin(partial_CS_partial_b_allbw(default_task_id)) - 0.002
max_y_CS_b = np.nanmax(partial_CS_partial_b_allbw(default_task_id)) + 0.01
axs[0,0].set_ylim((min_y_Pif_bf, max_y_Pif_bf))
axs[0,1].set_ylim((min_y_Pif_b, max_y_Pif_b))
axs[0,2].set_ylim((min_y_CS_b, max_y_CS_b))
for i in range(3):
    axs[0,i].set_xticks(num_firms_array)
        
plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_bw_deriv_allbw_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
if save_:
    create_file(f"{paths.stats_path}auction_val_allbw.tex", f"{partial_diffPif_partial_bf_allbw(default_task_id)[3] * 201.0:.2f}")
    
def int_to_en(num):
    """Given an int32 number, print it in English. Taken from https://stackoverflow.com/questions/8982163/how-do-i-tell-python-to-convert-integers-into-words"""
    d = { 0 : "zero", 1 : "one", 2 : "two", 3 : "three", 4 : "four", 5 : "five",
          6 : "six", 7 : "seven", 8 : "eight", 9 : "nine", 10 : "ten",
          11 : "eleven", 12 : "twelve", 13 : "thirteen", 14 : "fourteen",
          15 : "fifteen", 16 : "sixteen", 17 : "seventeen", 18 : "eighteen",
          19 : "nineteen", 20 : "twenty",
          30 : "thirty", 40 : "forty", 50 : "fifty", 60 : "sixty",
          70 : "seventy", 80 : "eighty", 90 : "ninety" }
    k = 1000
    m = k * 1000

    assert(0 <= num)

    if (num < 20):
        return d[num]

    if (num < 100):
        if num % 10 == 0: return d[num]
        else: return d[num // 10 * 10] + "-" + d[num % 10]

    if (num < k):
        if num % 100 == 0: return d[num // 100] + " hundred"
        else: return d[num // 100] + " hundred and " + int_to_en(num % 100)

    raise AssertionError("num is too large: %s" % str(num))

ratio_CS_to_Pif = int_to_en(np.round(partial_CS_partial_b_allbw(default_task_id)[3] / partial_diffPif_partial_bf_allbw(default_task_id)[3], 0).astype(int))
if print_:
    print(ratio_CS_to_Pif)

if save_:
    create_file(f"{paths.stats_path}ratio_CS_to_Pif.tex", ratio_CS_to_Pif)
    create_file(f"{paths.stats_path}CS_present_discounted.tex", f"{(partial_CS_partial_b_allbw(default_task_id)[3] * 201.0):.2f}")
    
ratio_CS_to_Pif_allfixed = int_to_en(np.round(partial_CS_partial_b_allfixed(default_task_id)[3] / partial_diffPif_partial_bf_allfixed(default_task_id)[3], 0).astype(int))
if print_:
    print(ratio_CS_to_Pif_allfixed)

if save_:
    create_file(f"{paths.stats_path}ratio_CS_to_Pif_allfixed.tex", ratio_CS_to_Pif_allfixed)
    
# %%
# Welfare for number of firms

fig, axs = plt.subplots(1, 3, figsize=(9.0,3.25), sharex=True, squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

# consumer surplus
axs[0,0].plot(num_firms_array_extend, cs(default_task_id), color="black", lw=lw, alpha=alpha)
#axs[0,0].plot(num_firms_array_extend, cs(default_task_id) + 1.96 * cs_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[0,0].plot(num_firms_array_extend, cs(default_task_id) - 1.96 * cs_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].axvline(x=num_firms_array_extend[np.nanargmax(cs(default_task_id))], color="black", linestyle="--", alpha=0.25)
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
axs[0,1].plot(num_firms_array_extend, ps(default_task_id), color="black", lw=lw, alpha=alpha)
#axs[0,1].plot(num_firms_array_extend, ps(default_task_id) + 1.96 * ps_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[0,1].plot(num_firms_array_extend, ps(default_task_id) - 1.96 * ps_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].axvline(x=num_firms_array_extend[np.nanargmax(ps(default_task_id))], color="black", linestyle="--", alpha=0.25)
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
axs[0,2].plot(num_firms_array_extend, ts(default_task_id), color="black", lw=lw, alpha=alpha)
#axs[0,2].plot(num_firms_array_extend, ts(default_task_id) + 1.96 * ts_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[0,2].plot(num_firms_array_extend, ts(default_task_id) - 1.96 * ts_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].axvline(x=num_firms_array_extend[np.nanargmax(ts(default_task_id))], color="black", linestyle="--", alpha=0.25)
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,2].set_title("total surplus", fontsize=title_fontsize)

# Set axis limits
min_y_cs = np.nanmin(cs(default_task_id)[1:]) # don't include monopoly case
max_y_cs = np.nanmax(cs(default_task_id)[1:])
min_y_ps = np.nanmin(ps(default_task_id)[1:])
max_y_ps = np.nanmax(ps(default_task_id)[1:])
min_y_ts = np.nanmin(ts(default_task_id)[1:])
max_y_ts = np.nanmax(ts(default_task_id)[1:])
diff_cs = max_y_cs - min_y_cs
diff_ps = max_y_ps - min_y_ps
diff_ts = max_y_ts - min_y_ts
axs[0,0].set_ylim((min_y_cs - margin * diff_cs, max_y_cs + margin * diff_cs))
axs[0,1].set_ylim((min_y_ps - margin * diff_ps, max_y_ps + margin * diff_ps))
axs[0,2].set_ylim((min_y_ts - margin * diff_ts, max_y_ts + margin * diff_ts))
for i in range(3):
    axs[0,i].set_xticks(num_firms_array_extend)
        
plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_welfare_1gb10gb.pdf", bbox_inches = "tight", transparent=True)
    
if print_:
    plt.show()
    
# %%
# Welfare for number of firms - all fixed

fig, axs = plt.subplots(1, 3, figsize=(9.0,3.25), sharex=True, squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

# consumer surplus
axs[0,0].plot(num_firms_array_extend, cs_allfixed(default_task_id), color="black", lw=lw, alpha=alpha)
#axs[0,0].plot(num_firms_array_extend, cs_allfixed(default_task_id) + 1.96 * cs_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[0,0].plot(num_firms_array_extend, cs_allfixed(default_task_id) - 1.96 * cs_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].axvline(x=num_firms_array_extend[np.nanargmax(cs_allfixed(default_task_id))], color="black", linestyle="--", alpha=0.25)
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
axs[0,1].plot(num_firms_array_extend, ps_allfixed(default_task_id), color="black", lw=lw, alpha=alpha)
#axs[0,1].plot(num_firms_array_extend, ps_allfixed(default_task_id) + 1.96 * ps_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[0,1].plot(num_firms_array_extend, ps_allfixed(default_task_id) - 1.96 * ps_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].axvline(x=num_firms_array_extend[np.nanargmax(ps_allfixed(default_task_id))], color="black", linestyle="--", alpha=0.25)
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
axs[0,2].plot(num_firms_array_extend, ts_allfixed(default_task_id), color="black", lw=lw, alpha=alpha)
#axs[0,2].plot(num_firms_array_extend, ts_allfixed(default_task_id) + 1.96 * ts_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[0,2].plot(num_firms_array_extend, ts_allfixed(default_task_id) - 1.96 * ts_allfixed_se(default_task_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].axvline(x=num_firms_array_extend[np.nanargmax(ts_allfixed(default_task_id))], color="black", linestyle="--", alpha=0.25)
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,2].set_title("total surplus", fontsize=title_fontsize)

# Set axis limits
min_y_cs = np.nanmin(cs_allfixed(default_task_id)[1:]) # don't include monopoly case
max_y_cs = np.nanmax(cs_allfixed(default_task_id)[1:])
min_y_ps = np.nanmin(ps_allfixed(default_task_id)[1:])
max_y_ps = np.nanmax(ps_allfixed(default_task_id)[1:])
min_y_ts = np.nanmin(ts_allfixed(default_task_id)[1:])
max_y_ts = np.nanmax(ts_allfixed(default_task_id)[1:])
diff_cs = max_y_cs - min_y_cs
diff_ps = max_y_ps - min_y_ps
diff_ts = max_y_ts - min_y_ts
axs[0,0].set_ylim((min_y_cs - margin * diff_cs, max_y_cs + margin * diff_cs))
axs[0,1].set_ylim((min_y_ps - margin * diff_ps, max_y_ps + margin * diff_ps))
axs[0,2].set_ylim((min_y_ts - margin * diff_ts, max_y_ts + margin * diff_ts))
for i in range(3):
    axs[0,i].set_xticks(num_firms_array_extend)
        
plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_welfare_1gb10gb_allfixed.pdf", bbox_inches = "tight", transparent=True)
    
if print_:
    plt.show()
    
# %%
# Consumer surplus by type for number of firms

fig, axs = plt.subplots(1, 3, figsize=(9.0,3.25), sharex=True, squeeze=False)

axs[0,0].plot(num_firms_array_extend, cs_by_type(default_task_id)[:,0], color="black", lw=lw, alpha=alpha)
#axs[i,0].plot(num_firms_array_extend, cs_by_type(default_task_id)[:,0] + 1.96 * cs_by_type_se(default_task_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[i,0].plot(num_firms_array_extend, cs_by_type(default_task_id)[:,0] - 1.96 * cs_by_type_se(default_task_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].axvline(x=num_firms_array_extend[np.argmax(cs_by_type(default_task_id)[:,0])], color="black", linestyle="--", alpha=0.25)
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("consumer surplus (\u20ac / person)", fontsize=y_fontsize)
axs[0,0].set_title("10th percentile", fontsize=title_fontsize)

axs[0,1].plot(num_firms_array_extend, cs_by_type(default_task_id)[:,4], color="black", lw=lw, alpha=alpha)
#axs[i,1].plot(num_firms_array_extend, cs_by_type(default_task_id)[:,4] + 1.96 * cs_by_type_se(default_task_id)[:,4], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[i,1].plot(num_firms_array_extend, cs_by_type(default_task_id)[:,4] - 1.96 * cs_by_type_se(default_task_id)[:,4], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].axvline(x=num_firms_array_extend[np.argmax(cs_by_type(default_task_id)[:,4])], color="black", linestyle="--", alpha=0.25)
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("consumer surplus (\u20ac / person)", fontsize=y_fontsize)
axs[0,1].set_title("50th percentile", fontsize=title_fontsize)

axs[0,2].plot(num_firms_array_extend, cs_by_type(default_task_id)[:,8], color="black", lw=lw, alpha=alpha)
#axs[i,2].plot(num_firms_array_extend, cs_by_type(default_task_id)[:,8] + 1.96 * cs_by_type_se(default_task_id)[:,8], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[i,2].plot(num_firms_array_extend, cs_by_type(default_task_id)[:,8] - 1.96 * cs_by_type_se(default_task_id)[:,8], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].axvline(x=num_firms_array_extend[np.argmax(cs_by_type(default_task_id)[:,8])], color="black", linestyle="--", alpha=0.25)
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("consumer surplus (\u20ac / person)", fontsize=y_fontsize)
axs[0,2].set_title("90th percentile", fontsize=title_fontsize)
    
# Set axis limits
for i, income_idx in enumerate([0,4,8]):
    margin = 0.1
    min_cs = np.min(cs_by_type(default_task_id)[1:,income_idx])
    max_cs = np.max(cs_by_type(default_task_id)[1:,income_idx])
    diff = max_cs - min_cs
    axs[0,i].set_ylim((min_cs - margin * diff, max_cs + margin * diff)) # don't include the monopoly case
    axs[0,i].set_xticks(num_firms_array_extend)
        
plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_cs_by_income_1gb10gb.pdf", bbox_inches = "tight", transparent=True)
    
if print_:
    plt.show()
    
# %%
# Consumer surplus by type for number of firms - all fixed

fig, axs = plt.subplots(1, 3, figsize=(9.0,3.25), sharex=True, squeeze=False)

axs[0,0].plot(num_firms_array_extend, cs_by_type_allfixed(default_task_id)[:,0], color="black", lw=lw, alpha=alpha)
#axs[i,0].plot(num_firms_array_extend, cs_by_type_allfixed(default_task_id)[:,0] + 1.96 * cs_by_type_allfixed_se(default_task_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[i,0].plot(num_firms_array_extend, cs_by_type_allfixed(default_task_id)[:,0] - 1.96 * cs_by_type_allfixed_se(default_task_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].axvline(x=num_firms_array_extend[np.argmax(cs_by_type_allfixed(default_task_id)[:,0])], color="black", linestyle="--", alpha=0.25)
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("consumer surplus (\u20ac / person)", fontsize=y_fontsize)
axs[0,0].set_title("10th percentile", fontsize=title_fontsize)

axs[0,1].plot(num_firms_array_extend, cs_by_type_allfixed(default_task_id)[:,4], color="black", lw=lw, alpha=alpha)
#axs[i,1].plot(num_firms_array_extend, cs_by_type_allfixed(default_task_id)[:,4] + 1.96 * cs_by_type_allfixed_se(default_task_id)[:,4], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[i,1].plot(num_firms_array_extend, cs_by_type_allfixed(default_task_id)[:,4] - 1.96 * cs_by_type_allfixed_se(default_task_id)[:,4], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].axvline(x=num_firms_array_extend[np.argmax(cs_by_type_allfixed(default_task_id)[:,4])], color="black", linestyle="--", alpha=0.25)
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("consumer surplus (\u20ac / person)", fontsize=y_fontsize)
axs[0,1].set_title("50th percentile", fontsize=title_fontsize)

axs[0,2].plot(num_firms_array_extend, cs_by_type_allfixed(default_task_id)[:,8], color="black", lw=lw, alpha=alpha)
#axs[i,2].plot(num_firms_array_extend, cs_by_type_allfixed(default_task_id)[:,8] + 1.96 * cs_by_type_allfixed_se(default_task_id)[:,8], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[i,2].plot(num_firms_array_extend, cs_by_type_allfixed(default_task_id)[:,8] - 1.96 * cs_by_type_allfixed_se(default_task_id)[:,8], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].axvline(x=num_firms_array_extend[np.argmax(cs_by_type_allfixed(default_task_id)[:,8])], color="black", linestyle="--", alpha=0.25)
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("consumer surplus (\u20ac / person)", fontsize=y_fontsize)
axs[0,2].set_title("90th percentile", fontsize=title_fontsize)
    
# Set axis limits
for i, income_idx in enumerate([0,4,8]):
    margin = 0.1
    min_cs = np.min(cs_by_type_allfixed(default_task_id)[1:,income_idx])
    max_cs = np.max(cs_by_type_allfixed(default_task_id)[1:,income_idx])
    diff = max_cs - min_cs
    axs[0,i].set_ylim((min_cs - margin * diff, max_cs + margin * diff)) # don't include the monopoly case
    axs[0,i].set_xticks(num_firms_array_extend)
        
plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_cs_by_income_1gb10gb_allfixed.pdf", bbox_inches = "tight", transparent=True)
    
if print_:
    plt.show()
    
# %%
# Endogenous Variables in "Short-run" Simulations

to_tex = "\\begin{tabular}{c c c c} \n"
to_tex += " & $\\Delta$ 1$\\,$000 MB plan & $\\Delta$ 10$\\,$000 MB plan & $\\Delta$ download  \\\\ \n" 
to_tex += " & prices (in \euro{}) & prices (in \euro{}) & speeds (in Mbps) \\\\ \n"
to_tex += "\\hline \n"
to_tex += "short-run"
to_tex += f" & ${round_var(p_stars_shortrun(default_task_id)[0,0], num_digits_round)}$ ${round_var(p_stars_shortrun_se(default_task_id)[0,0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(p_stars_shortrun(default_task_id)[0,1], num_digits_round)}$ ${round_var(p_stars_shortrun_se(default_task_id)[0,1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(q_stars_shortrun(default_task_id)[0], num_digits_round)}$ ${round_var(q_stars_shortrun_se(default_task_id)[0], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "long-run"
to_tex += f" & ${round_var(p_stars_shortrun(default_task_id)[1,0], num_digits_round)}$ ${round_var(p_stars_shortrun_se(default_task_id)[1,0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(p_stars_shortrun(default_task_id)[1,1], num_digits_round)}$ ${round_var(p_stars_shortrun_se(default_task_id)[1,1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(q_stars_shortrun(default_task_id)[1], num_digits_round)}$ ${round_var(q_stars_shortrun_se(default_task_id)[1], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "\\hline \n" 
to_tex += "difference"
to_tex += f" & ${round_var(p_stars_shortrun(default_task_id)[2,0], num_digits_round)}$ ${round_var(p_stars_shortrun_se(default_task_id)[2,0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(p_stars_shortrun(default_task_id)[2,1], num_digits_round)}$ ${round_var(p_stars_shortrun_se(default_task_id)[2,1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(q_stars_shortrun(default_task_id)[2], num_digits_round)}$ ${round_var(q_stars_shortrun_se(default_task_id)[2], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "\\hline \n" 
to_tex += "\\end{tabular} \n"
if save_:
    create_file(f"{paths.tables_path}counterfactual_shortrun_variables_preferredspecification.tex", to_tex)
if print_:
    print(to_tex)
    
# %%
# Welfare in "Short-run" simulations

to_tex = "\\begin{tabular}{c c c c} \n"
to_tex += " & $\\Delta$ CS & $\\Delta$ PS & $\\Delta$ TS \\\\ \n"
to_tex += "\\hline \n"
to_tex += "short-run"
to_tex += f" & ${round_var(cs_shortrun(default_task_id)[0], num_digits_round)}$ ${round_var(cs_shortrun_se(default_task_id)[0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ps_shortrun(default_task_id)[0], num_digits_round)}$ ${round_var(ps_shortrun_se(default_task_id)[0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ts_shortrun(default_task_id)[0], num_digits_round)}$ ${round_var(ts_shortrun_se(default_task_id)[0], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "long-run"
to_tex += f" & ${round_var(cs_shortrun(default_task_id)[1], num_digits_round)}$ ${round_var(cs_shortrun_se(default_task_id)[1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ps_shortrun(default_task_id)[1], num_digits_round)}$ ${round_var(ps_shortrun_se(default_task_id)[1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ts_shortrun(default_task_id)[1], num_digits_round)}$ ${round_var(ts_shortrun_se(default_task_id)[1], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "\\hline \n" 
to_tex += "difference"
to_tex += f" & ${round_var(cs_shortrun(default_task_id)[2], num_digits_round)}$ ${round_var(cs_shortrun_se(default_task_id)[2], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ps_shortrun(default_task_id)[2], num_digits_round)}$ ${round_var(ps_shortrun_se(default_task_id)[2], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ts_shortrun(default_task_id)[2], num_digits_round)}$ ${round_var(ts_shortrun_se(default_task_id)[2], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "\\hline \n" 
to_tex += "\\end{tabular} \n"
if save_:
    create_file(f"{paths.tables_path}counterfactual_shortrun_welfare_preferredspecification.tex", to_tex)
if print_:
    print(to_tex)
    
# %%
# CS by Income Level in "Short-run" Simulations

to_tex = "\\begin{tabular}{c c c c c c} \n"
to_tex += " & $\\Delta$ CS & $\\Delta$ CS & $\\Delta$ CS & $\\Delta$ CS & $\\Delta$ CS \\\\ \n" 
to_tex += " & 10 \\%ile & 30 \\%ile & 50 \\%ile & 70 \\%ile & 90 \\%ile \\\\ \n"
to_tex += "\\hline \n"
to_tex += "short-run & "
for i in range(5):
    to_tex += f"${round_var(cs_by_type_shortrun(default_task_id)[0,2*i], num_digits_round)}$ ${round_var(cs_by_type_shortrun_se(default_task_id)[1,2*i], num_digits_round, stderrs=True)}$"
    if i < 4:
        to_tex += " & "
to_tex += " \\\\ \n"
to_tex += "long-run & "
for i in range(5):
    to_tex += f"${round_var(cs_by_type_shortrun(default_task_id)[1,2*i], num_digits_round)}$ ${round_var(cs_by_type_shortrun_se(default_task_id)[0,2*i], num_digits_round, stderrs=True)}$"
    if i < 4:
        to_tex += " & "
to_tex += " \\\\ \n"
to_tex += "\\hline \n" 
to_tex += "difference & "
for i in range(5):
    to_tex += f"${round_var(cs_by_type_shortrun(default_task_id)[2,2*i], num_digits_round)}$ ${round_var(cs_by_type_shortrun_se(default_task_id)[0,2*i], num_digits_round, stderrs=True)}$"
    if i < 4:
        to_tex += " & "
to_tex += " \\\\ \n"
to_tex += "\\hline \n" 
to_tex += "\\end{tabular} \n"
if save_:
    create_file(f"{paths.tables_path}counterfactual_shortrun_cs_by_income_preferredspecification.tex", to_tex)
if print_:
    print(to_tex)
    
# %%
# "Add Free" endogenous variables (all fixed)

fig, axs = plt.subplots(2, 3, figsize=(12.0,6.5), squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

x_pos = [i for i in range(2)]
x_ticklabels = ["$3$ firms, $\\frac{4}{3}$ b", "$4$ firms, b"]

# dlim = 1,000 prices
axs[0,0].bar(x_pos, p_stars_free_allfixed(default_task_id)[:,0], yerr=1.96 * p_stars_free_allfixed_se(default_task_id)[:,0], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,0].set_xticks(x_pos)
axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,0].set_ylabel("$\\Delta p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
axs[0,1].bar(x_pos, p_stars_free_allfixed(default_task_id)[:,1], yerr=1.96 * p_stars_free_allfixed_se(default_task_id)[:,1], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,1].set_xticks(x_pos)
axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,1].set_ylabel("$\\Delta p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

axs[0,2].bar(x_pos, num_stations_per_firm_stars_free_allfixed(default_task_id) * 1000.0, yerr=1.96 * num_stations_per_firm_stars_free_allfixed_se(default_task_id) * 1000.0, capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,2].set_xticks(x_pos)
axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,2].set_ylabel("$\\Delta$ number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[0,2].set_title("number of stations / firm", fontsize=title_fontsize)

# total number of stations
axs[1,0].bar(x_pos, num_stations_stars_free_allfixed(default_task_id) * 1000.0, yerr=1.96 * num_stations_stars_free_allfixed_se(default_task_id) * 1000.0, capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,0].set_xticks(x_pos)
axs[1,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,0].set_ylabel("$\\Delta$ number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[1,0].set_title("total number of stations", fontsize=title_fontsize)

# average path loss
axs[1,1].bar(x_pos, ccs_per_bw_free_allfixed(default_task_id), yerr=1.96 * ccs_per_bw_free_allfixed_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,1].set_xticks(x_pos)
axs[1,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,1].set_ylabel("$\\Delta$ Mbps / MHz", fontsize=y_fontsize)
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)

# download speeds
axs[1,2].bar(x_pos, q_stars_free_allfixed(default_task_id), yerr=1.96 * q_stars_free_allfixed_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,2].set_xticks(x_pos)
axs[1,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,2].set_ylabel("$\\Delta Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.min(p_stars_free_allfixed(default_task_id)) - 0.75
max_y_p = np.max(p_stars_free_allfixed(default_task_id)) + 0.7
for i in range(2): # first two columns
    axs[0,i].set_ylim((min_y_p, max_y_p))

plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_free_variables_allfixed_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# "Add Free" endogenous variables (all bw)

fig, axs = plt.subplots(2, 3, figsize=(12.0,6.5), squeeze=False)

x_pos = [i for i in range(2)]
x_ticklabels = ["$3$ firms, $\\frac{4}{3}$ b", "$4$ firms, b"]

# dlim = 1,000 prices
axs[0,0].bar(x_pos, p_stars_free_allbw(default_task_id)[:,0], yerr=1.96 * p_stars_free_allbw_se(default_task_id)[:,0], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,0].set_xticks(x_pos)
axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,0].set_ylabel("$\\Delta p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
axs[0,1].bar(x_pos, p_stars_free_allbw(default_task_id)[:,1], yerr=1.96 * p_stars_free_allbw_se(default_task_id)[:,1], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,1].set_xticks(x_pos)
axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,1].set_ylabel("$\\Delta p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

axs[0,2].bar(x_pos, num_stations_per_firm_stars_free_allbw(default_task_id) * 1000.0, yerr=1.96 * num_stations_per_firm_stars_free_allbw_se(default_task_id) * 1000.0, capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,2].set_xticks(x_pos)
axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,2].set_ylabel("$\\Delta$ number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[0,2].set_title("number of stations / firm", fontsize=title_fontsize)

# total number of stations
axs[1,0].bar(x_pos, num_stations_stars_free_allbw(default_task_id) * 1000.0, yerr=1.96 * num_stations_stars_free_allbw_se(default_task_id) * 1000.0, capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,0].set_xticks(x_pos)
axs[1,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,0].set_ylabel("$\\Delta$ number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[1,0].set_title("total number of stations", fontsize=title_fontsize)

# average path loss
axs[1,1].bar(x_pos, ccs_per_bw_free_allbw(default_task_id), yerr=1.96 * ccs_per_bw_free_allbw_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,1].set_xticks(x_pos)
axs[1,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,1].set_ylabel("$\\Delta$ Mbps / MHz", fontsize=y_fontsize)
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)

# download speeds
axs[1,2].bar(x_pos, q_stars_free_allbw(default_task_id), yerr=1.96 * q_stars_free_allbw_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,2].set_xticks(x_pos)
axs[1,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,2].set_ylabel("$\\Delta Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.min(p_stars_free_allbw(default_task_id)) - 0.65
max_y_p = np.max(p_stars_free_allbw(default_task_id)) + 0.6
for i in range(2): # first two columns
    axs[0,i].set_ylim((min_y_p, max_y_p))

plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_free_variables_allbw_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Welfare for "Add Free" (all fixed)

fig, axs = plt.subplots(1, 3, figsize=(9.0,3.25), squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

x_pos = [i for i in range(2)]
x_ticklabels = ["$3$ firms, $\\frac{4}{3}$ b", "$4$ firms, b"]

margin = 0.1

# consumer surplus
axs[0,0].bar(x_pos, cs_free_allfixed(default_task_id), yerr=1.96*cs_free_allfixed_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,0].set_xticks(x_pos)
axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,0].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_free_allfixed(default_task_id) + 1.96 * cs_free_allfixed_se(default_task_id))
min_cs = np.min(cs_free_allfixed(default_task_id) - 1.96 * cs_free_allfixed_se(default_task_id))
diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
axs[0,0].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
axs[0,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
axs[0,1].bar(x_pos, ps_free_allfixed(default_task_id), yerr=1.96*ps_free_allfixed_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,1].set_xticks(x_pos)
axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,1].set_ylabel("$\\Delta$ PS (in \u20ac / person)", fontsize=y_fontsize)
max_ps = np.max(ps_free_allfixed(default_task_id) + 1.96 * ps_free_allfixed_se(default_task_id))
min_ps = np.min(ps_free_allfixed(default_task_id) - 1.96 * ps_free_allfixed_se(default_task_id))
diff = np.maximum(max_ps, 0.0) - np.minimum(min_ps, 0.0)
axs[0,1].set_ylim((np.minimum(min_ps - margin * diff, 0.0), np.maximum(max_ps + margin * diff, 0.0)))
axs[0,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
axs[0,2].bar(x_pos, ts_free_allfixed(default_task_id), yerr=1.96*ts_free_allfixed_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,2].set_xticks(x_pos)
axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,2].set_ylabel("$\\Delta$ TS (in \u20ac / person)", fontsize=y_fontsize)
max_ts = np.max(ts_free_allfixed(default_task_id) + 1.96 * ts_free_allfixed_se(default_task_id))
min_ts = np.min(ts_free_allfixed(default_task_id) - 1.96 * ts_free_allfixed_se(default_task_id))
diff = np.maximum(max_ts, 0.0) - np.minimum(min_ts, 0.0)
axs[0,2].set_ylim((np.minimum(min_ts - margin * diff, 0.0), np.maximum(max_ts + margin * diff, 0.0)))
axs[0,2].set_title("total surplus", fontsize=title_fontsize)
        
plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_free_welfare_allfixed_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Welfare for "Add Free" (all bw)

fig, axs = plt.subplots(1, 3, figsize=(9.0,3.25), squeeze=False)

x_pos = [i for i in range(2)]
x_ticklabels = ["$3$ firms, $\\frac{4}{3}$ b", "$4$ firms, b"]

margin = 0.1

# consumer surplus
axs[0,0].bar(x_pos, cs_free_allbw(default_task_id), yerr=1.96*cs_free_allbw_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,0].set_xticks(x_pos)
axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,0].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_free_allbw(default_task_id) + 1.96 * cs_free_allbw_se(default_task_id))
min_cs = np.min(cs_free_allbw(default_task_id) - 1.96 * cs_free_allbw_se(default_task_id))
diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
axs[0,0].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
axs[0,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
axs[0,1].bar(x_pos, ps_free_allbw(default_task_id), yerr=1.96*ps_free_allbw_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,1].set_xticks(x_pos)
axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,1].set_ylabel("$\\Delta$ PS (in \u20ac / person)", fontsize=y_fontsize)
max_ps = np.max(ps_free_allbw(default_task_id) + 1.96 * ps_free_allbw_se(default_task_id))
min_ps = np.min(ps_free_allbw(default_task_id) - 1.96 * ps_free_allbw_se(default_task_id))
diff = np.maximum(max_ps, 0.0) - np.minimum(min_ps, 0.0)
axs[0,1].set_ylim((np.minimum(min_ps - margin * diff, 0.0), np.maximum(max_ps + margin * diff, 0.0)))
axs[0,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
axs[0,2].bar(x_pos, ts_free_allbw(default_task_id), yerr=1.96*ts_free_allbw_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,2].set_xticks(x_pos)
axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,2].set_ylabel("$\\Delta$ TS (in \u20ac / person)", fontsize=y_fontsize)
max_ts = np.max(ts_free_allbw(default_task_id) + 1.96 * ts_free_allbw_se(default_task_id))
min_ts = np.min(ts_free_allbw(default_task_id) - 1.96 * ts_free_allbw_se(default_task_id))
diff = np.maximum(max_ts, 0.0) - np.minimum(min_ts, 0.0)
axs[0,2].set_ylim((np.minimum(min_ts - margin * diff, 0.0), np.maximum(max_ts + margin * diff, 0.0)))
axs[0,2].set_title("total surplus", fontsize=title_fontsize)
        
plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_free_welfare_allbw_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Consumer surplus by type for "Add Free" (all bw)

fig, axs = plt.subplots(1, 3, figsize=(9.0,3.25), squeeze=False)

x_pos = [i for i in range(2)]
x_ticklabels = ["$3$ firms, $\\frac{4}{3}$ b", "$4$ firms, b"]

axs[0,0].bar(x_pos, cs_by_type_free_allbw(default_task_id)[:,0], yerr=1.96*cs_by_type_free_allbw_se(default_task_id)[:,0], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,0].set_xticks(x_pos)
axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,0].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_by_type_free_allbw(default_task_id)[:,0] + 1.96 * cs_by_type_free_allbw_se(default_task_id)[:,0])
min_cs = np.min(cs_by_type_free_allbw(default_task_id)[:,0] - 1.96 * cs_by_type_free_allbw_se(default_task_id)[:,0])
diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
axs[0,0].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
axs[0,0].set_title("10th percentile", fontsize=title_fontsize)

axs[0,1].bar(x_pos, cs_by_type_free_allbw(default_task_id)[:,4], yerr=1.96*cs_by_type_free_allbw_se(default_task_id)[:,4], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,1].set_xticks(x_pos)
axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,1].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_by_type_free_allbw(default_task_id)[:,4] + 1.96 * cs_by_type_free_allbw_se(default_task_id)[:,4])
min_cs = np.min(cs_by_type_free_allbw(default_task_id)[:,4] - 1.96 * cs_by_type_free_allbw_se(default_task_id)[:,4])
diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
axs[0,1].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
axs[0,1].set_title("50th percentile", fontsize=title_fontsize)

axs[0,2].bar(x_pos, cs_by_type_free_allbw(default_task_id)[:,8], yerr=1.96*cs_by_type_free_allbw_se(default_task_id)[:,8], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,2].set_xticks(x_pos)
axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,2].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_by_type_free_allbw(default_task_id)[:,8] + 1.96 * cs_by_type_free_allbw_se(default_task_id)[:,8])
min_cs = np.min(cs_by_type_free_allbw(default_task_id)[:,8] - 1.96 * cs_by_type_free_allbw_se(default_task_id)[:,8])
diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
axs[0,2].ticklabel_format(style="plain", useOffset=False, axis="y")
axs[0,2].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
axs[0,2].set_title("90th percentile", fontsize=title_fontsize)
        
plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_free_cs_by_income_allbw_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Endogenous variables - number of firms - by density

fig, axs = plt.subplots(2, 3, figsize=(12.0, 6.5), squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

densities_argsort = np.argsort(densities(default_task_id))
densities_sort = densities(default_task_id)[densities_argsort]
default_dens_id = np.where(densities_sort == densities(default_task_id)[0])[0][0] # we saved the default density as the first one in the original file
dens_legend_ = ["continental USA density", "France density", "France contraharmonic mean density", "Paris density"] # b/c sorted
dens_legend = np.array([f"$\\bf{{{dens_legend_[i]}}}$".replace(" ", "\\:") if i == default_dens_id else f"{dens_legend_[i]}" for i, dens in enumerate(densities_sort)])
dens_use = np.ones(densities_sort.shape, dtype=bool) # default: use all
# dens_use[-1] = False

dens_color_p = "Greys" # plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
dens_color_R = "Greys"
dens_color_Rtot = "Greys"
dens_color_q = "Greys"
dens_color_pl = "Greys"
alphas_dens = np.linspace(0.25, 0.75, densities(default_task_id)[dens_use].shape[0])

# dlim = 1,000 prices
for i, dens in enumerate(densities_sort[dens_use]):
    axs[0,0].plot(num_firms_array, p_stars_dens(default_task_id)[:,densities_argsort[dens_use][i],0], color=cm.get_cmap(dens_color_p)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
for i, dens in enumerate(densities_sort[dens_use]):
    axs[0,1].plot(num_firms_array, p_stars_dens(default_task_id)[:,densities_argsort[dens_use][i],1], color=cm.get_cmap(dens_color_p)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

# radius
for i, dens in enumerate(densities_sort[dens_use]):
    axs[0,2].plot(num_firms_array, num_stations_per_firm_stars_dens(default_task_id)[:,densities_argsort[dens_use][i]] * 1000.0, color=cm.get_cmap(dens_color_R)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[0,2].set_title("number of stations / firm / person", fontsize=title_fontsize)

# total number of stations
for i, dens in enumerate(densities_sort[dens_use]):
    axs[1,0].plot(num_firms_array, num_stations_stars_dens(default_task_id)[:,densities_argsort[dens_use][i]] * 1000.0, color=cm.get_cmap(dens_color_Rtot)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
axs[1,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,0].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[1,0].set_title("total number of stations / person", fontsize=title_fontsize)

# for i, dens in enumerate(densities_sort[dens_use]):
#     axs[1,1].plot(num_firms_array, avg_path_losses_dens(default_task_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_pl)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
# axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
# axs[1,1].set_ylabel("dB", fontsize=y_fontsize)
# axs[1,1].set_title("average path loss", fontsize=title_fontsize)
for i, dens in enumerate(densities_sort[dens_use]):
    axs[1,1].plot(num_firms_array, ccs_per_bw_dens(default_task_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_pl)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,1].set_ylabel("Mbps / MHz", fontsize=y_fontsize)
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)

# download speeds
for i, dens in enumerate(densities_sort[dens_use]):
    axs[1,2].plot(num_firms_array, q_stars_dens(default_task_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_q)(alphas_dens[i]), lw=lw, label="download speed" if i == np.sum(dens_use) - 1 else None)
    axs[1,2].plot(num_firms_array, ccs_dens(default_task_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_q)(alphas_dens[i]), lw=0.6 * lw, ls="--", label="channel capacity" if i == np.sum(dens_use) - 1 else None)
axs[1,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,2].set_ylabel("$Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.nanmin(p_stars_dens(default_task_id)[:,densities_argsort,:][1:,dens_use,:])
max_y_p = np.nanmax(p_stars_dens(default_task_id)[:,densities_argsort,:][1:,dens_use,:])
min_y_num_stations_per_firm = 1000.0 * np.nanmin(num_stations_per_firm_stars_dens(default_task_id)[:,densities_argsort][:,dens_use])
max_y_num_stations_per_firm = 1000.0 * np.nanmax(num_stations_per_firm_stars_dens(default_task_id)[:,densities_argsort][:,dens_use])
min_y_num_stations = 1000.0 * np.nanmin(num_stations_stars_dens(default_task_id)[:,densities_argsort][:,dens_use])
max_y_num_stations = 1000.0 * np.nanmax(num_stations_stars_dens(default_task_id)[:,densities_argsort][:,dens_use])
# min_y_pl = np.nanmin(avg_path_losses_dens(default_task_id)[:,densities_argsort][:,dens_use])
# max_y_pl = np.nanmax(avg_path_losses_dens(default_task_id)[:,densities_argsort][:,dens_use])
min_y_q = np.nanmin(q_stars_dens(default_task_id)[:,densities_argsort][:,dens_use])
max_y_q = np.nanmax(q_stars_dens(default_task_id)[:,densities_argsort][:,dens_use])
diff_p = max_y_p - min_y_p
diff_num_stations_per_firm = max_y_num_stations_per_firm - min_y_num_stations_per_firm
diff_num_stations = max_y_num_stations - min_y_num_stations
# diff_pl = max_y_pl - min_y_pl
diff_q = max_y_q - min_y_q
margin = 0.1
for i in range(2):
    axs[0,i].set_ylim((min_y_p - margin * diff_p, max_y_p + margin * diff_p))
axs[0,2].set_ylim((min_y_num_stations_per_firm - margin * diff_num_stations_per_firm, max_y_num_stations_per_firm + margin * diff_num_stations_per_firm))
axs[1,0].set_ylim((min_y_num_stations - margin * diff_num_stations, max_y_num_stations + margin * diff_num_stations))
# axs[1,1].set_ylim((min_y_pl - margin * diff_pl, max_y_pl + margin * diff_pl + 5.0))
# axs[1,2].set_ylim((min_y_q - margin * diff_q, max_y_q + margin * diff_q))
for i in range(2):
    for j in range(3):
        axs[i,j].set_xticks(num_firms_array)
        
# Legends
# axs[0,0].legend(loc="best")
# axs[0,1].legend(loc="best")
# axs[0,2].legend(loc="best")
# axs[1,0].legend(loc="best")
# axs[1,1].legend(loc="best")
axs[1,2].legend(loc="best")
lines_1 = axs[0,0].get_lines()
lines_2 = axs[0,2].get_lines()
lines_3 = axs[1,1].get_lines()
lines_list = [(lines_1[i], lines_2[i], lines_3[i]) for i in range(np.sum(dens_use))]
labels_list = list(dens_legend[dens_use])
leg = fig.legend(lines_list, labels_list, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)}, ncol=4, loc="lower center", bbox_to_anchor=(-0.02, -0.065, 1, 1), bbox_transform = plt.gcf().transFigure, prop={'size': 12.5})
leg.get_frame().set_alpha(0.0)

plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_dens_1gb10gb.pdf", bbox_inches = "tight", transparent=True)
    
if save_:
    create_file(f"{paths.stats_path}rep_dens.tex", "{:,.0f}".format(densities(default_task_id)[0]).replace(",","\\,"))

if print_:
    plt.show()
    
# %%
# Welfare for number of firms by density

fig, axs = plt.subplots(1, 3, figsize=(9.0,3.25), sharex=True, squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

dens_color_cs = "Greys"
dens_color_ps = "Greys"
dens_color_ts = "Greys"
alphas_dens = np.linspace(0.25, 0.75, densities(default_task_id)[dens_use].shape[0])

# consumer surplus
for i, dens in enumerate(densities_sort[dens_use]):
    axs[0,0].plot(num_firms_array_extend, cs_dens(default_task_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_cs)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
    axs[0,0].axvline(x=num_firms_array_extend[np.nanargmax(cs_dens(default_task_id)[:,densities_argsort[dens_use][i]])], color=cm.get_cmap(dens_color_cs)(alphas_dens[i]), linestyle="--")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
for i, dens in enumerate(densities_sort[dens_use]):
    axs[0,1].plot(num_firms_array_extend, ps_dens(default_task_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_ps)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
    axs[0,1].axvline(x=num_firms_array_extend[np.argmax(ps_dens(default_task_id)[:,densities_argsort[dens_use][i]])], color=cm.get_cmap(dens_color_ps)(alphas_dens[i]), linestyle="--")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
for i, dens in enumerate(densities_sort[dens_use]):
    axs[0,2].plot(num_firms_array_extend, ts_dens(default_task_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_ts)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
    axs[0,2].axvline(x=num_firms_array_extend[np.nanargmax(ts_dens(default_task_id)[:,densities_argsort[dens_use][i]])], color=cm.get_cmap(dens_color_ts)(alphas_dens[i]), linestyle="--")
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,2].set_title("total surplus", fontsize=title_fontsize)

# Set axis limits
# margin_cs = 1.0 - np.sort(np.concatenate(tuple([normalize_var(cs_dens(default_task_id)[:,densities_argsort[dens_use][i]])])))[-6]
# margin_ts = 1.0 - np.sort(np.concatenate(tuple([normalize_var(ts_dens(default_task_id)[:,densities_argsort[dens_use][i]])])))[-6]
# axs[0,0].set_ylim((1.0 - margin_cs, 1.0 + 0.33 * margin_cs))
# axs[0,2].set_ylim((1.0 - margin_ts, 1.0 + 0.33 * margin_ts))
axs[0,0].set_ylim((28.5, 29.5))
axs[0,1].set_ylim((-16.0, -14.0))
axs[0,2].set_ylim((13.0, 14.5))

for i in range(3):
    axs[0,i].set_xticks(num_firms_array_extend)
#     axs[0,i].set_yticks([])
    
# Legends
# axs[0,0].legend(loc="best")
# axs[0,1].legend(loc="best")
# axs[0,2].legend(loc="best")
lines_1 = axs[0,0].get_lines()
lines_2 = axs[0,1].get_lines()
lines_3 = axs[0,2].get_lines()
lines_list = [(lines_1[i*2], lines_2[i*2], lines_3[i*2]) for i in range(np.sum(dens_use))]
labels_list = list(dens_legend[dens_use])
leg = fig.legend(lines_list, labels_list, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)}, ncol=2, loc="lower center", bbox_to_anchor=(0.025, -0.2, 1, 1), bbox_transform = plt.gcf().transFigure, prop={'size': 12.5})
leg.get_frame().set_alpha(0.0)

plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_dens_welfare_1gb10gb.pdf", bbox_inches = "tight", transparent=True)
    
if print_:
    plt.show()
    
# %%
# Endogenous variables - number of firms - by bandwidth

bw_vals_argsort = np.argsort(bw_vals(default_task_id))
bw_vals_sort = bw_vals(default_task_id)[bw_vals_argsort]
default_bw_id = np.where(bw_vals_sort == bw_vals(default_task_id)[0])[0][0] # we saved the default density as the first one in the original file
bw_legend_ = ["0.5 * bw", "bw", "1.5 * bw"] # b/c sorted
bw_legend = np.array([f"$\\bf{{{bw_legend_[i]}}}$" if i == default_bw_id else f"{bw_legend_[i]}" for i, bw in enumerate(bw_vals_sort)])
bw_use = np.ones(bw_vals_sort.shape, dtype=bool)

fig, axs = plt.subplots(2, 3, figsize=(12.0, 6.5), squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

bw_color_p = "Greys"
bw_color_R = "Greys"
bw_color_Rtot = "Greys"
bw_color_q = "Greys"
bw_color_pl = "Greys"
alphas_bw = np.linspace(0.25, 0.75, bw_vals(default_task_id)[bw_use].shape[0])

# dlim = 1,000 prices
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[0,0].plot(num_firms_array, p_stars_bw(default_task_id)[:,bw_vals_argsort[bw_use][i],0], color=cm.get_cmap(bw_color_p)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[0,1].plot(num_firms_array, p_stars_bw(default_task_id)[:,bw_vals_argsort[bw_use][i],1], color=cm.get_cmap(bw_color_p)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

# radius
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[0,2].plot(num_firms_array, num_stations_per_firm_stars_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]] * 1000.0, color=cm.get_cmap(bw_color_R)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[0,2].set_title("number of stations / firm", fontsize=title_fontsize)

# total number of stations
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[1,0].plot(num_firms_array, num_stations_stars_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]] * 1000.0, color=cm.get_cmap(bw_color_Rtot)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
axs[1,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,0].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[1,0].set_title("total number of stations", fontsize=title_fontsize)

# average path loss
# for i, bw in enumerate(bw_vals_sort[bw_use]):
#     axs[1,1].plot(num_firms_array, avg_path_losses_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]], color=cm.get_cmap(bw_color_pl)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
# axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
# axs[1,1].set_ylabel("dB", fontsize=y_fontsize)
# axs[1,1].set_title("average path loss")
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[1,1].plot(num_firms_array, ccs_per_bw_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]], color=cm.get_cmap(bw_color_pl)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,1].set_ylabel("Mbps / MHz", fontsize=y_fontsize)
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)

# download speeds
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[1,2].plot(num_firms_array, q_stars_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]], color=cm.get_cmap(bw_color_q)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
axs[1,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,2].set_ylabel("$Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.nanmin(p_stars_bw(default_task_id)[1:,bw_use,:])
max_y_p = np.nanmax(p_stars_bw(default_task_id)[1:,bw_use,:])
min_y_num_stations_per_firm = np.nanmin(num_stations_per_firm_stars_bw(default_task_id)[:,bw_use]) * 1000.0
max_y_num_stations_per_firm = np.nanmax(num_stations_per_firm_stars_bw(default_task_id)[:,bw_use]) * 1000.0
min_y_num_stations = np.nanmin(num_stations_stars_bw(default_task_id)[:,bw_use]) * 1000.0
max_y_num_stations = np.nanmax(num_stations_stars_bw(default_task_id)[:,bw_use]) * 1000.0
# min_y_pl = np.nanmin(avg_path_losses_bw(default_task_id)[:,bw_use]) - 2.
# max_y_pl = np.nanmax(avg_path_losses_bw(default_task_id)[:,bw_use]) + 2.
min_y_q = np.nanmin(q_stars_bw(default_task_id)[:,bw_use])
max_y_q = np.nanmax(q_stars_bw(default_task_id)[:,bw_use])
diff_p = max_y_p - min_y_p
diff_num_stations_per_firm = max_y_num_stations_per_firm - min_y_num_stations_per_firm
diff_num_stations = max_y_num_stations - min_y_num_stations
# diff_pl = max_y_pl - min_y_pl
diff_q = max_y_q - min_y_q
margin = 0.1
for i in range(2):
    axs[0,i].set_ylim((min_y_p - margin * diff_p, max_y_p + margin * diff_p))
axs[0,2].set_ylim((min_y_num_stations_per_firm - margin * diff_num_stations_per_firm, max_y_num_stations_per_firm + margin * diff_num_stations_per_firm))
axs[1,0].set_ylim((min_y_num_stations - margin * diff_num_stations, max_y_num_stations + margin * diff_num_stations))
# axs[1,1].set_ylim((min_y_pl - margin * diff_pl, max_y_pl + margin * diff_pl))
axs[1,2].set_ylim((min_y_q - margin * diff_q, max_y_q + margin * diff_q))
for i in range(2):
    for j in range(3):
        axs[i,j].set_xticks(num_firms_array)

# Legends
axs[0,0].legend(loc="best")
axs[0,1].legend(loc="best")
axs[0,2].legend(loc="best")
axs[1,0].legend(loc="best")
axs[1,1].legend(loc="best")
axs[1,2].legend(loc="best")

plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_bw_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Welfare for number of firms by bandwidth

fig, axs = plt.subplots(1, 3, figsize=(9.0,3.25), sharex=True, squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

bw_use = np.ones(bw_vals_sort.shape, dtype=bool)

bw_color_cs = "Greys"
bw_color_ps = "Greys"
bw_color_ts = "Greys"
alphas_bw = np.linspace(0.25, 0.75, bw_vals(default_task_id)[bw_use].shape[0])

normalize_var = lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

# consumer surplus
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[0,0].plot(num_firms_array_extend, normalize_var(cs_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]]), color=cm.get_cmap(bw_color_cs)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
    axs[0,0].axvline(x=num_firms_array_extend[np.nanargmax(cs_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]])], color=cm.get_cmap(bw_color_cs)(alphas_bw[i]), linestyle="--")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[0,1].plot(num_firms_array_extend, normalize_var(ps_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]]), color=cm.get_cmap(bw_color_ps)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
    axs[0,1].axvline(x=num_firms_array_extend[np.nanargmax(ps_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]])], color=cm.get_cmap(bw_color_ps)(alphas_bw[i]), linestyle="--")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[0,2].plot(num_firms_array_extend, normalize_var(ts_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]]), color=cm.get_cmap(bw_color_ts)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
    axs[0,2].axvline(x=num_firms_array_extend[np.nanargmax(ts_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]])], color=cm.get_cmap(bw_color_ts)(alphas_bw[i]), linestyle="--")
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,2].set_title("total surplus", fontsize=title_fontsize)

# Set axis limits
margin_cs = 1.0 - np.sort(np.concatenate(tuple([normalize_var(cs_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]])])))[-6]
margin_ts = 1.0 - np.sort(np.concatenate(tuple([normalize_var(ts_bw(default_task_id)[:,bw_vals_argsort[bw_use][i]])])))[-6]
axs[0,0].set_ylim((1.0 - margin_cs, 1.0 + 0.33 * margin_cs))
axs[0,2].set_ylim((1.0 - margin_ts, 1.0 + 0.33 * margin_ts))

for i in range(3):
    axs[0,i].set_xticks(num_firms_array_extend)
    axs[0,i].set_yticks([])

# Legends
axs[0,0].legend(loc="best")
axs[0,1].legend(loc="best")
axs[0,2].legend(loc="best")

plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_bw_welfare_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Channel capacity as function of radius
radii = np.linspace(0.01, 4.0, 100)
hata_losses = infr.hata_loss(radii, infr.freq_rep, infr.height_rep)

# Path loss as function of radius graph
fig, axs = plt.subplots(1,1)

axs.plot(radii, infr.A0 - hata_losses, color="black")

axs.axhline(y=infr.JN_noise, color="black", linestyle="dashed")

df_inf = pd.read_csv(f"{paths.data_path}infrastructure_clean.csv", engine="python") # engine helps encoding, error with commune names, but doesn't matter b/c not used
df_inf = df_inf[df_inf['market'] > 0] # don't include Rest-of-France market
area = df_inf['area_effective'].values # adjusted commune area
stations = df_inf[[f"stations{i}" for i in range(1,5)]].values # number of base stations
radius = np.sqrt(area[:,np.newaxis] / stations / (np.sqrt(3.) * 3. / 2.)) # cell radius assuming homogeneous hexagonal cells, in km
axs.axvline(x=np.mean(radius[np.isfinite(radius)]), color="red", linestyle="dashed")

default_elast_id = paths.default_elast_id
default_nest_id = paths.default_nest_id
densities = lambda task_id: np.load(f"{paths.arrays_path}cntrfctl_densities_{task_id}.npy")
R_stars_dens = lambda task_id: np.load(f"{paths.arrays_path}R_stars_dens_{task_id}.npy")
axs.axvline(x=R_stars_dens(default_task_id)[3,np.argsort(densities(default_task_id))[1]], color="red", linestyle="dashed")
axs.axvline(x=R_stars_dens(default_task_id)[3,np.argsort(densities(default_task_id))[-2]], color="red", linestyle="dashed")

axs.set_ylim((-115.0, 0.0))
axs.set_xlabel("distance (km)")

if save_:
    plt.savefig(f"{paths.graphs_path}path_loss_radius.pdf", bbox_inches="tight", transparent=True)

if print_:
    plt.show()
    
channel_capacities = np.zeros(radii.shape)
gamma = np.load(f"{paths.arrays_path}cntrfctl_gamma.npy")[0]
bw = np.load(f"{paths.arrays_path}cntrfctl_bw_vals_e{default_elast_id}_n{default_nest_id}.npy")[0] / 4.0
for i, radius_ in enumerate(radii):
    channel_capacities[i] = infr.rho_C_hex(bw, radius_, gamma)

# Channel capacity graph
fig, axs = plt.subplots(1,1)

axs.plot(radii, channel_capacities, color="black")

low_R = R_stars_dens(default_task_id)[3,np.argsort(densities(default_task_id))[1]] # France density
high_R = R_stars_dens(default_task_id)[3,np.argsort(densities(default_task_id))[-2]] # France contraharmonic mean density
data_R = np.mean(radius[np.isfinite(radius)])
axs.axvline(x=data_R, color="grey", linestyle="dashed", alpha=0.65)
axs.axvline(x=low_R, color="grey", linestyle="dashed", alpha=0.65)
axs.axvline(x=high_R, color="grey", linestyle="dashed", alpha=0.65)

axs.set_ylim((36.0, 44.0))
axs.set_xlabel("distance (km)")
axs.set_ylabel("channel capacity (Mbps)")

axs.text(low_R + 0.1, channel_capacities[np.argmin(np.abs(radii - low_R))], "$R_{low\ density}^{*}$")
axs.text(high_R + 0.1, channel_capacities[np.argmin(np.abs(radii - high_R))] + 0.3, "$R_{high\ density}^{*}$")
axs.text(data_R + 0.1, channel_capacities[np.argmin(np.abs(radii - data_R))] + 0.075, "$R_{data}$")

if save_:
    plt.savefig(f"{paths.graphs_path}channel_capacity_radius.pdf", bbox_inches="tight", transparent=True)

if print_:
    plt.show()
    
# Process information about xis
xis = np.load(f"{paths.arrays_path}xis_e{default_elast_id}_n{default_nest_id}.npy")
xis_Org_stdev = np.std(xis[:,ds.firms == 1], ddof=1) # standard deviation of Orange xis
xis_across_firms = np.unique(xis[:,ds.firms != 1]) # non-Orange xis
thetahat = np.load(f"{paths.arrays_path}thetahat_e{default_elast_id}_n{default_nest_id}.npy")
xis_across_firms = np.concatenate((xis_across_firms, np.array([thetahat[coef.O]])))
xis_across_firms_stdev = np.std(xis_across_firms, ddof=1)
if save_:
    create_file(f"{paths.stats_path}xis_across_firms_stdev.tex", f"{xis_across_firms_stdev:.3}")
    create_file(f"{paths.stats_path}xis_Org_stdev.tex", f"{xis_Org_stdev:.3}")

# %%
# Difference in diversion ratios when average particular ways

xis = blp.xi(ds, thetahat, ds.data, None)
div_ratios = blp.div_ratio(ds, thetahat, ds.data, xis=xis)
div_ratios_alt = blp.div_ratio_numdenom(ds, thetahat, ds.data, xis=xis)
diff_div_ratios = np.abs(np.mean(div_ratios[1:]) + np.mean(div_ratios_alt[0][1:]) / np.mean(div_ratios_alt[1][1:]))
if save_:
    create_file(f"{paths.stats_path}diff_div_ratios_averaging.tex", f"{diff_div_ratios:.5f}")
    
# %%
# What discount rate would rationalize the bids based on our number for firm WTP for spectrum?

idx_4firms = 3
firm_wtp_monthly = partial_diffPif_partial_bf_allbw(default_task_id)[idx_4firms]
firm_wtp_discounted = 0.70
discount_factor_monthly = 1.0 - firm_wtp_monthly / firm_wtp_discounted
discount_factor_yearly = discount_factor_monthly**12.0
discount_rate_yearly = 1.0 / discount_factor_yearly - 1.0
if save_:
    create_file(f"{paths.stats_path}auction_firm_wtp_allbw.tex", f"{firm_wtp_monthly:.5f}")
    create_file(f"{paths.stats_path}auction_implied_discount_rate_allbw.tex", f"{discount_rate_yearly * 100.0:.2f}")
    
# %%
# Asymmetric equilibrium graph

fig, axs = plt.subplots(2, 3, figsize=(12.0,6.5), squeeze=False)

x_pos = np.arange(2)
x_ticklabels = ["symmetric", "aymmetric"]
patterns = ["", "/"]
labels = ["$\\frac{1}{3}$bw", "$\\frac{1}{6}$bw"]
alpha_pattern = [1.0, 0.6]
width = 0.4 # the width of the bars

# dlim = 1,000 prices
multiplier = 0.0
for i in range(2):
    offset = width * multiplier
    idx_use = np.ones(x_pos.shape, dtype=bool)
    patterns_use = patterns[i]
    if i == 0:
        offset = np.array([offset + 0.5 * width * (multiplier + 1.0), offset])
        patterns_use = np.copy(patterns)
        patterns_use[0] = patterns_use[1]
    else:
        idx_use[0] = False # don't plot the first one
    bars = axs[0,0].bar((x_pos + offset)[idx_use], p_stars_asymmetric_allbw(default_task_id)[:2,:,:][idx_use,i,0], width, yerr=1.96 * p_stars_asymmetric_allbw_se(default_task_id)[:2,:,:][idx_use,i,0], capsize=7.0, color="black", alpha=0.8 * alpha * alpha_pattern[i], hatch=patterns[i], edgecolor="black", label=labels[i])
    multiplier += 1.0
    for j, bar in enumerate(bars):  # loop over bars and hatches to set hatches in correct order
        if (i == 0) and (j == 0):
            bar.set_hatch(patterns[1])
axs[0,0].set_xticks(x_pos + 0.5 * width)
axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize, rotation_mode="anchor")
# axs[0,0].legend(loc="upper center", ncol=2)
axs[0,0].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
multiplier = 0.0
for i in range(2):
    offset = width * multiplier
    idx_use = np.ones(x_pos.shape, dtype=bool)
    patterns_use = patterns[i]
    if i == 0:
        offset = np.array([offset + 0.5 * width * (multiplier + 1.0), offset])
        patterns_use = np.copy(patterns)
        patterns_use[0] = patterns_use[1]
    else:
        idx_use[0] = False # don't plot the first one
    bars = axs[0,1].bar((x_pos + offset)[idx_use], p_stars_asymmetric_allbw(default_task_id)[:2,:,:][idx_use,i,1], width, yerr=1.96 * p_stars_asymmetric_allbw_se(default_task_id)[:2,:,:][idx_use,i,1], capsize=7.0, color="black", alpha=0.8 * alpha * alpha_pattern[i], hatch=patterns[i], edgecolor="black", label=labels[i])
    multiplier += 1.0
    for j, bar in enumerate(bars):  # loop over bars and hatches to set hatches in correct order
        if (i == 0) and (j == 0):
            bar.set_hatch(patterns[1])
axs[0,1].set_xticks(x_pos + 0.5 * width)
axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize, rotation_mode="anchor")
# axs[0,1].legend(loc="upper center", ncol=2)
axs[0,1].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

# download speeds
multiplier = 0.0
for i in range(2):
    offset = width * multiplier
    idx_use = np.ones(x_pos.shape, dtype=bool)
    patterns_use = patterns[i]
    if i == 0:
        offset = np.array([offset + 0.5 * width * (multiplier + 1.0), offset])
        patterns_use = np.copy(patterns)
        patterns_use[0] = patterns_use[1]
    else:
        idx_use[0] = False # don't plot the first one
    bars = axs[0,2].bar((x_pos + offset)[idx_use], q_stars_asymmetric_allbw(default_task_id)[:2,:][idx_use,i], width, yerr=1.96 * q_stars_asymmetric_allbw_se(default_task_id)[:2,:][idx_use,i], capsize=7.0, color="black", alpha=0.8 * alpha * alpha_pattern[i], hatch=patterns[i], edgecolor="black", label=labels[i])
    multiplier += 1.0
    for j, bar in enumerate(bars):  # loop over bars and hatches to set hatches in correct order
        if (i == 0) and (j == 0):
            bar.set_hatch(patterns[1])
axs[0,2].set_xticks(x_pos + 0.5 * width)
axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize, rotation_mode="anchor")
# axs[0,2].legend(loc="upper center", ncol=2)
axs[0,2].set_ylabel("$Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[0,2].set_title("download speeds", fontsize=title_fontsize)

# consumer surplus
axs[1,0].bar(x_pos, cs_asymmetric_allbw(default_task_id)[:2], color="black", alpha=0.8 * alpha)
# axs[1,0].bar(x_pos, cs_asymmetric_allbw(default_task_id), yerr=1.96 * cs_asymmetric_allbw_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,0].set_xticks(x_pos)
axs[1,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,0].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[1,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
axs[1,1].bar(x_pos, ps_asymmetric_allbw(default_task_id)[:2], color="black", alpha=0.8 * alpha)
# axs[1,1].bar(x_pos, ps_asymmetric_allbw(default_task_id), yerr=1.96 * ps_asymmetric_allbw_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,1].set_xticks(x_pos)
axs[1,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,1].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[1,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
axs[1,2].bar(x_pos, ts_asymmetric_allbw(default_task_id)[:2], color="black", alpha=0.8 * alpha)
# axs[1,2].bar(x_pos, ts_asymmetric_allbw(default_task_id), yerr=1.96 * ts_asymmetric_allbw_se(default_task_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,2].set_xticks(x_pos)
axs[1,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,2].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[1,2].set_title("total surplus", fontsize=title_fontsize)

# Set axis limits endogenous variables
for i in range(3): # first two columns
    axs[0,i].set_xlim((-0.25, 1.75))
    if i > 1:
        continue
    min_y_p = np.min(p_stars_asymmetric_allbw(default_task_id)[:2,:,:][:,:,i] - 1.96 * p_stars_asymmetric_allbw_se(default_task_id)[:2,:,:][:,:,i])
    max_y_p = np.max(p_stars_asymmetric_allbw(default_task_id)[:2,:,:][:,:,i] + 1.96 * p_stars_asymmetric_allbw_se(default_task_id)[:2,:,:][:,:,i])
    diff = max_y_p - min_y_p
    axs[0,i].set_ylim((min_y_p - 0.25 * diff, max_y_p + 0.25 * diff))
    
# Set axis limits endogenous variables
max_diff = 0.0
welfare_vars = [cs_asymmetric_allbw(default_task_id), ps_asymmetric_allbw(default_task_id), ts_asymmetric_allbw(default_task_id)]
for i, welfare_var in enumerate(welfare_vars): 
    diff = np.max(welfare_var) - np.min(welfare_var)
    if diff > max_diff:
        max_diff = diff
for i, welfare_var in enumerate(welfare_vars): 
    min_y = np.min(welfare_var)
    max_y = np.max(welfare_var)
    axs[1,i].set_ylim((min_y - 0.25 * max_diff, max_y + 0.25 * max_diff))

plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_asymmetric.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Asymmetric equilibrium table

to_tex = "\\begin{tabular}{l c c c c} \n"
to_tex += " & firm's & 1$\\,$000 MB plan & 10$\\,$000 MB plan & download  \\\\ \n" 
to_tex += " & bandwidth & price (in \euro{}) & price (in \euro{}) & speed (in Mbps) \\\\ \n"
to_tex += "\\hline \n"
to_tex += "symmetric allocation & & & &  \\\\ \n"
to_tex += "$\\quad$ equal allocation firm & $\\frac{1}{3}B$"
to_tex += f" & ${round_var(p_stars_asymmetric_allbw(default_task_id)[0,0,0], num_digits_round)}$ ${round_var(p_stars_asymmetric_allbw_se(default_task_id)[0,0,0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(p_stars_asymmetric_allbw(default_task_id)[0,0,1], num_digits_round)}$ ${round_var(p_stars_asymmetric_allbw_se(default_task_id)[0,0,1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(q_stars_asymmetric_allbw(default_task_id)[0,0], num_digits_round)}$ ${round_var(q_stars_asymmetric_allbw_se(default_task_id)[0,0], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += " & & & & \\\\ \n"
to_tex += "asymmetric allocation & & & & \\\\ \n"
to_tex += "$\\quad$ large allocation firm & $\\frac{1}{2}B$"
to_tex += f" & ${round_var(p_stars_asymmetric_allbw(default_task_id)[1,0,0], num_digits_round)}$ ${round_var(p_stars_asymmetric_allbw_se(default_task_id)[1,0,0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(p_stars_asymmetric_allbw(default_task_id)[1,0,1], num_digits_round)}$ ${round_var(p_stars_asymmetric_allbw_se(default_task_id)[1,0,1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(q_stars_asymmetric_allbw(default_task_id)[1,0], num_digits_round)}$ ${round_var(q_stars_asymmetric_allbw_se(default_task_id)[1,0], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "$\\quad$ small allocation firm & $\\frac{1}{4}B$"
to_tex += f" & ${round_var(p_stars_asymmetric_allbw(default_task_id)[1,1,0], num_digits_round)}$ ${round_var(p_stars_asymmetric_allbw_se(default_task_id)[1,1,0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(p_stars_asymmetric_allbw(default_task_id)[1,1,1], num_digits_round)}$ ${round_var(p_stars_asymmetric_allbw_se(default_task_id)[1,1,1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(q_stars_asymmetric_allbw(default_task_id)[1,1], num_digits_round)}$ ${round_var(q_stars_asymmetric_allbw_se(default_task_id)[1,1], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "\\hline \n" 
to_tex += " & & & & \\\\ \n"
to_tex += " & & $\\Delta$ CS & $\\Delta$ PS & $\\Delta$ TS \\\\ \n" 
to_tex += " & & (in \euro{}/person) & (in \euro{}/person) & (in \euro{}/person) \\\\ \n"
to_tex += "\\hline \n"
to_tex += "symmetric allocation & "
to_tex += f" & ${round_var(cs_asymmetric_allbw(default_task_id)[0], num_digits_round)}$ ${round_var(cs_asymmetric_allbw_se(default_task_id)[0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ps_asymmetric_allbw(default_task_id)[0], num_digits_round)}$ ${round_var(ps_asymmetric_allbw_se(default_task_id)[0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ts_asymmetric_allbw(default_task_id)[0], num_digits_round)}$ ${round_var(ts_asymmetric_allbw_se(default_task_id)[0], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "asymmetric allocation & "
to_tex += f" & ${round_var(cs_asymmetric_allbw(default_task_id)[1], num_digits_round)}$ ${round_var(cs_asymmetric_allbw_se(default_task_id)[1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ps_asymmetric_allbw(default_task_id)[1], num_digits_round)}$ ${round_var(ps_asymmetric_allbw_se(default_task_id)[1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ts_asymmetric_allbw(default_task_id)[1], num_digits_round)}$ ${round_var(ts_asymmetric_allbw_se(default_task_id)[1], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "difference & "
to_tex += f" & ${round_var(cs_asymmetric_allbw(default_task_id)[2], num_digits_round)}$ ${round_var(cs_asymmetric_allbw_se(default_task_id)[2], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ps_asymmetric_allbw(default_task_id)[2], num_digits_round)}$ ${round_var(ps_asymmetric_allbw_se(default_task_id)[2], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ts_asymmetric_allbw(default_task_id)[2], num_digits_round)}$ ${round_var(ts_asymmetric_allbw_se(default_task_id)[2], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "\\hline \n" 
to_tex += "\\end{tabular} \n"
if save_:
    create_file(f"{paths.tables_path}counterfactual_asymmetric.tex", to_tex)
if print_:
    print(to_tex)
    
# Save decrease in CS from 4 to 3 firms if asymmetric allocation
if save_:
    create_file(f"{paths.stats_path}four_to_three_asymmetric.tex", f"{-cs_asymmetric_allbw(default_task_id)[0]:.2f}")
    
# %%
# Short-run merger

# Construct arrays
mnos = ["Orange", "Bouygues", "Free", "SFR", "MVNO"]
mno_num = len(mnos) - 1
cs_shortrun = np.zeros((mno_num, mno_num))
triu_idx = np.triu_indices(mno_num, k=1)
tril_idx = np.tril_indices(mno_num, k=-1)
cs_shortrun[triu_idx[0], triu_idx[1]] = cs_shortrunall(default_task_id)[1:]
cs_shortrun[tril_idx] = cs_shortrun.T[tril_idx]
constructed_markets_idx = np.load(f"{paths.arrays_path}constructed_markets_idx.npy")
radius_mergers_all = np.load(f"{paths.arrays_path}radius_mergers_all.npy")
market_weights = np.load(f"{paths.arrays_path}mimic_market_weights.npy")
bw_4g_equiv = np.load(f"{paths.arrays_path}bw_4g_equiv.npy")
radius_mergers = np.average(radius_mergers_all[constructed_markets_idx,:,:], weights=market_weights, axis=0)
radius_shortrun = np.diag(radius_mergers[:,0])
ctr = 1
for i in range(radius_shortrun.shape[0]):
    for j in range(i, radius_shortrun.shape[0]):
        if i != j:
            radius_shortrun[i,j] = radius_mergers[i,ctr]
            radius_shortrun[j,i] = radius_mergers[i,ctr]
            ctr = ctr + 1
radius_shortrun_transform = np.zeros(radius_shortrun.shape)
for i in range(radius_shortrun.shape[0]):
    for j in range(radius_shortrun.shape[1]):
        if i != j: # a merger
            radius_mergers_all[constructed_markets_idx,i,0]
            radius_shortrun_transform[i,j] = np.average((radius_mergers_all[constructed_markets_idx,i,0]**2.0 + radius_mergers_all[constructed_markets_idx,j,0]**2.0)**-0.5, weights=market_weights)
        else:
            radius_shortrun_transform[i,j] = np.nan
            
bw_mergers = np.diag(np.average(bw_4g_equiv[constructed_markets_idx,:], weights=market_weights, axis=0))
for i in range(bw_mergers.shape[0]):
    for j in range(i, bw_mergers.shape[0]):
        if i != j:
            bw_ij = np.average(bw_4g_equiv[constructed_markets_idx,i] + bw_4g_equiv[constructed_markets_idx,j], weights=market_weights, axis=0)
            bw_mergers[i,j] = bw_ij
            bw_mergers[j,i] = bw_ij

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), squeeze=False)

minmin = np.nanmin([np.nanmin(cs_shortrun)])
maxmax = np.nanmax([np.nanmax(cs_shortrun)])
im_bw = axs[0,0].imshow(bw_mergers, vmin=np.nanmin(bw_mergers), vmax=np.nanmax(bw_mergers), cmap="Greys_r")
im_radius = axs[0,1].imshow(radius_shortrun, vmin=np.nanmin(radius_shortrun), vmax=np.nanmax(radius_shortrun), cmap="Greys_r")
im_shortrun = axs[0,2].imshow(cs_shortrun, vmin=np.nanmin(cs_shortrun), vmax=np.nanmax(cs_shortrun), cmap="Greys_r")

mno_names = np.array(mnos[:-1])
for i, mno in enumerate(mno_names):
    if mno == "Orange":
        mno_names[i] = "ORG"
    if mno == "SFR":
        mno_names[i] = "SFR"
    if mno == "Free":
        mno_names[i] = "FREE"
    if mno == "Bouygues":
        mno_names[i] = "BYG"
for i in range(3):
    axs[0,i].set_xticks(range(mno_num))
    axs[0,i].set_yticks(range(mno_num))
    axs[0,i].set_xticklabels(mno_names)
    axs[0,i].set_yticklabels(mno_names)
    
axs[0,0].set_title(f"bandwidth", fontsize=title_fontsize)
axs[0,1].set_title(f"radius", fontsize=title_fontsize)
axs[0,2].set_title(f"consumer surplus", fontsize=title_fontsize)

for ax_idx, var in enumerate([bw_mergers, radius_shortrun, cs_shortrun]):#enumerate([cs_shortrun, cs_longrun]):
    for i in range(var.shape[0]):
        for j in range(var.shape[1]):
            if np.isnan(var[i,j]):
                axs[0,ax_idx].plot([j-0.5, j+0.5], [i-0.5, i+0.5], color="black")
                axs[0,ax_idx].plot([j-0.5, j+0.5], [i+0.5, i-0.5], color="black")
            else:
                text_display = f"{var[i,j]:.1f}"
                if ax_idx == 0: # bandwidth
                    text_display = f"{var[i,j]:.0f}"
                text = axs[0,ax_idx].text(j, i, text_display, ha="center", va="center", color="black" if (var[i,j] > 0.67 * np.nanmin(var) + 0.33 * np.nanmax(var)) else "white")

# cbar_ax_radius = fig.add_axes([axs[0,0].get_position().x0, axs[0,0].get_position().y0 - 0.15, axs[0,0].get_position().width, 0.03])
# cbar_radius = fig.colorbar(im_radius, cax=cbar_ax_radius, orientation="horizontal")
# cbar_radius.set_label("km", labelpad=2, fontsize=y_fontsize)
# cbar_ax_shortrun = fig.add_axes([axs[0,1].get_position().x0, axs[0,1].get_position().y0 - 0.15, axs[0,1].get_position().width, 0.03])
# cbar_shortrun = fig.colorbar(im_shortrun, cax=cbar_ax_shortrun, orientation="horizontal")
# cbar_shortrun.set_label("\u20ac / person", labelpad=2, fontsize=y_fontsize)

pad = 0.06
colorbar_height = 0.02
for i, (ax, im) in enumerate(zip(axs.flatten(), [im_bw, im_radius, im_shortrun])):
    pos = ax.get_position()
    addl_horizontal = 0.0
    if i == 0: # bandwidth
        addl_horizontal = -0.04
    if i == 1: # radius
        addl_horizontal = 0.015
    if i == 2: # consumer surplus
        addl_horizontal = 0.07
    cax = fig.add_axes([pos.x0 + addl_horizontal, pos.y0 - pad - colorbar_height, pos.width, colorbar_height])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("\u20ac / person", labelpad=2, fontsize=10)
    if i == 0:
        cbar.set_label("MHz", labelpad=2, fontsize=y_fontsize)
    elif i == 1: 
        cbar.set_label("km", labelpad=2, fontsize=y_fontsize)
    elif i == 2:
        cbar.set_label("\u20ac / person", labelpad=2, fontsize=y_fontsize)
    
plt.tight_layout(rect=[0, 0.1, 1, 1])

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_shortrunmerger.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
merger_combos = [list(combo) for combo in combinations(mnos[:-1], 2)]
min_decrease = np.argmax(cs_shortrunall(default_task_id)[1:])
min_decrease_mnos = merger_combos[min_decrease]
max_decrease = np.argmin(cs_shortrunall(default_task_id)[1:])
max_decrease_mnos = merger_combos[max_decrease]
if save_:
    create_file(f"{paths.stats_path}min_decrease_merger.tex", f"{-cs_shortrunall(default_task_id)[1:][min_decrease]:.2f}")
    create_file(f"{paths.stats_path}min_decrease_mnos_merger.tex", f"{min_decrease_mnos[0]} and {min_decrease_mnos[1]}")
    create_file(f"{paths.stats_path}max_decrease_merger.tex", f"{-cs_shortrunall(default_task_id)[1:][max_decrease]:.2f}")
    create_file(f"{paths.stats_path}max_decrease_mnos_merger.tex", f"{max_decrease_mnos[0]} and {max_decrease_mnos[1]}")
    
population_categories = np.load(f"{paths.arrays_path}mimic_market_population_categories.npy")[:-1]
population_categories = [f"{int(pop):,}".replace(",", "\\,") for pop in population_categories]
population_categories_text = ""
for i in range(len(population_categories)+1):
    if i == 0:
        population_categories_text += f"$<${population_categories[i]}, "
    elif i < len(population_categories):
        population_categories_text += f"{population_categories[i-1]}--{population_categories[i]}, "
    else:
        population_categories_text += f"and $\\geq${population_categories[i-1]}"
        
market_weights = np.load(f"{paths.arrays_path}mimic_market_weights.npy")
market_weights_text = ""
for i, weight in enumerate(market_weights):
    if i < market_weights.shape[0] - 1:
        market_weights_text += f"{(weight * 100.0):.2f}\\%, "
    else:
        market_weights_text += f"and {(weight * 100.0):.2f}\\%"
market_weights_text

market_names = np.load(f"{paths.arrays_path}mimic_market_names.npy")
market_names_text = ""
for i, name in enumerate(market_names):
    if i < market_names.shape[0] - 1:
        market_names_text += f"{name}, "
    else:
        market_names_text += f"and {name}"
market_names_text = market_names_text.replace("É", "\\\'{E}")
        
if save_:
    create_file(f"{paths.stats_path}merger_population_categories.tex", population_categories_text)
    create_file(f"{paths.stats_path}merger_market_weights.tex", market_weights_text)
    create_file(f"{paths.stats_path}merger_market_names.tex", market_names_text)
    
# %%
# Compare results for different allocations of MVNO

demand_parameters = "\\begin{tabular}{l c c c c c c c c c c c c c c c c} \\hline \n" 
demand_parameters += "Specification & & $\\hat{\\theta}_{p0}$ & & $\\hat{\\theta}_{pz}$ & & $\\hat{\\theta}_{v}$ & & $\\hat{\\theta}_{O}$ & & $\\hat{\\theta}_{d 0}$ & & $\\hat{\\theta}_{d z}$ & & $\\hat{\\log\\left(\\theta_{c}\\right)}$ & & $\\hat{\\sigma}$ \\\\ \n"
demand_parameters += "\\cline{3-3} \\cline{5-5} \\cline{7-7} \\cline{9-9} \\cline{11-11} \\cline{13-13} \\cline{15-15} \\cline{17-17} \n"
for i, mvno_description in enumerate(["original", "expanded MVNOs"]):
    demand_parameters += f"$\\quad${mvno_description} & & " + " & & ".join(f"${param:.3f}$" for param in np.load(f"{paths.arrays_path}thetahat_{i}.npy")[:-1]) # not \theta_\sigma
    transform_theta_sigma_est = np.load(f"{paths.arrays_path}thetahat_{i}.npy")[-1]
    theta_sigma_est = np.exp(transform_theta_sigma_est) / (1.0 + np.exp(transform_theta_sigma_est))
    demand_parameters += f" & & ${theta_sigma_est:.3f}$"
    demand_parameters += " \\\\ \n"
demand_parameters += "\\hline \n"
demand_parameters += "\\end{tabular} \n"
create_file(f"{paths.tables_path}demand_parameters_diff_specs.tex", demand_parameters)
print(demand_parameters)

counterfactual_results = "\\begin{tabular}{l c c c c c c c c c c c c} \\hline \n"
#counterfactual_results += "Specification & & 1 GB $P^{*}_{n=4}$ & & 10 GB $P^{*}_{n=4}$ & & $Q^{*}_{n=4}$ & & $\\begin{array}{c}\nCS_{n=4} - \\\\ CS_{n=3}\n\\end{array}$ & & $\\begin{array}{c}\n\\arg\\max_{n} \\\\ \\{CS_{n}\\}\n\\end{array}$ & & $\\begin{array}{c}\n\\arg\\max_{n} \\\\ \\{TS_{n}\\}\n\\end{array}$ \\\\ \n"
counterfactual_results += "Specification & & $\\begin{array}{c}\nP^{*, 1\\text{GB}}_{n=4} - \\\\ P^{*, 1\\text{GB}}_{n=3}\n\\end{array}$ & & $\\begin{array}{c}\nP^{*, 10\\text{GB}}_{n=4} - \\\\ P^{*, 10\\text{GB}}_{n=3}\n\\end{array}$ & & $\\begin{array}{c}\n Q^{*}_{n=4} - \\\\ Q^{*}_{n=3}\n\\end{array}$ & & $\\begin{array}{c}\nCS_{n=4} - \\\\ CS_{n=3}\n\\end{array}$ & & $\\begin{array}{c}\n\\arg\\max_{n} \\\\ \\{CS_{n}\\}\n\\end{array}$ & & $\\begin{array}{c}\n\\arg\\max_{n} \\\\ \\{TS_{n}\\}\n\\end{array}$ \\\\ \n"
counterfactual_results += "\\cline{3-3} \\cline{5-5} \\cline{7-7} \\cline{9-9} \\cline{11-11} \\cline{13-13} \n"
for i, mvno_description in enumerate(["original", "host MNO"]):
    counterfactual_results += f"$\\quad${mvno_description} "
    counterfactual_results += f" & & ${(p_stars(i)[:,0][3] - p_stars(i)[:,0][2]):.3f}$"
    counterfactual_results += f" & & ${(p_stars(i)[:,1][3] - p_stars(i)[:,1][2]):.3f}$"
    counterfactual_results += f" & & ${(q_stars(i)[3] - q_stars(i)[2]):.3f}$"
    counterfactual_results += f" & & ${(cs(i)[3] - cs(i)[2]):.3f}$"
    counterfactual_results += f" & & ${(np.argmax(cs(i)) + 1)}$"
    counterfactual_results += f" & & ${(np.argmax(ts(i)) + 1)}$"
    counterfactual_results += " \\\\ \n"
counterfactual_results += "\\hline \n"
counterfactual_results += "\\end{tabular} \n"
create_file(f"{paths.tables_path}counterfactual_results_diff_specs.tex", counterfactual_results)
print(counterfactual_results)

# %%
# Check for evidence of multiplicity

success_test = np.load(f"{paths.arrays_path}success_test.npy")
p_stars_test = np.load(f"{paths.arrays_path}p_stars_test.npy")
R_stars_test = np.load(f"{paths.arrays_path}R_stars_test.npy")
success_test = success_test & np.all(R_stars_test > 0.0, axis=0) & np.all(R_stars_test < 5.0, axis=0) & np.all(p_stars_test < 200.0, axis=0)

print(np.round(p_stars_test[:,success_test], 5))
print(np.round(R_stars_test[:,success_test], 5))

print(f"Are any values different?")
print(f"\tprices: {not np.allclose(p_stars_test[:,success_test], p_stars_test[:,success_test][:,0][:,np.newaxis])}")
print(f"\tradii: {not np.allclose(R_stars_test[:,success_test], R_stars_test[:,success_test][:,0][:,np.newaxis])}")
