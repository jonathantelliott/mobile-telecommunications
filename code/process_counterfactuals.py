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
with open(f"{paths.data_path}demandsystem.obj", "rb") as file_ds:
    ds = pickle.load(file_ds)
    
# Drop Rest-of-France market
market_idx = ds.dim3.index(ds.marketname)
market_numbers = np.max(ds.data[:,:,market_idx], axis=1)
ds.data = ds.data[market_numbers > 0,:,:] # drop "Rest of France"# %%

# %%
# Define functions to load results
c_u = lambda x,y: np.load(f"{paths.arrays_path}c_u_e{x}_n{y}.npy")
c_R = lambda x,y: np.load(f"{paths.arrays_path}c_R_e{x}_n{y}.npy")

p_stars = lambda x,y: np.load(f"{paths.arrays_path}p_stars_e{x}_n{y}.npy")
R_stars = lambda x,y: np.load(f"{paths.arrays_path}R_stars_e{x}_n{y}.npy")
num_stations_stars = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_e{x}_n{y}.npy")
num_stations_per_firm_stars = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_e{x}_n{y}.npy")
q_stars = lambda x,y: np.load(f"{paths.arrays_path}q_stars_e{x}_n{y}.npy")
cs_by_type = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_e{x}_n{y}.npy")
cs = lambda x,y: np.load(f"{paths.arrays_path}cs_e{x}_n{y}.npy")
ps = lambda x,y: np.load(f"{paths.arrays_path}ps_e{x}_n{y}.npy")
ts = lambda x,y: np.load(f"{paths.arrays_path}ts_e{x}_n{y}.npy")
ccs = lambda x,y: np.load(f"{paths.arrays_path}ccs_e{x}_n{y}.npy")
ccs_per_bw = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_e{x}_n{y}.npy")
avg_path_losses = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_e{x}_n{y}.npy")
avg_SINR = lambda x,y: np.load(f"{paths.arrays_path}avg_SINR_e{x}_n{y}.npy")
partial_elasts = lambda x,y: np.load(f"{paths.arrays_path}partial_elasts_e{x}_n{y}.npy")
full_elasts = lambda x,y: np.load(f"{paths.arrays_path}full_elasts_e{x}_n{y}.npy")
p_stars_allfixed = lambda x,y: np.load(f"{paths.arrays_path}p_stars_allfixed_e{x}_n{y}.npy")
R_stars_allfixed = lambda x,y: np.load(f"{paths.arrays_path}R_stars_allfixed_e{x}_n{y}.npy")
num_stations_stars_allfixed = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_allfixed_e{x}_n{y}.npy")
num_stations_per_firm_stars_allfixed = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_allfixed_e{x}_n{y}.npy")
q_stars_allfixed = lambda x,y: np.load(f"{paths.arrays_path}q_stars_allfixed_e{x}_n{y}.npy")
cs_by_type_allfixed = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_allfixed_e{x}_n{y}.npy")
cs_allfixed = lambda x,y: np.load(f"{paths.arrays_path}cs_allfixed_e{x}_n{y}.npy")
ps_allfixed = lambda x,y: np.load(f"{paths.arrays_path}ps_allfixed_e{x}_n{y}.npy")
ts_allfixed = lambda x,y: np.load(f"{paths.arrays_path}ts_allfixed_e{x}_n{y}.npy")
ccs_allfixed = lambda x,y: np.load(f"{paths.arrays_path}ccs_allfixed_e{x}_n{y}.npy")
ccs_per_bw_allfixed = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_allfixed_e{x}_n{y}.npy")
avg_path_losses_allfixed = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_allfixed_e{x}_n{y}.npy")
avg_SINR_allfixed = lambda x,y: np.load(f"{paths.arrays_path}avg_SINR_allfixed_e{x}_n{y}.npy")
partial_Pif_partial_bf_allfixed = lambda x,y: np.load(f"{paths.arrays_path}partial_Pif_partial_bf_allfixed_e{x}_n{y}.npy")
partial_Piotherf_partial_bf_allfixed = lambda x,y: np.load(f"{paths.arrays_path}partial_Piotherf_partial_bf_allfixed_e{x}_n{y}.npy")
partial_diffPif_partial_bf_allfixed = lambda x,y: np.load(f"{paths.arrays_path}partial_diffPif_partial_bf_allfixed_e{x}_n{y}.npy")
partial_Pif_partial_b_allfixed = lambda x,y: np.load(f"{paths.arrays_path}partial_Pif_partial_b_allfixed_e{x}_n{y}.npy")
partial_CS_partial_b_allfixed = lambda x,y: np.load(f"{paths.arrays_path}partial_CS_partial_b_allfixed_e{x}_n{y}.npy")
p_stars_shortrun = lambda x,y: np.load(f"{paths.arrays_path}p_stars_shortrun_e{x}_n{y}.npy")
R_stars_shortrun = lambda x,y: np.load(f"{paths.arrays_path}R_stars_shortrun_e{x}_n{y}.npy")
num_stations_stars_shortrun = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_shortrun_e{x}_n{y}.npy")
num_stations_per_firm_stars_shortrun = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_shortrun_e{x}_n{y}.npy")
q_stars_shortrun = lambda x,y: np.load(f"{paths.arrays_path}q_stars_shortrun_e{x}_n{y}.npy")
cs_by_type_shortrun = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_shortrun_e{x}_n{y}.npy")
cs_shortrun = lambda x,y: np.load(f"{paths.arrays_path}cs_shortrun_e{x}_n{y}.npy")
ps_shortrun = lambda x,y: np.load(f"{paths.arrays_path}ps_shortrun_e{x}_n{y}.npy")
ts_shortrun = lambda x,y: np.load(f"{paths.arrays_path}ts_shortrun_e{x}_n{y}.npy")
ccs_shortrun = lambda x,y: np.load(f"{paths.arrays_path}ccs_shortrun_e{x}_n{y}.npy")
ccs_per_bw_shortrun = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_shortrun_e{x}_n{y}.npy")
avg_path_losses_shortrun = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_shortrun_e{x}_n{y}.npy")
p_stars_free_allfixed = lambda x,y: np.load(f"{paths.arrays_path}p_stars_free_allfixed_e{x}_n{y}.npy")
R_stars_free_allfixed = lambda x,y: np.load(f"{paths.arrays_path}R_stars_free_allfixed_e{x}_n{y}.npy")
num_stations_stars_free_allfixed = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_free_allfixed_e{x}_n{y}.npy")
num_stations_per_firm_stars_free_allfixed = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_free_allfixed_e{x}_n{y}.npy")
q_stars_free_allfixed = lambda x,y: np.load(f"{paths.arrays_path}q_stars_free_allfixed_e{x}_n{y}.npy")
cs_by_type_free_allfixed = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_free_allfixed_e{x}_n{y}.npy")
cs_free_allfixed = lambda x,y: np.load(f"{paths.arrays_path}cs_free_allfixed_e{x}_n{y}.npy")
ps_free_allfixed = lambda x,y: np.load(f"{paths.arrays_path}ps_free_allfixed_e{x}_n{y}.npy")
ts_free_allfixed = lambda x,y: np.load(f"{paths.arrays_path}ts_free_allfixed_e{x}_n{y}.npy")
ccs_free_allfixed = lambda x,y: np.load(f"{paths.arrays_path}ccs_free_allfixed_e{x}_n{y}.npy")
ccs_per_bw_free_allfixed = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_free_allfixed_e{x}_n{y}.npy")
avg_path_losses_free_allfixed = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_free_allfixed_e{x}_n{y}.npy")
partial_Pif_partial_bf_allbw = lambda x,y: np.load(f"{paths.arrays_path}partial_Pif_partial_bf_allbw_e{x}_n{y}.npy")
partial_Piotherf_partial_bf_allbw = lambda x,y: np.load(f"{paths.arrays_path}partial_Piotherf_partial_bf_allbw_e{x}_n{y}.npy")
partial_diffPif_partial_bf_allbw = lambda x,y: np.load(f"{paths.arrays_path}partial_diffPif_partial_bf_allbw_e{x}_n{y}.npy")
partial_Pif_partial_b_allbw = lambda x,y: np.load(f"{paths.arrays_path}partial_Pif_partial_b_allbw_e{x}_n{y}.npy")
partial_CS_partial_b_allbw = lambda x,y: np.load(f"{paths.arrays_path}partial_CS_partial_b_allbw_e{x}_n{y}.npy")
p_stars_free_allbw = lambda x,y: np.load(f"{paths.arrays_path}p_stars_free_allbw_e{x}_n{y}.npy")
R_stars_free_allbw = lambda x,y: np.load(f"{paths.arrays_path}R_stars_free_allbw_e{x}_n{y}.npy")
num_stations_stars_free_allbw = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_free_allbw_e{x}_n{y}.npy")
num_stations_per_firm_stars_free_allbw = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_free_allbw_e{x}_n{y}.npy")
q_stars_free_allbw = lambda x,y: np.load(f"{paths.arrays_path}q_stars_free_allbw_e{x}_n{y}.npy")
cs_by_type_free_allbw = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_free_allbw_e{x}_n{y}.npy")
cs_free_allbw = lambda x,y: np.load(f"{paths.arrays_path}cs_free_allbw_e{x}_n{y}.npy")
ps_free_allbw = lambda x,y: np.load(f"{paths.arrays_path}ps_free_allbw_e{x}_n{y}.npy")
ts_free_allbw = lambda x,y: np.load(f"{paths.arrays_path}ts_free_allbw_e{x}_n{y}.npy")
ccs_free_allbw = lambda x,y: np.load(f"{paths.arrays_path}ccs_free_allbw_e{x}_n{y}.npy")
ccs_per_bw_free_allbw = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_free_allbw_e{x}_n{y}.npy")
avg_path_losses_free_allbw = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_free_allbw_e{x}_n{y}.npy")
p_stars_dens = lambda x,y: np.load(f"{paths.arrays_path}p_stars_dens_e{x}_n{y}.npy")
R_stars_dens = lambda x,y: np.load(f"{paths.arrays_path}R_stars_dens_e{x}_n{y}.npy")
num_stations_stars_dens = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_dens_e{x}_n{y}.npy")
num_stations_per_firm_stars_dens = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_dens_e{x}_n{y}.npy")
q_stars_dens = lambda x,y: np.load(f"{paths.arrays_path}q_stars_dens_e{x}_n{y}.npy")
cs_by_type_dens = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_dens_e{x}_n{y}.npy")
cs_dens = lambda x,y: np.load(f"{paths.arrays_path}cs_dens_e{x}_n{y}.npy")
ps_dens = lambda x,y: np.load(f"{paths.arrays_path}ps_dens_e{x}_n{y}.npy")
ts_dens = lambda x,y: np.load(f"{paths.arrays_path}ts_dens_e{x}_n{y}.npy")
ccs_dens = lambda x,y: np.load(f"{paths.arrays_path}ccs_dens_e{x}_n{y}.npy")
ccs_per_bw_dens = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_dens_e{x}_n{y}.npy")
avg_path_losses_dens = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_dens_e{x}_n{y}.npy")
avg_SINR_dens = lambda x,y: np.load(f"{paths.arrays_path}avg_SINR_dens_e{x}_n{y}.npy")
p_stars_bw = lambda x,y: np.load(f"{paths.arrays_path}p_stars_bw_e{x}_n{y}.npy")
R_stars_bw = lambda x,y: np.load(f"{paths.arrays_path}R_stars_bw_e{x}_n{y}.npy")
num_stations_stars_bw = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_bw_e{x}_n{y}.npy")
num_stations_per_firm_stars_bw = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_bw_e{x}_n{y}.npy")
q_stars_bw = lambda x,y: np.load(f"{paths.arrays_path}q_stars_bw_e{x}_n{y}.npy")
cs_by_type_bw = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_bw_e{x}_n{y}.npy")
cs_bw = lambda x,y: np.load(f"{paths.arrays_path}cs_bw_e{x}_n{y}.npy")
ps_bw = lambda x,y: np.load(f"{paths.arrays_path}ps_bw_e{x}_n{y}.npy")
ts_bw = lambda x,y: np.load(f"{paths.arrays_path}ts_bw_e{x}_n{y}.npy")
ccs_bw = lambda x,y: np.load(f"{paths.arrays_path}ccs_bw_e{x}_n{y}.npy")
ccs_per_bw_bw = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_bw_e{x}_n{y}.npy")
avg_path_losses_bw = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_bw_e{x}_n{y}.npy")
avg_SINR_bw = lambda x,y: np.load(f"{paths.arrays_path}avg_SINR_bw_e{x}_n{y}.npy")
p_stars_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}p_stars_dens_1p_e{x}_n{y}.npy")
R_stars_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}R_stars_dens_1p_e{x}_n{y}.npy")
num_stations_stars_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_dens_1p_e{x}_n{y}.npy")
num_stations_per_firm_stars_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_dens_1p_e{x}_n{y}.npy")
q_stars_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}q_stars_dens_1p_e{x}_n{y}.npy")
cs_by_type_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_dens_1p_e{x}_n{y}.npy")
cs_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}cs_dens_1p_e{x}_n{y}.npy")
ps_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}ps_dens_1p_e{x}_n{y}.npy")
ts_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}ts_dens_1p_e{x}_n{y}.npy")
ccs_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}ccs_dens_1p_e{x}_n{y}.npy")
ccs_per_bw_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_dens_1p_e{x}_n{y}.npy")
avg_path_losses_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_dens_1p_e{x}_n{y}.npy")
avg_SINR_dens_1p = lambda x,y: np.load(f"{paths.arrays_path}avg_SINR_dens_1p_e{x}_n{y}.npy")
per_user_costs = lambda x,y: np.load(f"{paths.arrays_path}per_user_costs_e{x}_n{y}.npy")

c_u_se = lambda x,y: np.load(f"{paths.arrays_path}c_u_se_e{x}_n{y}.npy")
c_R_se = lambda x,y: np.load(f"{paths.arrays_path}c_R_se_e{x}_n{y}.npy")

p_stars_se = lambda x,y: np.load(f"{paths.arrays_path}p_stars_se_e{x}_n{y}.npy")
R_stars_se = lambda x,y: np.load(f"{paths.arrays_path}R_stars_se_e{x}_n{y}.npy")
num_stations_stars_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_se_e{x}_n{y}.npy")
num_stations_per_firm_stars_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_se_e{x}_n{y}.npy")
q_stars_se = lambda x,y: np.load(f"{paths.arrays_path}q_stars_se_e{x}_n{y}.npy")
cs_by_type_se = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_se_e{x}_n{y}.npy")
cs_se = lambda x,y: np.load(f"{paths.arrays_path}cs_se_e{x}_n{y}.npy")
ps_se = lambda x,y: np.load(f"{paths.arrays_path}ps_se_e{x}_n{y}.npy")
ts_se = lambda x,y: np.load(f"{paths.arrays_path}ts_se_e{x}_n{y}.npy")
ccs_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_se_e{x}_n{y}.npy")
ccs_per_bw_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_se_e{x}_n{y}.npy")
avg_path_losses_se = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_se_e{x}_n{y}.npy")
partial_elasts_se = lambda x,y: np.load(f"{paths.arrays_path}partial_elasts_se_e{x}_n{y}.npy")
full_elasts_se = lambda x,y: np.load(f"{paths.arrays_path}full_elasts_se_e{x}_n{y}.npy")
p_stars_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}p_stars_allfixed_se_e{x}_n{y}.npy")
R_stars_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}R_stars_allfixed_se_e{x}_n{y}.npy")
num_stations_stars_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_allfixed_se_e{x}_n{y}.npy")
num_stations_per_firm_stars_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_allfixed_se_e{x}_n{y}.npy")
q_stars_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}q_stars_allfixed_se_e{x}_n{y}.npy")
cs_by_type_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_allfixed_se_e{x}_n{y}.npy")
cs_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}cs_allfixed_se_e{x}_n{y}.npy")
ps_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}ps_allfixed_se_e{x}_n{y}.npy")
ts_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}ts_allfixed_se_e{x}_n{y}.npy")
ccs_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_allfixed_se_e{x}_n{y}.npy")
ccs_per_bw_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_allfixed_se_e{x}_n{y}.npy")
avg_path_losses_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_allfixed_se_e{x}_n{y}.npy")
partial_Pif_partial_bf_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}partial_Pif_partial_bf_allfixed_se_e{x}_n{y}.npy")
partial_Piotherf_partial_bf_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}partial_Piotherf_partial_bf_allfixed_se_e{x}_n{y}.npy")
partial_diffPif_partial_bf_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}partial_diffPif_partial_bf_allfixed_se_e{x}_n{y}.npy")
partial_Pif_partial_b_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}partial_Pif_partial_b_allfixed_se_e{x}_n{y}.npy")
partial_CS_partial_b_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}partial_CS_partial_b_allfixed_se_e{x}_n{y}.npy")
p_stars_shortrun_se = lambda x,y: np.load(f"{paths.arrays_path}p_stars_shortrun_se_e{x}_n{y}.npy")
R_stars_shortrun_se = lambda x,y: np.load(f"{paths.arrays_path}R_stars_shortrun_se_e{x}_n{y}.npy")
num_stations_stars_shortrun_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_shortrun_se_e{x}_n{y}.npy")
num_stations_per_firm_stars_shortrun_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_shortrun_se_e{x}_n{y}.npy")
q_stars_shortrun_se = lambda x,y: np.load(f"{paths.arrays_path}q_stars_shortrun_se_e{x}_n{y}.npy")
cs_by_type_shortrun_se = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_shortrun_se_e{x}_n{y}.npy")
cs_shortrun_se = lambda x,y: np.load(f"{paths.arrays_path}cs_shortrun_se_e{x}_n{y}.npy")
ps_shortrun_se = lambda x,y: np.load(f"{paths.arrays_path}ps_shortrun_se_e{x}_n{y}.npy")
ts_shortrun_se = lambda x,y: np.load(f"{paths.arrays_path}ts_shortrun_se_e{x}_n{y}.npy")
ccs_shortrun_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_shortrun_se_e{x}_n{y}.npy")
ccs_per_bw_shortrun_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_shortrun_se_e{x}_n{y}.npy")
avg_path_losses_shortrun_se = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_shortrun_se_e{x}_n{y}.npy")
p_stars_free_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}p_stars_free_allfixed_se_e{x}_n{y}.npy")
R_stars_free_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}R_stars_free_allfixed_se_e{x}_n{y}.npy")
num_stations_stars_free_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_free_allfixed_se_e{x}_n{y}.npy")
num_stations_per_firm_stars_free_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_free_allfixed_se_e{x}_n{y}.npy")
q_stars_free_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}q_stars_free_allfixed_se_e{x}_n{y}.npy")
cs_by_type_free_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_free_allfixed_se_e{x}_n{y}.npy")
cs_free_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}cs_free_se_allfixed_e{x}_n{y}.npy")
ps_free_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}ps_free_se_allfixed_e{x}_n{y}.npy")
ts_free_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}ts_free_se_allfixed_e{x}_n{y}.npy")
ccs_free_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_free_allfixed_se_e{x}_n{y}.npy")
ccs_per_bw_free_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_free_allfixed_se_e{x}_n{y}.npy")
avg_path_losses_free_allfixed_se = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_free_allfixed_se_e{x}_n{y}.npy")
partial_Pif_partial_bf_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}partial_Pif_partial_bf_allbw_se_e{x}_n{y}.npy")
partial_Piotherf_partial_bf_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}partial_Piotherf_partial_bf_allbw_se_e{x}_n{y}.npy")
partial_diffPif_partial_bf_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}partial_diffPif_partial_bf_allbw_se_e{x}_n{y}.npy")
partial_Pif_partial_b_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}partial_Pif_partial_b_allbw_se_e{x}_n{y}.npy")
partial_CS_partial_b_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}partial_CS_partial_b_allbw_se_e{x}_n{y}.npy")
p_stars_free_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}p_stars_free_allbw_se_e{x}_n{y}.npy")
R_stars_free_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}R_stars_free_allbw_se_e{x}_n{y}.npy")
num_stations_stars_free_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_free_allbw_se_e{x}_n{y}.npy")
num_stations_per_firm_stars_free_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_free_allbw_se_e{x}_n{y}.npy")
q_stars_free_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}q_stars_free_allbw_se_e{x}_n{y}.npy")
cs_by_type_free_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_free_allbw_se_e{x}_n{y}.npy")
cs_free_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}cs_free_se_allbw_e{x}_n{y}.npy")
ps_free_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}ps_free_se_allbw_e{x}_n{y}.npy")
ts_free_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}ts_free_se_allbw_e{x}_n{y}.npy")
ccs_free_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_free_allbw_se_e{x}_n{y}.npy")
ccs_per_bw_free_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_free_allbw_se_e{x}_n{y}.npy")
avg_path_losses_free_allbw_se = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_free_allbw_se_e{x}_n{y}.npy")
p_stars_dens_se = lambda x,y: np.load(f"{paths.arrays_path}p_stars_dens_se_e{x}_n{y}.npy")
R_stars_dens_se = lambda x,y: np.load(f"{paths.arrays_path}R_stars_dens_se_e{x}_n{y}.npy")
num_stations_stars_dens_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_dens_se_e{x}_n{y}.npy")
num_stations_per_firm_stars_dens_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_dens_se_e{x}_n{y}.npy")
q_stars_dens_se = lambda x,y: np.load(f"{paths.arrays_path}q_stars_dens_se_e{x}_n{y}.npy")
cs_by_type_dens_se = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_dens_se_e{x}_n{y}.npy")
cs_dens_se = lambda x,y: np.load(f"{paths.arrays_path}cs_dens_se_e{x}_n{y}.npy")
ps_dens_se = lambda x,y: np.load(f"{paths.arrays_path}ps_dens_se_e{x}_n{y}.npy")
ts_dens_se = lambda x,y: np.load(f"{paths.arrays_path}ts_dens_se_e{x}_n{y}.npy")
ccs_dens_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_dens_se_e{x}_n{y}.npy")
ccs_per_bw_dens_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_dens_se_e{x}_n{y}.npy")
avg_path_losses_dens_se = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_dens_se_e{x}_n{y}.npy")
p_stars_bw_se = lambda x,y: np.load(f"{paths.arrays_path}p_stars_bw_se_e{x}_n{y}.npy")
R_stars_bw_se = lambda x,y: np.load(f"{paths.arrays_path}R_stars_bw_se_e{x}_n{y}.npy")
num_stations_stars_bw_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_bw_se_e{x}_n{y}.npy")
num_stations_per_firm_stars_bw_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_bw_se_e{x}_n{y}.npy")
q_stars_bw_se = lambda x,y: np.load(f"{paths.arrays_path}q_stars_bw_se_e{x}_n{y}.npy")
cs_by_type_bw_se = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_bw_se_e{x}_n{y}.npy")
cs_bw_se = lambda x,y: np.load(f"{paths.arrays_path}cs_bw_se_e{x}_n{y}.npy")
ps_bw_se = lambda x,y: np.load(f"{paths.arrays_path}ps_bw_se_e{x}_n{y}.npy")
ts_bw_se = lambda x,y: np.load(f"{paths.arrays_path}ts_bw_se_e{x}_n{y}.npy")
ccs_bw_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_bw_se_e{x}_n{y}.npy")
ccs_per_bw_bw_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_bw_se_e{x}_n{y}.npy")
avg_path_losses_bw_se = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_bw_se_e{x}_n{y}.npy")
p_stars_dens_1p_se = lambda x,y: np.load(f"{paths.arrays_path}p_stars_dens_1p_se_e{x}_n{y}.npy")
R_stars_dens_1p_se = lambda x,y: np.load(f"{paths.arrays_path}R_stars_dens_1p_se_e{x}_n{y}.npy")
num_stations_stars_dens_1p_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_stars_dens_1p_se_e{x}_n{y}.npy")
num_stations_per_firm_stars_dens_1p_se = lambda x,y: np.load(f"{paths.arrays_path}num_stations_per_firm_stars_dens_1p_se_e{x}_n{y}.npy")
q_stars_dens_1p_se = lambda x,y: np.load(f"{paths.arrays_path}q_stars_dens_1p_se_e{x}_n{y}.npy")
cs_by_type_dens_1p_se = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_dens_1p_se_e{x}_n{y}.npy")
cs_dens_1p_se = lambda x,y: np.load(f"{paths.arrays_path}cs_dens_1p_se_e{x}_n{y}.npy")
ps_dens_1p_se = lambda x,y: np.load(f"{paths.arrays_path}ps_dens_1p_se_e{x}_n{y}.npy")
ts_dens_1p_se = lambda x,y: np.load(f"{paths.arrays_path}ts_dens_1p_se_e{x}_n{y}.npy")
ccs_dens_1p_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_dens_1p_se_e{x}_n{y}.npy")
ccs_per_bw_dens_1p_se = lambda x,y: np.load(f"{paths.arrays_path}ccs_per_bw_dens_1p_se_e{x}_n{y}.npy")
avg_path_losses_dens_1p_se = lambda x,y: np.load(f"{paths.arrays_path}avg_path_losses_dens_1p_se_e{x}_n{y}.npy")
per_user_costs_se = lambda x,y: np.load(f"{paths.arrays_path}per_user_costs_se_e{x}_n{y}.npy")

densities = lambda x,y: np.load(f"{paths.arrays_path}cntrfctl_densities_e{x}_n{y}.npy")
densities_pops = lambda x,y: np.load(f"{paths.arrays_path}cntrfctl_densities_pop_e{x}_n{y}.npy")
bw_vals = lambda x,y: np.load(f"{paths.arrays_path}cntrfctl_bw_vals_e{x}_n{y}.npy")

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
default_elast_id = paths.default_elast_id
default_nest_id = paths.default_nest_id
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
barlist = ax.bar(x_pos, c_u(default_elast_id,default_nest_id), yerr=1.96 * c_u_se(default_elast_id,default_nest_id), capsize=4.0)
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
    y_loc = 0.85 * np.max(c_u(default_elast_id,default_nest_id))
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
    axs[row,col].hist(c_R(default_elast_id,default_nest_id)[:,i] * 75.0, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], alpha=0.7)
    axs[row,col].set_xlabel("$\hat{c}_{fm}$ (in \u20ac)" if row == 1 else "", fontsize=12)
    axs[row,col].set_title(mno, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], fontsize=20, y=0.85)
    axs[row,col].axvline(x=np.mean(c_R(default_elast_id,default_nest_id)[:,i]) * 75.0, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], linestyle="--", linewidth=2.)
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
to_tex +=  " & & & & " + " & & ".join(f"${c_u:.2f}$ " for c_u in process_c_u(c_u(default_elast_id,default_nest_id)))
to_tex += " \\\\ \n"
to_tex += " & & & & (" + ") & & (".join(f"${c_u:.2f}$" for c_u in process_c_u(c_u_se(default_elast_id,default_nest_id))) + ") \\\\ \n"
to_tex += " & & & & & & & & \\\\ \n"
    
if save_:
    c_u_prefspec = process_c_u(c_u(default_elast_id, default_nest_id))
    create_file(f"{paths.stats_path}c_u_small.tex", f"{c_u_prefspec[0]:.2f}")
    create_file(f"{paths.stats_path}c_u_med.tex", f"{c_u_prefspec[1]:.2f}")
    create_file(f"{paths.stats_path}c_u_large.tex", f"{c_u_prefspec[2]:.2f}")
    
# Per-base station costs
    
def process_c_R(c_R):
    return 200.0 * np.mean(c_R * 75.0, axis=0) # 200 to convert from monthly to perpetuity, 75 MHz

def process_c_R_sd(c_R):
    return np.std(200.0 * c_R * 75.0, axis=0) # 200 to convert from monthly to perpetuity, 75 MHz

to_tex += "\\textit{Per-base station costs} & & Orange & & SFR & & Free & & Bouygues \\\\ \n"
to_tex += "$\\qquad \\hat{C}_{f}$ & & (in \\euro{}) & & (in \\euro{}) & & (in \\euro{}) & & (in \\euro{}) \\\\ \n"
to_tex += "\\cline{3-3} \\cline{5-5} \\cline{7-7} \\cline{9-9}"
to_tex += " & & " + " & & ".join(f"${c_R:,.0f}$ ".replace(",","\\,") for c_R in process_c_R(c_R(default_elast_id,default_nest_id))) + " \\\\ \n"
to_tex += " & & (" + ") & & (".join(f"${c_R:,.0f}$".replace(",","\\,") for c_R in process_c_R_sd(c_R(default_elast_id,default_nest_id))) + ") \\\\ \n"
to_tex += "\\hline \n"
to_tex += "\\end{tabular} \n"
if save_:
    create_file(f"{paths.tables_path}costs_estimates_table.tex", to_tex)
if print_:
    print(to_tex)
    
if save_:
    c_R_prefspec = process_c_R(c_R(default_elast_id, default_nest_id))
    create_file(f"{paths.stats_path}c_R_ORG.tex", f"{np.round(c_R_prefspec[0], -3):,.0f}".replace(",","\\,"))
    create_file(f"{paths.stats_path}c_R_SFR.tex", f"{np.round(c_R_prefspec[1], -3):,.0f}".replace(",","\\,"))
    create_file(f"{paths.stats_path}c_R_FREE.tex", f"{np.round(c_R_prefspec[2], -3):,.0f}".replace(",","\\,"))
    create_file(f"{paths.stats_path}c_R_BYG.tex", f"{np.round(c_R_prefspec[3], -3):,.0f}".replace(",","\\,"))

    c_R_prefspec_std = process_c_R_sd(c_R(default_elast_id, default_nest_id))
    create_file(f"{paths.stats_path}c_R_std_ORG.tex", f"{np.round(c_R_prefspec_std[0], -3):,.0f}".replace(",","\\,"))
    create_file(f"{paths.stats_path}c_R_std_SFR.tex", f"{np.round(c_R_prefspec_std[1], -3):,.0f}".replace(",","\\,"))
    create_file(f"{paths.stats_path}c_R_std_FREE.tex", f"{np.round(c_R_prefspec_std[2], -3):,.0f}".replace(",","\\,"))
    create_file(f"{paths.stats_path}c_R_std_BYG.tex", f"{np.round(c_R_prefspec_std[3], -3):,.0f}".replace(",","\\,"))
    
# %%
# Values used in counterfactuals

print("Values used in counterfactuals")
print(f"c_R (per unit of bw): {np.mean(c_R(default_elast_id,default_nest_id)[:,np.array([True,True,False,True])]) * 200.0}")
print(f"1 GB c_u: {np.mean(c_u(default_elast_id,default_nest_id)[ds.data[0,:,dlimidx] < 5000.0])}")
print(f"10 GB c_u: {np.mean(c_u(default_elast_id,default_nest_id)[ds.data[0,:,dlimidx] >= 5000.0])}")

if save_:
    create_file(f"{paths.stats_path}per_user_cost_lowdlim.tex", f"{per_user_costs(default_elast_id,default_nest_id)[0]:.2f}")
    create_file(f"{paths.stats_path}per_user_cost_highdlim.tex", f"{per_user_costs(default_elast_id,default_nest_id)[1]:.2f}")
    
# %%
# Endogenous variables - number of firms

fig, axs = plt.subplots(2, 3, figsize=(12.0, 8.0), squeeze=False)

x_fontsize = "x-large"
y_fontsize = "x-large"
title_fontsize = "xx-large"

# dlim = 1,000 prices
axs[0,0].plot(num_firms_array, p_stars(default_elast_id,default_nest_id)[:,0], color="black", lw=lw, alpha=alpha)
axs[0,0].plot(num_firms_array, p_stars(default_elast_id,default_nest_id)[:,0] + 1.96 * p_stars_se(default_elast_id,default_nest_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].plot(num_firms_array, p_stars(default_elast_id,default_nest_id)[:,0] - 1.96 * p_stars_se(default_elast_id,default_nest_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
axs[0,1].plot(num_firms_array, p_stars(default_elast_id,default_nest_id)[:,1], color="black", lw=lw, alpha=alpha)
axs[0,1].plot(num_firms_array, p_stars(default_elast_id,default_nest_id)[:,1] + 1.96 * p_stars_se(default_elast_id,default_nest_id)[:,1], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].plot(num_firms_array, p_stars(default_elast_id,default_nest_id)[:,1] - 1.96 * p_stars_se(default_elast_id,default_nest_id)[:,1], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

# radius
axs[0,2].plot(num_firms_array, num_stations_per_firm_stars(default_elast_id,default_nest_id) * 1000.0, color="black", lw=lw, alpha=alpha)
axs[0,2].plot(num_firms_array, num_stations_per_firm_stars(default_elast_id,default_nest_id) * 1000.0 + 1.96 * num_stations_per_firm_stars_se(default_elast_id,default_nest_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].plot(num_firms_array, num_stations_per_firm_stars(default_elast_id,default_nest_id) * 1000.0 - 1.96 * num_stations_per_firm_stars_se(default_elast_id,default_nest_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[0,2].set_title("number of stations / firm", fontsize=title_fontsize)

# total number of stations
axs[1,0].plot(num_firms_array, num_stations_stars(default_elast_id,default_nest_id) * 1000.0, color="black", lw=lw, alpha=alpha)
axs[1,0].plot(num_firms_array, num_stations_stars(default_elast_id,default_nest_id) * 1000.0 + 1.96 * num_stations_stars_se(default_elast_id,default_nest_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,0].plot(num_firms_array, num_stations_stars(default_elast_id,default_nest_id) * 1000.0 - 1.96 * num_stations_stars_se(default_elast_id,default_nest_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,0].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[1,0].set_title("total number of stations", fontsize=title_fontsize)

# path loss
# axs[1,1].plot(num_firms_array, avg_path_losses(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
# axs[1,1].plot(num_firms_array, avg_path_losses(default_elast_id,default_nest_id) + 1.96 * avg_path_losses_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
# axs[1,1].plot(num_firms_array, avg_path_losses(default_elast_id,default_nest_id) - 1.96 * avg_path_losses_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
# axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
# axs[1,1].set_ylabel("dB", fontsize=y_fontsize)
# axs[1,1].set_title("average path loss", fontsize=title_fontsize)
axs[1,1].plot(num_firms_array, ccs_per_bw(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
axs[1,1].plot(num_firms_array, ccs_per_bw(default_elast_id,default_nest_id) + 1.96 * ccs_per_bw_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,1].plot(num_firms_array, ccs_per_bw(default_elast_id,default_nest_id) - 1.96 * ccs_per_bw_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,1].set_ylabel("Mbps / MHz", fontsize=y_fontsize)
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)
axs[1,1].ticklabel_format(useOffset=False)

# download speeds
axs[1,2].plot(num_firms_array, q_stars(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha, label="download speed")
axs[1,2].plot(num_firms_array, q_stars(default_elast_id,default_nest_id) + 1.96 * q_stars_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,2].plot(num_firms_array, q_stars(default_elast_id,default_nest_id) - 1.96 * q_stars_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,2].plot(num_firms_array, ccs(default_elast_id,default_nest_id), color="black", lw=lw, alpha=0.9, ls=(0, (3, 1, 1, 1)), label="channel capacity")
# axs[1,2].plot(num_firms_array, ccs(default_elast_id,default_nest_id) + 1.96 * ccs_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls=(0, (3, 1, 1, 1)))
# axs[1,2].plot(num_firms_array, ccs(default_elast_id,default_nest_id) - 1.96 * ccs_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls=(0, (3, 1, 1, 1)))
axs[1,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,2].set_ylabel("$Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.nanmin(p_stars(default_elast_id,default_nest_id)[1:,:]) - 2.0
max_y_p = np.nanmax(p_stars(default_elast_id,default_nest_id)[1:,:]) + 5.0
min_y_num_stations_per_firm = np.nanmin(num_stations_per_firm_stars(default_elast_id,default_nest_id) * 1000.0)
max_y_num_stations_per_firm = np.nanmax(num_stations_per_firm_stars(default_elast_id,default_nest_id) * 1000.0)
min_y_num_stations = np.nanmin(num_stations_stars(default_elast_id,default_nest_id)[:] * 1000.0)
max_y_num_stations = np.nanmax(num_stations_stars(default_elast_id,default_nest_id)[:] * 1000.0)
# min_y_pl = np.nanmin(avg_path_losses(default_elast_id,default_nest_id)[:]) - 2.
# max_y_pl = np.nanmax(avg_path_losses(default_elast_id,default_nest_id)[:]) + 2.
min_y_q = np.minimum(np.nanmin(q_stars(default_elast_id,default_nest_id)[:]), np.nanmin(ccs(default_elast_id,default_nest_id)[1:]))
max_y_q = np.maximum(np.nanmax(q_stars(default_elast_id,default_nest_id)[:]), np.nanmax(ccs(default_elast_id,default_nest_id)[1:]))
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

fig, axs = plt.subplots(2, 3, figsize=(12.0, 8.0), squeeze=False)

x_fontsize = "x-large"
y_fontsize = "x-large"
title_fontsize = "xx-large"

# dlim = 1,000 prices
axs[0,0].plot(num_firms_array, p_stars_allfixed(default_elast_id,default_nest_id)[:,0], color="black", lw=lw, alpha=alpha)
axs[0,0].plot(num_firms_array, p_stars_allfixed(default_elast_id,default_nest_id)[:,0] + 1.96 * p_stars_allfixed_se(default_elast_id,default_nest_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].plot(num_firms_array, p_stars_allfixed(default_elast_id,default_nest_id)[:,0] - 1.96 * p_stars_allfixed_se(default_elast_id,default_nest_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
axs[0,1].plot(num_firms_array, p_stars_allfixed(default_elast_id,default_nest_id)[:,1], color="black", lw=lw, alpha=alpha)
axs[0,1].plot(num_firms_array, p_stars_allfixed(default_elast_id,default_nest_id)[:,1] + 1.96 * p_stars_allfixed_se(default_elast_id,default_nest_id)[:,1], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].plot(num_firms_array, p_stars_allfixed(default_elast_id,default_nest_id)[:,1] - 1.96 * p_stars_allfixed_se(default_elast_id,default_nest_id)[:,1], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

# radius
axs[0,2].plot(num_firms_array, num_stations_per_firm_stars_allfixed(default_elast_id,default_nest_id) * 1000.0, color="black", lw=lw, alpha=alpha)
axs[0,2].plot(num_firms_array, num_stations_per_firm_stars_allfixed(default_elast_id,default_nest_id) * 1000.0 + 1.96 * num_stations_per_firm_stars_allfixed_se(default_elast_id,default_nest_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].plot(num_firms_array, num_stations_per_firm_stars_allfixed(default_elast_id,default_nest_id) * 1000.0 - 1.96 * num_stations_per_firm_stars_allfixed_se(default_elast_id,default_nest_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[0,2].set_title("number of stations / firm", fontsize=title_fontsize)

# total number of stations
axs[1,0].plot(num_firms_array, num_stations_stars_allfixed(default_elast_id,default_nest_id) * 1000.0, color="black", lw=lw, alpha=alpha)
axs[1,0].plot(num_firms_array, num_stations_stars_allfixed(default_elast_id,default_nest_id) * 1000.0 + 1.96 * num_stations_stars_allfixed_se(default_elast_id,default_nest_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,0].plot(num_firms_array, num_stations_stars_allfixed(default_elast_id,default_nest_id) * 1000.0 - 1.96 * num_stations_stars_allfixed_se(default_elast_id,default_nest_id) * 1000.0, color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,0].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[1,0].set_title("total number of stations", fontsize=title_fontsize)

# path loss
# axs[1,1].plot(num_firms_array, avg_path_losses_allfixed(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
# axs[1,1].plot(num_firms_array, avg_path_losses_allfixed(default_elast_id,default_nest_id) + 1.96 * avg_path_losses_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
# axs[1,1].plot(num_firms_array, avg_path_losses_allfixed(default_elast_id,default_nest_id) - 1.96 * avg_path_losses_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
# axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
# axs[1,1].set_ylabel("dB", fontsize=y_fontsize)
# axs[1,1].set_title("average path loss", fontsize=title_fontsize)
axs[1,1].plot(num_firms_array, ccs_per_bw_allfixed(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
axs[1,1].plot(num_firms_array, ccs_per_bw_allfixed(default_elast_id,default_nest_id) + 1.96 * ccs_per_bw_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,1].plot(num_firms_array, ccs_per_bw_allfixed(default_elast_id,default_nest_id) - 1.96 * ccs_per_bw_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,1].set_ylabel("Mbps / MHz", fontsize=y_fontsize)
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)
axs[1,1].ticklabel_format(useOffset=False)

# download speeds
axs[1,2].plot(num_firms_array, q_stars_allfixed(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha, label="download speed")
axs[1,2].plot(num_firms_array, q_stars_allfixed(default_elast_id,default_nest_id) + 1.96 * q_stars_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,2].plot(num_firms_array, q_stars_allfixed(default_elast_id,default_nest_id) - 1.96 * q_stars_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[1,2].plot(num_firms_array, ccs_allfixed(default_elast_id,default_nest_id), color="black", lw=lw, alpha=0.9, ls=(0, (3, 1, 1, 1)), label="channel capacity")
# axs[1,2].plot(num_firms_array, ccs_allfixed(default_elast_id,default_nest_id) + 1.96 * ccs_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls=(0, (3, 1, 1, 1)))
# axs[1,2].plot(num_firms_array, ccs_allfixed(default_elast_id,default_nest_id) - 1.96 * ccs_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls=(0, (3, 1, 1, 1)))
axs[1,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,2].set_ylabel("$Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.nanmin(p_stars_allfixed(default_elast_id,default_nest_id)[1:,:]) - 2.0
max_y_p = np.nanmax(p_stars_allfixed(default_elast_id,default_nest_id)[1:,:]) + 5.0
min_y_num_stations_per_firm = np.nanmin(num_stations_per_firm_stars_allfixed(default_elast_id,default_nest_id) * 1000.0)
max_y_num_stations_per_firm = np.nanmax(num_stations_per_firm_stars_allfixed(default_elast_id,default_nest_id) * 1000.0)
min_y_num_stations = np.nanmin(num_stations_stars_allfixed(default_elast_id,default_nest_id)[:] * 1000.0)
max_y_num_stations = np.nanmax(num_stations_stars_allfixed(default_elast_id,default_nest_id)[:] * 1000.0)
# min_y_pl = np.nanmin(avg_path_losses(default_elast_id,default_nest_id)[:]) - 2.
# max_y_pl = np.nanmax(avg_path_losses(default_elast_id,default_nest_id)[:]) + 2.
min_y_q = np.minimum(np.nanmin(q_stars_allfixed(default_elast_id,default_nest_id)[:]), np.nanmin(ccs_allfixed(default_elast_id,default_nest_id)[1:]))
max_y_q = np.maximum(np.nanmax(q_stars_allfixed(default_elast_id,default_nest_id)[:]), np.nanmax(ccs_allfixed(default_elast_id,default_nest_id)[1:]))
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
axs[0,0].plot(num_firms_array, partial_elasts(default_elast_id,default_nest_id)[:,0], lw=lw, alpha=alpha, color="black", ls="--", label="partial")
axs[0,0].plot(num_firms_array, full_elasts(default_elast_id,default_nest_id)[:,0], lw=lw, alpha=alpha, color="black", label="full")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].legend(loc="upper right")
axs[0,0].set_title("1$\,$000 MB plan", fontsize=title_fontsize)

# dlim = 10,000 elasticities
axs[0,1].plot(num_firms_array, partial_elasts(default_elast_id,default_nest_id)[:,1], lw=lw, alpha=alpha, color="black", ls="--", label="partial")
axs[0,1].plot(num_firms_array, full_elasts(default_elast_id,default_nest_id)[:,1], lw=lw, alpha=alpha, color="black", label="full")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].legend(loc="upper right")
axs[0,1].set_title("10$\,$000 MB plan", fontsize=title_fontsize)

# Set axis limits
min_y = np.nanmin(np.concatenate(tuple([full_elasts(default_elast_id,default_nest_id)] + [partial_elasts(default_elast_id,default_nest_id)]))) - 0.3
max_y = np.nanmax(np.concatenate(tuple([full_elasts(default_elast_id,default_nest_id) for elast_id in elast_ids] + [partial_elasts(default_elast_id,default_nest_id)]))) + 0.3
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

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), sharex=True, squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "xx-large"
title_pad = 15.0

# partial_Pif_partial_bf
axs[0,0].plot(num_firms_array, partial_diffPif_partial_bf_allfixed(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
axs[0,0].plot(num_firms_array, partial_diffPif_partial_bf_allfixed(default_elast_id,default_nest_id) + 1.96 * partial_diffPif_partial_bf_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].plot(num_firms_array, partial_diffPif_partial_bf_allfixed(default_elast_id,default_nest_id) - 1.96 * partial_diffPif_partial_bf_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("\u20ac / person / MHz", fontsize=y_fontsize)
axs[0,0].set_title("$d \\Pi_{f} / d B_{f} - d \\Pi_{f} / d B_{f^{\\prime}}$", fontsize=title_fontsize, pad=title_pad)

# partial_Pif_partial_b
axs[0,1].plot(num_firms_array, partial_Pif_partial_b_allfixed(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
axs[0,1].plot(num_firms_array, partial_Pif_partial_b_allfixed(default_elast_id,default_nest_id) + 1.96 * partial_Pif_partial_b_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].plot(num_firms_array, partial_Pif_partial_b_allfixed(default_elast_id,default_nest_id) - 1.96 * partial_Pif_partial_b_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("\u20ac / person / MHz", fontsize=y_fontsize)
axs[0,1].set_title("$d \\Pi_{f} / d B$", fontsize=title_fontsize, pad=title_pad)

# partial_CS_partial_b
axs[0,2].plot(num_firms_array, partial_CS_partial_b_allfixed(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
axs[0,2].plot(num_firms_array, partial_CS_partial_b_allfixed(default_elast_id,default_nest_id) + 1.96 * partial_CS_partial_b_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].plot(num_firms_array, partial_CS_partial_b_allfixed(default_elast_id,default_nest_id) - 1.96 * partial_CS_partial_b_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("\u20ac / person / MHz", fontsize=y_fontsize)
axs[0,2].set_title("$d CS / d B$", fontsize=title_fontsize, pad=title_pad)

# Set axis limits
min_y_Pif_bf = np.nanmin(partial_diffPif_partial_bf_allfixed(default_elast_id,default_nest_id)) - 0.006
max_y_Pif_bf = np.nanmax(partial_diffPif_partial_bf_allfixed(default_elast_id,default_nest_id)) + 0.006
min_y_Pif_b = np.nanmin(partial_Pif_partial_b_allfixed(default_elast_id,default_nest_id)) - 0.001
max_y_Pif_b = np.nanmax(partial_Pif_partial_b_allfixed(default_elast_id,default_nest_id)) + 0.0005
min_y_CS_b = np.nanmin(partial_CS_partial_b_allfixed(default_elast_id,default_nest_id)) - 0.01
max_y_CS_b = np.nanmax(partial_CS_partial_b_allfixed(default_elast_id,default_nest_id)) + 0.02
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
    create_file(f"{paths.stats_path}auction_val_allfixed.tex", f"{partial_diffPif_partial_bf_allfixed(default_elast_id,default_nest_id)[3] * 200.0:.2f}")
    
# %%
# Bandwidth derivatives (all bw)

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), sharex=True, squeeze=False)

# partial_Pif_partial_bf
axs[0,0].plot(num_firms_array, partial_diffPif_partial_bf_allbw(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
axs[0,0].plot(num_firms_array, partial_diffPif_partial_bf_allbw(default_elast_id,default_nest_id) + 1.96 * partial_diffPif_partial_bf_allbw_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].plot(num_firms_array, partial_diffPif_partial_bf_allbw(default_elast_id,default_nest_id) - 1.96 * partial_diffPif_partial_bf_allbw_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("\u20ac / person / MHz", fontsize=y_fontsize)
axs[0,0].set_title("$d \\Pi_{f} / d B_{f} - d \\Pi_{f} / d B_{f^{\\prime}}$", fontsize=title_fontsize, pad=title_pad)

# partial_Pif_partial_b
axs[0,1].plot(num_firms_array, partial_Pif_partial_b_allbw(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
axs[0,1].plot(num_firms_array, partial_Pif_partial_b_allbw(default_elast_id,default_nest_id) + 1.96 * partial_Pif_partial_b_allbw_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].plot(num_firms_array, partial_Pif_partial_b_allbw(default_elast_id,default_nest_id) - 1.96 * partial_Pif_partial_b_allbw_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("\u20ac / person / MHz", fontsize=y_fontsize)
axs[0,1].set_title("$d \\Pi_{f} / d B$", fontsize=title_fontsize, pad=title_pad)

# partial_CS_partial_b
axs[0,2].plot(num_firms_array, partial_CS_partial_b_allbw(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
axs[0,2].plot(num_firms_array, partial_CS_partial_b_allbw(default_elast_id,default_nest_id) + 1.96 * partial_CS_partial_b_allbw_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].plot(num_firms_array, partial_CS_partial_b_allbw(default_elast_id,default_nest_id) - 1.96 * partial_CS_partial_b_allbw_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("\u20ac / person / MHz", fontsize=y_fontsize)
axs[0,2].set_title("$d CS / d B$", fontsize=title_fontsize, pad=title_pad)

# Set axis limits
min_y_Pif_bf = np.nanmin(partial_diffPif_partial_bf_allbw(default_elast_id,default_nest_id)) - 0.002
max_y_Pif_bf = np.nanmax(partial_diffPif_partial_bf_allbw(default_elast_id,default_nest_id)) + 0.003
min_y_Pif_b = np.nanmin(partial_Pif_partial_b_allbw(default_elast_id,default_nest_id)) - 0.001
max_y_Pif_b = np.nanmax(partial_Pif_partial_b_allbw(default_elast_id,default_nest_id)) + 0.001
min_y_CS_b = np.nanmin(partial_CS_partial_b_allbw(default_elast_id,default_nest_id)) - 0.002
max_y_CS_b = np.nanmax(partial_CS_partial_b_allbw(default_elast_id,default_nest_id)) + 0.01
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
    create_file(f"{paths.stats_path}auction_val_allbw.tex", f"{partial_diffPif_partial_bf_allbw(default_elast_id,default_nest_id)[3] * 200.0:.2f}")
    
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

ratio_CS_to_Pif = int_to_en(np.round(partial_CS_partial_b_allbw(default_elast_id,default_nest_id)[3] / partial_diffPif_partial_bf_allbw(default_elast_id,default_nest_id)[3], 0).astype(int))
if print_:
    print(ratio_CS_to_Pif)

if save_:
    create_file(f"{paths.stats_path}ratio_CS_to_Pif.tex", ratio_CS_to_Pif)
    
ratio_CS_to_Pif_allfixed = int_to_en(np.round(partial_CS_partial_b_allfixed(default_elast_id,default_nest_id)[3] / partial_diffPif_partial_bf_allfixed(default_elast_id,default_nest_id)[3], 0).astype(int))
if print_:
    print(ratio_CS_to_Pif_allfixed)

if save_:
    create_file(f"{paths.stats_path}ratio_CS_to_Pif_allfixed.tex", ratio_CS_to_Pif_allfixed)
    
# %%
# Welfare for number of firms

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), sharex=True, squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

# consumer surplus
axs[0,0].plot(num_firms_array_extend, cs(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
#axs[0,0].plot(num_firms_array_extend, cs(default_elast_id,default_nest_id) + 1.96 * cs_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[0,0].plot(num_firms_array_extend, cs(default_elast_id,default_nest_id) - 1.96 * cs_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].axvline(x=num_firms_array_extend[np.nanargmax(cs(default_elast_id,default_nest_id))], color="black", linestyle="--", alpha=0.25)
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
axs[0,1].plot(num_firms_array_extend, ps(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
#axs[0,1].plot(num_firms_array_extend, ps(default_elast_id,default_nest_id) + 1.96 * ps_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[0,1].plot(num_firms_array_extend, ps(default_elast_id,default_nest_id) - 1.96 * ps_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].axvline(x=num_firms_array_extend[np.nanargmax(ps(default_elast_id,default_nest_id))], color="black", linestyle="--", alpha=0.25)
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
axs[0,2].plot(num_firms_array_extend, ts(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
#axs[0,2].plot(num_firms_array_extend, ts(default_elast_id,default_nest_id) + 1.96 * ts_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[0,2].plot(num_firms_array_extend, ts(default_elast_id,default_nest_id) - 1.96 * ts_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].axvline(x=num_firms_array_extend[np.nanargmax(ts(default_elast_id,default_nest_id))], color="black", linestyle="--", alpha=0.25)
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,2].set_title("total surplus", fontsize=title_fontsize)

# Set axis limits
min_y_cs = np.nanmin(cs(default_elast_id,default_nest_id)[1:]) # don't include monopoly case
max_y_cs = np.nanmax(cs(default_elast_id,default_nest_id)[1:])
min_y_ps = np.nanmin(ps(default_elast_id,default_nest_id)[1:])
max_y_ps = np.nanmax(ps(default_elast_id,default_nest_id)[1:])
min_y_ts = np.nanmin(ts(default_elast_id,default_nest_id)[1:])
max_y_ts = np.nanmax(ts(default_elast_id,default_nest_id)[1:])
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

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), sharex=True, squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

# consumer surplus
axs[0,0].plot(num_firms_array_extend, cs_allfixed(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
#axs[0,0].plot(num_firms_array_extend, cs_allfixed(default_elast_id,default_nest_id) + 1.96 * cs_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[0,0].plot(num_firms_array_extend, cs_allfixed(default_elast_id,default_nest_id) - 1.96 * cs_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].axvline(x=num_firms_array_extend[np.nanargmax(cs_allfixed(default_elast_id,default_nest_id))], color="black", linestyle="--", alpha=0.25)
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
axs[0,1].plot(num_firms_array_extend, ps_allfixed(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
#axs[0,1].plot(num_firms_array_extend, ps_allfixed(default_elast_id,default_nest_id) + 1.96 * ps_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[0,1].plot(num_firms_array_extend, ps_allfixed(default_elast_id,default_nest_id) - 1.96 * ps_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].axvline(x=num_firms_array_extend[np.nanargmax(ps_allfixed(default_elast_id,default_nest_id))], color="black", linestyle="--", alpha=0.25)
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
axs[0,2].plot(num_firms_array_extend, ts_allfixed(default_elast_id,default_nest_id), color="black", lw=lw, alpha=alpha)
#axs[0,2].plot(num_firms_array_extend, ts_allfixed(default_elast_id,default_nest_id) + 1.96 * ts_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[0,2].plot(num_firms_array_extend, ts_allfixed(default_elast_id,default_nest_id) - 1.96 * ts_allfixed_se(default_elast_id,default_nest_id), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].axvline(x=num_firms_array_extend[np.nanargmax(ts_allfixed(default_elast_id,default_nest_id))], color="black", linestyle="--", alpha=0.25)
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,2].set_title("total surplus", fontsize=title_fontsize)

# Set axis limits
min_y_cs = np.nanmin(cs_allfixed(default_elast_id,default_nest_id)[1:]) # don't include monopoly case
max_y_cs = np.nanmax(cs_allfixed(default_elast_id,default_nest_id)[1:])
min_y_ps = np.nanmin(ps_allfixed(default_elast_id,default_nest_id)[1:])
max_y_ps = np.nanmax(ps_allfixed(default_elast_id,default_nest_id)[1:])
min_y_ts = np.nanmin(ts_allfixed(default_elast_id,default_nest_id)[1:])
max_y_ts = np.nanmax(ts_allfixed(default_elast_id,default_nest_id)[1:])
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

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), sharex=True, squeeze=False)

axs[0,0].plot(num_firms_array_extend, cs_by_type(default_elast_id,default_nest_id)[:,0], color="black", lw=lw, alpha=alpha)
#axs[i,0].plot(num_firms_array_extend, cs_by_type(default_elast_id,default_nest_id)[:,0] + 1.96 * cs_by_type_se(default_elast_id,default_nest_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[i,0].plot(num_firms_array_extend, cs_by_type(default_elast_id,default_nest_id)[:,0] - 1.96 * cs_by_type_se(default_elast_id,default_nest_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].axvline(x=num_firms_array_extend[np.argmax(cs_by_type(default_elast_id,default_nest_id)[:,0])], color="black", linestyle="--", alpha=0.25)
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("consumer surplus (\u20ac / person)", fontsize=y_fontsize)
axs[0,0].set_title("10th percentile", fontsize=title_fontsize)

axs[0,1].plot(num_firms_array_extend, cs_by_type(default_elast_id,default_nest_id)[:,4], color="black", lw=lw, alpha=alpha)
#axs[i,1].plot(num_firms_array_extend, cs_by_type(default_elast_id,default_nest_id)[:,4] + 1.96 * cs_by_type_se(default_elast_id,default_nest_id)[:,4], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[i,1].plot(num_firms_array_extend, cs_by_type(default_elast_id,default_nest_id)[:,4] - 1.96 * cs_by_type_se(default_elast_id,default_nest_id)[:,4], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].axvline(x=num_firms_array_extend[np.argmax(cs_by_type(default_elast_id,default_nest_id)[:,4])], color="black", linestyle="--", alpha=0.25)
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("consumer surplus (\u20ac / person)", fontsize=y_fontsize)
axs[0,1].set_title("50th percentile", fontsize=title_fontsize)

axs[0,2].plot(num_firms_array_extend, cs_by_type(default_elast_id,default_nest_id)[:,8], color="black", lw=lw, alpha=alpha)
#axs[i,2].plot(num_firms_array_extend, cs_by_type(default_elast_id,default_nest_id)[:,8] + 1.96 * cs_by_type_se(default_elast_id,default_nest_id)[:,8], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[i,2].plot(num_firms_array_extend, cs_by_type(default_elast_id,default_nest_id)[:,8] - 1.96 * cs_by_type_se(default_elast_id,default_nest_id)[:,8], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].axvline(x=num_firms_array_extend[np.argmax(cs_by_type(default_elast_id,default_nest_id)[:,8])], color="black", linestyle="--", alpha=0.25)
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("consumer surplus (\u20ac / person)", fontsize=y_fontsize)
axs[0,2].set_title("90th percentile", fontsize=title_fontsize)
    
# Set axis limits
for i, income_idx in enumerate([0,4,8]):
    margin = 0.1
    min_cs = np.min(cs_by_type(default_elast_id,default_nest_id)[1:,income_idx])
    max_cs = np.max(cs_by_type(default_elast_id,default_nest_id)[1:,income_idx])
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

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), sharex=True, squeeze=False)

axs[0,0].plot(num_firms_array_extend, cs_by_type_allfixed(default_elast_id,default_nest_id)[:,0], color="black", lw=lw, alpha=alpha)
#axs[i,0].plot(num_firms_array_extend, cs_by_type_allfixed(default_elast_id,default_nest_id)[:,0] + 1.96 * cs_by_type_allfixed_se(default_elast_id,default_nest_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[i,0].plot(num_firms_array_extend, cs_by_type_allfixed(default_elast_id,default_nest_id)[:,0] - 1.96 * cs_by_type_allfixed_se(default_elast_id,default_nest_id)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,0].axvline(x=num_firms_array_extend[np.argmax(cs_by_type_allfixed(default_elast_id,default_nest_id)[:,0])], color="black", linestyle="--", alpha=0.25)
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("consumer surplus (\u20ac / person)", fontsize=y_fontsize)
axs[0,0].set_title("10th percentile", fontsize=title_fontsize)

axs[0,1].plot(num_firms_array_extend, cs_by_type_allfixed(default_elast_id,default_nest_id)[:,4], color="black", lw=lw, alpha=alpha)
#axs[i,1].plot(num_firms_array_extend, cs_by_type_allfixed(default_elast_id,default_nest_id)[:,4] + 1.96 * cs_by_type_allfixed_se(default_elast_id,default_nest_id)[:,4], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[i,1].plot(num_firms_array_extend, cs_by_type_allfixed(default_elast_id,default_nest_id)[:,4] - 1.96 * cs_by_type_allfixed_se(default_elast_id,default_nest_id)[:,4], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,1].axvline(x=num_firms_array_extend[np.argmax(cs_by_type_allfixed(default_elast_id,default_nest_id)[:,4])], color="black", linestyle="--", alpha=0.25)
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("consumer surplus (\u20ac / person)", fontsize=y_fontsize)
axs[0,1].set_title("50th percentile", fontsize=title_fontsize)

axs[0,2].plot(num_firms_array_extend, cs_by_type_allfixed(default_elast_id,default_nest_id)[:,8], color="black", lw=lw, alpha=alpha)
#axs[i,2].plot(num_firms_array_extend, cs_by_type_allfixed(default_elast_id,default_nest_id)[:,8] + 1.96 * cs_by_type_allfixed_se(default_elast_id,default_nest_id)[:,8], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
#axs[i,2].plot(num_firms_array_extend, cs_by_type_allfixed(default_elast_id,default_nest_id)[:,8] - 1.96 * cs_by_type_allfixed_se(default_elast_id,default_nest_id)[:,8], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
axs[0,2].axvline(x=num_firms_array_extend[np.argmax(cs_by_type_allfixed(default_elast_id,default_nest_id)[:,8])], color="black", linestyle="--", alpha=0.25)
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("consumer surplus (\u20ac / person)", fontsize=y_fontsize)
axs[0,2].set_title("90th percentile", fontsize=title_fontsize)
    
# Set axis limits
for i, income_idx in enumerate([0,4,8]):
    margin = 0.1
    min_cs = np.min(cs_by_type_allfixed(default_elast_id,default_nest_id)[1:,income_idx])
    max_cs = np.max(cs_by_type_allfixed(default_elast_id,default_nest_id)[1:,income_idx])
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
to_tex += f" & ${round_var(p_stars_shortrun(default_elast_id,default_nest_id)[0,0], num_digits_round)}$ ${round_var(p_stars_shortrun_se(default_elast_id,default_nest_id)[0,0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(p_stars_shortrun(default_elast_id,default_nest_id)[0,1], num_digits_round)}$ ${round_var(p_stars_shortrun_se(default_elast_id,default_nest_id)[0,1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(q_stars_shortrun(default_elast_id,default_nest_id)[0], num_digits_round)}$ ${round_var(q_stars_shortrun_se(default_elast_id,default_nest_id)[0], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "long-run"
to_tex += f" & ${round_var(p_stars_shortrun(default_elast_id,default_nest_id)[1,0], num_digits_round)}$ ${round_var(p_stars_shortrun_se(default_elast_id,default_nest_id)[1,0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(p_stars_shortrun(default_elast_id,default_nest_id)[1,1], num_digits_round)}$ ${round_var(p_stars_shortrun_se(default_elast_id,default_nest_id)[1,1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(q_stars_shortrun(default_elast_id,default_nest_id)[1], num_digits_round)}$ ${round_var(q_stars_shortrun_se(default_elast_id,default_nest_id)[1], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "\\hline \n" 
to_tex += "difference"
to_tex += f" & ${round_var(p_stars_shortrun(default_elast_id,default_nest_id)[2,0], num_digits_round)}$ ${round_var(p_stars_shortrun_se(default_elast_id,default_nest_id)[2,0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(p_stars_shortrun(default_elast_id,default_nest_id)[2,1], num_digits_round)}$ ${round_var(p_stars_shortrun_se(default_elast_id,default_nest_id)[2,1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(q_stars_shortrun(default_elast_id,default_nest_id)[2], num_digits_round)}$ ${round_var(q_stars_shortrun_se(default_elast_id,default_nest_id)[2], num_digits_round, stderrs=True)}$"
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
to_tex += f" & ${round_var(cs_shortrun(default_elast_id,default_nest_id)[0], num_digits_round)}$ ${round_var(cs_shortrun_se(default_elast_id,default_nest_id)[0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ps_shortrun(default_elast_id,default_nest_id)[0], num_digits_round)}$ ${round_var(ps_shortrun_se(default_elast_id,default_nest_id)[0], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ts_shortrun(default_elast_id,default_nest_id)[0], num_digits_round)}$ ${round_var(ts_shortrun_se(default_elast_id,default_nest_id)[0], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "long-run"
to_tex += f" & ${round_var(cs_shortrun(default_elast_id,default_nest_id)[1], num_digits_round)}$ ${round_var(cs_shortrun_se(default_elast_id,default_nest_id)[1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ps_shortrun(default_elast_id,default_nest_id)[1], num_digits_round)}$ ${round_var(ps_shortrun_se(default_elast_id,default_nest_id)[1], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ts_shortrun(default_elast_id,default_nest_id)[1], num_digits_round)}$ ${round_var(ts_shortrun_se(default_elast_id,default_nest_id)[1], num_digits_round, stderrs=True)}$"
to_tex += " \\\\ \n"
to_tex += "\\hline \n" 
to_tex += "difference"
to_tex += f" & ${round_var(cs_shortrun(default_elast_id,default_nest_id)[2], num_digits_round)}$ ${round_var(cs_shortrun_se(default_elast_id,default_nest_id)[2], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ps_shortrun(default_elast_id,default_nest_id)[2], num_digits_round)}$ ${round_var(ps_shortrun_se(default_elast_id,default_nest_id)[2], num_digits_round, stderrs=True)}$"
to_tex += f" & ${round_var(ts_shortrun(default_elast_id,default_nest_id)[2], num_digits_round)}$ ${round_var(ts_shortrun_se(default_elast_id,default_nest_id)[2], num_digits_round, stderrs=True)}$"
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
    to_tex += f"${round_var(cs_by_type_shortrun(default_elast_id,default_nest_id)[0,2*i], num_digits_round)}$ ${round_var(cs_by_type_shortrun_se(default_elast_id,default_nest_id)[1,2*i], num_digits_round, stderrs=True)}$"
    if i < 4:
        to_tex += " & "
to_tex += " \\\\ \n"
to_tex += "long-run & "
for i in range(5):
    to_tex += f"${round_var(cs_by_type_shortrun(default_elast_id,default_nest_id)[1,2*i], num_digits_round)}$ ${round_var(cs_by_type_shortrun_se(default_elast_id,default_nest_id)[0,2*i], num_digits_round, stderrs=True)}$"
    if i < 4:
        to_tex += " & "
to_tex += " \\\\ \n"
to_tex += "\\hline \n" 
to_tex += "difference & "
for i in range(5):
    to_tex += f"${round_var(cs_by_type_shortrun(default_elast_id,default_nest_id)[2,2*i], num_digits_round)}$ ${round_var(cs_by_type_shortrun_se(default_elast_id,default_nest_id)[0,2*i], num_digits_round, stderrs=True)}$"
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

fig, axs = plt.subplots(2, 3, figsize=(12.0,8.0), squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

x_pos = [i for i in range(2)]
x_ticklabels = ["$3$ firms, $\\frac{4}{3}$ b", "$4$ firms, b"]

# dlim = 1,000 prices
axs[0,0].bar(x_pos, p_stars_free_allfixed(default_elast_id,default_nest_id)[:,0], yerr=1.96 * p_stars_free_allfixed_se(default_elast_id,default_nest_id)[:,0], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,0].set_xticks(x_pos)
axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,0].set_ylabel("$\\Delta p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
axs[0,1].bar(x_pos, p_stars_free_allfixed(default_elast_id,default_nest_id)[:,1], yerr=1.96 * p_stars_free_allfixed_se(default_elast_id,default_nest_id)[:,1], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,1].set_xticks(x_pos)
axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,1].set_ylabel("$\\Delta p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

axs[0,2].bar(x_pos, num_stations_per_firm_stars_free_allfixed(default_elast_id,default_nest_id) * 1000.0, yerr=1.96 * num_stations_per_firm_stars_free_allfixed_se(default_elast_id,default_nest_id) * 1000.0, capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,2].set_xticks(x_pos)
axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,2].set_ylabel("$\\Delta$ number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[0,2].set_title("number of stations / firm", fontsize=title_fontsize)

# total number of stations
axs[1,0].bar(x_pos, num_stations_stars_free_allfixed(default_elast_id,default_nest_id) * 1000.0, yerr=1.96 * num_stations_stars_free_allfixed_se(default_elast_id,default_nest_id) * 1000.0, capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,0].set_xticks(x_pos)
axs[1,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,0].set_ylabel("$\\Delta$ number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[1,0].set_title("total number of stations", fontsize=title_fontsize)

# average path loss
axs[1,1].bar(x_pos, ccs_per_bw_free_allfixed(default_elast_id,default_nest_id), yerr=1.96 * ccs_per_bw_free_allfixed_se(default_elast_id,default_nest_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,1].set_xticks(x_pos)
axs[1,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,1].set_ylabel("$\\Delta$ Mbps / MHz", fontsize=y_fontsize)
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)

# download speeds
axs[1,2].bar(x_pos, q_stars_free_allfixed(default_elast_id,default_nest_id), yerr=1.96 * q_stars_free_allfixed_se(default_elast_id,default_nest_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,2].set_xticks(x_pos)
axs[1,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,2].set_ylabel("$\\Delta Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.min(p_stars_free_allfixed(default_elast_id,default_nest_id)) - 0.75
max_y_p = np.max(p_stars_free_allfixed(default_elast_id,default_nest_id)) + 0.7
for i in range(2): # first two columns
    axs[0,i].set_ylim((min_y_p, max_y_p))

plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_free_variables_allfixed_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# "Add Free" endogenous variables (all bw)

fig, axs = plt.subplots(2, 3, figsize=(12.0,8.0), squeeze=False)

x_pos = [i for i in range(2)]
x_ticklabels = ["$3$ firms, $\\frac{4}{3}$ b", "$4$ firms, b"]

# dlim = 1,000 prices
axs[0,0].bar(x_pos, p_stars_free_allbw(default_elast_id,default_nest_id)[:,0], yerr=1.96 * p_stars_free_allbw_se(default_elast_id,default_nest_id)[:,0], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,0].set_xticks(x_pos)
axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,0].set_ylabel("$\\Delta p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
axs[0,1].bar(x_pos, p_stars_free_allbw(default_elast_id,default_nest_id)[:,1], yerr=1.96 * p_stars_free_allbw_se(default_elast_id,default_nest_id)[:,1], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,1].set_xticks(x_pos)
axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,1].set_ylabel("$\\Delta p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

axs[0,2].bar(x_pos, num_stations_per_firm_stars_free_allbw(default_elast_id,default_nest_id) * 1000.0, yerr=1.96 * num_stations_per_firm_stars_free_allbw_se(default_elast_id,default_nest_id) * 1000.0, capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,2].set_xticks(x_pos)
axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,2].set_ylabel("$\\Delta$ number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[0,2].set_title("number of stations / firm", fontsize=title_fontsize)

# total number of stations
axs[1,0].bar(x_pos, num_stations_stars_free_allbw(default_elast_id,default_nest_id) * 1000.0, yerr=1.96 * num_stations_stars_free_allbw_se(default_elast_id,default_nest_id) * 1000.0, capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,0].set_xticks(x_pos)
axs[1,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,0].set_ylabel("$\\Delta$ number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[1,0].set_title("total number of stations", fontsize=title_fontsize)

# average path loss
axs[1,1].bar(x_pos, ccs_per_bw_free_allbw(default_elast_id,default_nest_id), yerr=1.96 * ccs_per_bw_free_allbw_se(default_elast_id,default_nest_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,1].set_xticks(x_pos)
axs[1,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,1].set_ylabel("$\\Delta$ Mbps / MHz", fontsize=y_fontsize)
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)

# download speeds
axs[1,2].bar(x_pos, q_stars_free_allbw(default_elast_id,default_nest_id), yerr=1.96 * q_stars_free_allbw_se(default_elast_id,default_nest_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,2].set_xticks(x_pos)
axs[1,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[1,2].set_ylabel("$\\Delta Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.min(p_stars_free_allbw(default_elast_id,default_nest_id)) - 0.65
max_y_p = np.max(p_stars_free_allbw(default_elast_id,default_nest_id)) + 0.6
for i in range(2): # first two columns
    axs[0,i].set_ylim((min_y_p, max_y_p))

plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_free_variables_allbw_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# %%
# Welfare for "Add Free" (all fixed)

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

x_pos = [i for i in range(2)]
x_ticklabels = ["$3$ firms, $\\frac{4}{3}$ b", "$4$ firms, b"]

margin = 0.1

# consumer surplus
axs[0,0].bar(x_pos, cs_free_allfixed(default_elast_id,default_nest_id), yerr=1.96*cs_free_allfixed_se(default_elast_id,default_nest_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,0].set_xticks(x_pos)
axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,0].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_free_allfixed(default_elast_id,default_nest_id) + 1.96 * cs_free_allfixed_se(default_elast_id,default_nest_id))
min_cs = np.min(cs_free_allfixed(default_elast_id,default_nest_id) - 1.96 * cs_free_allfixed_se(default_elast_id,default_nest_id))
diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
axs[0,0].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
axs[0,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
axs[0,1].bar(x_pos, ps_free_allfixed(default_elast_id,default_nest_id), yerr=1.96*ps_free_allfixed_se(default_elast_id,default_nest_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,1].set_xticks(x_pos)
axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,1].set_ylabel("$\\Delta$ PS (in \u20ac / person)", fontsize=y_fontsize)
max_ps = np.max(ps_free_allfixed(default_elast_id,default_nest_id) + 1.96 * ps_free_allfixed_se(default_elast_id,default_nest_id))
min_ps = np.min(ps_free_allfixed(default_elast_id,default_nest_id) - 1.96 * ps_free_allfixed_se(default_elast_id,default_nest_id))
diff = np.maximum(max_ps, 0.0) - np.minimum(min_ps, 0.0)
axs[0,1].set_ylim((np.minimum(min_ps - margin * diff, 0.0), np.maximum(max_ps + margin * diff, 0.0)))
axs[0,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
axs[0,2].bar(x_pos, ts_free_allfixed(default_elast_id,default_nest_id), yerr=1.96*ts_free_allfixed_se(default_elast_id,default_nest_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,2].set_xticks(x_pos)
axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,2].set_ylabel("$\\Delta$ TS (in \u20ac / person)", fontsize=y_fontsize)
max_ts = np.max(ts_free_allfixed(default_elast_id,default_nest_id) + 1.96 * ts_free_allfixed_se(default_elast_id,default_nest_id))
min_ts = np.min(ts_free_allfixed(default_elast_id,default_nest_id) - 1.96 * ts_free_allfixed_se(default_elast_id,default_nest_id))
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

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), squeeze=False)

x_pos = [i for i in range(2)]
x_ticklabels = ["$3$ firms, $\\frac{4}{3}$ b", "$4$ firms, b"]

margin = 0.1

# consumer surplus
axs[0,0].bar(x_pos, cs_free_allbw(default_elast_id,default_nest_id), yerr=1.96*cs_free_allbw_se(default_elast_id,default_nest_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,0].set_xticks(x_pos)
axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,0].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_free_allbw(default_elast_id,default_nest_id) + 1.96 * cs_free_allbw_se(default_elast_id,default_nest_id))
min_cs = np.min(cs_free_allbw(default_elast_id,default_nest_id) - 1.96 * cs_free_allbw_se(default_elast_id,default_nest_id))
diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
axs[0,0].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
axs[0,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
axs[0,1].bar(x_pos, ps_free_allbw(default_elast_id,default_nest_id), yerr=1.96*ps_free_allbw_se(default_elast_id,default_nest_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,1].set_xticks(x_pos)
axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,1].set_ylabel("$\\Delta$ PS (in \u20ac / person)", fontsize=y_fontsize)
max_ps = np.max(ps_free_allbw(default_elast_id,default_nest_id) + 1.96 * ps_free_allbw_se(default_elast_id,default_nest_id))
min_ps = np.min(ps_free_allbw(default_elast_id,default_nest_id) - 1.96 * ps_free_allbw_se(default_elast_id,default_nest_id))
diff = np.maximum(max_ps, 0.0) - np.minimum(min_ps, 0.0)
axs[0,1].set_ylim((np.minimum(min_ps - margin * diff, 0.0), np.maximum(max_ps + margin * diff, 0.0)))
axs[0,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
axs[0,2].bar(x_pos, ts_free_allbw(default_elast_id,default_nest_id), yerr=1.96*ts_free_allbw_se(default_elast_id,default_nest_id), capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,2].set_xticks(x_pos)
axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,2].set_ylabel("$\\Delta$ TS (in \u20ac / person)", fontsize=y_fontsize)
max_ts = np.max(ts_free_allbw(default_elast_id,default_nest_id) + 1.96 * ts_free_allbw_se(default_elast_id,default_nest_id))
min_ts = np.min(ts_free_allbw(default_elast_id,default_nest_id) - 1.96 * ts_free_allbw_se(default_elast_id,default_nest_id))
diff = np.maximum(max_ts, 0.0) - np.minimum(min_ts, 0.0)
axs[0,2].set_ylim((np.minimum(min_ts - margin * diff, 0.0), np.maximum(max_ts + margin * diff, 0.0)))
axs[0,2].set_title("total surplus", fontsize=title_fontsize)
        
plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_free_welfare_allbw_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    
# # %%
# # Consumer surplus by type for "Add Free" (all fixed)

# fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), squeeze=False)

# x_fontsize = "large"
# y_fontsize = "large"
# title_fontsize = "x-large"

# x_pos = [i for i in range(2)]
# x_ticklabels = ["$3$ firms, $\\frac{4}{3}$ b", "$4$ firms, b"]

# axs[0,0].bar(x_pos, cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,0], yerr=1.96*cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,0], capsize=7.0, color="black", alpha=0.8 * alpha)
# axs[0,0].set_xticks(x_pos)
# axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
# axs[0,0].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
# max_cs = np.max(cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,0] + 1.96 * cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,0])
# min_cs = np.min(cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,0] - 1.96 * cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,0])
# diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
# axs[0,0].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
# axs[0,0].set_title("10th percentile", fontsize=title_fontsize)

# axs[0,1].bar(x_pos, cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,4], yerr=1.96*cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,4], capsize=7.0, color="black", alpha=0.8 * alpha)
# axs[0,1].set_xticks(x_pos)
# axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
# axs[0,1].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
# max_cs = np.max(cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,4] + 1.96 * cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,4])
# min_cs = np.min(cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,4] - 1.96 * cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,4])
# diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
# axs[0,1].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
# axs[0,1].set_title("50th percentile", fontsize=title_fontsize)

# axs[0,2].bar(x_pos, cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,8], yerr=1.96*cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,8], capsize=7.0, color="black", alpha=0.8 * alpha)
# axs[0,2].set_xticks(x_pos)
# axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
# axs[0,2].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
# max_cs = np.max(cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,8] + 1.96 * cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,8])
# min_cs = np.min(cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,8] - 1.96 * cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,8])
# diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
# axs[0,2].ticklabel_format(style="plain", useOffset=False, axis="y")
# axs[0,2].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
# axs[0,2].set_title("90th percentile", fontsize=title_fontsize)
        
# plt.tight_layout()

# if save_:
#     plt.savefig(f"{paths.graphs_path}counterfactual_free_cs_by_income_allfixed_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

# if print_:
#     plt.show()

# %%
# Consumer surplus by type for "Add Free" (all bw)

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), squeeze=False)

x_pos = [i for i in range(2)]
x_ticklabels = ["$3$ firms, $\\frac{4}{3}$ b", "$4$ firms, b"]

axs[0,0].bar(x_pos, cs_by_type_free_allbw(default_elast_id,default_nest_id)[:,0], yerr=1.96*cs_by_type_free_allbw_se(default_elast_id,default_nest_id)[:,0], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,0].set_xticks(x_pos)
axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,0].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_by_type_free_allbw(default_elast_id,default_nest_id)[:,0] + 1.96 * cs_by_type_free_allbw_se(default_elast_id,default_nest_id)[:,0])
min_cs = np.min(cs_by_type_free_allbw(default_elast_id,default_nest_id)[:,0] - 1.96 * cs_by_type_free_allbw_se(default_elast_id,default_nest_id)[:,0])
diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
axs[0,0].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
axs[0,0].set_title("10th percentile", fontsize=title_fontsize)

axs[0,1].bar(x_pos, cs_by_type_free_allbw(default_elast_id,default_nest_id)[:,4], yerr=1.96*cs_by_type_free_allbw_se(default_elast_id,default_nest_id)[:,4], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,1].set_xticks(x_pos)
axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,1].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_by_type_free_allbw(default_elast_id,default_nest_id)[:,4] + 1.96 * cs_by_type_free_allbw_se(default_elast_id,default_nest_id)[:,4])
min_cs = np.min(cs_by_type_free_allbw(default_elast_id,default_nest_id)[:,4] - 1.96 * cs_by_type_free_allbw_se(default_elast_id,default_nest_id)[:,4])
diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
axs[0,1].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
axs[0,1].set_title("50th percentile", fontsize=title_fontsize)

axs[0,2].bar(x_pos, cs_by_type_free_allbw(default_elast_id,default_nest_id)[:,8], yerr=1.96*cs_by_type_free_allbw_se(default_elast_id,default_nest_id)[:,8], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,2].set_xticks(x_pos)
axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,2].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_by_type_free_allbw(default_elast_id,default_nest_id)[:,8] + 1.96 * cs_by_type_free_allbw_se(default_elast_id,default_nest_id)[:,8])
min_cs = np.min(cs_by_type_free_allbw(default_elast_id,default_nest_id)[:,8] - 1.96 * cs_by_type_free_allbw_se(default_elast_id,default_nest_id)[:,8])
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

fig, axs = plt.subplots(2, 3, figsize=(12.0, 8.0), squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

densities_argsort = np.argsort(densities(default_elast_id,default_nest_id))
densities_sort = densities(default_elast_id,default_nest_id)[densities_argsort]
default_dens_id = np.where(densities_sort == densities(default_elast_id,default_nest_id)[0])[0][0] # we saved the default density as the first one in the original file
dens_legend_ = ["continental USA density", "France density", "France contraharmonic mean density", "Paris density"] # b/c sorted
dens_legend = np.array([f"$\\bf{{{dens_legend_[i]}}}$".replace(" ", "\\:") if i == default_dens_id else f"{dens_legend_[i]}" for i, dens in enumerate(densities_sort)])
dens_use = np.ones(densities_sort.shape, dtype=bool) # default: use all
# dens_use[-1] = False

dens_color_p = "Blues" # plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
dens_color_R = "Reds"
dens_color_Rtot = "Reds"
dens_color_q = "Greens"
dens_color_pl = "Greens"
alphas_dens = np.linspace(0.25, 0.75, densities(default_elast_id,default_nest_id)[dens_use].shape[0])

# dlim = 1,000 prices
for i, dens in enumerate(densities_sort[dens_use]):
    axs[0,0].plot(num_firms_array, p_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i],0], color=cm.get_cmap(dens_color_p)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
for i, dens in enumerate(densities_sort[dens_use]):
    axs[0,1].plot(num_firms_array, p_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i],1], color=cm.get_cmap(dens_color_p)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

# radius
for i, dens in enumerate(densities_sort[dens_use]):
    axs[0,2].plot(num_firms_array, num_stations_per_firm_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]] * 1000.0, color=cm.get_cmap(dens_color_R)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[0,2].set_title("number of stations / firm / person", fontsize=title_fontsize)

# total number of stations
for i, dens in enumerate(densities_sort[dens_use]):
    axs[1,0].plot(num_firms_array, num_stations_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]] * 1000.0, color=cm.get_cmap(dens_color_Rtot)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
axs[1,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,0].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[1,0].set_title("total number of stations / person", fontsize=title_fontsize)

# for i, dens in enumerate(densities_sort[dens_use]):
#     axs[1,1].plot(num_firms_array, avg_path_losses_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_pl)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
# axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
# axs[1,1].set_ylabel("dB", fontsize=y_fontsize)
# axs[1,1].set_title("average path loss", fontsize=title_fontsize)
for i, dens in enumerate(densities_sort[dens_use]):
    axs[1,1].plot(num_firms_array, ccs_per_bw_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_pl)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,1].set_ylabel("Mbps / MHz", fontsize=y_fontsize)
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)

# download speeds
for i, dens in enumerate(densities_sort[dens_use]):
    axs[1,2].plot(num_firms_array, q_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_q)(alphas_dens[i]), lw=lw, label="download speed" if i == np.sum(dens_use) - 1 else None)
    axs[1,2].plot(num_firms_array, ccs_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_q)(alphas_dens[i]), lw=0.6 * lw, ls="--", label="channel capacity" if i == np.sum(dens_use) - 1 else None)
axs[1,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,2].set_ylabel("$Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.nanmin(p_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort,:][1:,dens_use,:])
max_y_p = np.nanmax(p_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort,:][1:,dens_use,:])
min_y_num_stations_per_firm = 1000.0 * np.nanmin(num_stations_per_firm_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort][:,dens_use])
max_y_num_stations_per_firm = 1000.0 * np.nanmax(num_stations_per_firm_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort][:,dens_use])
min_y_num_stations = 1000.0 * np.nanmin(num_stations_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort][:,dens_use])
max_y_num_stations = 1000.0 * np.nanmax(num_stations_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort][:,dens_use])
# min_y_pl = np.nanmin(avg_path_losses_dens(default_elast_id,default_nest_id)[:,densities_argsort][:,dens_use])
# max_y_pl = np.nanmax(avg_path_losses_dens(default_elast_id,default_nest_id)[:,densities_argsort][:,dens_use])
min_y_q = np.nanmin(q_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort][:,dens_use])
max_y_q = np.nanmax(q_stars_dens(default_elast_id,default_nest_id)[:,densities_argsort][:,dens_use])
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
    create_file(f"{paths.stats_path}rep_dens.tex", "{:,.0f}".format(densities(default_elast_id, default_nest_id)[0]).replace(",","\\,"))

if print_:
    plt.show()
    
# %%
# Welfare for number of firms by density

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), sharex=True, squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

dens_color_cs = "Blues"
dens_color_ps = "Reds"
dens_color_ts = "Greens"
alphas_dens = np.linspace(0.25, 0.75, densities(default_elast_id,default_nest_id)[dens_use].shape[0])

# consumer surplus
for i, dens in enumerate(densities_sort[dens_use]):
    axs[0,0].plot(num_firms_array_extend, cs_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_cs)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
    axs[0,0].axvline(x=num_firms_array_extend[np.nanargmax(cs_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]])], color=cm.get_cmap(dens_color_cs)(alphas_dens[i]), linestyle="--")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
for i, dens in enumerate(densities_sort[dens_use]):
    axs[0,1].plot(num_firms_array_extend, ps_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_ps)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
    axs[0,1].axvline(x=num_firms_array_extend[np.argmax(ps_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]])], color=cm.get_cmap(dens_color_ps)(alphas_dens[i]), linestyle="--")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
for i, dens in enumerate(densities_sort[dens_use]):
    axs[0,2].plot(num_firms_array_extend, ts_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]], color=cm.get_cmap(dens_color_ts)(alphas_dens[i]), lw=lw, label=dens_legend[dens_use][i])
    axs[0,2].axvline(x=num_firms_array_extend[np.nanargmax(ts_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]])], color=cm.get_cmap(dens_color_ts)(alphas_dens[i]), linestyle="--")
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,2].set_title("total surplus", fontsize=title_fontsize)

# Set axis limits
# margin_cs = 1.0 - np.sort(np.concatenate(tuple([normalize_var(cs_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]])])))[-6]
# margin_ts = 1.0 - np.sort(np.concatenate(tuple([normalize_var(ts_dens(default_elast_id,default_nest_id)[:,densities_argsort[dens_use][i]])])))[-6]
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

bw_vals_argsort = np.argsort(bw_vals(default_elast_id,default_nest_id))
bw_vals_sort = bw_vals(default_elast_id,default_nest_id)[bw_vals_argsort]
default_bw_id = np.where(bw_vals_sort == bw_vals(default_elast_id,default_nest_id)[0])[0][0] # we saved the default density as the first one in the original file
bw_legend_ = ["0.5 * bw", "bw", "1.5 * bw"] # b/c sorted
bw_legend = np.array([f"$\\bf{{{bw_legend_[i]}}}$" if i == default_bw_id else f"{bw_legend_[i]}" for i, bw in enumerate(bw_vals_sort)])
bw_use = np.ones(bw_vals_sort.shape, dtype=bool)

fig, axs = plt.subplots(2, 3, figsize=(12.0, 8.0), squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

bw_color_p = "Blues"
bw_color_R = "Reds"
bw_color_Rtot = "Reds"
bw_color_q = "Greens"
bw_color_pl = "Greens"
alphas_bw = np.linspace(0.25, 0.75, bw_vals(default_elast_id,default_nest_id)[bw_use].shape[0])

# dlim = 1,000 prices
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[0,0].plot(num_firms_array, p_stars_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i],0], color=cm.get_cmap(bw_color_p)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[0,1].plot(num_firms_array, p_stars_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i],1], color=cm.get_cmap(bw_color_p)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize=y_fontsize)
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

# radius
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[0,2].plot(num_firms_array, num_stations_per_firm_stars_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]] * 1000.0, color=cm.get_cmap(bw_color_R)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[0,2].set_title("number of stations / firm", fontsize=title_fontsize)

# total number of stations
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[1,0].plot(num_firms_array, num_stations_stars_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]] * 1000.0, color=cm.get_cmap(bw_color_Rtot)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
axs[1,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,0].set_ylabel("number of stations\n(per 1000 people)", fontsize=y_fontsize)
axs[1,0].set_title("total number of stations", fontsize=title_fontsize)

# average path loss
# for i, bw in enumerate(bw_vals_sort[bw_use]):
#     axs[1,1].plot(num_firms_array, avg_path_losses_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]], color=cm.get_cmap(bw_color_pl)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
# axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
# axs[1,1].set_ylabel("dB", fontsize=y_fontsize)
# axs[1,1].set_title("average path loss")
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[1,1].plot(num_firms_array, ccs_per_bw_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]], color=cm.get_cmap(bw_color_pl)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
axs[1,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,1].set_ylabel("Mbps / MHz", fontsize=y_fontsize)
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)

# download speeds
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[1,2].plot(num_firms_array, q_stars_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]], color=cm.get_cmap(bw_color_q)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
axs[1,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[1,2].set_ylabel("$Q_{f}^{*}$ (in Mbps)", fontsize=y_fontsize)
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.nanmin(p_stars_bw(default_elast_id,default_nest_id)[1:,bw_use,:])
max_y_p = np.nanmax(p_stars_bw(default_elast_id,default_nest_id)[1:,bw_use,:])
min_y_num_stations_per_firm = np.nanmin(num_stations_per_firm_stars_bw(default_elast_id,default_nest_id)[:,bw_use]) * 1000.0
max_y_num_stations_per_firm = np.nanmax(num_stations_per_firm_stars_bw(default_elast_id,default_nest_id)[:,bw_use]) * 1000.0
min_y_num_stations = np.nanmin(num_stations_stars_bw(default_elast_id,default_nest_id)[:,bw_use]) * 1000.0
max_y_num_stations = np.nanmax(num_stations_stars_bw(default_elast_id,default_nest_id)[:,bw_use]) * 1000.0
# min_y_pl = np.nanmin(avg_path_losses_bw(default_elast_id,default_nest_id)[:,bw_use]) - 2.
# max_y_pl = np.nanmax(avg_path_losses_bw(default_elast_id,default_nest_id)[:,bw_use]) + 2.
min_y_q = np.nanmin(q_stars_bw(default_elast_id,default_nest_id)[:,bw_use])
max_y_q = np.nanmax(q_stars_bw(default_elast_id,default_nest_id)[:,bw_use])
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

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), sharex=True, squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

bw_use = np.ones(bw_vals_sort.shape, dtype=bool)

bw_color_cs = "Blues"
bw_color_ps = "Reds"
bw_color_ts = "Greens"
alphas_bw = np.linspace(0.25, 0.75, bw_vals(default_elast_id,default_nest_id)[bw_use].shape[0])

normalize_var = lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

# consumer surplus
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[0,0].plot(num_firms_array_extend, normalize_var(cs_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]]), color=cm.get_cmap(bw_color_cs)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
    axs[0,0].axvline(x=num_firms_array_extend[np.nanargmax(cs_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]])], color=cm.get_cmap(bw_color_cs)(alphas_bw[i]), linestyle="--")
axs[0,0].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,0].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,0].set_title("consumer surplus", fontsize=title_fontsize)

# producer surplus
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[0,1].plot(num_firms_array_extend, normalize_var(ps_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]]), color=cm.get_cmap(bw_color_ps)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
    axs[0,1].axvline(x=num_firms_array_extend[np.nanargmax(ps_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]])], color=cm.get_cmap(bw_color_ps)(alphas_bw[i]), linestyle="--")
axs[0,1].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,1].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,1].set_title("producer surplus", fontsize=title_fontsize)

# total surplus
for i, bw in enumerate(bw_vals_sort[bw_use]):
    axs[0,2].plot(num_firms_array_extend, normalize_var(ts_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]]), color=cm.get_cmap(bw_color_ts)(alphas_bw[i]), lw=lw, label=bw_legend[bw_use][i])
    axs[0,2].axvline(x=num_firms_array_extend[np.nanargmax(ts_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]])], color=cm.get_cmap(bw_color_ts)(alphas_bw[i]), linestyle="--")
axs[0,2].set_xlabel("number of firms", fontsize=x_fontsize)
axs[0,2].set_ylabel("\u20ac / person", fontsize=y_fontsize)
axs[0,2].set_title("total surplus", fontsize=title_fontsize)

# Set axis limits
margin_cs = 1.0 - np.sort(np.concatenate(tuple([normalize_var(cs_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]])])))[-6]
margin_ts = 1.0 - np.sort(np.concatenate(tuple([normalize_var(ts_bw(default_elast_id,default_nest_id)[:,bw_vals_argsort[bw_use][i]])])))[-6]
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
densities = lambda x,y: np.load(f"{paths.arrays_path}cntrfctl_densities_e{x}_n{y}.npy")
R_stars_dens = lambda x,y: np.load(f"{paths.arrays_path}R_stars_dens_e{x}_n{y}.npy")
axs.axvline(x=R_stars_dens(default_elast_id, default_nest_id)[3,np.argsort(densities(default_elast_id, default_nest_id))[1]], color="red", linestyle="dashed")
axs.axvline(x=R_stars_dens(default_elast_id, default_nest_id)[3,np.argsort(densities(default_elast_id, default_nest_id))[-2]], color="red", linestyle="dashed")

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

low_R = R_stars_dens(default_elast_id, default_nest_id)[3,np.argsort(densities(default_elast_id, default_nest_id))[1]] # France density
high_R = R_stars_dens(default_elast_id, default_nest_id)[3,np.argsort(densities(default_elast_id, default_nest_id))[-2]] # France contraharmonic mean density
data_R = np.mean(radius[np.isfinite(radius)])
axs.axvline(x=data_R, color="red", linestyle="dashed")
axs.axvline(x=low_R, color="red", linestyle="dashed")
axs.axvline(x=high_R, color="red", linestyle="dashed")

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
firm_wtp_monthly = partial_diffPif_partial_bf_allbw(default_elast_id,default_nest_id)[idx_4firms]
firm_wtp_discounted = 0.70
discount_factor_monthly = 1.0 - firm_wtp_monthly / firm_wtp_discounted
discount_factor_yearly = discount_factor_monthly**12.0
discount_rate_yearly = 1.0 / discount_factor_yearly - 1.0
if save_:
    create_file(f"{paths.stats_path}auction_firm_wtp_allbw.tex", f"{firm_wtp_monthly:.5f}")
    create_file(f"{paths.stats_path}auction_implied_discount_rate_allbw.tex", f"{discount_rate_yearly * 100.0:.2f}")
