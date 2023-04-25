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

import pickle




# %%
print_ = False
save_ = True

# %%
avg_price_elasts = paths.avg_price_elasts
sigmas = paths.sigmas

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
bw_vals = lambda x,y: np.load(f"{paths.arrays_path}cntrfctl_bw_vals_e{elast_id}_n{nest_id}.npy")




# %%
# Define common graph features
num_firms_to_simulate = 6
num_firms_to_simulate_extend = 9
num_firms_array = np.arange(num_firms_to_simulate, dtype=int) + 1
num_firms_array_extend = np.arange(num_firms_to_simulate_extend, dtype=int) + 1
elast_ids = np.arange(avg_price_elasts.shape[0])
nest_ids = np.arange(sigmas.shape[0])
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

dlimidx = ds.chars.index(ds.dlimname)
dlims = ds.data[0,:,dlimidx]

def process_c_u(c_u):
    c_u_small = np.mean(c_u[dlims < 1000.0])
    c_u_medium = np.mean(c_u[(dlims >= 1000.0) & (dlims < 5000.0)])
    c_u_big = np.mean(c_u[dlims >= 5000.0])
    return np.array([c_u_small, c_u_medium, c_u_big])

to_tex = "\\begin{tabular}{c c c c c} \n"
to_tex += " & \\textbf{Nesting} & $\\bar{d} < 1\\,000$ & $1\\,000 \\leq \\bar{d} < 5\\,000$ & $\\bar{d} \\geq 5\\,000$ \\\\ \n"
to_tex += "\\textbf{Elasticity} & \\textbf{Parameter} & (in \\euro{}) & (in \\euro{}) & (in \\euro{}) \\\\ \n \t \\hline \n"
for i, elast_id in enumerate(elast_ids):
    to_tex += f"$\\boldsymbol{{{avg_price_elasts[elast_id]}}}$" if (elast_id == default_elast_id) else f"${avg_price_elasts[elast_id]}$"
    for j, nest_id in enumerate(nest_ids):
        bold_ = (elast_id == default_elast_id) and (nest_id == default_nest_id)
        to_tex += f" & $\\boldsymbol{{{sigmas[nest_id]}}}$ & " if bold_ else f" & ${sigmas[nest_id]}$ & "
        to_tex +=  " & ".join(f"$\\boldsymbol{{{c_u:.2f}}}$ " for c_u in process_c_u(c_u(elast_id,nest_id))) if bold_ else " & ".join(f"${c_u:.2f}$ " for c_u in process_c_u(c_u(elast_id,nest_id)))
        to_tex += " \\\\ \n"
        to_tex += " & & \\textbf{(}" if bold_ else " & & ("
        to_tex += "\\textbf{)} & \\textbf{(}".join(f"$\\boldsymbol{{{c_u:.2f}}}$" for c_u in process_c_u(c_u_se(elast_id,nest_id))) if bold_ else ") & (".join(f"${c_u:.2f}$" for c_u in process_c_u(c_u_se(elast_id,nest_id)))
        to_tex += "\\textbf{)} \\\\ \n" if bold_ else ") \\\\ \n"
to_tex += "\\hline \n" 
to_tex += "\\end{tabular} \n"
if save_:
    create_file(f"{paths.tables_path}c_u.tex", to_tex)
if print_:
    print(to_tex)
    
if save_:
    c_u_prefspec = process_c_u(c_u(default_elast_id, default_nest_id))
    create_file(f"{paths.stats_path}c_u_small.tex", f"{c_u_prefspec[0]:.2f}")
    create_file(f"{paths.stats_path}c_u_med.tex", f"{c_u_prefspec[1]:.2f}")
    create_file(f"{paths.stats_path}c_u_large.tex", f"{c_u_prefspec[2]:.2f}")




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

def process_c_R(c_R):
    return 200. * np.mean(c_R * 75.0, axis=0) # 200 to convert from monthly to perpetuity

def process_c_R_sd(c_R):
    return np.std(200. * c_R * 75.0, axis=0) # 200 to convert from monthly to perpetuity

to_tex = "\\begin{tabular}{c c c c c c} \n"
to_tex += " & \\textbf{Nesting} & \\textbf{Orange} & \\textbf{SFR} & \\textbf{Free} & \\textbf{Bouygues} \\\\ \n"
to_tex += "\\textbf{Elasticity} & \\textbf{Parameter} & (in \\euro{}) & (in \\euro{}) & (in \\euro{}) & (in \\euro{}) \\\\ \n \t \\hline \n"
for i, elast_id in enumerate(elast_ids):
    to_tex += f"$\\boldsymbol{{{avg_price_elasts[elast_id]}}}$" if (elast_id == default_elast_id) else f"${avg_price_elasts[elast_id]}$"
    for j, nest_id in enumerate(nest_ids):
        bold_ = (elast_id == default_elast_id) and (nest_id == default_nest_id)
        to_tex += f" & $\\boldsymbol{{{sigmas[nest_id]}}}$ & " if bold_ else f" & ${sigmas[nest_id]}$ & "
        to_tex += " & ".join(f"$\\boldsymbol{{{c_R:,.0f}}}$ ".replace(",","\\,") for c_R in process_c_R(c_R(elast_id,nest_id))) if bold_ else " & ".join(f"${c_R:,.0f}$ ".replace(",","\\,") for c_R in process_c_R(c_R(elast_id,nest_id)))
        to_tex += " \\\\ \n"
        to_tex += " & & \\textbf{(}" if bold_ else " & & ("
        to_tex += "\\textbf{)} & \\textbf{(}".join(f"$\\boldsymbol{{{c_R:,.0f}}}$".replace(",","\\,") for c_R in process_c_R_sd(c_R(elast_id,nest_id))) if bold_ else ") & (".join(f"${c_R:,.0f}$".replace(",","\\,") for c_R in process_c_R_sd(c_R(elast_id,nest_id)))
        to_tex += "\\textbf{)} \\\\ \n" if bold_ else ") \\\\ \n"
to_tex += "\\hline \n \\end{tabular}"
if save_:
    create_file(f"{paths.tables_path}c_R.tex", to_tex)
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




# %%
# Consumer surplus by type for "Add Free" (all fixed)

fig, axs = plt.subplots(1, 3, figsize=(9.0,4.0), squeeze=False)

x_fontsize = "large"
y_fontsize = "large"
title_fontsize = "x-large"

x_pos = [i for i in range(2)]
x_ticklabels = ["$3$ firms, $\\frac{4}{3}$ b", "$4$ firms, b"]

axs[0,0].bar(x_pos, cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,0], yerr=1.96*cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,0], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,0].set_xticks(x_pos)
axs[0,0].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,0].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,0] + 1.96 * cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,0])
min_cs = np.min(cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,0] - 1.96 * cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,0])
diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
axs[0,0].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
axs[0,0].set_title("10th percentile", fontsize=title_fontsize)

axs[0,1].bar(x_pos, cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,4], yerr=1.96*cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,4], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,1].set_xticks(x_pos)
axs[0,1].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,1].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,4] + 1.96 * cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,4])
min_cs = np.min(cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,4] - 1.96 * cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,4])
diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
axs[0,1].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
axs[0,1].set_title("50th percentile", fontsize=title_fontsize)

axs[0,2].bar(x_pos, cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,8], yerr=1.96*cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,8], capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,2].set_xticks(x_pos)
axs[0,2].set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=x_fontsize)
axs[0,2].set_ylabel("$\\Delta$ CS (in \u20ac / person)", fontsize=y_fontsize)
max_cs = np.max(cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,8] + 1.96 * cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,8])
min_cs = np.min(cs_by_type_free_allfixed(default_elast_id,default_nest_id)[:,8] - 1.96 * cs_by_type_free_allfixed_se(default_elast_id,default_nest_id)[:,8])
diff = np.maximum(max_cs, 0.0) - np.minimum(min_cs, 0.0)
axs[0,2].ticklabel_format(style="plain", useOffset=False, axis="y")
axs[0,2].set_ylim((np.minimum(min_cs - margin * diff, 0.0), np.maximum(max_cs + margin * diff, 0.0)))
axs[0,2].set_title("90th percentile", fontsize=title_fontsize)
        
plt.tight_layout()

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_free_cs_by_income_allfixed_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()




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
axs[0,0].set_ylim((35.25, 35.75))
axs[0,1].set_ylim((-20.0, -18.0))
axs[0,2].set_ylim((16.0, 17.0))

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
# Endogenous variables for all imputations

fig, axs = plt.subplots(2, 2, figsize=(11.0 * 1.5, 8.0 * 1.5), squeeze=False)

# Create axis indices and labels
num_x_axis_pts = elast_ids_sparse.shape[0] * nest_ids_sparse.shape[0] + (elast_ids_sparse.shape[0] - 1)
x_pos = [i for i in range(num_x_axis_pts)]
x_axis_idx = np.arange(elast_ids_sparse.shape[0] * nest_ids_sparse.shape[0] + (elast_ids_sparse.shape[0] - 1))
nest_idx = x_axis_idx % (nest_ids_sparse.shape[0] + 1)
x_ticklabels_nest = np.concatenate((sigmas, np.ones(1) * np.nan))[nest_idx]
elast_idx = x_axis_idx // (nest_ids_sparse.shape[0] + 1)
elast_idx_last_idx = np.unique(elast_idx, return_index=True)[1] - 1
elast_idx_last_idx = elast_idx_last_idx[elast_idx_last_idx >= 0]
elast_idx[elast_idx_last_idx] = np.max(elast_idx) + 1
x_ticklabels_elast = np.concatenate((avg_price_elasts, np.ones(1) * np.nan))[elast_idx]
use_x_pos = ~np.isnan(x_ticklabels_nest[np.array(x_pos)])
pos_preferred = np.array(x_pos)[(elast_idx == default_elast_id) & (nest_idx == default_nest_id)][0]

# Create position indices / labels for graph
x_ticklabels_elast_expand = np.concatenate((np.ones(1) * np.nan, x_ticklabels_elast, np.ones(1) * np.nan))
x_pos_expand = np.arange(-1, num_x_axis_pts + 1)
pos_begin = x_pos_expand[:-1][np.isnan(x_ticklabels_elast_expand[:-1])]
pos_end = x_pos_expand[1:][np.isnan(x_ticklabels_elast_expand[1:])]
pos_lines = np.unique(np.concatenate((pos_begin, pos_end)))[1:-1]
pos_begin = pos_begin + 1 - 0.5 # correct position, then want to begin a little to the left
pos_end = pos_end - 1 + 0.5 # correct position, then want to end a little to the right
elasts_vals_ = [f"E = {elast_}" for i, elast_ in enumerate(avg_price_elasts[elast_ids_sparse])]

# Populate arrays
p_stars_1gb_ = np.zeros(num_x_axis_pts)
p_stars_1gb_se_ = np.zeros(num_x_axis_pts)
p_stars_10gb_ = np.zeros(num_x_axis_pts)
p_stars_10gb_se_ = np.zeros(num_x_axis_pts)
num_stations_per_firm_stars_ = np.zeros(num_x_axis_pts)
num_stations_per_firm_stars_se_ = np.zeros(num_x_axis_pts)
q_stars_ = np.zeros(num_x_axis_pts)
q_stars_se_ = np.zeros(num_x_axis_pts)
for i in range(num_x_axis_pts):
    if (nest_idx[i] < nest_ids_sparse.shape[0]) and (elast_idx[i] < elast_ids_sparse.shape[0]):
        p_stars_1gb_[i] = p_stars(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx,0]
        p_stars_1gb_se_[i] = p_stars_se(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx,0]
        
        p_stars_10gb_[i] = p_stars(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx,1]
        p_stars_10gb_se_[i] = p_stars_se(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx,1]
        
        num_stations_per_firm_stars_[i] = num_stations_per_firm_stars(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        num_stations_per_firm_stars_se_[i] = num_stations_per_firm_stars_se(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        
        q_stars_[i] = q_stars(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        q_stars_se_[i] = q_stars_se(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
    else:
        p_stars_1gb_[i] = np.nan
        p_stars_1gb_se_[i] = np.nan
        
        p_stars_10gb_[i] = np.nan
        p_stars_10gb_se_[i] = np.nan
        
        num_stations_per_firm_stars_[i] = np.nan
        num_stations_per_firm_stars_se_[i] = np.nan
        
        q_stars_[i] = np.nan
        q_stars_se_[i] = np.nan
        
axs[0,0].bar(x_pos, p_stars_1gb_, yerr=p_stars_1gb_se_ * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,0].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize="x-large")
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize="xx-large")

axs[0,1].bar(x_pos, p_stars_10gb_, yerr=p_stars_10gb_se_ * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)
axs[0,1].set_ylabel("$p_{j}^{*}$ (in \u20ac)", fontsize="x-large")
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize="xx-large")

axs[1,0].bar(x_pos, num_stations_per_firm_stars_ * 1000.0, yerr=num_stations_per_firm_stars_se_ * 1.96 * 1000.0, capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,0].set_ylabel("number of stations (per 1000 people)", fontsize="x-large")
axs[1,0].set_title("number of stations / firm", fontsize="xx-large")

axs[1,1].bar(x_pos, q_stars_, yerr=q_stars_se_ * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)
axs[1,1].set_ylabel("$Q_{f}^{*}$ (in Mbps)", fontsize="x-large")
axs[1,1].set_title("download speeds", fontsize="xx-large")

# Set y axes
margin = 0.1
min_y_00 = np.nanmin(p_stars_1gb_ - 1.96 * p_stars_1gb_se_)
max_y_00 = np.nanmax(p_stars_1gb_ + 1.96 * p_stars_1gb_se_)
diff_y_00 = max_y_00 - min_y_00
min_y_01 = np.nanmin(p_stars_10gb_ - 1.96 * p_stars_10gb_se_)
max_y_01 = np.nanmax(p_stars_10gb_ + 1.96 * p_stars_10gb_se_)
diff_y_01 = max_y_01 - min_y_01
min_y_10 = np.nanmin(num_stations_per_firm_stars_ - 1.96 * num_stations_per_firm_stars_se_) * 1000.0
max_y_10 = np.nanmax(num_stations_per_firm_stars_ + 1.96 * num_stations_per_firm_stars_se_) * 1000.0
diff_y_10 = max_y_10 - min_y_10
min_y_11 = np.nanmin(q_stars_ - 1.96 * q_stars_se_)
max_y_11 = np.nanmax(q_stars_ + 1.96 * q_stars_se_)
diff_y_11 = max_y_11 - min_y_11
axs[0,0].set_ylim((min_y_00 - margin * diff_y_00, max_y_00 + margin * diff_y_00))
axs[0,1].set_ylim((min_y_01 - margin * diff_y_01, max_y_01 + margin * diff_y_01))
axs[1,0].set_ylim((min_y_10 - margin * diff_y_10, max_y_10 + margin * diff_y_10))
axs[1,1].set_ylim((min_y_11 - margin * diff_y_11, max_y_11 + margin * diff_y_11))

# Operations to all subplots
for i in range(2):
    for j in range(2):
        # Set axis labels
        axs[i,j].set_xticks(np.array(x_pos)[use_x_pos].tolist())
        axs[i,j].set_xticklabels(x_ticklabels_nest[use_x_pos].tolist())
        axs[i,j].set_xlabel("")
#         axs[i,j].get_xticklabels()[pos_preferred - 1].set_color("white")
#         axs[i,j].get_xticklabels()[pos_preferred - 1].set_fontsize("x-large")
        axs[i,j].get_xticklabels()[pos_preferred - 1].set_weight("bold")
#         axs[i,j].get_xticklabels()[pos_preferred - 1].set_bbox(dict(facecolor="black", alpha=0.25))
        axs[i,j].vlines(pos_lines, 0, -0.1, color="black", lw=0.8, clip_on=False, transform=axs[i,j].get_xaxis_transform())
        for elast_, pos0, pos1 in zip(elasts_vals_, pos_begin, pos_end):
            text_add = axs[i,j].text((pos0 + pos1) / 2, -0.12, elast_, ha="center", clip_on=False, transform=axs[i,j].get_xaxis_transform(), fontsize="x-large")
            if elast_ == f"E = {avg_price_elasts[default_elast_id]}":
                text_add.set_weight("bold")
        
plt.tight_layout()

plt.subplots_adjust(hspace = 0.25)

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_variables_imputations.pdf", bbox_inches="tight", transparent=True)

if print_:
    plt.show()
    
if save_:
    create_file(f"{paths.stats_path}p_star_low_elastic.tex", "{:.2f}".format(p_stars(0,default_nest_id)[default_num_firm_idx,:][0]))
    create_file(f"{paths.stats_path}p_star_low_inelastic.tex", "{:.2f}".format(p_stars(avg_price_elasts.shape[0]-1,default_nest_id)[default_num_firm_idx,:][0]))
    create_file(f"{paths.stats_path}p_star_high_elastic.tex", "{:.2f}".format(p_stars(0,default_nest_id)[default_num_firm_idx,:][1]))
    create_file(f"{paths.stats_path}p_star_high_inelastic.tex", "{:.2f}".format(p_stars(avg_price_elasts.shape[0]-1,default_nest_id)[default_num_firm_idx,:][1]))




# %%
# Endogenous variables - number of firms - by elasticity

fig, axs = plt.subplots(2, 3, figsize=(12.0, 8.0), squeeze=False)

elast_color_p = "Blues" # plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
elast_color_R = "Reds"
elast_color_Rtot = "Reds"
elast_color_q = "Greens"
elast_color_pl = "Greens"
alphas_elast = np.linspace(0.25, 0.75, elast_ids_sparse.shape[0])

elast_legend = [f"E={avg_price_elasts[elast_id]}" for i, elast_id in enumerate(elast_ids_sparse)]

# dlim = 1,000 prices
for i, elast_id in enumerate(elast_ids_sparse):
    axs[0,0].plot(num_firms_array, p_stars(elast_id,default_nest_id)[:,0], color=cm.get_cmap(elast_color_p)(alphas_elast[i]), lw=lw, label=elast_legend[i])
axs[0,0].set_xlabel("number of firms")
axs[0,0].set_ylabel("$p_{j}^{*}$ (in \u20ac)")
axs[0,0].set_title("1$\,$000 MB plan prices", fontsize=title_fontsize)

# dlim = 10,000 prices
for i, elast_id in enumerate(elast_ids_sparse):
    axs[0,1].plot(num_firms_array, p_stars(elast_id,default_nest_id)[:,1], color=cm.get_cmap(elast_color_p)(alphas_elast[i]), lw=lw, label=elast_legend[i])
axs[0,1].set_xlabel("number of firms")
axs[0,1].set_ylabel("$p_{j}^{*}$ (in \u20ac)")
axs[0,1].set_title("10$\,$000 MB plan prices", fontsize=title_fontsize)

# radius
for i, elast_id in enumerate(elast_ids_sparse):
    axs[0,2].plot(num_firms_array, num_stations_per_firm_stars(elast_id,default_nest_id) * 1000.0, color=cm.get_cmap(elast_color_R)(alphas_elast[i]), lw=lw, label=elast_legend[i])
axs[0,2].set_xlabel("number of firms")
axs[0,2].set_ylabel("number of stations\n(per 1000 people)")
axs[0,2].set_title("number of stations / firm", fontsize=title_fontsize)

# total number of stations
for i, elast_id in enumerate(elast_ids_sparse):
    axs[1,0].plot(num_firms_array, num_stations_stars(elast_id,default_nest_id) * 1000.0, color=cm.get_cmap(elast_color_Rtot)(alphas_elast[i]), lw=lw, label=elast_legend[i])
axs[1,0].set_xlabel("number of firms")
axs[1,0].set_ylabel("number of stations\n(per 1000 people)")
axs[1,0].set_title("total number of stations", fontsize=title_fontsize)

# average path loss
# for i, elast_id in enumerate(elast_ids_sparse):
#     axs[1,1].plot(num_firms_array, avg_path_losses(elast_id,default_nest_id), color=cm.get_cmap(elast_color_pl)(alphas_elast[i]), lw=lw, label=elast_legend[i])
# axs[1,1].set_xlabel("number of firms")
# axs[1,1].set_ylabel("dB")
# axs[1,1].set_title("average path loss")
for i, elast_id in enumerate(elast_ids_sparse):
    axs[1,1].plot(num_firms_array, ccs_per_bw(elast_id,default_nest_id), color=cm.get_cmap(elast_color_pl)(alphas_elast[i]), lw=lw, label=elast_legend[i])
axs[1,1].set_xlabel("number of firms")
axs[1,1].set_ylabel("Mbps / MHz")
axs[1,1].set_title("channel capacity / unit bw", fontsize=title_fontsize)

# download speeds
for i, elast_id in enumerate(elast_ids_sparse):
    axs[1,2].plot(num_firms_array, q_stars(elast_id,default_nest_id), color=cm.get_cmap(elast_color_q)(alphas_elast[i]), lw=lw, label=elast_legend[i])
axs[1,2].set_xlabel("number of firms")
axs[1,2].set_ylabel("$Q_{f}^{*}$ (in Mbps)")
axs[1,2].set_title("download speeds", fontsize=title_fontsize)

# Set axis limits
min_y_p = np.nanmin(np.concatenate(tuple([p_stars(elast_id,default_nest_id)[1:,:] for i, elast_id in enumerate(elast_ids_sparse)])))
max_y_p = np.nanmax(np.concatenate(tuple([p_stars(elast_id,default_nest_id)[1:,:] for i, elast_id in enumerate(elast_ids_sparse)])))
min_y_num_stations_per_firm = np.nanmin(np.concatenate(tuple([num_stations_per_firm_stars(elast_id,default_nest_id) for i, elast_id in enumerate(elast_ids_sparse)]))) * 1000.0
max_y_num_stations_per_firm = np.nanmax(np.concatenate(tuple([num_stations_per_firm_stars(elast_id,default_nest_id) for i, elast_id in enumerate(elast_ids_sparse)]))) * 1000.0
min_y_num_stations = np.nanmin(np.concatenate(tuple([num_stations_stars(elast_id,default_nest_id) for i, elast_id in enumerate(elast_ids_sparse)]))) * 1000.0
max_y_num_stations = np.nanmax(np.concatenate(tuple([num_stations_stars(elast_id,default_nest_id) for i, elast_id in enumerate(elast_ids_sparse)]))) * 1000.0
min_y_q = np.nanmin(np.concatenate(tuple([q_stars(elast_id,default_nest_id) for i, elast_id in enumerate(elast_ids_sparse)])))
max_y_q = np.nanmax(np.concatenate(tuple([q_stars(elast_id,default_nest_id) for i, elast_id in enumerate(elast_ids_sparse)])))
# min_y_pl = np.nanmin(np.concatenate(tuple([avg_path_losses(elast_id,default_nest_id) for i, elast_id in enumerate(elast_ids_sparse)])))
# max_y_pl = np.nanmax(np.concatenate(tuple([avg_path_losses(elast_id,default_nest_id) for i, elast_id in enumerate(elast_ids_sparse)])))
diff_p = max_y_p - min_y_p
diff_num_stations_per_firm = max_y_num_stations_per_firm - min_y_num_stations_per_firm
diff_num_stations = max_y_num_stations - min_y_num_stations
diff_q = max_y_q - min_y_q
# diff_pl = max_y_pl - min_y_pl
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
    plt.savefig(f"{paths.graphs_path}counterfactual_elast_1gb10gb.pdf", bbox_inches = "tight", transparent=True)

if print_:
    plt.show()
    




# %%
# Welfare for all imputations

fig, axs = plt.subplots(3, 3, figsize=(8.0 * 1.5, 11.0 * 1.5), squeeze=False)

sigmas_colors = ["Blues", "Reds", "Greens"]#plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
alphas_sigmas = np.linspace(0.25, 0.75, sigmas.shape[0])
legend_loc = ["upper left", "upper right", "upper right"]

normalize_var = lambda x: (x - np.mean(x[1:])) / np.std(x[1:])

nest_legends = [f"$\\bf{{\\sigma = {sigmas[nest_id]}}}$" if nest_id == default_nest_id else f"$\\sigma = {sigmas[nest_id]}$" for nest_id in nest_ids_sparse]
for i, elast in enumerate(elast_ids_sparse):
    # Add lines by nesting parameter
    for j, nest_param in enumerate(nest_ids_sparse):
        axs[i,0].plot(num_firms_array_extend, normalize_var(cs(elast_ids_sparse[i],nest_ids_sparse[j])), color=cm.get_cmap(sigmas_colors[0])(alphas_sigmas[j]), lw=lw, label=nest_legends[j])
        axs[i,0].axvline(x=num_firms_array_extend[np.nanargmax(cs(elast_ids_sparse[i],nest_ids_sparse[j]))], color=cm.get_cmap(sigmas_colors[0])(alphas_sigmas[j]), linestyle="--", label=None)
        axs[i,1].plot(num_firms_array_extend, normalize_var(ps(elast_ids_sparse[i],nest_ids_sparse[j])), color=cm.get_cmap(sigmas_colors[1])(alphas_sigmas[j]), lw=lw, label=nest_legends[j])
        if not np.isnan(ps(elast_ids_sparse[i],nest_ids_sparse[j])[0]): # b/c sometimes monopoly case doesn't work, don't want misleading result
            axs[i,1].axvline(x=num_firms_array_extend[np.nanargmax(ps(elast_ids_sparse[i],nest_ids_sparse[j]))], color=cm.get_cmap(sigmas_colors[1])(alphas_sigmas[j]), linestyle="--", label=None)
        axs[i,2].plot(num_firms_array_extend, normalize_var(ts(elast_ids_sparse[i],nest_ids_sparse[j])), color=cm.get_cmap(sigmas_colors[2])(alphas_sigmas[j]), lw=lw, label=nest_legends[j])
        axs[i,2].axvline(x=num_firms_array_extend[np.nanargmax(ts(elast_ids_sparse[i],nest_ids_sparse[j]))], color=cm.get_cmap(sigmas_colors[2])(alphas_sigmas[j]), linestyle="--", label=None)
    
    # Label graphs
    for j in range(3):
        axs[i,j].set_xlabel("number of firms")
        #axs[i,j].set_ylabel("\u20ac")
        axs[i,j].set_yticks([])
        axs[i,j].set_xticks(num_firms_array_extend)
        axs[i,j].legend(loc=legend_loc[j])

# Set axis limits
for i, elast in enumerate(elast_ids_sparse):
    min_y_cs = np.nanmin(np.concatenate(tuple([normalize_var(cs(elast_ids_sparse[i],nest_ids_sparse[j]))[1:] for j, nest_param in enumerate(nest_ids_sparse)]))) # don't include monopoly case
    max_y_cs = np.nanmax(np.concatenate(tuple([normalize_var(cs(elast_ids_sparse[i],nest_ids_sparse[j]))[1:] for j, nest_param in enumerate(nest_ids_sparse)])))
    min_y_ps = np.nanmin(np.concatenate(tuple([normalize_var(ps(elast_ids_sparse[i],nest_ids_sparse[j]))[1:] for j, nest_param in enumerate(nest_ids_sparse)])))
    max_y_ps = np.nanmax(np.concatenate(tuple([normalize_var(ps(elast_ids_sparse[i],nest_ids_sparse[j]))[1:] for j, nest_param in enumerate(nest_ids_sparse)])))
    min_y_ts = np.nanmin(np.concatenate(tuple([normalize_var(ts(elast_ids_sparse[i],nest_ids_sparse[j]))[1:] for j, nest_param in enumerate(nest_ids_sparse)])))
    max_y_ts = np.nanmax(np.concatenate(tuple([normalize_var(ts(elast_ids_sparse[i],nest_ids_sparse[j]))[1:] for j, nest_param in enumerate(nest_ids_sparse)])))
    diff_cs = max_y_cs - min_y_cs
    diff_ps = max_y_ps - min_y_ps
    diff_ts = max_y_ts - min_y_ts
    axs[i,0].set_ylim((min_y_cs - margin * diff_cs, max_y_cs + margin * diff_cs))
    axs[i,1].set_ylim((min_y_ps - margin * diff_ps, max_y_ps + margin * diff_ps))
    axs[i,2].set_ylim((min_y_ts - margin * diff_ts, max_y_ts + margin * diff_ts))
    
cols = ["consumer surplus", "producer surplus", "total surplus"]
rows = ["E = {}".format(elast) for elast in avg_price_elasts[elast_ids_sparse]]
pad = 12
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords="axes fraction", textcoords="offset points",
                size="xx-large", ha="center", va="baseline")
for ax, row in zip(axs[:,0], rows):
    row_title = ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords="offset points",
                            size="xx-large", ha="right", va="center")
    if row == f"E = {avg_price_elasts[default_elast_id]}":
        row_title.set_weight("bold")
        
plt.tight_layout()

fig.subplots_adjust(left=0.15, top=0.95, hspace = 0.25)

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_welfare_imputations.pdf", bbox_inches="tight", transparent=True)

if print_:
    plt.show()
    
if save_:
    create_file(f"{paths.stats_path}elastic_val.tex", f"{avg_price_elasts[0]}")
    create_file(f"{paths.stats_path}max_cs_elastic.tex", f"{num_firms_array_extend[np.nanargmax(cs(0,default_nest_id))]}")
    create_file(f"{paths.stats_path}max_ts_elastic.tex", f"{num_firms_array_extend[np.nanargmax(ts(0,default_nest_id))]}")
    create_file(f"{paths.stats_path}midelastic_val.tex", f"{avg_price_elasts[1]}")
    create_file(f"{paths.stats_path}max_cs_midelastic.tex", f"{num_firms_array_extend[np.nanargmax(cs(1,default_nest_id))]}")
    create_file(f"{paths.stats_path}max_ts_midelastic.tex", f"{num_firms_array_extend[np.nanargmax(ts(1,default_nest_id))]}")
    create_file(f"{paths.stats_path}inelastic_val.tex", f"{avg_price_elasts[2]}")
    create_file(f"{paths.stats_path}max_cs_inelastic.tex", f"{num_firms_array_extend[np.nanargmax(cs(2,default_nest_id))]}")
    create_file(f"{paths.stats_path}max_ts_inelastic.tex", f"{num_firms_array_extend[np.nanargmax(ts(2,default_nest_id))]}")




# %%
# Bandwidth derivatives for all imputations

fig, axs = plt.subplots(3, 2, figsize=(8.0 * 1.5, 11.0 * 1.5), squeeze=False)

# Create axis indices and labels
num_x_axis_pts = elast_ids_sparse.shape[0] * nest_ids_sparse.shape[0] + (elast_ids_sparse.shape[0] - 1)
x_pos = [i for i in range(num_x_axis_pts)]
x_axis_idx = np.arange(elast_ids_sparse.shape[0] * nest_ids_sparse.shape[0] + (elast_ids_sparse.shape[0] - 1))
nest_idx = x_axis_idx % (nest_ids_sparse.shape[0] + 1)
x_ticklabels_nest = np.concatenate((sigmas, np.ones(1) * np.nan))[nest_idx]
elast_idx = x_axis_idx // (nest_ids_sparse.shape[0] + 1)
elast_idx_last_idx = np.unique(elast_idx, return_index=True)[1] - 1
elast_idx_last_idx = elast_idx_last_idx[elast_idx_last_idx >= 0]
elast_idx[elast_idx_last_idx] = np.max(elast_idx) + 1
x_ticklabels_elast = np.concatenate((avg_price_elasts, np.ones(1) * np.nan))[elast_idx]
use_x_pos = ~np.isnan(x_ticklabels_nest[np.array(x_pos)])
pos_preferred = np.array(x_pos)[(elast_idx == default_elast_id) & (nest_idx == default_nest_id)][0]

# Create position indices / labels for graph
x_ticklabels_elast_expand = np.concatenate((np.ones(1) * np.nan, x_ticklabels_elast, np.ones(1) * np.nan))
x_pos_expand = np.arange(-1, num_x_axis_pts + 1)
pos_begin = x_pos_expand[:-1][np.isnan(x_ticklabels_elast_expand[:-1])]
pos_end = x_pos_expand[1:][np.isnan(x_ticklabels_elast_expand[1:])]
pos_lines = np.unique(np.concatenate((pos_begin, pos_end)))[1:-1]
pos_begin = pos_begin + 1 - 0.5 # correct position, then want to begin a little to the left
pos_end = pos_end - 1 + 0.5 # correct position, then want to end a little to the right
elasts_vals_ = [f"E = {elast_}" for i, elast_ in enumerate(avg_price_elasts[elast_ids_sparse])]

# Populate arrays
partial_diffPif_partial_bf_allfixed_ = np.zeros(num_x_axis_pts)
partial_diffPif_partial_bf_allfixed_se_ = np.zeros(num_x_axis_pts)
partial_Pif_partial_b_allfixed_ = np.zeros(num_x_axis_pts)
partial_Pif_partial_b_allfixed_se_ = np.zeros(num_x_axis_pts)
partial_CS_partial_b_allfixed_ = np.zeros(num_x_axis_pts)
partial_CS_partial_b_allfixed_se_ = np.zeros(num_x_axis_pts)
partial_diffPif_partial_bf_allbw_ = np.zeros(num_x_axis_pts)
partial_diffPif_partial_bf_allbw_se_ = np.zeros(num_x_axis_pts)
partial_Pif_partial_b_allbw_ = np.zeros(num_x_axis_pts)
partial_Pif_partial_b_allbw_se_ = np.zeros(num_x_axis_pts)
partial_CS_partial_b_allbw_ = np.zeros(num_x_axis_pts)
partial_CS_partial_b_allbw_se_ = np.zeros(num_x_axis_pts)
for i in range(num_x_axis_pts):
    if (nest_idx[i] < nest_ids_sparse.shape[0]) and (elast_idx[i] < elast_ids_sparse.shape[0]):
        partial_diffPif_partial_bf_allfixed_[i] = partial_diffPif_partial_bf_allfixed(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        partial_diffPif_partial_bf_allfixed_se_[i] = partial_diffPif_partial_bf_allfixed_se(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        
        partial_Pif_partial_b_allfixed_[i] = partial_Pif_partial_b_allfixed(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        partial_Pif_partial_b_allfixed_se_[i] = partial_Pif_partial_b_allfixed_se(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        
        partial_CS_partial_b_allfixed_[i] = partial_CS_partial_b_allfixed(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        partial_CS_partial_b_allfixed_se_[i] = partial_CS_partial_b_allfixed_se(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        
        partial_diffPif_partial_bf_allbw_[i] = partial_diffPif_partial_bf_allbw(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        partial_diffPif_partial_bf_allbw_se_[i] = partial_diffPif_partial_bf_allbw_se(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        
        partial_Pif_partial_b_allbw_[i] = partial_Pif_partial_b_allbw(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        partial_Pif_partial_b_allbw_se_[i] = partial_Pif_partial_b_allbw_se(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        
        partial_CS_partial_b_allbw_[i] = partial_CS_partial_b_allbw(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
        partial_CS_partial_b_allbw_se_[i] = partial_CS_partial_b_allbw_se(elast_ids_sparse[elast_idx[i]],nest_ids_sparse[nest_idx[i]])[default_num_firm_idx]
    else:
        partial_diffPif_partial_bf_allfixed_[i] = np.nan
        partial_diffPif_partial_bf_allfixed_se_[i] = np.nan
        
        partial_Pif_partial_b_allfixed_[i] = np.nan
        partial_Pif_partial_b_allfixed_se_[i] = np.nan
        
        partial_CS_partial_b_allfixed_[i] = np.nan
        partial_CS_partial_b_allfixed_se_[i] = np.nan
        
        partial_diffPif_partial_bf_allbw_[i] = np.nan
        partial_diffPif_partial_bf_allbw_se_[i] = np.nan
        
        partial_Pif_partial_b_allbw_[i] = np.nan
        partial_Pif_partial_b_allbw_se_[i] = np.nan
        
        partial_CS_partial_b_allbw_[i] = np.nan
        partial_CS_partial_b_allbw_se_[i] = np.nan
        
axs[0,0].bar(x_pos, partial_diffPif_partial_bf_allfixed_, yerr=partial_diffPif_partial_bf_allfixed_se_ * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)

axs[0,1].bar(x_pos, partial_diffPif_partial_bf_allbw_, yerr=partial_diffPif_partial_bf_allbw_se_ * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)

axs[1,0].bar(x_pos, partial_Pif_partial_b_allfixed_, yerr=partial_Pif_partial_b_allfixed_se_ * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)

axs[1,1].bar(x_pos, partial_Pif_partial_b_allbw_, yerr=partial_Pif_partial_b_allbw_se_ * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)

axs[2,0].bar(x_pos, partial_CS_partial_b_allfixed_, yerr=partial_CS_partial_b_allfixed_se_ * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)

axs[2,1].bar(x_pos, partial_CS_partial_b_allbw_, yerr=partial_CS_partial_b_allbw_se_ * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)

# Set y axes
margin = 0.1
min_y_0 = np.minimum(np.nanmin(partial_diffPif_partial_bf_allfixed_ - 1.96 * partial_diffPif_partial_bf_allfixed_se_), np.nanmin(partial_diffPif_partial_bf_allbw_ - 1.96 * partial_diffPif_partial_bf_allbw_se_))
max_y_0 = np.maximum(np.nanmax(partial_diffPif_partial_bf_allfixed_ + 1.96 * partial_diffPif_partial_bf_allfixed_se_), np.nanmax(partial_diffPif_partial_bf_allbw_ + 1.96 * partial_diffPif_partial_bf_allbw_se_))
diff_y_0 = max_y_0 - min_y_0
min_y_1 = np.minimum(np.nanmin(partial_Pif_partial_b_allfixed_ - 1.96 * partial_Pif_partial_b_allfixed_se_), np.nanmin(partial_Pif_partial_b_allbw_ - 1.96 * partial_Pif_partial_b_allbw_se_))
max_y_1 = np.maximum(np.nanmax(partial_Pif_partial_b_allfixed_ + 1.96 * partial_Pif_partial_b_allfixed_se_), np.nanmax(partial_Pif_partial_b_allbw_ + 1.96 * partial_Pif_partial_b_allbw_se_))
diff_y_1 = max_y_1 - min_y_1
min_y_2 = np.minimum(np.nanmin(partial_CS_partial_b_allfixed_ - 1.96 * partial_CS_partial_b_allfixed_se_), np.nanmin(partial_CS_partial_b_allbw_ - 1.96 * partial_CS_partial_b_allbw_se_))
max_y_2 = np.maximum(np.nanmax(partial_CS_partial_b_allfixed_ + 1.96 * partial_CS_partial_b_allfixed_se_), np.nanmax(partial_CS_partial_b_allbw_ + 1.96 * partial_CS_partial_b_allbw_se_))
diff_y_2 = max_y_2 - min_y_2
for i in range(2):
    axs[0,i].set_ylim((min_y_0 - margin * diff_y_0, max_y_0 + margin * diff_y_0))
    axs[1,i].set_ylim((min_y_1 - margin * diff_y_1, max_y_1 + margin * diff_y_1))
    axs[2,i].set_ylim((min_y_2 - margin * diff_y_2, max_y_2 + margin * diff_y_2))

# Operations to all subplots
for i in range(3):
    for j in range(2):
        # Set axis labels
        axs[i,j].set_xticks(np.array(x_pos)[use_x_pos].tolist())
        axs[i,j].set_xticklabels(x_ticklabels_nest[use_x_pos].tolist())
        axs[i,j].set_xlabel("")
#         axs[i,j].get_xticklabels()[pos_preferred - 1].set_color("white")
#         axs[i,j].get_xticklabels()[pos_preferred - 1].set_fontsize("large")
        axs[i,j].get_xticklabels()[pos_preferred - 1].set_weight("bold")
#         axs[i,j].get_xticklabels()[pos_preferred - 1].set_bbox(dict(facecolor="black", alpha=0.25))
        axs[i,j].vlines(pos_lines, 0, -0.1, color="black", lw=0.8, clip_on=False, transform=axs[i,j].get_xaxis_transform())
        for elast_, pos0, pos1 in zip(elasts_vals_, pos_begin, pos_end):
            text_add = axs[i,j].text((pos0 + pos1) / 2, -0.12, elast_, ha="center", clip_on=False, transform=axs[i,j].get_xaxis_transform(), fontsize="x-large")
            if elast_ == f"E = {avg_price_elasts[default_elast_id]}":
                text_add.set_weight("bold")
        
cols = ["fixed cost", "bandwidth cost"]
rows = ["$\\frac{d \\Pi_{f}}{d B_{f}} - \\frac{d \\Pi_{f}}{d B_{f^{\\prime}}}$", "$\\frac{d \\Pi_{f}}{d B}$", "$\\frac{d CS}{d B}$"]
pad = 12
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords="axes fraction", textcoords="offset points",
                size="xx-large", ha="center", va="baseline")
for ax, row in zip(axs[:,0], rows):
    row_title = ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords="offset points",
                            size="xx-large", ha="right", va="center")
    row_title.set_fontsize(1.5 * row_title.get_fontsize()) # 50% larger
        
plt.tight_layout()

fig.subplots_adjust(left=0.15, top=0.95, hspace = 0.25)

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_bw_deriv_imputations.pdf", bbox_inches="tight", transparent=True)

if print_:
    plt.show()
    
if save_:
    # Reshape bandwidth derivative arrays to be elasticity x nesting param
    partial_diffPif_partial_bf_allfixed_reshape = np.reshape(partial_diffPif_partial_bf_allfixed_[use_x_pos], (elast_ids_sparse.shape[0], nest_ids_sparse.shape[0]))
    partial_diffPif_partial_bf_allbw_reshape = np.reshape(partial_diffPif_partial_bf_allbw_[use_x_pos], (elast_ids_sparse.shape[0], nest_ids_sparse.shape[0]))
    partial_CS_partial_b_allfixed_reshape = np.reshape(partial_CS_partial_b_allfixed_[use_x_pos], (elast_ids_sparse.shape[0], nest_ids_sparse.shape[0]))
    partial_CS_partial_b_allbw_reshape = np.reshape(partial_CS_partial_b_allbw_[use_x_pos], (elast_ids_sparse.shape[0], nest_ids_sparse.shape[0]))

    # Create ratios
    ratio_allfixed = np.round(partial_CS_partial_b_allfixed_reshape / partial_diffPif_partial_bf_allfixed_reshape, 1)
    ratio_allbw = np.round(partial_CS_partial_b_allbw_reshape / partial_diffPif_partial_bf_allbw_reshape, 1)

    ratio_allfixed_df = pd.DataFrame(ratio_allfixed, index=tuple([f"$E$ = {avg_price_elasts[elast_id]}" for i, elast_id in enumerate(elast_ids_sparse)]), columns=tuple([f"$\\sigma$ = {sigmas[nest_id]}" for i, nest_id in enumerate(nest_ids_sparse)]))
    ratio_allbw_df = pd.DataFrame(ratio_allbw, index=tuple([f"$E$ = {avg_price_elasts[elast_id]}" for i, elast_id in enumerate(elast_ids_sparse)]), columns=tuple([f"$\\sigma$ = {sigmas[nest_id]}" for i, nest_id in enumerate(nest_ids_sparse)]))

    create_file(f"{paths.stats_path}ratio_allfixed_elastic.tex", f"{ratio_allfixed[0,default_nest_id]:.1f}")
    create_file(f"{paths.stats_path}ratio_allfixed_midelastic.tex", f"{ratio_allfixed[1,default_nest_id]:.1f}")
    create_file(f"{paths.stats_path}ratio_allfixed_inelastic.tex", f"{ratio_allfixed[2,default_nest_id]:.1f}")

    create_file(f"{paths.stats_path}ratio_allbw_elastic.tex", f"{ratio_allbw[0,default_nest_id]:.1f}")
    create_file(f"{paths.stats_path}ratio_allbw_midelastic.tex", f"{ratio_allbw[1,default_nest_id]:.1f}")
    create_file(f"{paths.stats_path}ratio_allbw_inelastic.tex", f"{ratio_allbw[2,default_nest_id]:.1f}")
    
    if print_:
        print(ratio_allfixed_df)
        print(ratio_allbw_df)




# %%
# Welfare from "add Free" counterfactuals for all imputations

# Set up figure and grid specification
fig = plt.figure(figsize=(11.0 * 1.5, 8.0 * 1.5))
axs = {} # save axes in dictionary
gs_cs = GridSpec(elast_ids_sparse.shape[0],2)
gs_ps = GridSpec(elast_ids_sparse.shape[0],2)
gs_ts = GridSpec(elast_ids_sparse.shape[0],2)
leftmost = 0.02
rightmost = 1.0 - leftmost
inner_margin = 0.3
outer_margin = 0.075
section_size = (rightmost - leftmost) / 3.0
gs_cs.update(left=leftmost, right=leftmost + section_size - outer_margin / 2.0, wspace=inner_margin)
gs_ps.update(left=leftmost + section_size + outer_margin / 2.0, right=leftmost + 2.0 * section_size - outer_margin / 2.0, wspace=inner_margin)
gs_ts.update(left=leftmost + 2.0 * section_size + outer_margin / 2.0, right=rightmost, wspace=inner_margin)
for i in range(elast_ids_sparse.shape[0]):
    for j in range(2):
        axs[f"cs_{i},{j}"] = plt.subplot(gs_cs[i,j])
        axs[f"ps_{i},{j}"] = plt.subplot(gs_ps[i,j])
        axs[f"ts_{i},{j}"] = plt.subplot(gs_ts[i,j])
        
# Create axis indices and labels
num_x_axis_pts = 2 * nest_ids_sparse.shape[0] + (nest_ids_sparse.shape[0] - 1)
x_pos = [i for i in range(num_x_axis_pts)]
x_axis_idx = np.arange(num_x_axis_pts)
firms_idx = x_axis_idx % (2 + 1)
x_ticklabels_firms = np.array([3,4,np.nan])[firms_idx]
nest_idx = x_axis_idx // (2 + 1)
nest_idx_last_idx = np.unique(nest_idx, return_index=True)[1] - 1
nest_idx_last_idx = nest_idx_last_idx[nest_idx_last_idx >= 0]
nest_idx[nest_idx_last_idx] = np.max(nest_idx) + 1
x_ticklabels_nest = np.concatenate((sigmas, np.ones(1) * np.nan))[nest_idx]
use_x_pos = ~np.isnan(x_ticklabels_firms[np.array(x_pos)])

# Create position indices / labels for graph
x_ticklabels_nest_expand = np.concatenate((np.ones(1) * np.nan, x_ticklabels_nest, np.ones(1) * np.nan))
x_pos_expand = np.arange(-1, num_x_axis_pts + 1)
pos_begin = x_pos_expand[:-1][np.isnan(x_ticklabels_nest_expand[:-1])]
pos_end = x_pos_expand[1:][np.isnan(x_ticklabels_nest_expand[1:])]
pos_lines = np.unique(np.concatenate((pos_begin, pos_end)))[1:-1]
pos_begin = pos_begin + 1 - 0.5 # correct position, then want to begin a little to the left
pos_end = pos_end - 1 + 0.5 # correct position, then want to end a little to the right
nest_vals_ = [f"{nest_}" for i, nest_ in enumerate(sigmas[nest_ids_sparse])]
firms_vals_ = ["fixed cost", "bandwidth cost"]

# Populate arrays
cs_free_allfixed_ = np.zeros((elast_ids_sparse.shape[0],num_x_axis_pts))
cs_free_allfixed_se_ = np.zeros((elast_ids_sparse.shape[0],num_x_axis_pts))
ps_free_allfixed_ = np.zeros((elast_ids_sparse.shape[0],num_x_axis_pts))
ps_free_allfixed_se_ = np.zeros((elast_ids_sparse.shape[0],num_x_axis_pts))
ts_free_allfixed_ = np.zeros((elast_ids_sparse.shape[0],num_x_axis_pts))
ts_free_allfixed_se_ = np.zeros((elast_ids_sparse.shape[0],num_x_axis_pts))
cs_free_allbw_ = np.zeros((elast_ids_sparse.shape[0],num_x_axis_pts))
cs_free_allbw_se_ = np.zeros((elast_ids_sparse.shape[0],num_x_axis_pts))
ps_free_allbw_ = np.zeros((elast_ids_sparse.shape[0],num_x_axis_pts))
ps_free_allbw_se_ = np.zeros((elast_ids_sparse.shape[0],num_x_axis_pts))
ts_free_allbw_ = np.zeros((elast_ids_sparse.shape[0],num_x_axis_pts))
ts_free_allbw_se_ = np.zeros((elast_ids_sparse.shape[0],num_x_axis_pts))
for i in range(elast_ids_sparse.shape[0]):
    for j in range(num_x_axis_pts):
        if (firms_idx[j] < 2) and (nest_idx[j] < nest_ids_sparse.shape[0]):
            cs_free_allfixed_[i,j] = cs_free_allfixed(elast_ids_sparse[i],nest_ids_sparse[nest_idx[j]])[np.arange(2)[firms_idx[j]]]
            cs_free_allfixed_se_[i,j] = cs_free_allfixed_se(elast_ids_sparse[i],nest_ids_sparse[nest_idx[j]])[np.arange(2)[firms_idx[j]]]

            ps_free_allfixed_[i,j] = ps_free_allfixed(elast_ids_sparse[i],nest_ids_sparse[nest_idx[j]])[np.arange(2)[firms_idx[j]]]
            ps_free_allfixed_se_[i,j] = ps_free_allfixed_se(elast_ids_sparse[i],nest_ids_sparse[nest_idx[j]])[np.arange(2)[firms_idx[j]]]

            ts_free_allfixed_[i,j] = ts_free_allfixed(elast_ids_sparse[i],nest_ids_sparse[nest_idx[j]])[np.arange(2)[firms_idx[j]]]
            ts_free_allfixed_se_[i,j] = ts_free_allfixed_se(elast_ids_sparse[i],nest_ids_sparse[nest_idx[j]])[np.arange(2)[firms_idx[j]]]

            cs_free_allbw_[i,j] = cs_free_allbw(elast_ids_sparse[i],nest_ids_sparse[nest_idx[j]])[np.arange(2)[firms_idx[j]]]
            cs_free_allbw_se_[i,j] = cs_free_allbw_se(elast_ids_sparse[i],nest_ids_sparse[nest_idx[j]])[np.arange(2)[firms_idx[j]]]

            ps_free_allbw_[i,j] = ps_free_allbw(elast_ids_sparse[i],nest_ids_sparse[nest_idx[j]])[np.arange(2)[firms_idx[j]]]
            ps_free_allbw_se_[i,j] = ps_free_allbw_se(elast_ids_sparse[i],nest_ids_sparse[nest_idx[j]])[np.arange(2)[firms_idx[j]]]

            ts_free_allbw_[i,j] = ts_free_allbw(elast_ids_sparse[i],nest_ids_sparse[nest_idx[j]])[np.arange(2)[firms_idx[j]]]
            ts_free_allbw_se_[i,j] = ts_free_allbw_se(elast_ids_sparse[i],nest_ids_sparse[nest_idx[j]])[np.arange(2)[firms_idx[j]]]
        else:
            cs_free_allfixed_[i,j] = np.nan
            cs_free_allfixed_se_[i,j] = np.nan

            ps_free_allfixed_[i,j] = np.nan
            ps_free_allfixed_se_[i,j] = np.nan

            ts_free_allfixed_[i,j] = np.nan
            ts_free_allfixed_se_[i,j] = np.nan

            cs_free_allbw_[i,j] = np.nan
            cs_free_allbw_se_[i,j] = np.nan

            ps_free_allbw_[i,j] = np.nan
            ps_free_allbw_se_[i,j] = np.nan

            ts_free_allbw_[i,j] = np.nan
            ts_free_allbw_se_[i,j] = np.nan
        
# Fill in results
for i in range(elast_ids_sparse.shape[0]):
    axs[f"cs_{i},0"].bar(x_pos, cs_free_allfixed_[i,:], yerr=cs_free_allfixed_se_[i,:] * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)
    axs[f"cs_{i},1"].bar(x_pos, cs_free_allbw_[i,:], yerr=cs_free_allbw_se_[i,:] * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)
    
    axs[f"ps_{i},0"].bar(x_pos, ps_free_allfixed_[i,:], yerr=ps_free_allfixed_se_[i,:] * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)
    axs[f"ps_{i},1"].bar(x_pos, ps_free_allbw_[i,:], yerr=ps_free_allbw_se_[i,:] * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)
    
    axs[f"ts_{i},0"].bar(x_pos, ts_free_allfixed_[i,:], yerr=ts_free_allfixed_se_[i,:] * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)
    axs[f"ts_{i},1"].bar(x_pos, ts_free_allbw_[i,:], yerr=ts_free_allbw_se_[i,:] * 1.96, capsize=7.0, color="black", alpha=0.8 * alpha)

# Set y axes
margin = 0.1
for i in range(elast_ids_sparse.shape[0]):
    min_y_cs = np.minimum(np.nanmin(cs_free_allfixed_[i,:] - 1.96 * cs_free_allfixed_se_[i,:]), np.nanmin(cs_free_allbw_[i,:] - 1.96 * cs_free_allbw_se_[i,:]))
    max_y_cs = np.maximum(np.nanmax(cs_free_allfixed_[i,:] + 1.96 * cs_free_allfixed_se_[i,:]), np.nanmax(cs_free_allbw_[i,:] + 1.96 * cs_free_allbw_se_[i,:]))
    diff_y_cs = max_y_cs - min_y_cs
    min_y_ps = np.minimum(np.nanmin(ps_free_allfixed_[i,:] - 1.96 * ps_free_allfixed_se_[i,:]), np.nanmin(ps_free_allbw_[i,:] - 1.96 * ps_free_allbw_se_[i,:]))
    max_y_ps = np.maximum(np.nanmax(ps_free_allfixed_[i,:] + 1.96 * ps_free_allfixed_se_[i,:]), np.nanmax(ps_free_allbw_[i,:] + 1.96 * ps_free_allbw_se_[i,:]))
    diff_y_ps = max_y_ps - min_y_ps
    min_y_ts = np.minimum(np.nanmin(ts_free_allfixed_[i,:] - 1.96 * ts_free_allfixed_se_[i,:]), np.nanmin(ts_free_allbw_[i,:] - 1.96 * ts_free_allbw_se_[i,:]))
    max_y_ts = np.maximum(np.nanmax(ts_free_allfixed_[i,:] + 1.96 * ts_free_allfixed_se_[i,:]), np.nanmax(ts_free_allbw_[i,:] + 1.96 * ts_free_allbw_se_[i,:]))
    diff_y_ts = max_y_ts - min_y_ts
    for j in range(2):
        axs[f"cs_{i},{j}"].set_ylim((min_y_cs - margin * diff_y_cs, max_y_cs + margin * diff_y_cs))
        axs[f"ps_{i},{j}"].set_ylim((min_y_ps - margin * diff_y_ps, max_y_ps + margin * diff_y_ps))
        axs[f"ts_{i},{j}"].set_ylim((min_y_ts - margin * diff_y_ts, max_y_ts + margin * diff_y_ts))
    
# Operations to all subplots
for i in range(elast_ids_sparse.shape[0]):
    for surplus in ["cs", "ps", "ts"]:
        for j in range(2):
            # Set title
            if i == 0:
                axs[f"{surplus}_{i},{j}"].set_title(firms_vals_[j])
            
            # Set axis labels
            axs[f"{surplus}_{i},{j}"].set_xticks(np.array(x_pos)[use_x_pos].tolist())
            axs[f"{surplus}_{i},{j}"].set_xticklabels(x_ticklabels_firms[use_x_pos].astype(int).tolist())
            axs[f"{surplus}_{i},{j}"].set_xlabel("")
            axs[f"{surplus}_{i},{j}"].vlines(pos_lines, 0, -0.1, color="black", lw=0.8, clip_on=False, transform=axs[f"{surplus}_{i},{j}"].get_xaxis_transform())
            for nest_, pos0, pos1 in zip(nest_vals_, pos_begin, pos_end):
                text_add = axs[f"{surplus}_{i},{j}"].text((pos0 + pos1) / 2, -0.15, nest_, ha="center", clip_on=False, transform=axs[f"{surplus}_{i},{j}"].get_xaxis_transform(), fontsize="large")
                if nest_ == f"{sigmas[default_nest_id]}":
                    text_add.set_weight("bold")
        
cols = ["$\\Delta CS$", "$\\Delta PS$", "$\\Delta TS$"]
rows = ["E = {}".format(elast) for elast in avg_price_elasts]
pad = 12
axs_firstcol = [axs[f"{surplus}_0,0"] for surplus in ["cs", "ps", "ts"]]
add_pad = 10 # want more because used title for cost specification
for ax, col in zip(axs_firstcol, cols):
    title = ax.annotate(col, xy=(1 + inner_margin / 2.0, 1), xytext=(0, pad + add_pad), 
                        xycoords="axes fraction", textcoords="offset points", 
                        size="xx-large", ha="center", va="baseline")
    title.set_fontsize(1.5 * title.get_fontsize()) # 50% larger than xx-large
axs_firstcol = [axs[f"cs_{i},0"] for i in range(elast_ids_sparse.shape[0])]
for ax, row in zip(axs_firstcol, rows):
    row_title = ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords="offset points",
                            size="xx-large", ha="right", va="center")
    if row == f"E = {avg_price_elasts[default_elast_id]}":
        row_title.set_weight("bold")

if save_:
    plt.savefig(f"{paths.graphs_path}counterfactual_free_welfare_imputations.pdf", bbox_inches="tight", transparent=True)

if print_:
    plt.show()
    
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
create_file(f"{paths.stats_path}xis_across_firms_stdev.tex", f"{xis_across_firms_stdev:.3}")
create_file(f"{paths.stats_path}xis_Org_stdev.tex", f"{xis_Org_stdev:.3}")
