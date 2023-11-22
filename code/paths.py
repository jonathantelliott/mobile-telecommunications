# Import packages
import numpy as np

# List of paths to folders where things are saved
scratch_path = "/scratch/jte254/telecom/"
home_path = "/home/jte254/telecom/"
data_path = scratch_path + "data/"
results_path = scratch_path + "results/"
arrays_path = results_path + "arrays/"
graphs_path = home_path + "graphs/"
res_path = results_path + "res/"
asym_draws_path = results_path + "asym_draws/"
dbar_path = results_path + "dbar/"
tables_path = home_path + "tables/"
stats_path = home_path + "stats/"

# Arrays of imputed parameters
# avg_price_elasts = np.array([-3.2, -2.5, -1.8])
# sigmas = np.array([0., 0.2, 0.4, 0.6, 0.8, 0.9])
avg_price_elasts = np.array([-2.36])
div_ratios = np.array([0.0356])
default_elast_id = 0
default_nest_id = 0

# Initial values for demand estimation
thetainit_p0 = np.array([[-1.55]])
thetainit_pz = np.array([[-1.1]])
thetainit_v = np.array([[0.25]])
thetainit_O = np.array([[3.0]])
thetainit_d0 = np.array([[0.8]])
thetainit_dz = np.array([[0.32]])
thetainit_c = np.array([[-8.75]])
thetainit_sigma = np.array([[np.log(0.8 / (1.0 - 0.8))]])

include_ROF_in_moments = False
