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
avg_price_elasts = np.array([-3.2, -2.5, -1.8])
sigmas = np.array([0.0, 0.5, 0.75, 0.85])
default_elast_id = 1
default_nest_id = 2

# Initial values for demand estimation
thetainit_p0 = np.array([[-0.25 , -0.875, -1.1, -1.6  ],
       [-0.5  , -1.125, -1.2, -1.75 ],
       [-0.75 , -1.375, -1.625, -2.   ]])
thetainit_pz = np.array([[-0.8       , -0.84166667, -1.1, -1.0],
       [-0.8       , -0.84166667, -1.1, -1.1],
       [-0.8       , -0.84166667, -0.85833333, -1.1]])
thetainit_v = np.array([[1.7   , 1.1375, 0.5, 0.5 ],
       [1.6   , 1.0375, 0.5, 0.0 ],
       [1.5   , 0.9375, 0.7125, 0.0 ]])
thetainit_O = np.array([[3.75      , 2.91666667, 3.5, 3.0],
       [3.5       , 2.66666667, 3.0, 3.0],
       [3.25      , 2.41666667, 2.08333333, 3.0]])
thetainit_d0 = np.array([[-1.3 , -0.55, 0.2,  0.5 ],
       [-0.8 , -0.05,  0.7,  1.0 ],
       [-0.3 ,  0.45,  0.75,  1.3 ]])
thetainit_dz = np.array([[0.3, 0.3, 0.34, 0.3],
       [0.3, 0.3, 0.3, 0.34],
       [0.3, 0.3, 0.3, 0.34]])
thetainit_c = np.array([[-6.3       , -7.00833333, -8.3, -8.7],
       [-7.3       , -8.00833333, -8.8, -8.71666667],
       [-8.3       , -9.00833333, -9.29166667, -9.71666667]])

include_ROF_in_moments = False
