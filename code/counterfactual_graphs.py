import numpy as np

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paths

import counterfactuals.infrastructurefunctions as infr
import counterfactuals.infrastructureequilibrium as ie

# %%
avg_price_elasts = np.array([-4., -2.5, -1.8])
sigmas = np.array([0., 0.2, 0.4, 0.6, 0.8, 0.9])

# %%
# Define functions to load results
p_stars = lambda x,y: np.load(f"{paths.arrays_path}p_stars_e{x}_n{y}.npy")
R_stars = lambda x,y: np.load(f"{paths.arrays_path}R_stars_e{x}_n{y}.npy")
q_stars = lambda x,y: np.load(f"{paths.arrays_path}q_stars_e{x}_n{y}.npy")
cs_by_type = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_e{x}_n{y}.npy")
cs = lambda x,y: np.load(f"{paths.arrays_path}cs_e{x}_n{y}.npy")
ps = lambda x,y: np.load(f"{paths.arrays_path}ps_e{x}_n{y}.npy")
ts = lambda x,y: np.load(f"{paths.arrays_path}ts_e{x}_n{y}.npy")
partial_elasts = lambda x,y: np.load(f"{paths.arrays_path}partial_elasts_e{x}_n{y}.npy")
full_elasts = lambda x,y: np.load(f"{paths.arrays_path}full_elasts_e{x}_n{y}.npy")
partial_Pif_partial_bf = lambda x,y: np.load(f"{paths.arrays_path}partial_Pif_partial_bf_e{x}_n{y}.npy")
partial_Pif_partial_b = lambda x,y: np.load(f"{paths.arrays_path}partial_Pif_partial_b_e{x}_n{y}.npy")
partial_CS_partial_b = lambda x,y: np.load(f"{paths.arrays_path}partial_CS_partial_b_e{x}_n{y}.npy")

p_stars_se = lambda x,y: np.load(f"{paths.arrays_path}p_stars_se_e{x}_n{y}.npy")
R_stars_se = lambda x,y: np.load(f"{paths.arrays_path}R_stars_se_e{x}_n{y}.npy")
q_stars_se = lambda x,y: np.load(f"{paths.arrays_path}q_stars_se_e{x}_n{y}.npy")
cs_by_type_se = lambda x,y: np.load(f"{paths.arrays_path}cs_by_type_se_e{x}_n{y}.npy")
cs_se = lambda x,y: np.load(f"{paths.arrays_path}cs_se_e{x}_n{y}.npy")
ps_se = lambda x,y: np.load(f"{paths.arrays_path}ps_se_e{x}_n{y}.npy")
ts_se = lambda x,y: np.load(f"{paths.arrays_path}ts_se_e{x}_n{y}.npy")
partial_elasts_se = lambda x,y: np.load(f"{paths.arrays_path}partial_elasts_se_e{x}_n{y}.npy")
full_elasts_se = lambda x,y: np.load(f"{paths.arrays_path}full_elasts_se_e{x}_n{y}.npy")
partial_Pif_partial_bf_se = lambda x,y: np.load(f"{paths.arrays_path}partial_Pif_partial_bf_se_e{x}_n{y}.npy")
partial_Pif_partial_b_se = lambda x,y: np.load(f"{paths.arrays_path}partial_Pif_partial_b_se_e{x}_n{y}.npy")
partial_CS_partial_b_se = lambda x,y: np.load(f"{paths.arrays_path}partial_CS_partial_b_se_e{x}_n{y}.npy")

# %%
# Define common graph features
num_firms_to_simulate = 6
num_firms_array = np.arange(num_firms_to_simulate, dtype=int) + 1
elast_ids = np.array([1, 2])[::-1]
alpha = 0.6
lw = 3.

# %%
# Plot effect of number of firms

fig, axs = plt.subplots(elast_ids.shape[0], 4, figsize=(15,3.5 * elast_ids.shape[0]), sharex=True)

for i, elast_id in enumerate(elast_ids):
    # dlim = 2,000 prices
    axs[i,0].plot(num_firms_array, p_stars(elast_id,3)[:,0], color="black", lw=lw, alpha=alpha)
    axs[i,0].plot(num_firms_array, p_stars(elast_id,3)[:,0] + 1.96 * p_stars_se(elast_id,3)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,0].plot(num_firms_array, p_stars(elast_id,3)[:,0] - 1.96 * p_stars_se(elast_id,3)[:,0], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,0].set_xlabel("number of firms")
    axs[i,0].set_ylabel("$p_{j}^{*}$ (in \u20ac)")
    
    # dlim = 10,000 prices
    axs[i,1].plot(num_firms_array, p_stars(elast_id,3)[:,1], color="black", lw=lw, alpha=alpha)
    axs[i,1].plot(num_firms_array, p_stars(elast_id,3)[:,1] + 1.96 * p_stars_se(elast_id,3)[:,1], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,1].plot(num_firms_array, p_stars(elast_id,3)[:,1] - 1.96 * p_stars_se(elast_id,3)[:,1], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,1].set_xlabel("number of firms")
    axs[i,1].set_ylabel("$p_{j}^{*}$ (in \u20ac)")

    # investment
    axs[i,2].plot(num_firms_array, R_stars(elast_id,3), color="black", label=f"{-avg_price_elasts[i]}", lw=lw, alpha=alpha)
    axs[i,2].plot(num_firms_array, R_stars(elast_id,3) + 1.96 * R_stars_se(elast_id,3), color="black", label=f"{-avg_price_elasts[i]}", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,2].plot(num_firms_array, R_stars(elast_id,3) - 1.96 * R_stars_se(elast_id,3), color="black", label=f"{-avg_price_elasts[i]}", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,2].set_xlabel("number of firms")
    axs[i,2].set_ylabel("$R_{f}^{*}$ (in km)")

    # download speeds
    axs[i,3].plot(num_firms_array, q_stars(elast_id,3), color="black", lw=lw, alpha=alpha)
    axs[i,3].plot(num_firms_array, q_stars(elast_id,3) + 1.96 * q_stars_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,3].plot(num_firms_array, q_stars(elast_id,3) - 1.96 * q_stars_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,3].set_xlabel("number of firms")
    axs[i,3].set_ylabel("$q_{f}^{*}$ (in Mbps)")

# Set titles
fontsize = 13.5
pad = 14
cols = ["2$\,$000 MB plan prices", "10$\,$000 MB plan prices", "investment", "download speeds"]
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=fontsize, ha='center', va='baseline', weight="bold")
mathbfE = "$\\mathbf{E}$"
rows = [f"{mathbfE} = {-avg_price_elasts[elast_id]}" for elast_id in elast_ids]
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=fontsize, ha='right', va='center', weight="bold")

# Set axis limits
min_y_p = np.min(np.concatenate(tuple([p_stars(elast_id,3) for elast_id in elast_ids]))) - 5.
max_y_p = np.max(np.concatenate(tuple([p_stars(elast_id,3) for elast_id in elast_ids]))) + 3.
min_y_R = np.min(np.concatenate(tuple([R_stars(elast_id,3) for elast_id in elast_ids]))) - 0.1
max_y_R = np.max(np.concatenate(tuple([R_stars(elast_id,3) for elast_id in elast_ids]))) + 0.1
min_y_q = np.min(np.concatenate(tuple([q_stars(elast_id,3) for elast_id in elast_ids]))) - 5.
max_y_q = np.max(np.concatenate(tuple([q_stars(elast_id,3) for elast_id in elast_ids]))) + 5.
for i, elast_id in enumerate(elast_ids):
    for j in range(2): # first two columns
        axs[i,j].set_ylim((min_y_p, max_y_p))
    axs[i,2].set_ylim((min_y_R, max_y_R))
    axs[i,3].set_ylim((min_y_q, max_y_q))
    for j in range(4): # all columns
        axs[i,j].set_xticks(num_firms_array)

plt.tight_layout()

plt.savefig(f"{paths.graphs_path}counterfactual_variables.pdf", bbox_inches = "tight")

# %%
# Plot elasticities

fig, axs = plt.subplots(elast_ids.shape[0], 2, figsize=(8,3.5 * elast_ids.shape[0]), sharex=True)

for i, elast_id in enumerate(elast_ids):
    # dlim = 2,000 elasticities
    axs[i,0].plot(num_firms_array, partial_elasts(elast_id,3)[:,0], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], lw=lw, alpha=alpha, label="partial")
    axs[i,0].plot(num_firms_array, full_elasts(elast_id,3)[:,0], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], lw=lw, alpha=alpha, label="full")
    axs[i,0].set_xlabel("number of firms")
    axs[i,0].legend(loc="lower left")
    
    # dlim = 10,000 elasticities
    axs[i,1].plot(num_firms_array, partial_elasts(elast_id,3)[:,1], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], lw=lw, alpha=alpha, label="partial")
    axs[i,1].plot(num_firms_array, full_elasts(elast_id,3)[:,1], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], lw=lw, alpha=alpha, label="full")
    axs[i,1].set_xlabel("number of firms")
    axs[i,1].legend(loc="lower left")

# Set titles
fontsize = 13.5
pad = 14
cols = ["2$\,$000 MB plan", "10$\,$000 MB plan"]
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=fontsize, ha='center', va='baseline', weight="bold")
mathbfE = "$\\mathbf{E}$"
rows = [f"{mathbfE} = {-avg_price_elasts[elast_id]}" for elast_id in elast_ids]
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=fontsize, ha='right', va='center', weight="bold")

# Set axis limits
min_y = np.min(np.concatenate(tuple([full_elasts(elast_id,3) for elast_id in elast_ids] + [partial_elasts(elast_id,3) for elast_id in elast_ids]))) - 0.3
max_y = np.max(np.concatenate(tuple([full_elasts(elast_id,3) for elast_id in elast_ids] + [partial_elasts(elast_id,3) for elast_id in elast_ids]))) + 0.3
for i, elast_id in enumerate(elast_ids):
    for j in range(2): # all columns
        axs[i,j].set_ylim((min_y, max_y))
        axs[i,j].set_xticks(num_firms_array)
        

plt.tight_layout()

plt.savefig(f"{paths.graphs_path}counterfactual_elasticities.pdf", bbox_inches = "tight")

# %%
# Plot bw derivatives

fig, axs = plt.subplots(elast_ids.shape[0], 3, figsize=(11,3.5 * elast_ids.shape[0]), sharex=True)

for i, elast_id in enumerate(elast_ids):
    # partial_Pif_partial_bf
    axs[i,0].plot(num_firms_array, partial_Pif_partial_bf(elast_id,3), color="black", lw=lw, alpha=alpha)
    axs[i,0].plot(num_firms_array, partial_Pif_partial_bf(elast_id,3) + 1.96 * partial_Pif_partial_bf_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,0].plot(num_firms_array, partial_Pif_partial_bf(elast_id,3) - 1.96 * partial_Pif_partial_bf_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,0].set_xlabel("number of firms")
    axs[i,0].set_ylabel("\u20ac per person in market / MHz")
    
    # partial_Pif_partial_b
    axs[i,1].plot(num_firms_array, partial_Pif_partial_b(elast_id,3), color="black", lw=lw, alpha=alpha)
    axs[i,1].plot(num_firms_array, partial_Pif_partial_b(elast_id,3) + 1.96 * partial_Pif_partial_b_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,1].plot(num_firms_array, partial_Pif_partial_b(elast_id,3) - 1.96 * partial_Pif_partial_b_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,1].set_xlabel("number of firms")
    axs[i,1].set_ylabel("\u20ac per person in market / MHz")
    
    # partial_CS_partial_b
    axs[i,2].plot(num_firms_array, partial_CS_partial_b(elast_id,3), color="black", lw=lw, alpha=alpha)
    axs[i,2].plot(num_firms_array, partial_CS_partial_b(elast_id,3) + 1.96 * partial_CS_partial_b_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,2].plot(num_firms_array, partial_CS_partial_b(elast_id,3) - 1.96 * partial_CS_partial_b_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,2].set_xlabel("number of firms")
    axs[i,2].set_ylabel("\u20ac per person in market / MHz")

# Set titles
fontsize = 13.5
pad = 14
cols = ["$\\frac{\\partial \\Pi_{f}}{\\partial b_{f}}$", "$\\frac{\\partial \\Pi_{f}}{\\partial b}$", "$\\frac{\\partial CS}{\\partial b}$"]
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=fontsize + 3., ha='center', va='baseline', weight="bold")
mathbfE = "$\\mathbf{E}$"
rows = [f"{mathbfE} = {-avg_price_elasts[elast_id]}" for elast_id in elast_ids]
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=fontsize, ha='right', va='center', weight="bold")

# Set axis limits
min_y_Pif_bf = np.min(np.concatenate(tuple([partial_Pif_partial_bf(elast_id,3) for elast_id in elast_ids]))) - 0.005
max_y_Pif_bf = np.max(np.concatenate(tuple([partial_Pif_partial_bf(elast_id,3) for elast_id in elast_ids]))) + 0.008
min_y_Pif_b = np.min(np.concatenate(tuple([partial_Pif_partial_b(elast_id,3) for elast_id in elast_ids]))) - 0.002
max_y_Pif_b = np.max(np.concatenate(tuple([partial_Pif_partial_b(elast_id,3) for elast_id in elast_ids]))) + 0.002
min_y_CS_b = np.min(np.concatenate(tuple([partial_CS_partial_b(elast_id,3) for elast_id in elast_ids]))) - 0.02
max_y_CS_b = np.max(np.concatenate(tuple([partial_CS_partial_b(elast_id,3) for elast_id in elast_ids]))) + 0.03
for i, elast_id in enumerate(elast_ids):
    axs[i,0].set_ylim((min_y_Pif_bf, max_y_Pif_bf))
    axs[i,1].set_ylim((min_y_Pif_b, max_y_Pif_b))
    axs[i,2].set_ylim((min_y_CS_b, max_y_CS_b))
    for j in range(3):
        axs[i,j].set_xticks(num_firms_array)
        
plt.tight_layout()

plt.savefig(f"{paths.graphs_path}counterfactual_bw_deriv.pdf", bbox_inches = "tight")

# %%
# Plot welfare for number of firms

fig, axs = plt.subplots(elast_ids.shape[0], 3, figsize=(11,3.5 * elast_ids.shape[0]), sharex=True)

for i, elast_id in enumerate(elast_ids):
    # consumer surplus
    axs[i,0].plot(num_firms_array, cs(elast_id,3), color="black", lw=lw, alpha=alpha)
    axs[i,0].plot(num_firms_array, cs(elast_id,3) + 1.96 * cs_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,0].plot(num_firms_array, cs(elast_id,3) - 1.96 * cs_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,0].axvline(x=num_firms_array[np.argmax(cs(elast_id,3))], color="black", linestyle="--", alpha=0.25)
    axs[i,0].set_xlabel("number of firms")
    axs[i,0].set_ylabel("\u20ac")
    
    # producer surplus
    axs[i,1].plot(num_firms_array, ps(elast_id,3), color="black", lw=lw, alpha=alpha)
    axs[i,1].plot(num_firms_array, ps(elast_id,3) + 1.96 * ps_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,1].plot(num_firms_array, ps(elast_id,3) - 1.96 * ps_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,1].axvline(x=num_firms_array[np.argmax(ps(elast_id,3))], color="black", linestyle="--", alpha=0.25)
    axs[i,1].set_xlabel("number of firms")
    axs[i,1].set_ylabel("\u20ac")
    
    # total surplus
    axs[i,2].plot(num_firms_array, ts(elast_id,3), color="black", lw=lw, alpha=alpha)
    axs[i,2].plot(num_firms_array, ts(elast_id,3) + 1.96 * ts_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,2].plot(num_firms_array, ts(elast_id,3) - 1.96 * ts_se(elast_id,3), color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
    axs[i,2].axvline(x=num_firms_array[np.argmax(ts(elast_id,3))], color="black", linestyle="--", alpha=0.25)
    axs[i,2].set_xlabel("number of firms")
    axs[i,2].set_ylabel("\u20ac")

# Set titles
fontsize = 13.5
pad = 14
cols = ["consumer surplus", "producer surplus", "total surplus"]
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=fontsize, ha='center', va='baseline', weight="bold")
mathbfE = "$\\mathbf{E}$"
rows = [f"{mathbfE} = {-avg_price_elasts[elast_id]}" for elast_id in elast_ids]
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=fontsize, ha='right', va='center', weight="bold")

# Set axis limits
min_y_cs = np.min(np.concatenate(tuple([cs(elast_id,3) for elast_id in elast_ids]))) - 5.
max_y_cs = np.max(np.concatenate(tuple([cs(elast_id,3) for elast_id in elast_ids]))) + 20.
min_y_ps = np.min(np.concatenate(tuple([ps(elast_id,3) for elast_id in elast_ids]))) - 5.
max_y_ps = np.max(np.concatenate(tuple([ps(elast_id,3) for elast_id in elast_ids]))) + 5.
min_y_ts = np.min(np.concatenate(tuple([ts(elast_id,3) for elast_id in elast_ids]))) - 5.
max_y_ts = np.max(np.concatenate(tuple([ts(elast_id,3) for elast_id in elast_ids]))) + 15.
for i, elast_id in enumerate(elast_ids):
    axs[i,0].set_ylim((min_y_cs, max_y_cs))
    axs[i,1].set_ylim((min_y_ps, max_y_ps))
    axs[i,2].set_ylim((min_y_ts, max_y_ts))
    for j in range(3):
        axs[i,j].set_xticks(num_firms_array)
        
plt.tight_layout()

plt.savefig(f"{paths.graphs_path}counterfactual_welfare.pdf", bbox_inches = "tight")

# %%
# Plot consumer surplus by type for number of firms

fig, axs = plt.subplots(elast_ids.shape[0], 5, figsize=(15,2.5 * elast_ids.shape[0]), sharex=True)

for i, elast_id in enumerate(elast_ids):
    for j in range(5):
        axs[i,j].plot(num_firms_array, cs_by_type(elast_id,3)[:,2*j], color="black", lw=lw, alpha=alpha)
        axs[i,j].plot(num_firms_array, cs_by_type(elast_id,3)[:,2*j] + 1.96 * cs_by_type_se(elast_id,3)[:,2*j], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
        axs[i,j].plot(num_firms_array, cs_by_type(elast_id,3)[:,2*j] - 1.96 * cs_by_type_se(elast_id,3)[:,2*j], color="black", lw=0.7 * lw, alpha=0.5 * alpha, ls="--")
        axs[i,j].axvline(x=num_firms_array[np.argmax(cs_by_type(elast_id,3)[:,2*j])], color="black", linestyle="--", alpha=0.25)
        axs[i,j].set_xlabel("number of firms")
        axs[i,j].set_ylabel("\u20ac")

# Set titles
fontsize = 13.5
pad = 14
cols = [f"{((2*i)+1)*10}th percentile" for i in range(5)]
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=fontsize, ha='center', va='baseline', weight="bold")
mathbfE = "$\\mathbf{E}$"
rows = [f"{mathbfE} = {-avg_price_elasts[elast_id]}" for elast_id in elast_ids]
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=fontsize, ha='right', va='center', weight="bold")
    
for i, elast_id in enumerate(elast_ids):
    for j in range(5):
        axs[i,j].set_xticks(num_firms_array)
        
plt.tight_layout()

plt.savefig(f"{paths.graphs_path}counterfactual_cs_by_income.pdf", bbox_inches = "tight")

# %%
# Plot effect of number of firms

num_firms_to_simulate = 6
num_firms_array = np.arange(num_firms_to_simulate, dtype=int) + 1

fig, axs = plt.subplots(1, 4, figsize=(14,4), sharex=True)
alpha = 0.6
lw = 3.

min_y = np.min(np.concatenate((p_stars(1,0), p_stars(1,1), p_stars(1,2), p_stars(1,3)))) - 2.5
max_y = np.max(np.concatenate((p_stars(1,0), p_stars(1,1), p_stars(1,2), p_stars(1,3)))) + 5.

for i in range(5):
    axs[0].plot(num_firms_array, p_stars(1,i)[:,0], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha)
# custom_lines = [Line2D([0], [0], color="black", lw=1.5),
#                 Line2D([0], [0], color="black", lw=1.5, ls="--")]
# axs[0].legend(custom_lines, ["$\\bar{d} = 2\\,000$ MB", "$\\bar{d} = 10\\,000$ MB"], loc="upper right")
axs[0].set_xlabel("number of firms")
axs[0].set_ylabel("$p^{*}$ (in \u20ac)")
axs[0].set_ylim((min_y, max_y))
axs[0].set_title("$\\bar{d} = 2\\,000$ MB plan prices", fontsize=12)

for i in range(5):
    axs[1].plot(num_firms_array, p_stars(1,i)[:,1], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha)
axs[1].set_xlabel("number of firms")
axs[1].set_ylabel("$p^{*}$ (in \u20ac)")
axs[1].set_ylim((min_y, max_y))
axs[1].set_title("$\\bar{d} = 10\\,000$ MB plan prices", fontsize=12)

for i in range(5):
    axs[2].plot(num_firms_array, num_firms_array * infr.num_stations(R_stars(1,i), 16.299135), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], label=f"{sigmas[i]}", lw=lw, alpha=alpha)
axs[2].set_xlabel("number of firms")
axs[2].set_ylabel("total number of stations")
axs[2].set_title("investment", fontsize=12)

for i in range(5):
    axs[3].plot(num_firms_array, q_stars(1,i), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha)
axs[3].set_xlabel("number of firms")
axs[3].set_ylabel("$q^{*}$ (in Mbps)")
axs[3].set_title("download speeds", fontsize=12)

fig.legend(loc="center right", ncol=1, title="Nesting Parameters", fontsize=12, bbox_to_anchor=(3., 0.5), bbox_transform=axs[2].transAxes)

plt.tight_layout()

plt.savefig(f"{paths.graphs_path}counterfactual_variables_sigmas.pdf", bbox_inches = "tight")

# %%
# Plot elasticities

fig, axs = plt.subplots(1, 2, figsize=(8,4), sharex=True)
alpha = 0.6
lw = 3.

min_y = np.min(np.concatenate((partial_elasts(1,0), partial_elasts(1,1), partial_elasts(1,2), partial_elasts(1,3), partial_elasts(1,4), full_elasts(1,0), full_elasts(1,1), full_elasts(1,2), full_elasts(1,3), full_elasts(1,4)))) - 0.2
max_y = np.max(np.concatenate((partial_elasts(1,0), partial_elasts(1,1), partial_elasts(1,2), partial_elasts(1,3), partial_elasts(1,4), full_elasts(1,0), full_elasts(1,1), full_elasts(1,2), full_elasts(1,3), full_elasts(1,4)))) + 0.2

for i in range(5):
    axs[0].plot(num_firms_array, partial_elasts(1,i)[:,0], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha, label=f"{sigmas[i]} partial")
    axs[0].plot(num_firms_array, full_elasts(1,i)[:,0], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha, linestyle="--", label=f"      full")
axs[0].set_xlabel("number of firms")
axs[0].set_ylim((min_y, max_y))
axs[0].set_title("$\\bar{d} = 2\\,000$ MB plan", fontsize=12)

for i in range(5):
    axs[1].plot(num_firms_array, partial_elasts(1,i)[:,1], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha)
    axs[1].plot(num_firms_array, full_elasts(1,i)[:,1], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha, linestyle="--")
axs[1].set_xlabel("number of firms")
axs[1].set_ylim((min_y, max_y))
axs[1].set_title("$\\bar{d} = 10\\,000$ MB plan", fontsize=12)

fig.legend(loc="center right", ncol=1, title="Nesting Parameters", fontsize=12, bbox_to_anchor=(1.565, 0.5), bbox_transform=axs[1].transAxes)

plt.tight_layout()

plt.savefig(f"{paths.graphs_path}counterfactual_elasticities_sigmas.pdf", bbox_inches = "tight")

# %%
# Plot bw derivatives

fig, axs = plt.subplots(1, 3, figsize=(11,4), sharex=True)

for i in range(5):
    axs[0].plot(num_firms_array, partial_Pif_partial_bf(1,i), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha, label=f"{sigmas[i]}")
axs[0].set_xlabel("number of firms")
axs[0].set_ylabel("\u20ac per person in market / MHz")
axs[0].set_title("$\\frac{\\partial \\Pi_{f}}{\\partial b_{f}}$", fontsize=17, y=1.05)

for i in range(5):
    axs[1].plot(num_firms_array, partial_Pif_partial_b(1,i), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha)
axs[1].set_xlabel("number of firms")
axs[1].set_ylabel("\u20ac per person in market / MHz")
axs[1].set_title("$\\frac{\\partial \\Pi_{f}}{\\partial b}$", fontsize=17, y=1.05)

for i in range(5):
    axs[2].plot(num_firms_array, partial_CS_partial_b(1,i), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha)
axs[2].set_xlabel("number of firms")
axs[2].set_ylabel("\u20ac per person in market / MHz")
axs[2].set_title("$\\frac{\\partial CS}{\\partial b}$", fontsize=17, y=1.05)

fig.legend(loc="center right", ncol=1, title="Nesting Parameters", fontsize=12, bbox_to_anchor=(1.75, 0.5), bbox_transform=axs[2].transAxes)

plt.tight_layout()

plt.savefig(f"{paths.graphs_path}counterfactual_bw_deriv_sigmas.pdf", bbox_inches = "tight")

# %%
# Plot welfare for number of firms

fig, axs = plt.subplots(1, 3, figsize=(10,4), sharex=True)
alpha = 0.6
lw = 3.

for i in range(5):
    axs[0].plot(num_firms_array, cs(1,i) / 10000., color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha, label=f"{sigmas[i]}")
    axs[0].axvline(x=num_firms_array[np.argmax(cs(1,i))] + (-1. * (i/5.) + 1. * ((5.-i)/5.)) * 0.15, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], linestyle="--", alpha=0.75 * alpha)
axs[0].set_xlabel("number of firms")
axs[0].set_ylabel("$10\\,000$ \u20ac")
axs[0].set_title("consumer surplus", fontsize=12)

for i in range(5):
    axs[1].plot(num_firms_array, ps(1,i) / 10000., color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha)
    axs[1].axvline(x=num_firms_array[np.argmax(ps(1,i))] + (-1. * (i/5.) + 1. * ((5.-i)/5.)) * 0.15, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], linestyle="--", alpha=0.75 * alpha)
axs[1].set_xlabel("number of firms")
axs[1].set_ylabel("$10\\,000$ \u20ac")
axs[1].set_title("producer surplus", fontsize=12)

for i in range(5):
    axs[2].plot(num_firms_array, ts(1,i) / 10000., color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], lw=lw, alpha=alpha)
    axs[2].axvline(x=num_firms_array[np.argmax(ts(1,i))] + (-1. * (i/5.) + 1. * ((5.-i)/5.)) * 0.15, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], linestyle="--", alpha=0.75 * alpha)
axs[2].set_xlabel("number of firms")
axs[2].set_ylabel("$10\\,000$ \u20ac")
axs[2].set_title("total surplus", fontsize=12)

fig.legend(loc="center right", ncol=1, title="Nesting Parameters", fontsize=12, bbox_to_anchor=(1.8, 0.5), bbox_transform=axs[2].transAxes)

plt.tight_layout()

plt.savefig(f"{paths.graphs_path}counterfactual_welfare_sigmas.pdf", bbox_inches = "tight")

# %%
# Plot consumer surplus by type for number of firms

fig, axs = plt.subplots(1, 5, figsize=(15,4.5), sharex=True)

for i in range(5):
    for j in range(5):
        axs[i].plot(num_firms_array, cs_by_type(1,j)[:,2*i], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][j], lw=lw, alpha=alpha, label=f"{sigmas[j]}" if i == 0 else None)
        axs[i].axvline(x=num_firms_array[np.argmax(cs_by_type(1,j)[:,2*i])] + (-1. * (j/5.) + 1. * ((5.-j)/5.)) * 0.15, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][j], linestyle="--", alpha=0.75 * alpha)
    axs[i].set_xlabel("number of firms")
    axs[i].set_ylabel("\u20ac")
    axs[i].set_title(f"{((2*i)+1)*10}th percentile", fontsize=12)
    
fig.legend(loc="center right", ncol=1, title="Nesting Parameters", fontsize=12, bbox_to_anchor=(4.5, 0.5), bbox_transform=axs[2].transAxes)
    
plt.tight_layout()

plt.savefig(f"{paths.graphs_path}counterfactual_cs_by_income_sigmas.pdf", bbox_inches = "tight")
