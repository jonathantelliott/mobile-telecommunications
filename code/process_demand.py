# %%
# Import packages

import copy
from decimal import Decimal

import numpy as np
import pandas as pd

import matplotlib as mpl
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "serif"
plt.rcParams['mathtext.fontset'] = "dejavuserif"

import sys

import demand.demandsystem as demsys
import demand.dataexpressions as de
import demand.coefficients as coef
import demand.demandfunctions as blp

import variancematrix as vm

import paths

import pickle

# %%
# Imputated parameters

# Load the DemandSystem created when estimating demand
with open(f"{paths.data_path}demandsystem_0.obj", "rb") as file_ds:
    ds = pickle.load(file_ds)

yc1idx = ds.dim3.index(ds.demolist[0])
yclastidx = ds.dim3.index(ds.demolist[-1])
median_yc = np.median(ds.data[:,0,yc1idx:yclastidx+1], axis=0)
median_yc

qidx = ds.chars.index(ds.qname)
median_q = np.median(ds.data[:,:,qidx])
median_q

def create_file(file_name, file_contents):
    """Create file with name file_name and content file_contents"""
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()
    
# %%
# Process demand results

# Start table
demand_parameters = "\\begin{tabular}{l c c c c c c c c c c c c c c c c} \\hline \n" 

# Estimates
demand_parameters += "\\textit{Estimates} & & $\\hat{\\theta}_{p0}$ & & $\\hat{\\theta}_{pz}$ & & $\\hat{\\theta}_{v}$ & & $\\hat{\\theta}_{O}$ & & $\\hat{\\theta}_{d 0}$ & & $\\hat{\\theta}_{d z}$ & & $\\hat{\\theta}_{c}$ & & $\\hat{\\sigma}$ \\\\ \n"
demand_parameters += "\\cline{3-3} \\cline{5-5} \\cline{7-7} \\cline{9-9} \\cline{11-11} \\cline{13-13} \\cline{15-15} \\cline{17-17} \n"
demand_parameters += " & & " + " & & ".join(f"${param:.3f}$" for param in np.load(f"{paths.arrays_path}thetahat_0.npy")[:-2]) # not \theta_c or \theta_\sigma
log_theta_c_est = np.load(f"{paths.arrays_path}thetahat_0.npy")[-2]
theta_c_est = np.exp(log_theta_c_est)
demand_parameters += f" & & ${Decimal(theta_c_est):.3e}}}$".replace("e", "\\mathrm{e}{")
transform_theta_sigma_est = np.load(f"{paths.arrays_path}thetahat_0.npy")[-1]
theta_sigma_est = np.exp(transform_theta_sigma_est) / (1.0 + np.exp(transform_theta_sigma_est))
demand_parameters += f" & & ${theta_sigma_est:.3f}$"
demand_parameters += " \\\\ \n"

# Standard errors
demand_parameters += " & & " + " & & ".join(f"(${param:.3f}$)" for param in np.load(f"{paths.arrays_path}stderrs_0.npy")[:-2])
G = np.load(f"{paths.arrays_path}Gn_0.npy")
W = np.load(f"{paths.arrays_path}What_0.npy")
varmatrix = vm.V(G, W, np.linalg.inv(W))
varmatrix_log_theta_c = varmatrix[-2,-2]
varmatrix_exp_log_theta_c = varmatrix_log_theta_c * np.exp(log_theta_c_est)**2.0
exp_log_theta_c_hat_stderr = np.sqrt(varmatrix_exp_log_theta_c / ds.num_markets_moms)
demand_parameters += f" & & (${Decimal(exp_log_theta_c_hat_stderr):.3e}}}$)".replace("e", "\\mathrm{e}{")
varmatrix_transform_theta_sigma = varmatrix[-1,-1]
varmatrix_logit_transform_theta_sigma = varmatrix_transform_theta_sigma * (np.exp(transform_theta_sigma_est) / (1.0 + np.exp(transform_theta_sigma_est))**2.0)**2.0
logit_transform_theta_sigma_hat_stderr = np.sqrt(varmatrix_logit_transform_theta_sigma / ds.num_markets_moms)
demand_parameters += f" & & (${logit_transform_theta_sigma_hat_stderr:.3f}$)"
demand_parameters += f" \\\\ \n"

# Willingness to pay
demand_parameters += " & & & & & & & & & & & & & & & & \\\\ \n"
demand_parameters += f"\\multicolumn{{7}}{{l}}{{\\textit{{Willingness to pay for}}}} & & 10th \\%ile & & 30th \\%ile & & 50th \\%ile & & 70th \\%ile & & 90th \\%ile \\\\ \n"
demand_parameters += "\\cline{9-9} \\cline{11-11} \\cline{13-13} \\cline{15-15} \\cline{17-17} \n"

# Increase in data limit
create_file(f"{paths.stats_path}q_med.tex", f"{np.round(median_q, 1)}")
def wtp_dlim(theta):
    Q = np.array([[median_q]])
    xbar_low = np.array([[1000.]])
    xbar_high = np.array([[4000.]])
    E_u_high = de.E_u(ds, theta, ds.data, Q, xbar_high, median_yc[np.newaxis,np.newaxis,:])[0,0,:]
    E_u_low = de.E_u(ds, theta, ds.data, Q, xbar_low, median_yc[np.newaxis,np.newaxis,:])[0,0,:]
    theta_p = coef.theta_pi(ds, theta, median_yc[np.newaxis,np.newaxis,:])[0,0,:]
    wtp = (E_u_high - E_u_low) / theta_p
    return wtp
demand_parameters += f"\\multicolumn{{2}}{{l}}{{}} & \\multicolumn{{5}}{{l}}{{1 GB plan $\\rightarrow$ 4 GB plan}} & & "
demand_parameters += " & & ".join(f"${wtp:.2f}$ " + "\\euro{} " for wtp in wtp_dlim(np.load(f"{paths.arrays_path}thetahat_0.npy"))[(np.arange(9) + 1) % 2 == 1])
demand_parameters += " \\\\ \n"
wtp_dlim_low = wtp_dlim(np.load(f"{paths.arrays_path}thetahat_0.npy"))[0]
wtp_dlim_high = wtp_dlim(np.load(f"{paths.arrays_path}thetahat_0.npy"))[-1]
create_file(f"{paths.stats_path}wtp_dlim_low.tex", f"{np.round(wtp_dlim_low, 2):.2f}")
create_file(f"{paths.stats_path}wtp_dlim_high.tex", f"{np.round(wtp_dlim_high, 2):.2f}")

# Unlimited voice
def wtp_v(theta):
    theta_p = coef.theta_pi(ds, theta, median_yc[np.newaxis,np.newaxis,:])[0,0,:]
    wtp = theta[coef.v] / theta_p
    return wtp
demand_parameters += f"\\multicolumn{{2}}{{l}}{{}} & \\multicolumn{{5}}{{l}}{{unlimited voice}} & & "
demand_parameters += " & & ".join(f"${wtp:.2f}$ " + "\\euro{} " for wtp in wtp_v(np.load(f"{paths.arrays_path}thetahat_0.npy"))[(np.arange(9) + 1) % 2 == 1])
demand_parameters += " \\\\ \n"
wtp_v_med = wtp_v(np.load(f"{paths.arrays_path}thetahat_0.npy"))[4]
create_file(f"{paths.stats_path}wtp_v_med.tex", f"{np.round(wtp_v_med, 2):.2f}")

# Increase in download speed
def wtp_q(theta):
    xbar = np.array([[10000.]])
    Q_low = np.array([[10.]])
    Q_high = np.array([[20.]])
    E_u_high = de.E_u(ds, theta, ds.data, Q_high, xbar, median_yc[np.newaxis,np.newaxis,:])[0,0,:]
    E_u_low = de.E_u(ds, theta, ds.data, Q_low, xbar, median_yc[np.newaxis,np.newaxis,:])[0,0,:]
    theta_p = coef.theta_pi(ds, theta, median_yc[np.newaxis,np.newaxis,:])[0,0,:]
    wtp = (E_u_high - E_u_low) / theta_p
    return wtp
demand_parameters += f"\\multicolumn{{2}}{{l}}{{}} & \\multicolumn{{5}}{{l}}{{10 Mbps $\\rightarrow$ 20 Mbps}} & & "
demand_parameters += " & & ".join(f"${wtp:.2f}$ " + "\\euro{} " for wtp in wtp_q(np.load(f"{paths.arrays_path}thetahat_0.npy"))[(np.arange(9) + 1) % 2 == 1])
demand_parameters += " \\\\ \n"
wtp_q_low = wtp_q(np.load(f"{paths.arrays_path}thetahat_0.npy"))[0]
wtp_q_high = wtp_q(np.load(f"{paths.arrays_path}thetahat_0.npy"))[-1]
create_file(f"{paths.stats_path}wtp_q_low.tex", f"{np.round(wtp_q_low, 2):.2f}")
create_file(f"{paths.stats_path}wtp_q_high.tex", f"{np.round(wtp_q_high, 2):.2f}")

# Finish table
demand_parameters += "\\hline \n"
demand_parameters += "\\end{tabular} \n"

# Export table
create_file(f"{paths.tables_path}demand_parameters.tex", demand_parameters)
print(demand_parameters)

# %%
# Create predicted vs actual data consumption

x_fontsize = "x-large"
y_fontsize = "x-large"
title_fontsize = "xx-large"

predicted_dbar = np.load(f"{paths.arrays_path}predicted_dbar_0.npy")
actual_dbar = np.load(f"{paths.arrays_path}actual_dbar_0.npy")

fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)
title = ['$\\bar{d} = 1\\,000$', '$\\bar{d} = 4\\,000$', '$\\bar{d} = 8\\,000$']
for j in range(3):
    col = j % 3
    axs[col].scatter(actual_dbar[:,j+2], predicted_dbar[:,j+2] * 1000, alpha=0.6)
    axs[col].plot(np.arange(0,6000,50), np.arange(0,6000,50))
    axs[col].set_title(title[j], fontsize=title_fontsize)
    axs[col].set_xlabel("actual (MB)", fontsize=x_fontsize)
    axs[col].set_ylabel("predicted (MB)", fontsize=y_fontsize)
fig.tight_layout()
plt.savefig(f"{paths.graphs_path}predict_vs_actual_dbar.pdf", bbox_inches = "tight")

fig, axs = plt.subplots(1, 1, figsize=(8.5, 4))
products = ['$\\bar{d} = 1\\,000$', '$\\bar{d} = 4\\,000$', '$\\bar{d} = 8\\,000$']
markers = ["o", "o", "x"]
alphas = [0.5, 0.5, 0.5]
for j in range(3):
    col = j % 3
    if j == 1:
        axs.scatter(actual_dbar[:,j+2], predicted_dbar[:,j+2] * 1000, alpha=alphas[j], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][j], label=products[j], marker=markers[j], facecolors="none", edgecolors=plt.rcParams['axes.prop_cycle'].by_key()['color'][j])
    else:
        axs.scatter(actual_dbar[:,j+2], predicted_dbar[:,j+2] * 1000, alpha=alphas[j], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][j], label=products[j], marker=markers[j])
axs.plot(np.arange(0,7000,50), np.arange(0,7000,50), linestyle="--", color="black", label="$45^{\\circ}$ line")
axs.set_xlabel("actual consumption (MB)", fontsize=x_fontsize)
axs.set_ylabel("predicted consumption (MB)", fontsize=y_fontsize)
leg = axs.legend(loc="center right", ncol=1, fontsize=12, bbox_to_anchor=(1.3, 0.5), bbox_transform=axs.transAxes)
for lh in leg.legendHandles: 
    lh.set_alpha(1.0)
fig.tight_layout()
plt.savefig(f"{paths.graphs_path}predict_vs_actual_dbar_alt.pdf", bbox_inches="tight")
plt.savefig(f"{paths.graphs_path}figure6.pdf", bbox_inches="tight")

create_file(f"{paths.stats_path}dbar_corr_low.tex", "{:.3f}".format(np.round(np.corrcoef(predicted_dbar[:,2], actual_dbar[:,2])[0,1], 3)))
create_file(f"{paths.stats_path}dbar_corr_med.tex", "{:.3f}".format(np.round(np.corrcoef(predicted_dbar[:,3], actual_dbar[:,3])[0,1], 3)))
create_file(f"{paths.stats_path}dbar_corr_high.tex", "{:.3f}".format(np.round(np.corrcoef(predicted_dbar[:,4], actual_dbar[:,4])[0,1], 3)))
