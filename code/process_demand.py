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
import demand.blpextension as blp
import demand.variancematrix as vm

import paths

import pickle

# %%
# Imputated parameters

avg_price_elasts = paths.avg_price_elasts
sigmas = paths.sigmas
default_elast_id = paths.default_elast_id
default_nest_id = paths.default_nest_id

# Load the DemandSystem created when estimating demand
with open(f"{paths.data_path}demandsystem.obj", "rb") as file_ds:
    ds = pickle.load(file_ds)
    
# Drop Rest-of-France market
market_idx = ds.dim3.index(ds.marketname)
market_numbers = np.max(ds.data[:,:,market_idx], axis=1)
ds.data = ds.data[market_numbers > 0,:,:] # drop "Rest of France"

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

demand_parameters = "\\begin{tabular}{c c c c c c c c c c} \n & \\textbf{Nesting} & & & & & & & & \\\\ \n\t \\textbf{Elasticity} & \\textbf{Parameter} & $\\hat{\\theta}_{p0}$ & $\\hat{\\theta}_{pz}$ & $\\hat{\\theta}_{v}$ & $\\hat{\\theta}_{O}$ & $\\hat{\\theta}_{d 0}$ & $\\hat{\\theta}_{d z}$ & $\\hat{\\theta}_{c}$ \\\\ \n \t \\hline \n"
for i in range(avg_price_elasts.shape[0]):
    demand_parameters += f"$\\boldsymbol{{{avg_price_elasts[i]}}}$ & " if i == default_elast_id else f"${avg_price_elasts[i]}$ & "
    for j in range(sigmas.shape[0]):
        bold_ = (i == default_elast_id) and (j == default_nest_id)
        demand_parameters += "" if j == 0 else " & "
        demand_parameters += f"$\\boldsymbol{{{sigmas[j]}}}$ & " if bold_ else f"${sigmas[j]}$ & "
        demand_parameters += " & ".join(f"$\\boldsymbol{{{np.round(param, 3)}}}$" for param in np.load(f"{paths.arrays_path}thetahat_e{i}_n{j}.npy")[:-1]) if bold_ else " & ".join(f"${np.round(param, 3)}$" for param in np.load(f"{paths.arrays_path}thetahat_e{i}_n{j}.npy")[:-1])
        log_theta_c_est = np.load(f"{paths.arrays_path}thetahat_e{i}_n{j}.npy")[-1]
        theta_c_est = np.exp(log_theta_c_est)
        demand_parameters += f" & $\\boldsymbol{{{Decimal(theta_c_est):.3e}}}}}$".replace("e", "\\mathrm{e}{") if bold_ else f" & ${Decimal(theta_c_est):.3e}}}$".replace("e", "\\mathrm{e}{")
        demand_parameters += " \\\\ \n"
        demand_parameters += " & & "
        demand_parameters += " & ".join(f"\\textbf{{(}}$\\boldsymbol{{{np.round(param, 3)}}}$\\textbf{{)}}" for param in np.load(f"{paths.arrays_path}stderrs_e{i}_n{j}.npy")[:-1]) if bold_ else " & ".join(f"(${np.round(param, 3)}$)" for param in np.load(f"{paths.arrays_path}stderrs_e{i}_n{j}.npy")[:-1])
        G = np.load(f"{paths.arrays_path}Gn_e{i}_n{j}.npy")
        W = np.load(f"{paths.arrays_path}What_e{i}_n{j}.npy")
        varmatrix = vm.V(G, W, np.linalg.inv(W))
        varmatrix_log_theta_c = varmatrix[-1,-1]
        varmatrix_exp_log_theta_c = varmatrix_log_theta_c * np.exp(log_theta_c_est)**2.0
        exp_log_theta_c_hat_stderr = np.sqrt(varmatrix_exp_log_theta_c / ds.num_markets_moms)
        demand_parameters += f" & \\txtbf{{(}}$\\boldsymbol{{{Decimal(exp_log_theta_c_hat_stderr):.3e}}}}}$\\txtbf{{)}}".replace("e", "\\mathrm{e}{").replace("txtbf", "textbf") if bold_ else f" & (${Decimal(exp_log_theta_c_hat_stderr):.3e}}}$)".replace("e", "\\mathrm{e}{")
        demand_parameters += " \\\\ \n"
demand_parameters += "\\hline \n \\end{tabular}"

create_file(f"{paths.tables_path}demand_parameters.tex", demand_parameters)
print(demand_parameters)

# %%
# Process willingness to pay for dlim

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

wtp_dlim_table = "\\begin{tabular}{c c c c c c c} \n & \\textbf{Nesting} & & & & & \\\\ \n\t \\textbf{Elasticity} & \\textbf{Parameter} & 10th \\%ile & 30th \\%ile & 50th \\%ile & 70th \\%ile & 90th \\%ile \\\\ \n \t \\hline \n"
for i in range(avg_price_elasts.shape[0]):
    wtp_dlim_table += f"$\\boldsymbol{{{avg_price_elasts[i]}}}$ & " if i == default_elast_id else f"${avg_price_elasts[i]}$ & "
    for j in range(sigmas.shape[0]):
        bold_ = (i == default_elast_id) and (j == default_nest_id)
        wtp_dlim_table += "" if j == 0 else " & "
        wtp_dlim_table += f"$\\boldsymbol{{{sigmas[j]}}}$ & " if bold_ else f"${sigmas[j]}$ & "
        wtp_dlim_table += " & ".join(f"$\\boldsymbol{{{wtp:.2f}}}$ " + "\\textbf{\\euro{}} " for wtp in wtp_dlim(np.concatenate((np.load(f"{paths.arrays_path}thetahat_e{i}_n{j}.npy"), np.array([sigmas[j]]))))[(np.arange(9) + 1) % 2 == 1]) if bold_ else " & ".join(f"${wtp:.2f}$ " + "\\euro{} " for wtp in wtp_dlim(np.concatenate((np.load(f"{paths.arrays_path}thetahat_e{i}_n{j}.npy"), np.array([sigmas[j]]))))[(np.arange(9) + 1) % 2 == 1])
        wtp_dlim_table += " \\\\ \n"
wtp_dlim_table += "\\hline \n \\end{tabular}"

create_file(f"{paths.tables_path}wtp_dlim.tex", wtp_dlim_table)
print(wtp_dlim_table)

wtp_dlim_low = wtp_dlim(np.load(f"{paths.arrays_path}thetahat_e{default_elast_id}_n{default_nest_id}.npy"))[0]
wtp_dlim_high = wtp_dlim(np.load(f"{paths.arrays_path}thetahat_e{default_elast_id}_n{default_nest_id}.npy"))[-1]
create_file(f"{paths.stats_path}wtp_dlim_low.tex", f"{np.round(wtp_dlim_low, 2):.2f}")
create_file(f"{paths.stats_path}wtp_dlim_high.tex", f"{np.round(wtp_dlim_high, 2):.2f}")

# %%
# Process willingness to pay for v unlimited

def wtp_v(theta):
    theta_p = coef.theta_pi(ds, theta, median_yc[np.newaxis,np.newaxis,:])[0,0,:]
    wtp = theta[coef.v] / theta_p
    return wtp

wtp_v_table = "\\begin{tabular}{c c c c c c c} \n & \\textbf{Nesting} & & & & & \\\\ \n\t \\textbf{Elasticity} & \\textbf{Parameter} & 10th \\%ile & 30th \\%ile & 50th \\%ile & 70th \\%ile & 90th \\%ile \\\\ \n \t \\hline \n"
for i in range(avg_price_elasts.shape[0]):
    wtp_v_table += f"$\\boldsymbol{{{avg_price_elasts[i]}}}$ & " if i == default_elast_id else f"${avg_price_elasts[i]}$ & "
    for j in range(sigmas.shape[0]):
        bold_ = (i == default_elast_id) and (j == default_nest_id)
        wtp_v_table += "" if j == 0 else " & "
        wtp_v_table += f"$\\boldsymbol{{{sigmas[j]}}}$ & " if bold_ else f"${sigmas[j]}$ & "
        wtp_v_table += " & ".join(f"$\\boldsymbol{{{wtp:.2f}}}$ " + "\\textbf{\\euro{}} " for wtp in wtp_v(np.concatenate((np.load(f"{paths.arrays_path}thetahat_e{i}_n{j}.npy"), np.array([sigmas[j]]))))[(np.arange(9) + 1) % 2 == 1]) if bold_ else " & ".join(f"${wtp:.2f}$ " + "\\euro{} " for wtp in wtp_v(np.concatenate((np.load(f"{paths.arrays_path}thetahat_e{i}_n{j}.npy"), np.array([sigmas[j]]))))[(np.arange(9) + 1) % 2 == 1])
        wtp_v_table += " \\\\ \n"
wtp_v_table += "\\hline \n \\end{tabular}"

create_file(f"{paths.tables_path}wtp_v.tex", wtp_v_table)
print(wtp_v_table)

wtp_v_med = wtp_v(np.load(f"{paths.arrays_path}thetahat_e{default_elast_id}_n{default_nest_id}.npy"))[4]
create_file(f"{paths.stats_path}wtp_v_med.tex", f"{np.round(wtp_v_med, 2):.2f}")

# %%
# Process willingness to pay for quality

def wtp_q(theta):
    xbar = np.array([[10000.]])
    Q_low = np.array([[10.]])
    Q_high = np.array([[20.]])
    E_u_high = de.E_u(ds, theta, ds.data, Q_high, xbar, median_yc[np.newaxis,np.newaxis,:])[0,0,:]
    E_u_low = de.E_u(ds, theta, ds.data, Q_low, xbar, median_yc[np.newaxis,np.newaxis,:])[0,0,:]
    theta_p = coef.theta_pi(ds, theta, median_yc[np.newaxis,np.newaxis,:])[0,0,:]
    wtp = (E_u_high - E_u_low) / theta_p
    return wtp

wtp_q_table = "\\begin{tabular}{c c c c c c c} \n & \\textbf{Nesting} & & & & & \\\\ \n\t \\textbf{Elasticity} & \\textbf{Parameter} & 10th \\%ile & 30th \\%ile & 50th \\%ile & 70th \\%ile & 90th \\%ile \\\\ \n \t \\hline \n"
for i in range(avg_price_elasts.shape[0]):
    wtp_q_table += f"$\\boldsymbol{{{avg_price_elasts[i]}}}$ & " if i == default_elast_id else f"${avg_price_elasts[i]}$ & "
    for j in range(sigmas.shape[0]):
        bold_ = (i == default_elast_id) and (j == default_nest_id)
        wtp_q_table += "" if j == 0 else " & "
        wtp_q_table += f"$\\boldsymbol{{{sigmas[j]}}}$ & " if bold_ else f"${sigmas[j]}$ & "
        wtp_q_table += " & ".join(f"$\\boldsymbol{{{wtp:.2f}}}$ " + "\\textbf{\\euro{}} " for wtp in wtp_q(np.concatenate((np.load(f"{paths.arrays_path}thetahat_e{i}_n{j}.npy"), np.array([sigmas[j]]))))[(np.arange(9) + 1) % 2 == 1]) if bold_ else " & ".join(f"${wtp:.2f}$ " + "\\euro{} " for wtp in wtp_q(np.concatenate((np.load(f"{paths.arrays_path}thetahat_e{i}_n{j}.npy"), np.array([sigmas[j]]))))[(np.arange(9) + 1) % 2 == 1])
        wtp_q_table += " \\\\ \n"
wtp_q_table += "\\hline \n \\end{tabular}"

create_file(f"{paths.tables_path}wtp_q.tex", wtp_q_table)
print(wtp_q_table)

wtp_q_low = wtp_q(np.load(f"{paths.arrays_path}thetahat_e{default_elast_id}_n{default_nest_id}.npy"))[0]
wtp_q_high = wtp_q(np.load(f"{paths.arrays_path}thetahat_e{default_elast_id}_n{default_nest_id}.npy"))[-1]
create_file(f"{paths.stats_path}wtp_q_low.tex", f"{np.round(wtp_q_low, 2):.2f}")
create_file(f"{paths.stats_path}wtp_q_high.tex", f"{np.round(wtp_q_high, 2):.2f}")

# %%
# Create predicted vs actual data consumption

x_fontsize = "x-large"
y_fontsize = "x-large"
title_fontsize = "xx-large"

predicted_dbar = np.load(f"{paths.dbar_path}predicted_e{default_elast_id}_n{default_nest_id}.npy")
actual_dbar = np.load(f"{paths.dbar_path}actual_e{default_elast_id}_n{default_nest_id}.npy")

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
# plt.show()

fig, axs = plt.subplots(1, 1, figsize=(8.5, 4))
products = ['$\\bar{d} = 1\\,000$', '$\\bar{d} = 4\\,000$', '$\\bar{d} = 8\\,000$']
for j in range(3):
    col = j % 3
    axs.scatter(actual_dbar[:,j+2], predicted_dbar[:,j+2] * 1000, alpha=0.3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][j], label=products[j])
axs.plot(np.arange(0,6000,50), np.arange(0,6000,50), linestyle="--", color="black", label="$45^{\\circ}$ line")
axs.set_xlabel("actual consumption (MB)", fontsize=x_fontsize)
axs.set_ylabel("predicted consumption (MB)", fontsize=y_fontsize)
leg = axs.legend(loc="center right", ncol=1, fontsize=12, bbox_to_anchor=(1.3, 0.5), bbox_transform=axs.transAxes)
for lh in leg.legendHandles: 
    lh.set_alpha(1.0)
fig.tight_layout()
plt.savefig(f"{paths.graphs_path}predict_vs_actual_dbar_alt.pdf", bbox_inches="tight")
# plt.show()

create_file(f"{paths.stats_path}dbar_corr_low.tex", "{:.3f}".format(np.round(np.corrcoef(predicted_dbar[:,2], actual_dbar[:,2])[0,1], 3)))
create_file(f"{paths.stats_path}dbar_corr_med.tex", "{:.3f}".format(np.round(np.corrcoef(predicted_dbar[:,3], actual_dbar[:,3])[0,1], 3)))
create_file(f"{paths.stats_path}dbar_corr_high.tex", "{:.3f}".format(np.round(np.corrcoef(predicted_dbar[:,4], actual_dbar[:,4])[0,1], 3)))

# %%
# Create table of share that diverts to outside option after 10% price increase in all products

# Create DemandSystem with 10% increase in prices
ds_increase = copy.deepcopy(ds)
pidx = ds_increase.chars.index(ds_increase.pname)
ds_increase.data[:,:,pidx] = ds_increase.data[:,:,pidx] * 1.1

to_tex = "\\begin{tabular}{c"
to_tex += " c" * len(paths.sigmas)
to_tex += "} \n Elasticity & "
for i, nest_id in enumerate(paths.sigmas):
    to_tex += f"$\\boldsymbol{{\\sigma = {nest_id}}}$" if i == default_nest_id else f"$\\sigma = {nest_id}$"
    if i < len(paths.sigmas) - 1:
        to_tex += " & "
to_tex += " \\\\ \n \t \\hline \n"

for i, elast_id in enumerate(paths.avg_price_elasts):
    to_tex += f"$\\boldsymbol{{{elast_id}}}$ & " if i == default_elast_id else f"${elast_id}$ & "
    for j, nest_id in enumerate(paths.sigmas):
        bold_ = (i == default_elast_id) and (j == default_nest_id)
        # Determine theta for this imputation
        thetahat = np.load(f"{paths.arrays_path}thetahat_e{i}_n{j}.npy")
        sigma = paths.sigmas[j]
        theta_sigma = np.concatenate((thetahat, np.array([sigma])))
        
        # Calculate xis
        xis = blp.xi(ds, theta_sigma, ds.data, None)
        
        # Calculate shares at observed prices
        share = np.average(np.sum(blp.s_mj(ds, theta_sigma, ds.data, xis), axis=1), weights=ds.data[:,0,ds.dim3.index(ds.msizename)])
        
        # Calculate shares at 10% increase in prices
        share_increase = np.average(np.sum(blp.s_mj(ds_increase, theta_sigma, ds_increase.data, xis), axis=1), weights=ds_increase.data[:,0,ds_increase.dim3.index(ds_increase.msizename)])
        
        # Add fraction who switch to outside option
        to_tex += "\\textbf{" + "{:.2f}".format(-(share_increase - share) / share * 100.0) + "\\%}" if bold_ else "{:.2f}".format(-(share_increase - share) / share * 100.0) + "\\%"
        
        if j < len(paths.sigmas) - 1:
            to_tex += " & "
        else:
            to_tex += " \\\\ \n"
            
        if (i == default_elast_id) and (j == 0):
            mult_logit_switch = "{:.2f}".format(-(share_increase - share) / share * 100.0)
            create_file(f"{paths.stats_path}mult_logit_switch_share.tex", mult_logit_switch)
            
        if (i == default_elast_id) and (j == default_nest_id):
            pref_switch = "{:.2f}".format(-(share_increase - share) / share * 100.0)
            create_file(f"{paths.stats_path}pref_switch_share.tex", pref_switch)
        
to_tex += "\\hline \n \\end{tabular}"
    
create_file(f"{paths.tables_path}outside_option_substitution.tex", to_tex)
print(to_tex)
