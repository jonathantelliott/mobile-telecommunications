# Import packages
import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib as mpl
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "serif"
plt.rcParams['mathtext.fontset'] = "dejavuserif"

import paths

def round_var(var, round_dig):
    return "{0:,.3f}".format(np.round(var, round_dig))

def create_file(file_name, file_contents):
    """Create file with name file_name and content file_contents"""
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()

x_fontsize = "x-large"
y_fontsize = "x-large"
title_fontsize = "xx-large"

print_ = False
save_ = True

df_chset = pd.read_stata(f"{paths.data_path}chset_final_newj.dta")

# Orange prices over time
fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
months = np.unique(df_chset['month'])
df_months = pd.DataFrame({'month_num': months})
df_months['month'] = (df_months['month_num'] + 9) % 12 + 1
df_months['year'] = 2013 + ((df_months['month_num'] + 9) // 12)
df_months['day'] = 1
df_months = df_months[['month','year','day']]
df_months['monthyear'] = pd.to_datetime(df_months)
js = np.unique(df_chset['j'])
ps = np.reshape(df_chset['p'].values, (js.shape[0], months.shape[0]))
ps[ps > 100.0] = np.nan
opcodes = np.reshape(df_chset['opcode'].values, (js.shape[0], months.shape[0]))[:,0]
for j, j_num in enumerate(js):
    ax.plot(df_months['monthyear'], ps[j,:], color="black", alpha=0.8 if opcodes[j] == 1 else 0.2, linewidth=3.0 if opcodes[j] == 1 else 1.5)
ax.set_xlabel("month", fontsize=x_fontsize)
ax.set_ylabel("prices (in \u20ac)", fontsize=y_fontsize)
fig.tight_layout()
if print_:
    plt.show()
if save_:
    plt.savefig(f"{paths.graphs_path}p_over_time_Org.pdf", bbox_inches="tight", transparent=True)
    plt.savefig(f"{paths.graphs_path}figure5.pdf", bbox_inches="tight", transparent=True)

# Import data
df = pd.read_stata(f"{paths.data_path}demand_estimation_data.dta")
df = df[df['month'] == 24]

# Drop "Rest-of-France" market
df = df[df['market'] != 0]

# Create Orange dataframe
df_Org = df[df['opcode'] == 1]
df_Org['frac_downloaded'] = df_Org['dbar_new'] / df_Org['dlim']
dlims = np.unique(df_Org['dlim'])

figsize_adjust = 2.0 / 3.0

# Average data usage vs. measured quality across markets
fig, axs = plt.subplots(1, 3, figsize=(15 * figsize_adjust, 6 * figsize_adjust))
X = sm.add_constant(df_Org[df_Org['dlim'] == dlims[1]]['q_ookla'])
model = sm.OLS(df_Org[df_Org['dlim'] == dlims[1]]['dbar_new'], X)
res = model.fit()
axs[0].scatter(df_Org[df_Org['dlim'] == dlims[1]]['q_ookla'], df_Org[df_Org['dlim'] == dlims[1]]['dbar_new'], color="black", alpha=0.15)
if save_:
    create_file(f"{paths.stats_path}d_q_corr_1.tex", round_var(np.corrcoef(df_Org[df_Org['dlim'] == dlims[1]]['q_ookla'], df_Org[df_Org['dlim'] == dlims[1]]['dbar_new'])[0,1], 3))
axs[0].plot(df_Org[df_Org['dlim'] == dlims[1]]['q_ookla'], res.predict(X), color="black", alpha=0.85)
X = sm.add_constant(df_Org[df_Org['dlim'] == dlims[2]]['q_ookla'])
model = sm.OLS(df_Org[df_Org['dlim'] == dlims[2]]['dbar_new'], X)
res = model.fit()
axs[1].scatter(df_Org[df_Org['dlim'] == dlims[2]]['q_ookla'], df_Org[df_Org['dlim'] == dlims[2]]['dbar_new'], color="black", alpha=0.15)
if save_:
    create_file(f"{paths.stats_path}d_q_corr_2.tex", round_var(np.corrcoef(df_Org[df_Org['dlim'] == dlims[2]]['q_ookla'], df_Org[df_Org['dlim'] == dlims[2]]['dbar_new'])[0,1], 3))
axs[1].plot(df_Org[df_Org['dlim'] == dlims[2]]['q_ookla'], res.predict(X), color="black", alpha=0.85)
X = sm.add_constant(df_Org[df_Org['dlim'] == dlims[3]]['q_ookla'])
model = sm.OLS(df_Org[df_Org['dlim'] == dlims[3]]['dbar_new'], X)
res = model.fit()
axs[2].scatter(df_Org[df_Org['dlim'] == dlims[3]]['q_ookla'], df_Org[df_Org['dlim'] == dlims[3]]['dbar_new'], color="black", alpha=0.15)
if save_:
    create_file(f"{paths.stats_path}d_q_corr_3.tex", round_var(np.corrcoef(df_Org[df_Org['dlim'] == dlims[3]]['q_ookla'], df_Org[df_Org['dlim'] == dlims[3]]['dbar_new'])[0,1], 3))
axs[2].plot(df_Org[df_Org['dlim'] == dlims[3]]['q_ookla'], res.predict(X), color="black", alpha=0.85)
axs[0].set_title("data limit = " + str(int(dlims[1])), fontsize=title_fontsize)
axs[1].set_title("data limit = " + str(int(dlims[2])), fontsize=title_fontsize)
axs[2].set_title("data limit = " + str(int(dlims[3])), fontsize=title_fontsize)
axs[0].set_xlabel("avg. download speed (Mbps)", fontsize=x_fontsize)
axs[1].set_xlabel("avg. download speed (Mbps)", fontsize=x_fontsize)
axs[2].set_xlabel("avg. download speed (Mbps)", fontsize=x_fontsize)
axs[0].set_ylabel("avg. data usage (MB)", fontsize=y_fontsize)
axs[1].set_ylabel("avg. data usage (MB)", fontsize=y_fontsize)
axs[2].set_ylabel("avg. data usage (MB)", fontsize=y_fontsize)
fig.tight_layout()
if print_:
    plt.show()
if save_:
    plt.savefig(f"{paths.graphs_path}avg_data_vs_q.pdf", bbox_inches="tight", transparent=True)
    plt.savefig(f"{paths.graphs_path}figure2.pdf", bbox_inches="tight", transparent=True)

# Average data usage across markets
fig, axs = plt.subplots(1, 3, figsize=(15 * figsize_adjust, 6 * figsize_adjust))
axs[0].hist(df_Org[df_Org['dlim'] == dlims[1]]['dbar_new'], edgecolor="black", lw=2.0, fc=(0, 0, 0, 0))
axs[1].hist(df_Org[df_Org['dlim'] == dlims[2]]['dbar_new'], edgecolor="black", lw=2.0, fc=(0, 0, 0, 0))
axs[2].hist(df_Org[df_Org['dlim'] == dlims[3]]['dbar_new'], edgecolor="black", lw=2.0, fc=(0, 0, 0, 0))
axs[0].axvline(x=dlims[1], color="black", linestyle="dashed", linewidth=2.5)
axs[1].axvline(x=dlims[2], color="black", linestyle="dashed", linewidth=2.5)
axs[2].axvline(x=dlims[3], color="black", linestyle="dashed", linewidth=2.5)
axs[0].set_title("data limit = " + str(int(dlims[1])), fontsize=title_fontsize)
axs[1].set_title("data limit = " + str(int(dlims[2])), fontsize=title_fontsize)
axs[2].set_title("data limit = " + str(int(dlims[3])), fontsize=title_fontsize)
axs[0].set_xlabel("data consumption (MB)", fontsize=x_fontsize)
axs[1].set_xlabel("data consumption (MB)", fontsize=x_fontsize)
axs[2].set_xlabel("data consumption (MB)", fontsize=x_fontsize)
if save_:
    create_file(f"{paths.stats_path}frac_dlim_used_1.tex", round_var(np.mean(df_Org[df_Org['dlim'] == dlims[1]]['dbar_new'] / dlims[1]), 3))
    create_file(f"{paths.stats_path}frac_dlim_used_2.tex", round_var(np.mean(df_Org[df_Org['dlim'] == dlims[2]]['dbar_new'] / dlims[2]), 3))
    create_file(f"{paths.stats_path}frac_dlim_used_3.tex", round_var(np.mean(df_Org[df_Org['dlim'] == dlims[3]]['dbar_new'] / dlims[3]), 3))
fig.tight_layout()
if print_:
    plt.show()
if save_:
    plt.savefig(f"{paths.graphs_path}dbar_hist.pdf", bbox_inches="tight", transparent=True)
    plt.savefig(f"{paths.graphs_path}figure3.pdf", bbox_inches="tight", transparent=True)

# Histogram of download speeds
grp = df.sort_values(['market', 'month', 'opcode', 'j_new']).groupby(['market', 'opcode']).agg('mean').reset_index()
fig, axs = plt.subplots(2, 3, figsize=(15 * figsize_adjust, 10 * figsize_adjust), sharex=True)
bin_range = np.linspace(0, np.nanmax(grp['q_ookla']), 20)
axs[0, 0].hist(grp[grp['opcode'] == 1]['q_ookla'], bins=bin_range, edgecolor="black", lw=2.0, fc=(0, 0, 0, 0))
axs[0, 0].set_title('Orange', fontsize=title_fontsize)
axs[0, 0].axvline(x=np.mean(grp[grp['opcode'] == 1]['q_ookla']), color="black", linestyle="dashed", linewidth=2.5)
axs[0, 1].hist(grp[grp['opcode'] == 2]['q_ookla'], bins=bin_range, edgecolor="black", lw=2.0, fc=(0, 0, 0, 0))
axs[0, 1].set_title('Bouygues', fontsize=title_fontsize)
axs[0, 1].axvline(x=np.mean(grp[grp['opcode'] == 2]['q_ookla']), color="black", linestyle="dashed", linewidth=2.5)
axs[0, 2].hist(grp[grp['opcode'] == 3]['q_ookla'], bins=bin_range, edgecolor="black", lw=2.0, fc=(0, 0, 0, 0))
axs[0, 2].set_title('Free', fontsize=title_fontsize)
axs[0, 2].axvline(x=np.mean(grp[grp['opcode'] == 3]['q_ookla']), color="black", linestyle="dashed", linewidth=2.5)
axs[1, 0].hist(grp[grp['opcode'] == 4]['q_ookla'], bins=bin_range, edgecolor="black", lw=2.0, fc=(0, 0, 0, 0))
axs[1, 0].set_title('SFR', fontsize=title_fontsize)
axs[1, 0].axvline(x=np.mean(grp[grp['opcode'] == 4]['q_ookla']), color="black", linestyle="dashed", linewidth=2.5)
axs[1, 1].hist(grp[grp['opcode'] == 5]['q_ookla'], bins=bin_range, edgecolor="black", lw=2.0, fc=(0, 0, 0, 0))
axs[1, 1].set_title('MVNO Operators', fontsize=title_fontsize)
axs[1, 1].axvline(x=np.mean(grp[grp['opcode'] == 5]['q_ookla']), color="black", linestyle="dashed", linewidth=2.5)
for row in range(2):
    for column in range(3):
        if not (row == 1 and column == 2):
            axs[row, column].set_xlabel('quality (Mbps)', fontsize=x_fontsize)
axs[1, 2].set_visible(False)
fig.tight_layout()
if print_:
    plt.show()
if save_:
    plt.savefig(f"{paths.graphs_path}quality_hist_new.pdf", bbox_inches="tight", transparent=True)
    plt.savefig(f"{paths.graphs_path}figure1.pdf", bbox_inches="tight", transparent=True)

# Median income vs. expensive contract market share
fig, ax = plt.subplots(1, 1, figsize=(8 * figsize_adjust, 6 * figsize_adjust))

js = np.unique(df_Org['j_new'])
js_use = np.array([3,4])
df_Org_use = df_Org[np.isin(df_Org['j_new'], js_use)]

yc5_expensive = np.reshape(df_Org_use['yc5'].values, (-1,2))[:,0]
mktshare_expensive = np.sum(np.reshape(df_Org_use['mktshare_new'].values, (-1,2)), axis=1)
X = sm.add_constant(yc5_expensive[:,np.newaxis])
model = sm.OLS(mktshare_expensive, X)
res = model.fit()
ax.scatter(yc5_expensive, mktshare_expensive, color="black", alpha=0.15)
ax.plot(yc5_expensive, res.predict(X), color="black", alpha=0.85)
ax.set_xlabel("median income (\u20ac)", fontsize=x_fontsize)
ax.set_ylabel("market share", fontsize=y_fontsize)
print(f"{np.corrcoef(yc5_expensive, mktshare_expensive)[0,1]:.3f}")
if save_:
    create_file(f"{paths.stats_path}yc5_mktshare_corr.tex", f"{np.corrcoef(yc5_expensive, mktshare_expensive)[0,1]:.3f}")

fig.tight_layout()
if print_:
    plt.show()
if save_:
    plt.savefig(f"{paths.graphs_path}income_vs_share.pdf", bbox_inches="tight", transparent=True)
    plt.savefig(f"{paths.graphs_path}figure4.pdf", bbox_inches="tight", transparent=True)
