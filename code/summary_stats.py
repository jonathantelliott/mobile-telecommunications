# Load packages
import numpy as np
import pandas as pd
import paths

# Read in data processed by Stata
df = pd.read_stata(f"{paths.data_path}demand_estimation_data.dta")

# Drop "Rest-of-France" market
df = df[df['market'] != 0]

# Dimensions
num_market = np.unique(df['market']).shape[0]
num_month = np.unique(df['month']).shape[0]
num_j = np.unique(df['j_new']).shape[0]

# Create table to export
to_tex = "\\begin{tabular}{l c c c c } \\hline \n"
to_tex += " & \\textbf{Mean} & \\textbf{Std. Dev.} & \\textbf{Min.} & \\textbf{Max.} \\\\ \n"

to_tex += "\\multicolumn{5}{l}{\\textbf{Quality and market data}} \\\\ \n"
to_tex += "\\hline \n"

to_tex += "Market average usage (MB) & "
opcode = np.reshape(df['opcode'].values, (num_market,num_month,num_j))
org_prods = np.all(opcode == 1, axis=(0,1))
avg_usage = np.reshape(df['dbar_new'].values, (num_market,num_month,num_j))[:,-1,org_prods]
customers = np.reshape(df['customer'].values, (num_market,num_month,num_j))[:,-1,org_prods]
avg_usage = np.average(avg_usage, weights=customers, axis=1)
to_tex += "{:,.0f}".format(np.mean(avg_usage)).replace(",", "\\,") + " & "
to_tex += "{:,.0f}".format(np.std(avg_usage)).replace(",", "\\,") + " & "
to_tex += "{:,.0f}".format(np.min(avg_usage)).replace(",", "\\,") + " & "
to_tex += "{:,.0f}".format(np.max(avg_usage)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Quality Orange (Mbps) & "
qualities = np.mean(np.reshape(df['q_ookla'].values, (num_market,num_month,num_j))[:,-1,org_prods], axis=1)
to_tex += "{:,.2f}".format(np.mean(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.std(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.min(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.max(qualities)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Quality Bouygues (Mbps) & "
byg_prods = np.all(opcode == 2, axis=(0,1))
qualities = np.mean(np.reshape(df['q_ookla'].values, (num_market,num_month,num_j))[:,-1,byg_prods], axis=1)
to_tex += "{:,.2f}".format(np.mean(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.std(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.min(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.max(qualities)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Quality Free (Mbps) & "
free_prods = np.all(opcode == 3, axis=(0,1))
qualities = np.mean(np.reshape(df['q_ookla'].values, (num_market,num_month,num_j))[:,-1,free_prods], axis=1)
to_tex += "{:,.2f}".format(np.mean(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.std(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.min(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.max(qualities)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Quality SFR (Mbps) & "
sfr_prods = np.all(opcode == 4, axis=(0,1))
qualities = np.mean(np.reshape(df['q_ookla'].values, (num_market,num_month,num_j))[:,-1,sfr_prods], axis=1)
to_tex += "{:,.2f}".format(np.mean(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.std(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.min(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.max(qualities)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Quality MVNO (Mbps) & "
mvno_prods = np.all(opcode == 5, axis=(0,1))
qualities = np.mean(np.reshape(df['q_ookla'].values, (num_market,num_month,num_j))[:,-1,mvno_prods], axis=1)
to_tex += "{:,.2f}".format(np.mean(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.std(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.min(qualities)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.max(qualities)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Median income (Euros) & "
med_inc = np.mean(np.reshape(df['yc5'].values, (num_market,num_month,num_j))[:,-1,:], axis=1)
to_tex += "{:,.0f}".format(np.mean(med_inc)).replace(",", "\\,") + " & "
to_tex += "{:,.0f}".format(np.std(med_inc)).replace(",", "\\,") + " & "
to_tex += "{:,.0f}".format(np.min(med_inc)).replace(",", "\\,") + " & "
to_tex += "{:,.0f}".format(np.max(med_inc)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Number of markets & \\multicolumn{4}{c}{" + "{:,.0f}".format(med_inc.shape[0]).replace(",", "\\,") + "} \\\\ \n"


to_tex += "\\multicolumn{5}{l}{\\textbf{Tariff data}} \\\\ \n"
to_tex += "\\hline \n"

to_tex += "Price & "
price = np.reshape(df['p'].values, (num_market,num_month,num_j))[0,-1,:]
to_tex += "{:,.2f}".format(np.nanmean(price)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanstd(price)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanmin(price)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanmax(price)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Price (Orange) & "
price = np.reshape(df['p'].values, (num_market,num_month,num_j))[0,-1,org_prods]
to_tex += "{:,.2f}".format(np.nanmean(price)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanstd(price)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanmin(price)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanmax(price)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Price (Others) & "
price = np.reshape(df['p'].values, (num_market,num_month,num_j))[0,-1,~org_prods]
to_tex += "{:,.2f}".format(np.nanmean(price)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanstd(price)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanmin(price)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanmax(price)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Data limit & "
dlim = np.reshape(df['dlim'].values, (num_market,num_month,num_j))[0,-1,:]
to_tex += "{:,.0f}".format(np.nanmean(dlim)).replace(",", "\\,") + " & "
to_tex += "{:,.0f}".format(np.nanstd(dlim)).replace(",", "\\,") + " & "
to_tex += "{:,.0f}".format(np.nanmin(dlim)).replace(",", "\\,") + " & "
to_tex += "{:,.0f}".format(np.nanmax(dlim)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Number of phone plans & \\multicolumn{4}{c}{" + "{:,.0f}".format(dlim.shape[0]).replace(",", "\\,") + "} \\\\ \n"


df_inf = pd.read_csv(f"{paths.data_path}infrastructure_clean.csv", engine="python") # engine helps encoding, error with commune names, but doesn't matter b/c not used
df_inf = df_inf[df_inf['market'] > 0] # don't include Rest-of-France market

# Commune characteristics
population = df_inf['msize_24'].values
area = df_inf['area_effective'].values # adjusted commune area
pop_dens = df_inf['pdens_clean'].values 

# Infrastructure
stations = df_inf[[f"stations{i}" for i in range(1,5)]].values # number of base stations
radius = np.sqrt(area[:,np.newaxis] / stations / (np.sqrt(3.) * 3. / 2.)) # cell radius assuming homogeneous hexagonal cells, in km
stations = np.nan_to_num(stations) # replace NaNs with 0
bw_3g = df_inf[[f"bw3g{i}" for i in range(1,5)]].values # in MHz
bw_4g = df_inf[[f"bw4g{i}" for i in range(1,5)]].values # in MHz

bw_4g_equiv = np.nan_to_num(bw_3g) * 2.5 / 4.08 + np.nan_to_num(bw_4g) # aggregate to 4G equivalent based on spectral efficiencies from https://en.wikipedia.org/wiki/Spectral_efficiency 

to_tex += "\\multicolumn{5}{l}{\\textbf{Infrastructure data}} \\\\ \n"
to_tex += "\\hline \n"

to_tex += "Bandwidth per firm (MHz) & "
to_tex += "{:,.2f}".format(np.nanmean(bw_4g_equiv)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanstd(bw_4g_equiv)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanmin(bw_4g_equiv)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanmax(bw_4g_equiv)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Number of base stations & "
to_tex += "{:,.2f}".format(np.nanmean(stations)).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanstd(stations)).replace(",", "\\,") + " & "
to_tex += "{:,.0f}".format(np.nanmin(stations)).replace(",", "\\,") + " & "
to_tex += "{:,.0f}".format(np.nanmax(stations)).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "Effective cell radius (km) & "
to_tex += "{:,.2f}".format(np.nanmean(radius[~np.isinf(radius)])).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanstd(radius[~np.isinf(radius)])).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanmin(radius[~np.isinf(radius)])).replace(",", "\\,") + " & "
to_tex += "{:,.2f}".format(np.nanmax(radius[~np.isinf(radius)])).replace(",", "\\,")
to_tex += " \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

print(to_tex)

# Save table
f = open(paths.tables_path + "summary_stats.tex", "w")
f.write(to_tex)
f.close()

# Create aggregate market shares table
df = pd.read_stata(f"{paths.data_path}agg_shares.dta")
agg_table = "\\begin{tabular}{ccccccc} \n"
agg_table += "\\toprule \n"
agg_table += "Market Size (millions) & ORG   & SFR   & BYG   & FREE  & MVNO & Non-users  \\\\ \n"
agg_table += "\\midrule \n"
agg_table += f"56.5 & "
agg_table += f"{(df['mshare1'].values[0] * 100.0):,.1f}\\% & "
agg_table += f"{(df['mshare2'].values[0] * 100.0):,.1f}\\% & "
agg_table += f"{(df['mshare3'].values[0] * 100.0):,.1f}\\% & "
agg_table += f"{(df['mshare4'].values[0] * 100.0):,.1f}\\% & "
agg_table += f"{(df['mshare5'].values[0] * 100.0):,.1f}\\% & "
agg_table += f"{(df['mshare0'].values[0] * 100.0):,.1f}\\% \\\\ \n "
agg_table += "\\bottomrule \n"
agg_table += "\\end{tabular} \n"
f = open(paths.tables_path + "agg_shares_table.tex", "w")
f.write(agg_table)
f.close()
