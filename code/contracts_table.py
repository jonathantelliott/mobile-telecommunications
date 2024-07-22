# Load packages
import numpy as np
import pandas as pd
import paths

# Read in data processed by Stata
df = pd.read_stata(f"{paths.data_path}contracts_summary.dta")

# Process variables
operator = df['operator'].values
p = np.round(df['p'].values, 2)
dlim = df['dlim'].values
v_dum = df['v_dum'].values.astype("str")
v_dum[v_dum == "0.0"] = "No"
v_dum[v_dum == "1.0"] = "Yes"
num_contracts = df['num_contracts'].values.astype(int)
min_price = np.round(df['min_price'].values, 2)
max_price = np.round(df['max_price'].values, 2)
min_dlim = df['min_dlim'].values.astype(int)
max_dlim = df['max_dlim'].values.astype(int)

# Create table
to_tex = "\\begin{tabular}{lcccccccc} \\hline \n"
to_tex += "\\textbf{Operator} & \\textbf{Price} & \\textbf{Data} & \\textbf{Unlimited} & \\textbf{Contracts} & \\textbf{Min} & \\textbf{Max} & \\textbf{Min} & \\textbf{Max} \\\\ \n"
to_tex += " & & \\textbf{Limit} & \\textbf{Voice} & \\textbf{Represented} & \\textbf{Price} & \\textbf{Price} & \\textbf{Limit} & \\textbf{Limit} \\\\ \\hline \n"

to_tex = "\\begin{tabular}{lcccccccc} \\hline \n"
to_tex += " &  & \\textbf{Data} &  &  & \\textbf{Min} & \\textbf{Max} & \\textbf{Min} & \\textbf{Max} \\\\ \n"
to_tex += " & \\textbf{Price} & \\textbf{Limit} & \\textbf{Unlimited} & \\textbf{Plans} & \\textbf{Price} & \\textbf{Price} & \\textbf{Limit} & \\textbf{Limit} \\\\ \n"
to_tex += "\\textbf{Operator} & \\textbf{(\\euro{})} & \\textbf{(MB)} & \\textbf{Voice} & \\textbf{Represented} & \\textbf{(\\euro{})} & \\textbf{(\\euro{})} & \\textbf{(MB)} & \\textbf{(MB)} \\\\ \\hline \n"

for i in range(operator.shape[0]):
    to_tex += f"{operator[i]} & "
    to_tex += "{:.2f}".format(p[i]) + " & "
    to_tex += f"{dlim[i]} & "
    to_tex += f"{v_dum[i]} & "
    to_tex += f"{num_contracts[i]} & "
    to_tex += "{:.2f}".format(min_price[i]) + " & "
    to_tex += "{:.2f}".format(max_price[i]) + " & "
    to_tex += "{:,}".format(min_dlim[i]).replace(",", "\\,") + " & "
    to_tex += "{:,}".format(max_dlim[i]).replace(",", "\\,") + " "
    to_tex += "\\\\ \n"
to_tex += "\\hline \n\\end{tabular}"
print(to_tex)

# Save table
f = open(paths.tables_path + "contracts_table.tex", "w")
f.write(to_tex)
f.close()
