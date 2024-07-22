# Load packages
import numpy as np
import pandas as pd
import paths

# Read in data processed by Stata
df = pd.read_csv(f"{paths.data_path}hourly_data_shares.csv", sep="\t")

# Determine variables from shares
daytime_begin = 7
daytime_end = 23
hours = df['hour']
in_daytime = (hours >= daytime_begin) & (hours <= daytime_end)
avg_daytime = np.mean(df['datashare'][in_daytime])
stddev_daytime = np.std(df['datashare'][in_daytime])
avg_nighttime = np.mean(df['datashare'][~in_daytime])
stddev_nighttime = np.std(df['datashare'][~in_daytime])

# Save values
def create_file(file_name, file_contents):
    """Create file with name file_name and content file_contents"""
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()
create_file(f"{paths.stats_path}avg_daytime_data_consumption.tex", f"{avg_daytime:.3f}")
create_file(f"{paths.stats_path}stddev_daytime_data_consumption.tex", f"{stddev_daytime:.3f}")
create_file(f"{paths.stats_path}avg_nighttime_data_consumption.tex", f"{avg_nighttime:.3f}")
create_file(f"{paths.stats_path}stddev_nighttime_data_consumption.tex", f"{stddev_nighttime:.3f}")
