// This file plots data consumption over the day
// Last updated: 25 March 2024

// Location of relevant folders
local orange_idei_path "`1'"
local databases_path "`orange_idei_path'Databases/"
local osiris_path "`databases_path'quality data/raw data/OSIRIS/"
local processed_data_path "`orange_idei_path'data/"
local home_path "`2'"
local graphs_path "`home_path'graphs/"

set more off

use "`osiris_path'data.dta" , clear

gen date = substr(datehour,1,10)
gen hour = substr(datehour,12,2)
destring hour, replace

collapse (sum) bitps, by(codecell cellname date hour)
rename bitps bits 
replace bits = bits*60*60
bysort codecell: egen totalbits = total(bits)
gen datashare = bits/totalbits

collapse (mean) datashare [aw=totalbits], by(hour)

twoway (line datashare hour) (scatter datashare hour) , ytitle("Data consumption") xtitle("Hour of day") title("Share of data consumed by hour") legend(off) graphregion(color(white))
graph export "`graphs_path'datashare_hour.png", replace

outsheet using "`processed_data_path'hourly_data_shares.csv", replace
