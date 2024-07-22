// This file constructs datasets used for estimating demand for mobile service plans in France
// Last updated: 13 November 2023

// Location of relevant folders
local orange_idei_path "`1'"
local databases_path "`orange_idei_path'Databases/"
local tariffsdata_path "`databases_path'tariffs data/"
local insee_path "`databases_path'INSEE data/"
local osiris_path "`databases_path'quality data/raw data/OSIRIS/"
local infrastructure_path "`orange_idei_path'infrastructure/"
local ookla_path "`databases_path'Ookla data/"
local processed_data_path "`orange_idei_path'data/"

set more off


// Create choice set
use "`processed_data_path'chset_final.dta", clear
sort opcode cclass
by opcode cclass: egen maxcommit = max(commit24)
keep if commit24 == maxcommit // need to do this b/c some cclass don't have a 24-month category
gen j_new = _n
order j_new
drop maxcommit
rename j j_old
rename j_new j
reshape long commit choiceset p dlim vlim lcost, i(j) j(month)
keep j j_old opcode month dlim vlim p commit
save "`processed_data_path'chset_final_newj.dta", replace
use "`processed_data_path'market_ROF.dta", clear
keep market
duplicates drop
cross using "`processed_data_path'chset_final_newj.dta" // combine choice set for every market
sort j opcode month market
save "`processed_data_path'demand_estimation_data.dta", replace

// Add market size
use "`processed_data_path'msize_alt.dta", clear
merge 1:m market month using "`processed_data_path'demand_estimation_data.dta"
keep if _merge == 3
drop _merge
save "`processed_data_path'demand_estimation_data.dta", replace

// Add spectral efficiency
use "`osiris_path'spectreff.dta", clear
merge m:1 com_cd_insee using "`processed_data_path'market_ROF.dta", keep(match) nogenerate
keep market spectreff technoband
collapse (mean) spectreff, by(market technoband) // aggregate to market level
tab technoband if market > 0
drop if technoband=="4G 1800" // only 4 observations, so we won't use this
gen type = subinstr(technoband," ","",.) // need string to not have spaces in order to reshape
drop technoband
reshape wide spectreff, i(market) j(type) s
merge 1:m market using "`processed_data_path'demand_estimation_data.dta"
drop _merge
save "`processed_data_path'demand_estimation_data.dta", replace

// Add experienced population density
import delimited "`processed_data_path'infrastructure_clean.csv", clear
keep market pdens_clean
merge 1:m market using "`processed_data_path'demand_estimation_data.dta"
// keep if _merge == 3 - infrastructure data doesn't include "Rest of France" communes, but these aren't used in moments, so don't just drop
drop _merge
save "`processed_data_path'demand_estimation_data.dta", replace

// Add income
use "`processed_data_path'income_alt.dta", clear
merge 1:m market using "`processed_data_path'demand_estimation_data.dta"
drop _merge
save "`processed_data_path'demand_estimation_data.dta", replace

// Add Ookla q measure
merge m:1 market opcode using "`processed_data_path'ookla_quality.dta"
keep if _merge == 3
drop _merge
save "`processed_data_path'demand_estimation_data.dta", replace

// Add number of stations
import delimited "`processed_data_path'infrastructure_clean.csv", clear
keep market stations*
gen stations5 = .
reshape long stations, i(market) j(opcode)
gen month = 24 // we only have data for 24th month
rename stations num_stations
merge 1:m market opcode month using "`processed_data_path'demand_estimation_data.dta"
// keep if _merge == 3 - infrastructure data doesn't include "Rest of France" communes, but these aren't used in moments, so don't just drop
drop _merge
save "`processed_data_path'demand_estimation_data.dta", replace

// Add avg data used
use "`processed_data_path'dbar.dta", clear
rename j j_old
merge 1:1 market j_old month using "`processed_data_path'demand_estimation_data.dta"
keep if _merge == 2 | _merge == 3
drop _merge
save "`processed_data_path'demand_estimation_data.dta", replace

// Construct market shares
replace customer = 0 if customer == . & opcode == 1 & month >= 3 // change from . to 0 for data necessary for estimation
replace customer = 0 if customer == . & opcode == 1 & month >= 3
gen mktshare = customer / msize
//drop customer firm

// Make missing products not offered in a given month
replace vlim = . if vlim == 888 // these are products j that aren't offered in the month, i.e. missing
replace p = . if p == 888
replace dlim = . if dlim == 888

// Convert vlim to a dummy for finite vs. infinite data limits
gen vunlimited = 0
replace vunlimited = 1 if vlim == 999
replace vunlimited = . if vlim == .

// Generate Orange dummy
gen Orange = 0
replace Orange = 1 if opcode == 1

// Keep only month 24
keep if month == 24

// Reorder for readability
order market month j q_ookla num_stations p dlim vlim vunlimited Orange dbar mktshare msize spectreff* yc*
sort market month j

// Deal with the fact that j == 2 has mostly left the market by this point, get rid of class of tiny data, unlimited voice
gen dbar_new = dbar
replace dbar_new = (mktshare * dbar + mktshare[_n+1] * dbar[_n+1]) / (mktshare + mktshare[_n+1]) if j == 1
gen mktshare_new = mktshare 
replace mktshare_new = mktshare + mktshare[_n+1] if j == 1
drop if dlim < 500 & vunlimited == 1
egen j_new = group(j)
sort market month j_new

save "`processed_data_path'demand_estimation_data.dta", replace
export delimited using "`processed_data_path'demand_estimation_data.csv", replace

// Create aggregate market shares dataset
use "`processed_data_path'agg_shares.dta", clear
keep month mshare*
save "`processed_data_path'agg_market_shares.dta", replace
export delimited using "`processed_data_path'agg_data.csv", replace
