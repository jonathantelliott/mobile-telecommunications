// This file combines the processed Ookla speedtests data to create average MNO-market download speeds
// Last updated: 26 July 2022

// Location of relevant folders
local orange_idei_path "`1'"
local databases_path "`orange_idei_path'Databases/"
local insee_path "`databases_path'INSEE data/"
local processed_data_path "`orange_idei_path'data/"

set more off

use "`processed_data_path'phone_records_insee.dta", clear

drop if missing(insee) // should only be a few of these that didn't merge into commune shapefile

// Recode Paris, Lyon, and Marseille - copied from 3_msize.do in original files
gen byte notnumeric = real(insee) == .
gen insee_num = ""
replace insee_num = insee if !notnumeric
destring insee_num, replace
replace insee="75056" if insee_num>=75101 & insee_num<=75120
replace insee="69123" if insee_num>=69381 & insee_num<=69389
replace insee="13055" if insee_num>=13201 & insee_num<=13216
drop insee_num

rename insee com_cd_insee
merge m:1 com_cd_insee using "`processed_data_path'market_ROF.dta", keep(match) nogenerate

// Name of MNOs
local op1 = "Orange"
local op2 = "Bouygues Telecom"
local op3 = "Free"
local op4 = "SFR"

// Name of MNOs if on Windows Phone (Ookla records them differently)
local op1_wp = "Orange F"
local op2_wp = "Bouygues Telecom"
local op3_wp = "Free"
local op4_wp = "F SFR"

// Create MNO number (opcode)
forvalues i=1/4 {
    egen mnc_`i' = mode(mnc) if network_operator_name == "`op`i''" & phone == "iOS" // only iOS include network_operator_name and mnc to identify mnc for android
    egen temp = mode(mnc_`i')
    replace mnc_`i' = temp
    drop temp
    replace mnc = mnc_`i' if phone == "wp" & operator_name == "`op`i'_wp'" // add the correct mnc to the windows phones
}

gen opcode = .
forvalues i=1/4 {
    replace opcode = `i' if mnc == mnc_`i'
}
drop if missing(opcode)

// Aggregate to municipality-MNO level
egen max_download_speed = pctile(download_kbps), p(99.5)
drop if download_kbps > max_download_speed // a few observations are just unbelievably high
gen num_obs = 1
collapse (mean) download_kbps (sum) num_obs, by(market opcode)

// Drop markets in sample if they don't have all 4 MNOs
bys market: gen num = _N
drop if num < 4 & market > 0 // since going to aggregate for the out-of-sample communes, don't don't drop if lacking some MNOs in com_cd_insee
drop num

// Process into quality measures
sort market opcode
gen q_ookla = download_kbps / 1000 // kbps -> Mbps
reshape wide download_kbps num_obs q_ookla, i(market) j(opcode)
gen q_ookla5 = (q_ookla1 + q_ookla2 + q_ookla4) / 3 // download speeds used for MVNOs
reshape long
save "`processed_data_path'ookla_quality.dta", replace
