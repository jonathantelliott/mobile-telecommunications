// This file creates aggregate market shares
// Last updated: 26 July 2022

// Location of relevant folders
local orange_idei_path "`1'"
local databases_path "`orange_idei_path'Databases/"
local gsma_path "`databases_path'GSMA data/"
local processed_data_path "`orange_idei_path'data/"

// Import and clean GSMA aggregate shares data
import excel "`gsma_path'France-mobile-market-shares.xlsx", sheet("Custom-search-operator") cellrange(A3:DA27) firstrow clear
gen opcode = .
replace opcode = 1 if Operator_name == "Orange"
replace opcode = 2 if Operator_name == "Bouygues Telecom"
replace opcode = 3 if Operator_name == "Free Mobile (Iliad)"
replace opcode = 4 if Operator_name == "SFR (Altice Europe)"
keep if metric_name == "Market share" & attribute_name == "Contract"
keep opcode Q42015
rename Q42015 mshare
gen month = 24
reshape wide mshare, i(month) j(opcode)

// Add MVNO share and adjust shares accordingly
gen mshare5 = 0.115 // this number (2015 MVNO (residential) market share) comes from ARCEP 2016 report
forvalues i=1/4{
replace mshare`i' = mshare`i' * (1.0 - mshare5)
}

// Add outside option share and adjust shares accordingly
gen mshare0 = 0.08
forvalues i=1/5{
replace mshare`i' = mshare`i' * (1.0 - mshare0)
}

// Reshape to put in order
reshape long
reshape wide

// Save
save "`processed_data_path'agg_shares.dta", replace
