// This file creates a table that summarizes phone contracts
// Last updated: 27 May 2022

// Location of relevant folders
local orange_idei_path "`1'"
local databases_path "`orange_idei_path'Databases/"
local tariffs_path "`databases_path'tariffs data/"
local processed_data_path "`orange_idei_path'data/"

set more off

use "`processed_data_path'/contractclass.dta", clear
gen num_contracts = 1

// need to expand Orange offerings first because some not offered in final month(s)
sort opcode cclass commit t
by opcode cclass commit: egen max_month_in_group = max(t)
gen needed_months = 24
gen diff_months = needed_months - max_month_in_group 
gen num_expand = diff_months * (t == max_month_in_group) + 1 // + 1 because that's how many ADDITIONAL needed
replace num_expand = 0 if opcode > 1

// expand to the necessary months
gen tempvar = _n
expand num_expand
bys tempvar: gen new_obs = _n - 1
replace t = t + new_obs
drop tempvar

// take care of subsidized phones - this is just copied from Georges's code choiceset.do
g subz_iphone= (iphone_alon- iphone_subz)/24
g subz_samsung= (samsung_alon- samsung_subz)/24
bysort t cclass: egen subi_=mean(subz_iphone) if sim==0 & commit==24 & opcode==1
bysort t cclass: egen subi=min(subi_) if sim==0 & commit==24
g subz_i=subz_iphone if sim==0 & commit==24 & opcode==1
replace subz_i=subi if sim==0 & commit==24 & opcode!=1
replace subz_i=0 if subz_i==.
bysort t cclass: egen subs_=mean(subz_samsung) if sim==0 & commit==24 & opcode==1
bysort t cclass: egen subs=min(subs_) if sim==0 & commit==24
g subz_s=subz_samsung if sim==0 & commit==24 & opcode==1
replace subz_s=subs if sim==0 & commit==24 & opcode!=1
replace subz_s=0 if subz_s==.
replace price = price-subz_i

gen cclass_new = cclass
replace cclass_new = cclass - 1 if cclass >= 2

// calculate the statistics
keep if t == 24
sort opcode cclass_new
by opcode cclass_new: egen min_price = min(price)
by opcode cclass_new: egen max_price = max(price)
by opcode cclass_new: egen min_dlim = min(dlim)
by opcode cclass_new: egen max_dlim = max(dlim)
collapse (sum) num_contracts (min) min* (max) max*, by(opcode cclass_new)
save "`processed_data_path'temp_contract_data_stats.dta", replace

//merge 1:1 opcode cclass using 
use "`processed_data_path'chset_final_newj.dta", clear
drop if p == 888
g dclass=1 if dlim<500
replace dclass=2 if dlim>=500 & dlim<3000
replace dclass=3 if dlim>=3000 & dlim<7000
replace dclass=4 if dlim>=7000 & dlim!=.

g vclass=1 if vlim!=999
replace vclass=2 if vlim==999

g cclass=1 if dclass==1 & vclass==1
replace cclass=2 if dclass==1 & vclass==2
replace cclass=3 if dclass==2 & vclass==1
replace cclass=4 if dclass==2 & vclass==2
replace cclass=5 if dclass==3
replace cclass=6 if dclass==4

gen cclass_new = cclass
replace cclass_new = cclass - 1 if cclass >= 2
drop if cclass == 2

keep if month == 24

merge 1:1 opcode cclass_new using "`processed_data_path'temp_contract_data_stats.dta"
keep if _merge == 3

keep j opcode p dlim vlim num_contracts min_price min_dlim max_price max_dlim cclass
order j opcode p dlim vlim num_contracts min_price max_price min_dlim max_dlim

replace p = round(p, 0.01)
replace min_price = round(min_price, 0.01)
replace max_price = round(max_price, 0.01)

keep opcode p dlim vlim num_contracts min_price max_price min_dlim max_dlim
gen v_dum = 0
replace v_dum = 1 if vlim == 999
drop vlim 
order opcode p dlim v_dum 
gen operator = "Orange"
replace operator = "Bouygues" if opcode == 2
replace operator = "Free Mobile" if opcode == 3
replace operator = "SFR" if opcode == 4
replace operator = "MVNO" if opcode == 5
drop opcode 
order operator p dlim v_dum num_contracts min_price max_price min_dlim max_dlim

save "`processed_data_path'contracts_summary.dta", replace
