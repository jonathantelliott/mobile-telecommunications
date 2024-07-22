// This file computes market size using population data from INSEE, and mobile usage data from CREDOC
// Last updated: 30 July 2022

// Location of relevant folders
local orange_idei_path "`1'"
local databases_path "`orange_idei_path'Databases/"
local insee_path "`databases_path'INSEE data/"
local orange_customer_path "`databases_path'Orange customer data/"
local data_path "`orange_customer_path'data/"
local processed_data_path "`orange_idei_path'data/"

set more off

clear*

	* population per commune (above 12 and total, in 2012)
import delim "`insee_path'popu.csv", delim(";") clear  // data from 2012
	* harmoninzing the commune id
	g code=com	
	gen N=_n
	replace code="" if N>11155 & N<11516 // Corse
	destring code, replace
	g zero=0
	egen com_cd_insee=concat(zero com) if code<10000
	replace com_cd_insee= com if com_cd_insee==""
	drop com zero N code
	
	gen popu12_1=pop_1_12
	forvalues i=13/100{
		replace popu12_1=popu12_1+pop_1_`i'
	}
	gen popu12_2=pop_2_12
	forvalues i=13/100{
		replace popu12_2=popu12_2+pop_2_`i'
	}
	gen popu12=popu12_1+popu12_2

	gen popu_1=pop_1_0
	forvalues i=1/100{
		replace popu_1=popu_1+pop_1_`i'
	}
	gen popu_2=pop_2_0
	forvalues i=1/100{
		replace popu_2=popu_2+pop_2_`i'
	}
	gen popu=popu_1+popu_2
	drop pop_* popu12_1 popu12_2 popu_*
save "`processed_data_path'msize_alt.dta", replace

* Determine communes found in Orange customer data
forvalues i=3/24 {
    use "`data_path'data_`i'.dta", clear
    rename codeinsee com_cd_insee
    keep com_cd_insee
    if `i' > 3 {
        append using "`processed_data_path'temp_comcdinsee.dta"
    }
    duplicates drop
    save "`processed_data_path'temp_comcdinsee.dta", replace
}

* Create list of all markets, including "Rest of France", but only if show up in Orange customer data
use "`processed_data_path'msize_alt.dta", replace
keep com_cd_insee
merge 1:1 com_cd_insee using "`insee_path'market.dta"
replace market = 0 if _merge == 1 // this is a safe identifier for the rest-of-France market b/c in-sample markets start at 1
drop _merge popu
merge 1:1 com_cd_insee using "`processed_data_path'temp_comcdinsee.dta" // only com_cd_insee that show up in Orange customer data
keep if _merge == 3
drop _merge
sort com_cd_insee
save "`processed_data_path'market_ROF.dta", replace
rename com_cd_insee codeinsee
export delimited using "`processed_data_path'market_ROF.csv", replace // this is used in the infrastructure processing code

	* surface and density
import delim "`insee_path'surface.csv", delim(";") clear
	* paris, lyon, marseille
	gen cd_insee= com_cd_insee
	sort cd_insee
	gen counter=_n
	replace cd_insee="" if counter>11178 & counter<11539 // Corse
	destring cd_insee, replace
	replace com_cd_insee="75056" if cd_insee>=75101 & cd_insee<=75120
	replace com_cd_insee="69123" if cd_insee>=69381 & cd_insee<=69389
	replace com_cd_insee="13055" if cd_insee>=13201 & cd_insee<=13216
	drop cd_insee counter com_label
	collapse (sum) population surface, by(com_cd_insee)
	
	merge 1:1 com_cd_insee using "`processed_data_path'msize_alt.dta"
	keep if _merge==3 // dropping overseas from the master file
	drop _merge population labcom
    merge 1:1 com_cd_insee using "`processed_data_path'temp_comcdinsee.dta"
    keep if _merge == 3 // dropping communes not found in Orange customer data
    drop _merge
	drop if surface==0 | surface==.
	
	forvalues i=1/24{
		gen x`i'=.
	}
	*
	reshape long x, i(com_cd_insee) j(month)
	drop x
	
	gen year=2013 if month==1 | month==2
	replace year=2014 if month>2 & month<15
	replace year=2015 if month>=15
	gen psize=popu*((1.0005)^(month+10))
	label var psize "monthly population size"
	gen msize=popu12*((1.0005)^(month+10))
	label var msize "monthly market size"
	// assume that population growth is the same in all communes: 0.05% per month
	// this growth rate is estimated from the population date of above 12, 
	// obtained from INSEE data
	gen density=psize/surface
save "`processed_data_path'msize_alt.dta", replace

* introducing market identifiers
use "`processed_data_path'msize_alt.dta", clear
merge m:1 com_cd_insee using "`processed_data_path'market_ROF.dta"
keep if _merge == 3
drop _merge
replace psize=round(psize)
replace msize=round(msize)
sort market com_cd_insee month

* Save msize with all INSEE codes
save "`processed_data_path'msize_alt_allinsee.dta", replace

* Aggregate to market level
keep market month psize msize
collapse (sum) psize msize, by(market month)
save "`processed_data_path'msize_alt.dta", replace
