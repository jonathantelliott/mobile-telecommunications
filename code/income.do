// This file computes the distribution of income across communes
// Last updated: 26 July 2022

// Location of relevant folders
local orange_idei_path "`1'"
local databases_path "`orange_idei_path'Databases/"
local insee_path "`databases_path'INSEE data/"
local processed_data_path "`orange_idei_path'data/"

set more off

clear*

* income deciles per commune
import delim "`insee_path'income_com.csv", delim(";") clear
save "`processed_data_path'income_alt.dta", replace

* add income deciles per department
import delim "`insee_path'income_dep.csv", delim(";") clear
	* merge with commune decile of income
	merge 1:m dep using "`processed_data_path'income_alt.dta", gen(income_dep_merge)
	* fill missing commune incomes by department incomes
	foreach j in 1 2 3 4 6 7 8 9{
	replace income_c_`j'=income_d_`j'*(income_c_5/income_d_5) if income_c_`j'==. & income_c_5!=.
	}
	* data cleaning
	forvalues j=1/9{
	ren income_c_`j' yc`j'
	ren income_d_`j' yd`j'
	}
	ren com com_cd_insee
    
	* final variables
	
	* Aggregate to market level
	gen month = 24 // need to choose specific month, this is the month used later in estiimation
	merge m:1 com_cd_insee month using "`processed_data_path'msize_alt_allinsee.dta", keep(match) nogen
	keep market com_cd_insee yc* msize
	collapse (mean) yc* [fw=msize], by(market)
	
	keep market yc*
	order market yc1 yc2 yc3 yc4 yc5 yc6 yc7 yc8 yc9
	sort market
save "`processed_data_path'income_alt.dta", replace
