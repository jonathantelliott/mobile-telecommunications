// This file constructs average daa usage for Orange mobile service plans
// Last updated: 13 November 2023

// Location of relevant folders
local orange_idei_path "`1'"
local databases_path "`orange_idei_path'Databases/"
local tariffsdata_path "`databases_path'tariffs data/"
local orange_customer_path "`databases_path'Orange customer data/"
local data_path "`orange_customer_path'data/"
local insee_path "`databases_path'INSEE data/"
local processed_data_path "`orange_idei_path'data/"

set more off

// Go through each data file, process average data consumption, append to complete file of averages
forvalues i=3/24 {

	use "`processed_data_path'chset_final.dta", clear
	reshape long commit choiceset p dlim vlim lcost, i(j opcode cclass) j(month)
	rename choiceset choiceset_chset
	rename p p_chset
	rename dlim dlim_chset
	rename vlim vlim_chset
	rename lcost lcost_chset
	keep if opcode == 1
	save "`processed_data_path'temp1.dta", replace

	use "`data_path'data_`i'.dta", clear
	
	keep dataq voicemn commitmentstr totaldata codeinsee outcall remaining customer
	
	* mapping markets
	rename codeinsee com_cd_insee
	merge m:1 com_cd_insee using "`processed_data_path'market_ROF.dta"
	rename com_cd_insee codeinsee
	keep if _merge==3
	drop _merge

	rename dataq dlim
	rename voicemn vlim
	rename commitmentstr commit
	
	* Create cclass
	gen dclass = 1 if dlim < 500
	replace dclass = 2 if dlim >= 500 & dlim < 3000
	replace dclass = 3 if dlim >= 3000 & dlim < 7000
	replace dclass = 4 if dlim >= 7000 & dlim != .
	
	gen vclass = 1 if vlim != 999
	replace vclass = 2 if vlim == 999
	
	gen cclass = 1 if dclass == 1 & vclass == 1
	replace cclass = 2 if dclass == 1 & vclass == 2
	replace cclass = 3 if dclass == 2 & vclass == 1
	replace cclass = 4 if dclass == 2 & vclass == 2
	replace cclass = 5 if dclass == 3
	replace cclass = 6 if dclass == 4
	
	replace commit = 1 if commit == 0
	
	* mapping choiceset
	gen month = `i'
	merge m:1 month cclass commit using "`processed_data_path'temp1.dta"
	keep if _merge==3
	drop _merge
	
	g vpu=outcall/customer
	sum vpu [fw=customer],d
	g vcons=outcall
	replace vcons=r(p99)*customer if outcall!=0 & vpu>r(p99)
	
	replace totaldata = dlim_chset * 1000 * customer if (dlim_chset < 500) & ((totaldata/customer)/1000 > dlim_chset)
	g dpu=(totaldata/customer)/1000 if dlim!=0 
	g dcons=totaldata/1000
	su dpu [fw=customer],d
	g maxd=2*r(p99)
	replace dpu = 0 if missing(dpu)
	replace dcons=maxd*customer if totaldata!=0 & dpu>maxd
	
	g nbvcons=(vcons>0)
	g nbdcons=(dcons>0)
	replace remaining=0 if remaining==.
	
	* generate variable needed to construct data used
    gen dbar = dcons / customer
    replace dbar = dlim if (dlim < 500) & (dbar > dlim) // don't consider add-on packages
    replace dbar = . if (dlim != dlim_chset) // only want to consider the cases where we're considering a plan with the same data limit as the data limit of the choice set
    gen customer_use = .
    replace customer_use = customer if !missing(dbar)
	save "`processed_data_path'temp1.dta", replace
	collapse (sum) customer customer_use, by(market j cclass commit)
	save "`processed_data_path'temp2.dta", replace
	use "`processed_data_path'temp1.dta", clear
	collapse (mean) dbar [fw=customer], by(market j cclass commit)
	merge 1:1 market j cclass commit using "`processed_data_path'temp2.dta"
	drop _merge
	save "`processed_data_path'temp2.dta", replace
	
	* sum customers by cclass and use only data usage statistics for the 24-month contracts
	sort market cclass
	gen maxcommit = 24 // all Orange cclass have a 24-month j
	by market cclass: egen customer_sum = sum(customer)
	keep if commit == maxcommit // now that have summed customers, no longer need any data from non 24-month contracts
	drop maxcommit
	rename customer customer_24mcontract
	rename customer_sum customer
	
	gen month = `i'
	
	keep market j customer_24mcontract customer customer_use dbar month
	
	/* merge */
	if `i' == 3 {
		save "`processed_data_path'dbar.dta", replace
	}
	else{
		append using "`processed_data_path'dbar.dta"
		save "`processed_data_path'dbar.dta", replace
	}
}
sort j month market
save "`processed_data_path'dbar.dta", replace
