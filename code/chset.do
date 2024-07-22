// This file constructs the choice set used for demand estimation
// Last updated: 23 May 2022

// Location of relevant folders
local orange_idei_path "`1'"
local databases_path "`orange_idei_path'Databases/"
local tariffsdata_path "`databases_path'tariffs data/"
local orange_customer_path "`databases_path'Orange customer data/"
local processed_data_path "`orange_idei_path'data/"

set more off


******************************************
* Bottom-up approach
*import excel "C:\Users\gvhoungbonon\Dropbox\orange_idei\databases\tariffs data\raw data\tariffs.xlsx", sheet("alloper") firstrow clear
import excel "`tariffsdata_path'Raw data/tariffs.xlsx", sheet("alloper") firstrow clear
drop if label=="Initial" | label=="forfait bloquÈ spÈcial RSA"

egen t=group(date)
*drop if t<3

g opcode=1 if operator=="Orange"
replace opcode=2 if operator=="Bouygues"
replace opcode=3 if operator=="Free Mobile"
replace opcode=4 if operator=="SFR"
replace opcode=5 if operator=="CIC Mobile" | operator=="NRJ Mobile"

replace dlim=1000 if dlim==1024
replace dlim=2000 if dlim==2048
replace dlim=3000 if dlim==3072
replace dlim=4000 if dlim==4096
replace dlim=5000 if dlim==5120
replace dlim=6000 if dlim==6144
replace dlim=7000 if dlim==7168
replace dlim=8000 if dlim==8192
replace dlim=9000 if dlim==9216
replace dlim=12000 if dlim==12288
replace dlim=16000 if dlim==16384

replace dlim= dlim_option if opcode==5

replace commit=0 if commit==1 | commit==.

* categories of data limits
g dclass=1 if dlim<500
replace dclass=2 if dlim>=500 & dlim<3000
replace dclass=3 if dlim>=3000 & dlim<7000
replace dclass=4 if dlim>=7000 & dlim!=.

* categories of voice limits: limited or unlimited
g vclass=1 if unlimv==. | unlimv==0
replace vclass=2 if unlimv==1

replace vlim=999 if vlim==.

* categories of contracts: focus on data but account for voice allowance in lower data limits (less than 3GB)
g 		cclass=1 if dclass==1 & vclass==1
replace cclass=2 if dclass==1 & vclass==2
replace cclass=3 if dclass==2 & vclass==1
replace cclass=4 if dclass==2 & vclass==2
replace cclass=5 if dclass==3
replace cclass=6 if dclass==4

replace lcost=0 if lcost==.
g sim=simonly
replace sim=1 if simonly==.

* each bundled contract has its standalone version per category ==> drop all bundles
drop if fbb_included==1
drop fbb_included
drop if fbb_required==1
drop fbb_required
drop adsl

keep t opcode dclass vclass cclass sim dlim vlim price commit lcost ///
samsung_alon samsung_subz iphone_alon iphone_subz

order opcode cclass commit t dclass dlim vclass vlim price lcost sim 

sort opcode cclass commit t dclass dlim vclass vlim price lcost sim 

save "`processed_data_path'contractclass.dta", replace

* chossing the least expensive contracts

*import excel "C:\Users\gvhoungbonon\Dropbox\orange_idei\databases\tariffs data\raw data\tariffs.xlsx", sheet("choiceset") firstrow clear
import excel "`tariffsdata_path'Raw data/tariffs.xlsx", sheet("choiceset") firstrow clear
merge 1:m opcode cclass commit t using "`processed_data_path'contractclass.dta"
g available=(_merge==3)

g rp=round(price, .01)
bysort opcode cclass commit t: egen mp=min(rp)
keep if rp==mp // keep the least expensive contract in category and commitment, by operator and time

order opcode cclass commit t available dclass dlim vclass vlim price lcost sim 
sort opcode cclass commit t dclass dlim vclass vlim price lcost sim 
duplicates drop opcode cclass commit t, force

drop _merge rp mp

save "`processed_data_path'chset_v0.dta", replace


* drop alternative if not available for alternative operators
use "`processed_data_path'chset_v0.dta", clear
drop if available==0 & opcode!=1
save "`processed_data_path'chset_v1.dta", replace


// unavailable contracts
clear*
g dlim=.
save "`processed_data_path'popcontract_stay.dta", replace

forvalues i=3/24{
use "`orange_customer_path'data/data_`i'", clear
bysort dataq commit offre: egen n0=sum(stay)
bysort dataq commit : egen mn0=max(n0)
g offre_=offre if n0==mn0 
g p_=price_c if offre_!=""
g sim_=sim_only if offre_!=""
g vlim_=voicemn  if offre_!=""
g fbb_=openprincip if offre_!=""
g fbb__=openmulti if offre_!=""
g fiber_=openfiber if offre_!=""
g roam_=roaming if offre_!=""
g sosh_=sosh if offre_!=""
g n_=n0 if offre_!=""

collapse (firstnm) offre_ p_ sim_ vlim_ fbb_ fbb__ fiber_ roam_ sosh_ n_, by(dataq commit)
ren dataq dlim
ren commitmentstr commit
gen t=`i'
append using "`processed_data_path'popcontract_stay.dta", force
save "`processed_data_path'popcontract_stay.dta", replace
}


use "`processed_data_path'popcontract_stay.dta", clear
drop if offre_=="FIN"

g dclass=1 if dlim<500
replace dclass=2 if dlim>=500 & dlim<3000
replace dclass=3 if dlim>=3000 & dlim<7000
replace dclass=4 if dlim>=7000 & dlim!=.

g vclass=1 if vlim!=999
replace vclass=2 if vlim==999

g 		cclass=1 if dclass==1 & vclass==1
replace cclass=2 if dclass==1 & vclass==2
replace cclass=3 if dclass==2 & vclass==1
replace cclass=4 if dclass==2 & vclass==2
replace cclass=5 if dclass==3
replace cclass=6 if dclass==4

bysort cclass commit t : egen rank=rank(n), field  // most kept contracts
keep if rank==1

g opcode=1

g rp=round(p_, .01)
bysort opcode cclass commit t: egen mp=min(rp)
keep if rp==mp // and the least expensive
ren dlim dlim_
drop dclass n_ vclass rank
save "`processed_data_path'repcontract_stay.dta", replace



* dealing with ORG's contracts no longer available
use "`processed_data_path'repcontract_stay.dta", clear
merge 1:1 opcode cclass commit t using "`processed_data_path'chset_v1.dta"
order opcode cclass commit t available dclass dlim vclass vlim price lcost sim
sort opcode cclass commit t

	* imputing missing values for unavailable contracts
foreach i in dlim vlim price lcost sim iphone_subz iphone_alon samsung_subz samsung_alon{
replace `i'=`i'[_n-1] if opcode==1 & cclass==1 & commit==24 & t==24
}
*
replace dlim=dlim_ if opcode==1 & cclass==2 & commit==12 & t==12
replace dlim=dlim[_n-1] if opcode==1 & cclass==2 & commit==12 & t>12
foreach i in vlim price lcost sim  iphone_subz iphone_alon samsung_subz samsung_alon{
replace `i'=`i'[_n-1] if opcode==1 & cclass==2 & commit==12 & t>11
}
*
replace dlim=dlim_ if opcode==1 & cclass==2 & commit==24 & t==12
replace dlim=dlim[_n-1] if opcode==1 & cclass==2 & commit==24 & t>12
replace price=p_ if opcode==1 & cclass==2 & commit==24 & t==12
replace price=price[_n-1] if opcode==1 & cclass==2 & commit==24 & t>12
foreach i in vlim lcost sim iphone_subz iphone_alon samsung_subz samsung_alon{
replace `i'=`i'[_n-1] if opcode==1 & cclass==2 & commit==24 & t>11
}
*
replace dlim=dlim_ if opcode==1 & (cclass==3 | cclass==4 | cclass==6) & commit==0
replace vlim=vlim_ if opcode==1 & (cclass==3 | cclass==4 | cclass==6) & commit==0
replace lcost=sosh_ if opcode==1 & (cclass==3 | cclass==4 | cclass==6) & commit==0
replace sim=sim_ if opcode==1 & (cclass==3 | cclass==4 | cclass==6) & commit==0
replace price=p_ if opcode==1 & (cclass==3 | cclass==4) & commit==0
replace price=p_ if opcode==1 & cclass==6 & commit==0 & t==4
replace price=price[_n-1] if opcode==1 & cclass==6 & commit==0 & t>4
foreach i in dlim vlim price lcost sim  iphone_subz iphone_alon samsung_subz samsung_alon{
replace `i'=`i'[_n+1] if opcode==1 & cclass==6 & t==3
}
*
foreach i in dlim vlim price lcost sim  iphone_subz iphone_alon samsung_subz samsung_alon{
replace `i'=`i'[_n-1] if opcode==1 & (cclass==3 | cclass==4) & commit==24 & t==24
}
*
foreach i in dlim vlim price lcost sim  iphone_subz iphone_alon samsung_subz samsung_alon{
replace `i'=`i'[_n+1] if opcode==1 & (cclass==3 | cclass==4) & commit==0 & t==2
}
*
foreach i in dlim vlim price lcost sim  iphone_subz iphone_alon samsung_subz samsung_alon{
replace `i'=`i'[_n+1] if opcode==1 & (cclass==3 | cclass==4) & commit==0 & t==1
}
*
foreach i in dlim vlim price lcost sim  iphone_subz iphone_alon samsung_subz samsung_alon{
replace `i'=`i'[_n+1] if opcode==1 & cclass==6 & t==2
}
foreach i in dlim vlim price lcost sim  iphone_subz iphone_alon samsung_subz samsung_alon{
replace `i'=`i'[_n+1] if opcode==1 & cclass==6 & t==1
}
drop dlim_- mp _merge

replace sim=1 if commit==0 & sim==0
ren available choiceset

	* calculating handset subsidies
g subz_iphone= (iphone_alon- iphone_subz)/24
g subz_samsung= (samsung_alon- samsung_subz)/24

bysort t cclass: egen subi_=mean(subz_iphone) if sim==0 & commit==24 & opcode==1
bysort t cclass: egen subi=min(subi_) if sim==0 & commit==24 // applying ORG's average subsidy to competitors
g subz_i=subz_iphone if sim==0 & commit==24 & opcode==1
replace subz_i=subi if sim==0 & commit==24 & opcode!=1
replace subz_i=0 if subz_i==.

bysort t cclass: egen subs_=mean(subz_samsung) if sim==0 & commit==24 & opcode==1
bysort t cclass: egen subs=min(subs_) if sim==0 & commit==24
g subz_s=subz_samsung if sim==0 & commit==24 & opcode==1
replace subz_s=subs if sim==0 & commit==24 & opcode!=1
replace subz_s=0 if subz_s==.

g p=price-subz_i

save "`processed_data_path'chset.dta", replace



*** final choice set
use "`processed_data_path'chset.dta", clear
egen j=group(opcode cclass commit)
keep j opcode cclass commit t choiceset dlim vlim lcost p



reshape wide choiceset dlim vlim lcost p, i(j) j(t)
reshape long choiceset dlim vlim lcost p, i(j) j(t)
replace choiceset=0 if choiceset==.
replace dlim=888 if dlim==.
replace vlim=888 if vlim==.
replace lcost=888 if lcost==.
replace p=888 if p==.
replace commit=1 if commit==0
reshape wide commit choiceset dlim vlim lcost p, i(j) j(t)

order j opcode cclass commit* choiceset* p* dlim* vlim* lcost*
save "`processed_data_path'chset_final.dta"
