// This file creates the infrastructure dataset by market
// Last updated: 23 February 2024

// Location of relevant folders
local orange_idei_path "`1'"
local databases_path "`orange_idei_path'Databases/"
local insee_path "`databases_path'INSEE data/"
local osiris_path "`databases_path'quality data/raw data/OSIRIS/"
local antennas_path "`databases_path'antennas/"
local spatial_path "`databases_path'spatial/"
local cognos_path "`databases_path'quality data/raw data/COGNOS/"
local ookla_path "`databases_path'Ookla data/"
local processed_data_path "`orange_idei_path'data/"

set more off


* --------------------------
* ookla quality data

use "`processed_data_path'ookla_quality.dta", clear
reshape wide q_ookla download_kbps num_obs, i(market) j(opcode)
merge 1:m market using "`processed_data_path'market_ROF.dta"
keep if _merge == 3
drop _merge
rename com_cd_insee codeinsee

 * Create included markets - a few don't have Ookla measures
preserve
keep codeinsee
save "`processed_data_path'sample_markets.dta", replace

* Aggregate by market
restore
collapse (mean) q_ookla* download_kbps* num_obs*, by(market) // 0 is dropped later so no need to msize-weight
sort market
outsheet using "`processed_data_path'quality_ookla.csv", comma names replace



* -------------------------------------------------------
* arrange data set for base-station statistics

use "`antennas_path'anfrapr2016.dta", clear

* focus on metro france (things work differently in Guiana and Polynesia)
gen lat = cor_nb_dg_lat + cor_nb_mn_lat/60 + cor_nb_sc_lat/3600
replace lat = -lat if cor_cd_ns_lat=="S"
gen lon = cor_nb_dg_lon + cor_nb_mn_lon/60 + cor_nb_sc_lon/3600
replace lon = -lon if cor_cd_ew_lon=="W"
keep if lon>=-5 & lon<=10
keep if lat>=40 & lat<=53


* coordinate id
egen bid = group(cor_nb_dg_lat cor_nb_mn_lat cor_nb_sc_lat cor_cd_ns_lat cor_nb_dg_lon cor_nb_mn_lon cor_nb_sc_lon cor_cd_ew_lon), missing
drop adr_lb*

* coordinator-operator id
egen fbid = group(operator bid), missing

* to aggregate multiple listings for same location...
forvalues i = 790/2690 {
  gen band`i' = 0
  replace band`i' = 1 if (ban_nb_f_deb <= `i') & (ban_nb_f_fin >= `i'+1)
  replace band`i' = 1 + `i' - ban_nb_f_deb if (ban_nb_f_deb > `i') & (ban_nb_f_deb <= `i'+1) & (ban_nb_f_fin >= `i'+1)
  replace band`i' = ban_nb_f_fin - `i' if (ban_nb_f_deb <= `i') & (ban_nb_f_fin < `i'+1) & (ban_nb_f_fin >= `i')
  quietly su band`i'
  if r(max)==0 {
    drop band`i'
  }
}


* collapse by location-techno
* bwrep will be sum of bandwidth in ANFR listings, even if
* same band is used on two rows (two components/antennae) in same location
rename bandwidth bwrep
collapse (sum) bwrep (max) band*, by(operator com_cd_insee fbid techno lat lon)

* in contrast, bw will not double-count components operating the
* same frequencies in a given location
egen bw = rowtotal(band791-band2689)


* note that there are many, many cases where bwrep
* is much larger than bw. this indicates that the same
* frequencies are operated by multiple components at the
* same base station
* ...
* recent conversations suggest this probably comes from directional signals

order com_cd_insee lat lon techno operator bwrep bw
sort com_cd_insee lat lon techno

gen bw3g = bw if techno=="3G"
gen bw4g = bw if techno=="4G"
gen bwrep3g = bwrep if techno=="3G"
gen bwrep4g = bwrep if techno=="4G"
collapse (max) bwrep3g bwrep4g (max) bw3g bw4g band*, by(operator com_cd_insee fbid lat lon)


sort com_cd_insee operator
order operator com_cd_insee fbid lat lon bw3g bw4g bwrep3g bwrep4g band*

save "`processed_data_path'base_station_data.dta", replace


* -------------------------------------------------------
* Compute "effective area" of commune using population density
* Effective population density is contraharmonic mean of population
* density of population density over space (or mean of population density
* over people)
* Effective area is population density divided by effective area

use "`processed_data_path'msize_alt_allinsee.dta", clear
keep com_cd_insee month msize
rename msize msize_
reshape wide msize, i(com_cd_insee) j(month)
keep com_cd_insee msize_24
rename com_cd_insee codeinsee
save "`processed_data_path'msize_infr.dta", replace


insheet using "`processed_data_path'PopDensMeans.csv", clear
save "`processed_data_path'pop_means.dta", replace

* note: here the aggregate insee codes are already used
insheet using "`insee_path'surface.csv", clear delimiter(";")
rename com_cd_insee codeinsee
merge 1:1 codeinsee using "`processed_data_path'pop_means.dta"
keep if _merge==3
drop _merge

merge 1:1 codeinsee using "`processed_data_path'msize_infr.dta"
keep if _merge==3
drop _merge

destring chmean normalmean, ignore("NA") replace
gen demomean = msize_24/surface

//scatter normalmean demomean
//scatter normalmean chmean

* rather than using contraharmonic mean directly from raster data
* use ratio of contraharmonic and normal mean from raster data,
* and multiply the mean from the demographic data by that ratio.
gen pdens_clean = demomean
replace pdens_clean = pdens_clean*chmean/normalmean if normalmean!=.



gen area_effective = msize_24/pdens_clean

* based on brief inspection, this seems to work well
* it doesn't make a huge difference (very high correlations),
* and it leads to smaller effective areas, which makes sense
* given that many communes will have some areas which are
* effectively not served
//scatter surface area_effective
su surface area*

* the places with lowest effratios have lots of uninhabited area
* (Fontainebleu, Marseille, La Chappelle-sur-Erdre)
gen effratio = area_effective/surface
sort effratio



drop effratio
rename codeinsee com_cd_insee
order com_cd_insee population surface area_effective
outsheet using "`processed_data_path'effective_pop_dens.csv", comma names replace
save "`processed_data_path'effective_pop_dens.dta", replace

* -------------------------------------------------
* generate combined base stations data

use "`antennas_path'anfrapr2016.dta", clear

gen lat = cor_nb_dg_lat + cor_nb_mn_lat/60 + cor_nb_sc_lat/3600
replace lat = -lat if cor_cd_ns_lat=="S"
gen lon = cor_nb_dg_lon + cor_nb_mn_lon/60 + cor_nb_sc_lon/3600
replace lon = -lon if cor_cd_ew_lon=="W"
keep if lon>=-5 & lon<=10
keep if lat>=40 & lat<=53

* coordinate id
egen bid = group(cor_nb_dg_lat cor_nb_mn_lat cor_nb_sc_lat cor_cd_ns_lat cor_nb_dg_lon cor_nb_mn_lon cor_nb_sc_lon cor_cd_ew_lon), missing
drop adr_lb*

gen dO = (operator=="ORANGE")
gen dS = (operator=="SFR")
gen dB = (operator=="BOUYGUES TELECOM")
gen dF = (operator=="FREE MOBILE")

* collapse by base station location, indciator for whether each firm is operating there
collapse (max) dO dS dB dF, by(com_cd_insee bid)


* whether various pairs of firms have a base station at each location
gen stationsOS = min(dO + dS,1)
gen stationsOB = min(dO + dB,1)
gen stationsOF = min(dO + dF,1)
gen stationsSB = min(dS + dB,1)
gen stationsSF = min(dS + dF,1)
gen stationsBF = min(dB + dF,1)

* fix some commune coding
gen codeinsee = com_cd_insee
replace codeinsee = "" if missing(real(com_cd_insee))
destring codeinsee, replace
replace codeinsee=75056 if codeinsee>=75101 & codeinsee<=75120
replace codeinsee=69123 if codeinsee>=69381 & codeinsee<=69389
replace codeinsee=13055 if codeinsee>=13201 & codeinsee<=13216
tostring codeinsee, replace
replace com_cd_insee = codeinsee if codeinsee!="" & codeinsee!="."

* total number of base stations operated by either firm for each firm pair, by commune 
collapse (sum) stations*, by(com_cd_insee)

save "`processed_data_path'merged_base_stations.dta", replace



* -------------------------------------------------
* arrange and save market-level infrastructure info

use "`processed_data_path'base_station_data.dta", clear

gen codeinsee = com_cd_insee
replace codeinsee = "" if missing(real(com_cd_insee))
destring codeinsee, replace
replace codeinsee=75056 if codeinsee>=75101 & codeinsee<=75120
replace codeinsee=69123 if codeinsee>=69381 & codeinsee<=69389
replace codeinsee=13055 if codeinsee>=13201 & codeinsee<=13216
tostring codeinsee, replace
replace com_cd_insee = codeinsee if codeinsee!="" & codeinsee!="."
gen stations = 1

* by 3G/4G, operator and commune, compute mean bandwidth across towers
* both ignoring redundant antennas and counting the bandwidth repeatedly
collapse (count) stations (mean) bw3g bw4g bwrep3g bwrep4g (max) band*, by(operator com_cd_insee)

* also compute the maximum bandwidth operated across towers
egen bwmax = rowtotal(band791-band2689)
drop band*

merge m:1 com_cd_insee using "`processed_data_path'effective_pop_dens.dta"
keep if _merge==3
drop _merge

* select only communes in sample
merge m:1 com_cd_insee using "`processed_data_path'market_ROF.dta"
keep if _merge==3
drop _merge

gen opcode = 1 if operator =="ORANGE"
replace opcode=2 if operator=="BOUYGUES TELECOM"
replace opcode=3 if operator=="FREE MOBILE"
replace opcode=4 if operator=="SFR"

drop operator
reshape wide bw3g bw4g bwrep3g bwrep4g bwmax stations, i(com_cd_insee) j(opcode)

* fill in missings with zeros
forvalues i = 1/4 {
  replace stations`i' = 0 if stations`i'==.
  replace bw3g`i'=0 if bw3g`i'==.
  replace bw4g`i'=0 if bw4g`i'==.
  replace bwrep3g`i'=0 if bwrep3g`i'==.
  replace bwrep4g`i'=0 if bwrep4g`i'==.
  replace bwmax`i'=0 if bwmax`i'==.
}

merge 1:1 com_cd_insee using "`processed_data_path'merged_base_stations.dta"
keep if _merge == 3
drop _merge

rename com_cd_insee codeinsee

order market codeinsee population surface area_effective bw3g* bw4g* bwrep3g* bwrep4g* bwmax* stations1 stations2 stations3 stations4 pdens_clean
sort market


merge m:1 codeinsee using "`processed_data_path'sample_markets.dta"
keep if _merge==3
drop _merge


// aggregate by market
collapse (max) bwmax* population msize_24 surface area_effective pdens_clean (mean) bw3g* bw4g* bwrep3g* bwrep4g* stations* [fw=msize_24], by(market) // b/c Rest-of-France gets dropped later and doesn't enter moments, aggregating in this way okay b/c 1-1 for market and codeinsee in our sample
sort market

outsheet using "`processed_data_path'infrastructure_clean.csv", comma names replace
save "`processed_data_path'infrastructure.dta", replace

* clear positive relationship between pop dens and tower density
gen bsd1 = stations1/area_effective
//scatter bsd1 pdens_clean

* scatterplots look reasonable
* note: still have a few communes with bandwidth that is weirdly high
* seems that base stations per capita are decreasing with pop density
* slightly, at least for low population densities. bandwidth operated
* appears to be increasing with population density.
gen bspc1 = stations1/population
//scatter bspc1 pdens_clean
//scatter bw3g1 pdens_clean
//scatter bw4g1 pdens_clean



* -------------------------------------------------
* download speed data

insheet using "`cognos_path'sitecoordinates.csv", clear names delimiter(";")
duplicates drop nomsite codeinsee, force
drop nomsicellule
drop if codeinsee==""
gen site = subinstr(nomsite,"_","",.)
replace site = subinstr(site," ","",.)
bysort site: gen nobs = _N
drop if nobs!=1
drop nobs
save "`processed_data_path'sites.dta", replace




use "`osiris_path'data.dta", clear
merge m:1 site using "`processed_data_path'sites.dta"
keep if _merge==3
drop _merge

* keep noon-1pm
gen hour = substr(datehour,-2,2)
keep if hour =="12"

gen code_backup = codeinsee
replace codeinsee = "" if missing(real(codeinsee))
destring codeinsee, replace
replace codeinsee=75056 if codeinsee>=75101 & codeinsee<=75120
replace codeinsee=69123 if codeinsee>=69381 & codeinsee<=69389
replace codeinsee=13055 if codeinsee>=13201 & codeinsee<=13216
tostring codeinsee, replace
replace codeinsee = code_backup if missing(real(code_backup))


gen ncell = 1 if downspeed!=.
collapse (sum) usersps bitps ncell, by(codeinsee techno)
gen downspeed = bitps/usersps
gen tech = 3 if techno=="3G"
replace tech = 4 if techno=="4G"
drop techno
reshape wide downspeed usersps bitps ncell, i(codeinsee) j(tech)

save "`processed_data_path'download_speed.dta", replace
merge m:1 codeinsee using "`processed_data_path'sample_markets.dta"
keep if _merge==3
drop _merge

// aggregate by market
rename codeinsee com_cd_insee
merge 1:1 com_cd_insee using "`processed_data_path'market_ROF.dta", keep(match) nogen
rename com_cd_insee codeinsee
merge 1:1 codeinsee using "`processed_data_path'msize_infr.dta", keep(match) nogen
collapse (mean) downspeed* usersps* bitps* ncell* [fw=msize_24], by(market)
sort market

outsheet using "`processed_data_path'quality_clean.csv", comma names replace


* --------------------------
* determine median frequency

use "`antennas_path'anfrapr2016.dta", clear

* focus on metro france (things work differently in Guiana and Polynesia)
gen lat = cor_nb_dg_lat + cor_nb_mn_lat/60 + cor_nb_sc_lat/3600
replace lat = -lat if cor_cd_ns_lat=="S"
gen lon = cor_nb_dg_lon + cor_nb_mn_lon/60 + cor_nb_sc_lon/3600
replace lon = -lon if cor_cd_ew_lon=="W"
keep if lon>=-5 & lon<=10
keep if lat>=40 & lat<=53

gen codeinsee = com_cd_insee
replace codeinsee = "" if missing(real(com_cd_insee))
destring codeinsee, replace
replace codeinsee=75056 if codeinsee>=75101 & codeinsee<=75120
replace codeinsee=69123 if codeinsee>=69381 & codeinsee<=69389
replace codeinsee=13055 if codeinsee>=13201 & codeinsee<=13216
tostring codeinsee, replace
replace com_cd_insee = codeinsee if codeinsee!="" & codeinsee!="."

drop codeinsee

merge m:1 com_cd_insee using "`processed_data_path'market_ROF.dta"
keep if _merge==3
drop _merge
rename com_cd_insee codeinsee

* looks like median frequency is about 1900
keep if market > 0 // only interested in in-sample communes
gen fmid = (ban_nb_f_deb + ban_nb_f_fin)/2
su fmid, d
