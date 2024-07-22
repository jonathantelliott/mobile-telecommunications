// This file creates the coordinates of the antennas as a csv file
// Last updated: 24 May 2022

// Location of relevant folders
local orange_idei_path "`1'"
local databases_path "`orange_idei_path'Databases/"
local processed_data_path "`orange_idei_path'data/"

set more off

* population density source:
* http://sedac.ciesin.columbia.edu/data/collection/gpw-v4

* -------------------------------------------------------
* locations of antennas for spatial merge
use "`databases_path'antennas/anfrapr2016.dta", clear
contract cor_nb_dg_lat cor_nb_mn_lat cor_nb_sc_lat cor_cd_ns_lat cor_nb_dg_lon cor_nb_mn_lon cor_nb_sc_lon cor_cd_ew_lon
drop _freq
outsheet using "`processed_data_path'ant_coordinates.csv", comma names replace
