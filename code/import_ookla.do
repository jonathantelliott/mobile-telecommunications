// This file combines Ookla records together into a Stata dataset by coordinates
// Last updated: 23 May 2022

// Location of relevant folders
local orange_idei_path "`1'"
local databases_path "`orange_idei_path'Databases/"
local ookla_path "`databases_path'Ookla data/"
local ookla_rawdata_path "`ookla_path'raw_data/"
local processed_data_path "`orange_idei_path'data/"

set more off

local op1 = "android"
local op2 = "iOS"
local op3 = "wp"

forvalues i=1/3 {
	forvalues m=4/6 {
		display "`op`i''"
		display "`m'"
		import delimited "`ookla_rawdata_path'`op`i''_2016-0`m'-01.csv", clear
		gen phone = "`op`i''"
		if `i' != 1 | `m' != 4 {
			append using "`processed_data_path'phone_records.dta"
		}
		save "`processed_data_path'phone_records.dta", replace
	}
}
