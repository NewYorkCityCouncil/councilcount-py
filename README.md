<picture align="center">
  <img alt="NYCC Data Team Logo" src="https://raw.githubusercontent.com/NewYorkCityCouncil/councilcount-py/53afaa0e28072eb80d32d65c8d8638fa00f18436/data-council-logo.png">
</picture>

# Overview

`councilcount` is the companion package for CouncilCount, a webpage designed by the New York City Council Data Team that visualizes population data for around 200 demographic groups across various NYC geographic boundaries. Where possible, this data was sourced directly from the 5-Year American Community Survey (ACS). For geographic boundaries that are not available in the census hierarchy, like council districts, estimates were generated (see Methodology). This package allows for easy acces to the estimates displayed on CouncilCount, as well as the ability to generate new estimates using the same methodology. 

Visit CouncilCount at https://rnd.council.nyc.gov/councilcount/. 

# Installation

To install `councilcount` for Python, please use the following code:

Use pip to install the package in the terminal:
``` bash
pip install councilcount
```
Then import the package in Python:
``` python
import councilcount as cc
```

To install for R, please use the following code:

```R
install.packages('reticulate')
library(reticulate)
py_install("councilcount")
cc <- import("councilcount")
```
You may be prompted to import other Python packages using the same import structure before `councilcount` successfully installs.

To access functions while using R, use this template:

```R
cc$<FUNCTION>

# here is an example
acs_year = 2023 
cc$get_available_councilcount_codes(acs_year=acs_year)
```

**Note** When using R, you may need to wrap single item lists in this fashion: make_py_list(c("")).

# Prerequisites

- Python version 3.9 or above is needed.

### Functions

`councilcount` includes functions that allow users to pull from the existing database of estimates currently displayed on the CouncilCount webpage, as well as to generate completely new estimates.

**Note:** As per Census notation, variable codes ending in 'E' are number estimates. Those ending in 'M' are number margins of error (MOEs). Adding 'P' before 'E' or 'M' means the value is now a percent. The Data Team devised a new code ending in 'V' to represent coefficients of variation (CVs). Columns in the DataFrames produced by all `councilcount` functions will be named accordingly.

To explore the existing CouncilCount database:

* `get_councilcount_estimates()` – Creates a dataframe that provides demographic estimates, MOEs, and CVs for selected variables along chosen geographic boundaries (e.g. council district, borough, etc.) for a chosen ACS 5-Year survey. Pulls from existing data. Use this function if the estimates you are seeking have already been generated. 
* `get_available_councilcount_codes()` – Provides information on all of the available demographic variables that can be accessed via `get_councilcount_estimates()` for a specified survey year. 
* `get_bbl_population_estimates()` – Generates a DataFrame that provides total population estimates at the borough, block, lot (BBL) level (BBLs are more or less equivalent to buildings). There are latitude and longitude columns, which allow the dataset to be spatially joined with GeoDataFrames containing geographic boundaries provided by the user. This allows for the aggregation of population estimates to custom geographies. The estimates grow increasingly reliable as they are aggregated to larger geographic regions. Do not use estimates for individual BBLs. 

Here is an example, in which codes for “Female” and “Adults with
Bachelor’s degree or higher” will be used. The data will be requested along 2023
Council District boundaries for the 2019-2023 ACS.

First, review the ACS years available for existing estimates, which will be drawn from what is currently displayed on the CouncilCount webpage:
```Python
cc.available_years() 
```
Next, review the codes available in the CouncilCount database for the chosen year:
```Python
acs_year = 2023 
cc.get_available_councilcount_codes(acs_year=acs_year)
```

Then, retrieve the desired estimates for the selected year and geographic boundaries. Available geography inputs: 'councildist' (Council District), 'communitydist' (Community District), 'schooldist' (School District), 'policeprct' (Police Precinct), 'modzcta' (Modified ZIP Code Tabulation Area), 'nta' (Neighborhood Tabulation Area), 'borough' (Borough), and 'city' (New York City).
``` Python
var_codes = [
"DP05_0003E", # Female
"DP02_0065E" # Adults with Bachelor’s degree or higher
]
geo = "councildist" # "councildist", "policeprct", "schooldist", "nta", "communitydist", "modzcta", "borough", and "city" are acceptable inputs
boundary_year = 2023 # only necessary for Council District requests—2013 and 2023 are accetable inputs

cc.get_councilcount_estimates(acs_year=acs_year, geo=geo, var_codes=var_codes, boundary_year=boundary_year) 
```

**Note:** Percent estimate variables (codes ending in 'PE') in tables produced by `get_councilcount_estimates()` have varying denominators. To find the denominator used for a specific variable, view the 'denominator_var_code' column in the table generated by `get_available_councilcount_codes()`. For variables from the Data Profiles survey, which provides percent estimates for each variable, denominators will reflect those used by the ACS in order to keep the meaning of variable codes consistent with the survey. The Detailed and Subject Table surveys usually do not provide any percent estimates for variables directly, so the denominators for percent estimates from these surveys were selected by the Data Team based on information found in [this Census document](https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2023_ACSSubjectDefinitions.pdf) (edit the year in the URL to switch ACS surveys).

In a separate example, let's review how to use `get_bbl_population_estimates()`:

Simply enter the desired year (can be taken from `available_years()` as well). A DataFrame with BBL-level population estimates for the year will be produced. Remember not to use estimates for individual BBLs. Aggregation to larger geographic regions is highly encouraged.
```Python
year = 2016
cc.get_bbl_population_estimates(year=year)
```

To generate new estimates:

* `generate_new_estimates()` - Generates demographic estimates, MOEs, and CVs for a specified NYC geography. Use this function if the ACS demographic variable you are looking for is not already available in the CouncilCount database. Available surveys include the Detailed Tables, Data Profiles, and Subject Tables 5-Year ACS (for the `survey_key` parameter in this function, input '1' for Detailed, '2' for Profiles, and '3' for Subject). 
* `get_census_api_codes()`: Pulls from an ACS 5-Year data dictionary to show all available variable codes for a given survey and year. Use this function to search for variables to use in `generate_new_estimates()`. You may also visit this [link](https://api.census.gov/data/2023/acs/acs5.html) and click the "variables" hyperlink associated with the desired survey to search in a web format (edit the year in the URL to switch ACS surveys). To view the variables available in the existing CouncilCount database, please use `get_available_councilcount_codes()` instead.

Here is an example in which new estimates are created. The data is requested along school district boundaries for the 2007-2011 ACS Data Profiles, which was shown to be available by `available_years()` above. 

First, review the codes available in the ACS Data Profiles database. Generate your own census API key [here](https://api.census.gov/data/key_signup.html):
```Python
survey_key = '2' # code representing Data Profiles survey
acs_year = 2011
census_api_key = "<INSERT KEY>"
cc.get_census_api_codes(survey_key=survey_key, acs_year=acs_year, census_api_key=census_api_key)
```
 
Then, generate the new estimates. For each demographic code, indicate whether it is a household or person-level estimate. Codes for "total population" and/ or "total households" must be also included if person and/ or household-level estimates have been requested. Output columns for these variables will also be provided. All geographies listed above as options for `get_councilcount_estimates()` work for `generate_new_estimates()` as well.
```Python
demo_dict = {
    "DP02_0002E": "household", # Married-couple household
    "DP02_0024E": "person", # Males 15 and over
    "DP02_0025E": "person" # Never married males 15 and over
    }
geo = "schooldist" # "councildist", "policeprct", "schooldist", "nta", "communitydist", "modzcta", "borough", and "city" are acceptable inputs
# For Data Profiles, use "DP02_0088E" for years 2020 and above. Use "DP02_0086E" for 2018 and earlier surveys. Use "DP02_0087E" for 2019. 
# For Detailed Tables, use "B01001_001E" for all years. 
# For Subject Tables, use "S0101_C01_001E" for all years.
total_pop_code = "DP02_0086E"
# For Data Profiles, use "DP02_0001E" for all years. 
# For Detailed Tables, use "B19001_001E" for all years. 
# For Subject Tables, use "S1901_C01_001E" for all years.
total_house_code = "DP02_0001E" 
    
table = cc.generate_new_estimates(survey_key=survey_key, acs_year=acs_year, demo_dict=demo_dict, geo=geo, census_api_key=census_api_key, total_pop_code=total_pop_code, total_house_code=total_house_code, boundary_year=None)
```    

**Note:** `generate_new_estimates()` only produces number estimates and MOEs. In order to create custom percent estimates, use `calc_percent_estimate()`, as described below. The meaning of percent variable codes (codes ending in 'PE') will only match between the ACS and your dataset if you ensure that the denominators you choose match the denominators used for these variables in the ACS. You may create custom percent estimates for values that do not match those found in the ACS; just be aware that the outputed percent variable codes will no longer be referring to the same data points. Since Detailed and Subject Tables do not have percent variables, the outputted percent column names will not reflect real ACS variables.
    
Other functions:

* `available_years()`: When run, this function will print the list of available years for all functions that require year variables.
* `calc_percent_estimate()`: Calculates the percent estimate (variable ending in 'PE') and percent MOE (variable ending in 'PM') that results from dividing a numerator estimate by a denominator estimate, based on the Census Bureau's formula for doing so. Can be used to generate custom percent estimates.   

Drawing on the data generated in the previous example, let's create a custom percent estimate by dividing "DP02_0025E" (never married males 15 and over) by "DP02_0024E" (males 15 and over). This will create estimates of the percent of males 15 and over that have never been married. In order for the function to work, there must be existing estimate and MOE columns for both the numerator and denominator in the DataFrame (in this case, "DP02_0025E", "DP02_0025M", "DP02_0024E", and "DP02_0024M").
    
```Python
# generating the custom percent estimate and MOE    
# the percent estimate and MOE columns will be called "DP02_0025PE" and "DP02_0025PM" in the table
cc.calc_percent_estimate(geo_df=table, geo=geo, num_code="DP02_0025E", denom_code="DP02_0024E")    
```   
    
## Data Sources 

* [The Five Year American Community Survey (ACS) Data Profiles](https://www.census.gov/data/developers/data-sets/acs-5year.html)
* [Primary Land Use Tax Lot Output (PLUTO) datasets](https://www.nyc.gov/content/planning/pages/resources?search=pluto#datasets) 

## Methodology 

Estimates for around 200 ACS demographic variables were generated for the dashboard. Estimates are available at Council District, Community District, School District, Police Precinct, Modified ZIP Code Tabulation Area (MODZCTA), Neighborhood Tabulation Area (NTA), Borough, and New York City levels. CouncilCount utilizes the 5-Year ACS, meaning the data points presented on the dashboard represent 5-year averages for the listed demographic variables. Using the multiyear estimates increases the statistical reliability of the data, especially for small population subgroups and regions with low populations. 

These estimates were generated using the Detailed Tables, Subject Tables, or Data Profiles 5-Year ACS datasets, which provide demographic estimates by census tract. Estimates for some geographies, like neighborhood tabulation areas, which are built from census tracts, may be generated by directly aggregating census-tract-level data. However, this method does not work for geographies that have no relation to census tracts, like council districts and police precincts. In order to generate estimates for such geographies, ACS demographic data was synthesized with building data from PLUTO to approximate the distribution of subpopulations around the city for each time period. Estimates for all geographies (except for council districts, for which a boundary year must be specified) are available along boundary lines as they were drawn in 2020, regardless of the period chosen, in order to make comparisons possible across time. Consequently, pre-2020 ACS NTA requests will be fulfilled using the NYCC Data Team's methodology. This is because all NTA estimates from `councilcount` will be provided along 2020 NTA boundaries (which are directly comprised of 2020 census tracts), and pre-2020 ACS data is provided along 2010 census tract boundaries, making direct aggregation challenging. The same applies to MODZCTA estimates, yet the base geography units in this case are ZIP Code Tabulation Areas.
    
New estimates will be generated according to the same methodology. 
    
For more information on the method used to generate the demographic estimates presented on CouncilCount, please contact datainfo@council.nyc.gov.

