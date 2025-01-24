import numpy as np
from get_census_function import get_census
from gen_percent_estimate_function import _gen_percent_estimate

def _generate_bbl_estimates(acs_year, demo_dict, pop_est_df, census_api_key, total_pop_code = None, total_house_code = None):

    """

    This function generates BBL-level (Borough, Block, and Lot) demographic estimates using American Community Survey (ACS) data.
    It integrates census tract-level ACS data with BBL-level PLUTO data and calculates population or household estimates for given
    demographic variables. Called in `generate_new_estimates()`.

    Parameters:
    -----------
    acs_year : int
        The 5-Year ACS end-year to fetch data for (e.g., 2022 for the 2018-2022 ACS).
    demo_dict : dict
        A dictionary where keys are ACS variable codes and values specify whether the variable is 'person' or 'household' level.
        Example: {'DP05_0001E': 'person', 'DP02_0059E': 'household'}.
    pop_est_df : pandas.DataFrame
        A DataFrame containing BBL-level data. Must include columns 'borough' and 'ct{census_year}' for census tract identifiers.
    census_api_key : str
        API key for accessing the U.S. Census Bureau's API.
    total_pop_code : str, optional
        ACS variable code for total population in given ACS year. Must include if generating any person-level estimates. Default
        is None.
    total_house_code : str, optional
        ACS variable code for total households in given ACS year. Must include if generating any household-level estimates.
        Default is None.

    Returns:
    --------
    pandas.DataFrame
        An updated DataFrame with the following:
        - Added columns for proportions (prop_<variable_code>) of each demographic variable within census tracts.
        - Estimated BBL-level counts (pop_est_<variable_code> or hh_est_<variable_code>) for each demographic.

    Notes:
    ------
    - Census tract compatibility is determined by the acs_year. Pre-2020 ACS uses 2010 tracts; 2020 and later use 2020 tracts.
    
    """

    # setting census year (the year census tracts in the dataset are associated with) based on which ACS 5-Year it is 
    
    if (acs_year < 2020) and (acs_year >= 2010): # censuses from these years use 2010 census tracts 
        census_year = 2010
    elif acs_year >= 2020: # censuses from these years use 2020 census tracts 
        census_year = 2020
        
    # adding unique identifier column: '{census_year}_tract_id' for pop_est_df
    county_fips = {'BX':'5', 'BK':'47', 'MN':'61', 'QN':'81', 'SI':'85'}    
    pop_est_df['county_fip'] = pop_est_df['borough'].map(county_fips)
    pop_est_df[f'{census_year}_tract_id'] = pop_est_df[f'ct{census_year}'].astype(str) + '-' + pop_est_df['county_fip']
                   
    # picking which denoms to include
    denom_list = [code for code in (total_pop_code, total_house_code) if code is not None]

    # list of all codes entered in the demo_dict + denominators
    var_code_list = list(demo_dict.keys()) + denom_list
    
    # making api call
    demo_df = get_census(acs_year, census_year, var_code_list, census_api_key) 
    
    # creating bbl-level estimates in pop_est_df
    
    for var_code in list(demo_dict.keys()): # for each code in the list

        if var_code not in denom_list: # exclude total population and total households because they are the denominators for the other variables

            # turning raw number to percent (total population/ households is denominator)
            demo_df = _gen_percent_estimate(demo_dict, demo_df, var_code, total_pop_code, total_house_code) 
            demo_df[var_code] = demo_df[var_code] / 100 # creating proportion

        if demo_dict[var_code] == 'household': # for variables with total households as the denominator
            est_level = 'hh_est_' # household estimate
            total_pop = 'unitsres' # denominator is total units
        elif demo_dict[var_code] == 'person': # for variables with total population as the denominator
            est_level = 'pop_est_' # total population estimate
            total_pop = 'bbl_population_estimate' # denominator is total population

        # adding proportion by census tract (for given demo variable) to pop_est_df based on tract ID

        pop_est_df = pop_est_df.merge(demo_df[[var_code, str(census_year) + '_tract_id']], on = str(census_year) + '_tract_id')

        # proportion of the BBL that this demographic holds
        pop_est_df = pop_est_df.rename(columns={var_code: 'prop_' + var_code}) 
        # total number of people in this BBL from this demographic
        pop_est_df[est_level + var_code] = pop_est_df[total_pop] * pop_est_df['prop_' + var_code] 

    return pop_est_df
    