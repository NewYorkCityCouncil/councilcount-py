import numpy as np
from get_census_function import get_census
from gen_proportion_MOE_function import _gen_proportion_MOE

def _generate_bbl_variances(acs_year, demo_dict, pop_est_df, census_api_key, total_pop_code = None, total_house_code = None):

    """
    This function retrieves ACS 5-Year data for specified demographic variables, calculates the variances at the 
    census tract level, and generates proportion MOEs for demographic estimates (with total population or households 
    as the denominator). Called in `generate_new_estimates()`.
    
    Parameters:
    -----------
    acs_year : int
        The ACS 5-Year dataset end year (e.g., 2022 for the 2018-2022 ACS 5-Year dataset).
    demo_dict : dict
        Dictionary pairing each demographic variable name with its category ('person' or 'household').
    pop_est_df : dataframe
        DataFrame containing population estimates by BBL (Building Block Lot).
    census_api_key : str
        API key for accessing the U.S. Census Bureau's API.
    total_pop_code : str, optional
        API code for total population. Required if generating person-level estimates.
    total_house_code : str, optional
        API code for total households. Required if generating household-level estimates.

    Returns:
    -----------
    DataFrame: A DataFrame containing variances for all specified variables, with columns:
        - '{variable}_variance': Variance of the demographic variable proportion.

    Notes:
        - Census Tract raw number MOEs are converted to proportions using a census formula
        - Proportion MOEs are converted to variances using the formula: variance = (MOE / 1.645)^2.
    """
    
    # setting census year (the year census tracts are associated with) 
    
    if (acs_year < 2020) and (acs_year >= 2010): # censuses from these years use 2010 census tracts 
        census_year = 2010
    elif acs_year >= 2020: # censuses from these years use 2020 census tracts   
        census_year = 2020
        
    # picking which denoms to include
    denom_list = [code for code in (total_pop_code, total_house_code) if code is not None]
    denom_moe_list = [denom_code[:-1] + 'M' for denom_code in denom_list]
        
    var_code_list = list(demo_dict.keys()) + denom_list # list of all variable codes entered in the demo_dict
    MOE_code_list = [var_code[:-1] + 'M' for var_code in var_code_list] # converting to codes that access a variable's MOE (ending in M calls variable's MOE)

        # retrieving the MOE and estimate data by census tract (need this data for calculating MOE of proportion in gen_proportion_MOE)
    variance_df = get_census(acs_year, census_year, var_code_list + MOE_code_list, census_api_key)      
    
    for MOE_code in MOE_code_list: # for each code in the list, convert to proportion
        
        if MOE_code not in denom_moe_list: # exclude total population and total households because they are the denominators for the other variables
       
            # turning raw number MOE to MOE of proportion (total population or total households = denominator)
            variance_df = _gen_proportion_MOE(demo_dict, variance_df, MOE_code, total_pop_code, total_house_code) 

        var_code = MOE_code[:-1] + 'E' # creating column name based on estimate code
        
        variance_df[var_code + '_variance'] = (variance_df[MOE_code] / 1.645) ** 2 # converting MOE to variance
        
    variance_df = variance_df.drop(columns=var_code_list + MOE_code_list) # removing unnecesary columns
        
    return variance_df
                    