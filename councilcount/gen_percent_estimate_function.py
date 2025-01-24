import numpy as np

def _gen_percent_estimate(demo_dict, demo_df, var_code, total_pop_code = None, total_house_code = None):

    """

    This function calculates percent estimates for a demographic variable by dividing its population counts by the 
    appropriate denominator (total population or total households) and multiplying by 100. Helper function for
    `_generate_bbl_estimates()`.

    Parameters:
    -----------
    demo_dict : dict 
        A dictionary where keys are ACS variable codes and values specify whether the variable is 'person' or 'household' level.
        Example: {'DP05_0001E': 'person', 'DP02_0059E': 'household'}.
    demo_df : DataFrame
        DataFrame containing population numbers by census tract for demographic groups.
    var_code : str
        Census API code for the demographic variable.
    total_pop_code : str, optional
        ACS variable code for total population in given ACS year. Must include if generating any person-level estimates. Default
        is None.
    total_house_code : str, optional
        ACS variable code for total households in given ACS year. Must include if generating any household-level estimates.
        Default is None.

    Returns:
    -----------
        DataFrame: Updated DataFrame with the demographic variable's percent estimates added.

    Notes:
        - Percent estimates are calculated as (demographic count / denominator) * 100.
        - Any infinite values resulting from division by zero are replaced with NaN.

    """
   
    denom = demo_dict.get(var_code) # accessing denom
    
    if denom == 'household': # will divide by total households 
        demo_df[var_code] = 100 * demo_df[var_code] / demo_df[total_house_code] # creating percent by tract
    elif denom == 'person': # will divide by total population
        demo_df[var_code] = 100 * demo_df[var_code] / demo_df[total_pop_code]  

    demo_df.replace([np.inf, -np.inf], np.nan, inplace=True) # for any inf values created because of division by 0
   
    return demo_df