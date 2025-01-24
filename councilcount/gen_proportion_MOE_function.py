import numpy as np

def _gen_proportion_MOE(demo_dict, variance_df, MOE_code, total_pop_code = None, total_house_code = None): 
    
    """
    Calculates the margins of error (MOE) for proportions based on Census Bureau's formula. Helper function for
    `_generate_bbl_variance()`.
    
    Parameters:
    -----------
    demo_dict : dict
        A dictionary where keys are ACS variable codes and values specify whether the variable is 'person' or 'household' level.
        Example: {'DP05_0001E': 'person', 'DP02_0059E': 'household'}.
    variance_df: dataframe
        DataFrame containing estimates and MOEs pulled from the census API.
    MOE_code: str
        Code for the demographic variable's MOE in the census API.
    total_pop_code : str, optional
        ACS variable code for total population in given ACS year. Must include if generating any person-level estimates. Default
        is None.
    total_house_code : str, optional
        ACS variable code for total households in given ACS year. Must include if generating any household-level estimates.
        Default is None.

    Returns 
    --------
    pandas.DataFrame
        Updated DataFrame with calculated proportion MOEs.
    """
    
    # gathering column names needed to access the values necesary for the MOE calculation
    
    numerator_MOE = MOE_code # numerator MOE
    numerator_est = MOE_code[:-1] + 'E' # numerator estimate
    total_pop_code_MOE = total_pop_code[:-1] + 'M' if total_pop_code else None # MOE version of denominator
    total_house_code_MOE = total_house_code[:-1] + 'M' if total_house_code else None # MOE version of denominator
    
    # determine denominator columns
    if demo_dict.get(numerator_est) == 'household':
        denom_est, denom_MOE = total_house_code, total_house_code_MOE
    elif demo_dict.get(numerator_est) == 'person':
        denom_est, denom_MOE = total_pop_code, total_pop_code_MOE

    # census formula for MOE of a proportion: 
    # sqrt(numerator's MOE squared - proportion squared * denominator's MOE squared) / denominator estimate
    
    def calculate_moe(row):
        numerator_MOE_val = row[numerator_MOE]
        numerator_est_val = row[numerator_est]
        denom_est_val = row[denom_est]
        denom_MOE_val = row[denom_MOE]

        if denom_est_val == 0:
            return np.nan  # avoid division by zero

        under_sqrt = numerator_MOE_val**2 - (numerator_est_val / denom_est_val)**2 * denom_MOE_val**2
        if under_sqrt >= 0:
            return np.sqrt(under_sqrt) / denom_est_val
        else:
            return np.sqrt(numerator_MOE_val**2 + (numerator_est_val / denom_est_val)**2 * denom_MOE_val**2) / denom_est_val

    variance_df[MOE_code] = variance_df.apply(calculate_moe, axis=1) # apply function
    
    variance_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # for any inf values created because of division by 0

    return variance_df