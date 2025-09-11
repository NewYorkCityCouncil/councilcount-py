import numpy as np
from .estimates import _reorder_columns

def calc_percent_estimate(geo_df, geo, num_code, denom_code): 
    
    """
    Calculates the percent estimate and percent margin of error (MOE) that comes from dividing a numerator estimate by a
    denominator estimate (based on Census Bureau's formula). Use to make custom percent estimates.
    
    Parameters:
    -----------
    geo_df: dataframe
        DataFrame containing estimates and MOEs.
    geo : str
        The geographic level for estimates. Options currently include 'councildist', 'communitydist', 'schooldist',
        'policeprct', 'modzcta', 'nta', 'borough', 'city'.
    num_code: str
        Code for the numerator in the census API.
    denom_code: str
        Code for the denominator in the census API.

    Returns 
    --------
    pandas.DataFrame
        Updated DataFrame with calculated percent estimate and percent MOE. Columns will end in 'PE' and 'PM', respectively.
        
    Notes
    -----
        - Variable codes ending in 'E' are number estimates. Those ending in 'M' are number MOEs. Adding
        'P' before 'E' or 'M' means the value is a percent. Codes ending in 'V' are coefficients of variation.
    
    """

    # gathering column names needed to access the values necesary for the MOE calculation
    
    numerator_est = num_code # numerator estimate
    numerator_percent_est = numerator_est[:-1] + 'PE' # numerator & estimate
    numerator_MOE = numerator_est[:-1] + 'M' # numerator MOE
    numerator_percent_moe = numerator_est[:-1] + 'PM' # numerator % MOE 
    denom_est = denom_code # denominator estimate
    denom_MOE = denom_est[:-1] + 'M' # denominator MOE

    # calculate percent estimate
    geo_df[numerator_percent_est] = (100*geo_df[numerator_est] / geo_df[denom_est]).round(2) 

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
            return (100*(np.sqrt(under_sqrt) / denom_est_val)).round(2)
        else:
            return (100*(np.sqrt(numerator_MOE_val**2 + (numerator_est_val / denom_est_val)**2 * denom_MOE_val**2) / denom_est_val)).round(2)

    geo_df[numerator_percent_moe] = geo_df.apply(calculate_moe, axis=1) # apply function
    
    geo_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # for any inf values created because of division by 0

    return _reorder_columns(geo_df, geo)

