import numpy as np

def gen_percent_MOE(geo_df, MOE_num_code, MOE_denom_code): 
    
    """
    Calculates the percent margin of error (MOE) that comes from dividing a numerator MOE by a denominator MOE 
    based on Census Bureau's formula. Can be used when making custom percent estimates.
    
    Parameters:
    -----------
    geo_df: dataframe
        DataFrame containing estimates and MOEs.
    MOE_num_code: str
        Code for the numerator's MOE code in the census API.
    MOE_denom_code: str
        Code for the denominator's MOE code in the census API.

    Returns 
    --------
    pandas.DataFrame
        Updated DataFrame with calculated MOE proportions.
        
    Notes
    -----
        - Variable codes ending in 'E' are number estimates. Those ending in 'M' are number MOEs. Adding
        'P' before 'E' or 'M' means the value is a percent. Codes ending in 'V' are coefficients of variation.
    
    """

    # gathering column names needed to access the values necesary for the MOE calculation
    
    numerator_MOE = MOE_num_code # numerator MOE
    numerator_est = numerator_MOE[:-1] + 'E' # numerator estimate
    denom_MOE = MOE_denom_code # denominator MOE
    denom_est = denom_MOE[:-1] + 'E' # denominator estimate
    proportion_moe = numerator_MOE[:-1] + 'PM' # the code for the result

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

    geo_df[proportion_moe] = geo_df.apply(calculate_moe, axis=1) # apply function
    
    geo_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # for any inf values created because of division by 0

    return reorder_columns(geo_df)

