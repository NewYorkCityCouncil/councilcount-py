def _reorder_columns(geo_df):
    
    """
    
    Output: The inputed dataframe with correct column order (estimate variables are in alphabetical order). Used by multiple
    functions.
    
    Paramaters:
    ----------
    geo_df: DataFrame
        A dataframe that needs its ACS estimate columns to be organized
        
    Returns 
    --------
    pandas.DataFrame
        DataFrame with columns organized in alphabetical order of variable codes.
    
    """

    variable_col_string = 'DP0' # all estimate columns start with these 3 characters
    
    # separate columns with and without the estimates
    variable_cols = [col for col in geo_df.columns if variable_col_string in col]
    non_variable_cols = [col for col in geo_df.columns if variable_col_string not in col]

    # sort columns with variable estimates
    new_column_order = non_variable_cols + sorted(variable_cols) 
    
    return geo_df.reindex(columns=new_column_order)  # reindex the DataFrame