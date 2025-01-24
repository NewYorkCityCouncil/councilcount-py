def _get_MOE_and_CV(demo_dict, variance_df, pop_est_df, census_year, geo_df, geo, total_pop_code = None, total_house_code = None): 
    
    """
    This function is called by `_estimates_by_geography()` to calculate MOE and CV values for given demographic variables at a
    specified geography level. It uses population estimates and variance data to determine statistical reliability for each
    demographic.

    Parameters:
    -----------
    demo_dict : dict
        A dictionary mapping variable codes to their corresponding type ('person' or 'household').
    variance_df : pd.DataFrame
        DataFrame containing variance information for demographic variables at the census tract level.
    pop_est_df : pd.DataFrame
        DataFrame with population estimates and columns for geographic regions and census tracts.
    census_year : int
        The census year associated with the data (e.g., 2010 or 2020).
    geo_df : pd.DataFrame
        The DataFrame for the specified geography, where calculated values will be appended.
    geo : str
        The geographic level of aggregation (e.g., council districts, neighborhoods).
    total_pop_code : str, optional
        The variable code for total population. Required if any variables are person-level.
    total_house_code : str, optional
        The variable code for total households. Required if any variables are household-level.

    Returns:
    --------
        pd.DataFrame:
            The updated `geo_df` with appended MOE and CV columns for each variable in `demo_dict`.

    """
    
    # picking which denoms to include
    denom_list = [code for code in (total_pop_code, total_house_code) if code is not None]
    
    for var_code in demo_dict.keys(): # for all of the variables in the demo_dict
        
        if var_code not in denom_list: # excluding denominators

            denom_type = demo_dict[var_code] # to access type

            if denom_type == 'household': # will pull values for household-level estimates

                est_level = 'hh_est_' 
                total_pop = 'unitsres' # denominator is total residential units

            elif denom_type == 'person': # will pull correct values for person-level estimates

                est_level = 'pop_est_' 
                total_pop = 'bbl_population_estimate' # denominator is total population

        # following Chris' protocal for converting census tract variances to geo-level variances

        # df that displays the overlap between each geographic region and each census tract 
        # for each overlap, the estimated denominator population and the estimated population of the given demographic
        census_geo_overlap = pop_est_df.groupby([geo, str(census_year) + '_tract_id']).sum()[[total_pop, est_level + var_code]]

        # adding the variance by census tract (in proportion form, with total population/ households being the denominator) to each overlapping geo-tract region
        census_geo_overlap = census_geo_overlap.reset_index()
        census_geo_overlap = census_geo_overlap.merge(variance_df[[var_code + '_variance', str(census_year) + '_tract_id']], on = str(census_year) + '_tract_id')

        # population of each overlapping geo-tract region squared multiplied by the given demographic's variances for that region
        census_geo_overlap['n_squared_x_variance'] = census_geo_overlap[total_pop]**2 * census_geo_overlap[var_code + '_variance']

        # aggregating all values by selected geo
        by_geo = census_geo_overlap.groupby(geo).sum()

        # estimated proportion of the population in each council district that belongs to a given demographic
#             by_geo['prop_' + var_code] = by_geo[est_level + var_code] / by_geo[total_pop]

        # df of variances by geo region for given demographic variable and chosen geography   
        by_geo[geo + '_variance'] = by_geo['n_squared_x_variance'] / by_geo[total_pop]**2      

        var_code_base = var_code[:9] # preparing for naming -> taking first 9 digits, then adding appropriate final letter(s) below
        column_name_MOE = var_code_base + 'M'
        column_name_percent_MOE = var_code_base + 'PM'

        by_geo[column_name_percent_MOE] = round(100*((np.sqrt(by_geo[geo + '_variance'])) * 1.645),2) # creating MOE as % (square root of variance multiplied by 1.645, then 100)
        by_geo[column_name_MOE] = round((by_geo[column_name_percent_MOE]/100) * by_geo[total_pop]) # MOE as number

        # adding MOE by geo region to geo_df
        geo_df = geo_df.assign(new_col=by_geo[column_name_MOE]).rename(columns={'new_col':column_name_MOE}) # number MOE

        # making MOE null when estimate is 0
        mask = geo_df[var_code_base + 'E'] == 0
        # apply the mask to the desired columns and set those values to NaN
        geo_df.loc[mask, [column_name_MOE]] = np.nan

        # generating coefficient of variation column in geo_df (standard deviation / mean multiplied by 100)
        geo_df[var_code_base + 'V'] = round(100*((geo_df[column_name_MOE] / 1.645) / geo_df[var_code_base + 'E']), 2)
        geo_df[var_code_base + 'V'] = geo_df[var_code_base + 'V'].replace(np.inf, np.nan) # converting infinity to NaN (inf comes from estimate aka the denominator being 0)

    return geo_df