import pandas as pd
import geojson
from importlib.resources import files
from get_MOE_and_CV_function import _get_MOE_and_CV

def _estimates_by_geography(acs_year, demo_dict, geo, pop_est_df, variance_df, total_pop_code=None, total_house_code=None, boundary_year=None):
    
    """
    Aggregates population and household estimates by a specified geography and attaches these values to the corresponding
    geographic DataFrame. Called in `generate_new_estimates()`.

    Parameters:
    ----------
    acs_year : int
        The 5-Year ACS end-year to fetch data for (e.g., 2022 for the 2018-2022 ACS).
    demo_dict : dict
        A dictionary where keys are variable codes, and values are either 'person' or 'household', 
        indicating the type of denominator used for estimation.
    geo : str
        The geographic level to aggregate by (e.g., "borough", "communitydist").
    pop_est_df : pandas.DataFrame
        DataFrame containing demographic estimate data at the BBL level.
    variance_df : pandas.DataFrame
        DataFrame containing variance data for the estimates.
    total_pop_code : str, optional
        API code for total population. Required if generating person-level estimates.
    total_house_code : str, optional
        API code for total households. Required if generating household-level estimates.
    boundary_year : int
        Year for the geographic boundary (relevant only for geo = "councildist"). Options: 2013, 2023.
        
    Returns:
    -------
    pandas.DataFrame
        A DataFrame with aggregated demographic estimates, attached to the specified geography.
        
    Notes: 
    ------
        - To explore available variable codes use pull_census_api_codes() or visit 
        https://api.census.gov/data/<acs_year>/acs/acs5/profile/variables.html 
    
    """
    
    # setting census year (the year census tracts are associated with) 
    if (acs_year < 2020) and (acs_year >= 2010): # censuses from these years use 2010 census tracts 
        census_year = 2010
    elif acs_year >= 2020: # censuses from these years use 2020 census tracts 
        census_year = 2020
    
    # setting boundary year (only applies to councildist)
    boundary_ext = f'_{boundary_year}' if (boundary_year) and (geo == 'councildist') else ''
    
    # setting path
    data_path = files("councilcount").joinpath("data") # setting path
    file_path = f'{data_path}/{geo}-boundaries.geojson'
    
    # load GeoJSON file for geographic boundaries
    with open(file_path) as f:
        geo_data = geojson.load(f)

    # create dataframe
    features = geo_data["features"]
    geo_df = pd.json_normalize([feature["properties"] for feature in features])
    geo_df = geo_df.set_index(geo)

    # prepare denominators
    denom_list = [code for code in (total_pop_code, total_house_code) if code]

    # process each variable in demo_dict
    for var_code, denom_type in demo_dict.items():
        if var_code not in denom_list: # excluding denominators 
            if denom_type == "household":
                est_level = "hh_est_"
                total_col = "unitsres" # denominator is residential units
            elif denom_type == "person": 
                est_level = "pop_est_"
                total_col = "bbl_population_estimate" # denominator is total population

            # aggregating the estimated population by desired geography and adding it to the geo_df
            var_code_base = var_code[:9]  # preparing for naming -> taking first 9 digits, then adding appropriate final letter(s) below
            aggregated_data = pop_est_df.groupby(geo)[est_level + var_code].sum().round()
            geo_df = geo_df.assign(**{var_code_base + "E": aggregated_data})
        
    # adding Margin of Error and Coefficient of Variation to geo_df 
    geo_df = get_MOE_and_CV(demo_dict, variance_df, pop_est_df, census_year, geo_df, geo, total_pop_code, total_house_code)  
        
    # add total population and household data if applicable
    if total_pop_code:
        total_population = pop_est_df.groupby(geo)["bbl_population_estimate"].sum().round()
        geo_df = geo_df.assign(total_population=total_population)

    if total_house_code:
        total_households = pop_est_df.groupby(geo)["unitsres"].sum().round()
        geo_df = geo_df.assign(total_residences=total_households)
        
    # return the final DataFrame
    return geo_df
