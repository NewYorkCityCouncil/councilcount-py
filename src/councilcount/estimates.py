import os
from importlib.resources import files
import pandas as pd
import numpy as np
import geojson
from shapely.geometry import shape
import requests
from warnings import warn

# unique url segments for each available 5-Year ACS survey
surveys = {'1' : '',
           '2' : '/profile',
           '3' : '/subject'}

######## HELPER FUNCTIONS

def _pull_raw_census_data(survey_key, acs_year, census_year, var_code_list, level, census_api_key):
    
    """
    Fetches American Community Survey (ACS) data from the U.S. Census Bureau API and processes it into a pandas DataFrame. Used
    by `_generate_bbl_estimates()`, `_generate_bbl_variances()`, and `generate_new_estimates()`. Can be aggregated to geographies
    in the census hierachy, including census tract, neighborhood tabulation area, borough, and New York City.

    Parameters:
    -----------
    survey_key : str
        Code indicating which ACS 5-Year survey the requested variables come from.
        - Key:
            '1': Detailed Tables
            '2': Data Profiles
            '3': Subject Tables
    acs_year : int/str 
        The year of the ACS dataset to fetch (e.g., 2019 for 2019 ACS 5-year data).
    census_year : int 
        The decennial census year to associate with the unique identifier for census tracts. Enter '2010' for ACS surveys from
        before 2020 but after 2010. Enter '2020' for surveys 2020 and above.
    var_code_list : list of str
        A list of variable codes to retrieve from the ACS dataset (e.g., ['DP03_0045E', 'DP03_0032E']).
    level : str
        The level of geographic aggregation desired (options include 'tract', 'nta', 'modzcta', 'borough', and 'city')
    census_api_key : str
        A valid API key for accessing the U.S. Census Bureau's API.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the requested variable estimates for the selected geography (pulled directly from the ACS itself).
    
    Notes:
    ------
    - NTA and MODZCTA estimates are aggregated from tracts and ZCTAs, respectively. Tract, borough, and city estimates are taken directly from the survey.
    - The unique identifier for each census tract is created by concatenating the tract number and county FIPS number.
    
    """

    # define parameters
    base_url = "https://api.census.gov/data"
    dataset = surveys[survey_key]  # selected ACS 5-year dataset
    variables = ",".join(var_code_list)  # concatenate variables into a comma-separated string
    state = "36" # code for New York state
    
    # setting path for pulling data from internal folder
    data_path = files("councilcount").joinpath("data") 

    # setting values based on geography level chosen
    if level == 'city': 
        for_code = f"place:51000&in=state:{state}" # NYC code
    elif level == 'borough': 
        for_code = f"county:005,047,061,081,085&in=state:{state}" # county codes for NYC boroughs
        # to match counties to boroughs
        conversion_dict = {'5': 'The Bronx', '47': 'Brooklyn', '61': 'Manhattan', '81': 'Queens', '85': 'Staten Island'} 
    elif level == 'modzcta':
        # defining NYC ZCTAs to pull data for
        nyc_zctas = pd.read_csv(f'{data_path}/nyc-zcta-list.csv')
        nyc_zctas_str = str(nyc_zctas['ZCTA5CE20'].to_list()).replace(' ', '').replace('[', '').replace(']', '')
        for_code =f"zip code tabulation area:{nyc_zctas_str}" # all NYC ZCTAs
        # converts zcta to modzcta
        file_path = f'{data_path}/modzcta-boundaries.geojson'
        with open(file_path) as f: df = geojson.load(f)
        features = df["features"]
        zcta_to_modzcta_df = pd.json_normalize([feature["properties"] for feature in features])
        modzcta_dict = dict(zip(zcta_to_modzcta_df['ZCTA'],zcta_to_modzcta_df['MODZCTA']))
        # exploding the dict
        conversion_dict = {key: v for k, v in modzcta_dict.items() for key in k.split(', ')}
    elif level in ['tract', 'nta']: 
        for_code = f'tract:*&in=county:005,047,061,081,085&in=state:{state}' # all NYC census tracts
        if level == 'nta':
            # to help build NTAs out of census tracts
            nta_conversion = pd.read_csv(f'{data_path}/2020_Census_Tracts_to_2020_NTAs_and_CDTAs_Equivalency_20240905.csv')
            conversion_dict = pd.Series(nta_conversion['NTACode'].values,index=nta_conversion['GEOID'].astype(str)).to_dict() 
            # need to pull in df with both names and codes to create column with NTA full names 
            file_path = f'{data_path}/nta-boundaries.geojson'
            with open(file_path) as f: df = geojson.load(f)
            features = df["features"]
            nta_name_df = pd.json_normalize([feature["properties"] for feature in features])

    # combine    
    url = f'{base_url}/{acs_year}/acs/acs5{dataset}/variables?get={variables}&for={for_code}&key={census_api_key}'
    # request
    response = requests.get(url)

    # check the response
    if response.status_code == 200:
        try:
            data = response.json() # attempt to parse JSON response
            demo_df = pd.DataFrame(data[1:], columns=data[0]) # first row is the header
            demo_df.replace('-555555555', np.nan, inplace=True) # sometimes this number comes in when data is missing
            demo_df[var_code_list] = demo_df[var_code_list].astype(float) # setting dtype
                
            if level == 'tract':
                # create unique identifier for each tract (some counties have duplicate census tract numbers)
                demo_df[f'{census_year}_tract_id'] = demo_df['tract'].astype(int).astype(str) + '-' + demo_df['county'].astype(int).astype(str)
                demo_df = demo_df.drop(columns=['state', 'county', 'tract'])
            elif level == 'nta':
                # pair census tract GEOIDs to corresponding NTA
                demo_df['geoid'] = demo_df['state'] + demo_df['county'] + demo_df['tract']
                demo_df[level] = demo_df['geoid'].map(conversion_dict)
                demo_df = demo_df.drop(columns=['state', 'county', 'tract', 'geoid'])
                # census formula -> to aggregate multiple MOEs, sqrt the sum of all MOEs squared
                MOE_columns = [col for col in demo_df.columns if col[-1] == 'M'] # isolating the MOE columns
                demo_df[MOE_columns] = demo_df[MOE_columns]**2 # squaring MOE columns
                # aggregating estimates and MOE from tract-level to nta-level
                demo_df = demo_df.groupby(level).sum().reset_index()
                demo_df[MOE_columns] = np.sqrt(demo_df[MOE_columns]).round().astype(int) # sqrt the sum of all MOEs squared      
                # second conversion (adding full names using codes)
                nta_names = dict(zip(nta_name_df['nta2020'],nta_name_df['ntaname']))
                demo_df['ntaname'] = demo_df[level].map(nta_names)                      
                demo_df.insert(0, level, demo_df.pop(level)) # move region column to the beginning  
                demo_df.insert(1, 'ntaname', demo_df.pop('ntaname'))
            elif level == 'modzcta':
                demo_df[level] = demo_df['zip code tabulation area'].map(conversion_dict)
                demo_df = demo_df.groupby(level).sum().reset_index()
                demo_df = demo_df.drop(columns=['zip code tabulation area'])
                demo_df.insert(0, level, demo_df.pop(level)) # move region column to the beginning 
            elif level == 'borough':
                # pair county FIPS code to borough name
                demo_df['county'] = demo_df['county'].astype(int).astype(str)
                demo_df[level] = demo_df['county'].map(conversion_dict)
                demo_df = demo_df.drop(columns=['state', 'county'])
                demo_df.insert(0, level, demo_df.pop(level)) # move region column to the beginning  
            elif level == 'city':
                # renaming columns
                demo_df['place'] = 'New York City'
                demo_df = demo_df.drop(columns=['state']).rename(columns={'place':'city'})
                demo_df.insert(0, level, demo_df.pop(level)) # move region column to the beginning 
                
        except Exception as e:
            print("Error parsing JSON response:", e)
            print("Response text:", response.text)
    else:
        print(f"Error: {response.status_code}")
        print("Response text:", response.text)
        
    return demo_df

#

def _reorder_columns(geo_df, geo):
    
    """
    
    Customizes column order. Used by multiple functions.
    
    Paramaters:
    ----------
    geo_df: DataFrame
        A dataframe that needs its ACS estimate columns to be organized
    geo : str
        A string specifying the geographic region. Options currently include 'councildist', 'communitydist', 'schooldist',
        'policeprct', 'modzcta', 'nta', 'borough', 'city'.
        
    Returns 
    --------
    pandas.DataFrame
        DataFrame with columns organized in alphabetical order of variable codes.
    
    """

    ignore_cols = [geo]
    
    # separate columns with and without the estimates
    variable_cols = [col for col in geo_df.columns if col not in ignore_cols]
    non_variable_cols = [col for col in geo_df.columns if col in ignore_cols]
    
    # sort columns with variable estimates
    new_column_order = non_variable_cols + sorted(variable_cols) 
    
    return geo_df.reindex(columns=new_column_order) # reindex the DataFrame
        
#

def _calc_proportion_estimate(demo_dict, demo_df, var_code, total_pop_code = None, total_house_code = None):

    """

    This function calculates proportion estimates for a demographic variable by dividing its population counts by the 
    appropriate denominator (total population or total households). Helper function for `_generate_bbl_estimates()`.

    Parameters:
    -----------
    demo_dict : dict 
        A dictionary where keys are ACS variable codes and values specify whether the variable is 'person' or 'household' level.
        Example for Data Profiles survey: {'DP05_0001E': 'person', 'DP02_0059E': 'household'}.
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
        - Percent estimates are calculated as (demographic count / denominator).
        - Any infinite values resulting from division by zero are replaced with NaN.

    """
    
    if var_code == total_house_code:
        denom = 'household'
    elif var_code == total_pop_code: 
        denom = 'person'
    else:
        denom = demo_dict.get(var_code) # accessing denom
    
    if denom == 'household': # will divide by total households 
        demo_df[var_code] = (demo_df[var_code] / demo_df[total_house_code]).round(3) # creating percent by tract
    elif denom == 'person': # will divide by total population
        demo_df[var_code] = (demo_df[var_code] / demo_df[total_pop_code]).round(3)  

    demo_df.replace([np.inf, -np.inf], np.nan, inplace=True) # for any inf values created because of division by 0
   
    return demo_df

#

def _generate_bbl_estimates(survey_key, acs_year, demo_dict, pop_est_df, census_api_key, total_pop_code = None, total_house_code = None):

    """

    This function generates BBL-level (Borough, Block, and Lot) demographic estimates using American Community Survey (ACS) data.
    It integrates census tract-level ACS data with BBL-level PLUTO data and calculates population or household estimates for given
    demographic variables. Called in `generate_new_estimates()`.

    Parameters:
    -----------
    survey_key : str
        Code indicating which ACS 5-Year survey the requested variables come from.
        - Key:
            '1': Detailed Tables
            '2': Data Profiles
            '3': Subject Tables
    acs_year : int/str
        The 5-Year ACS end-year to fetch data for (e.g., 2022 for the 2018-2022 ACS).
    demo_dict : dict
        A dictionary where keys are ACS variable codes and values specify whether the variable is 'person' or 'household' level.
        Example for Data Profiles survey: {'DP05_0001E': 'person', 'DP02_0059E': 'household'}.
    pop_est_df : pandas.DataFrame
        A DataFrame containing BBL-level population data. Must include columns 'borough' and 'ct{census_year}' for census tract
        identifiers.
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
    acs_year = int(acs_year) # ensuring dtype int
    if (acs_year < 2020) and (acs_year >= 2010): # censuses from these years use 2010 census tracts 
        census_year = 2010
    elif acs_year >= 2020: # censuses from these years use 2020 census tracts 
        census_year = 2020
    elif acs_year < 2010: # probably won't come up, but including this as a safeguard
        raise ValueError(f"{acs_year} is not a supported input. Please choose from years 2010 or later.")
        
    # adding unique identifier column: '{census_year}_tract_id' for pop_est_df
    county_fips = {'BX':'5', 'BK':'47', 'MN':'61', 'QN':'81', 'SI':'85'}    
    pop_est_df['county_fip'] = pop_est_df['borough'].map(county_fips)
    pop_est_df[f'{census_year}_tract_id'] = pop_est_df[f'ct{census_year}'].astype(str) + '-' + pop_est_df['county_fip']
                   
    # picking which denoms to include
    denom_list = [code for code in (total_pop_code, total_house_code) if code is not None]

    # list of all codes entered in the demo_dict + denominators
    var_code_list = list(demo_dict.keys()) + denom_list
    
    # making api call
    demo_df = _pull_raw_census_data(survey_key, acs_year, census_year, var_code_list, 'tract', census_api_key)
    
    # creating bbl-level estimates in pop_est_df
    
    for var_code in list(demo_dict.keys()) + denom_list: # for each code in the list

#         if var_code not in denom_list: # exclude total population and total households because they are the denominators for the other variables

        # turning raw number to percent (total population/ households is denominator)
        demo_df = _calc_proportion_estimate(demo_dict, demo_df, var_code, total_pop_code, total_house_code) 
        
        # accessing denom
        if var_code == total_house_code:
            denom = 'household'
        elif var_code == total_pop_code: 
            denom = 'person'
        else:
            denom = demo_dict.get(var_code) 
        
        if denom == 'household': # for variables with total households as the denominator
            est_level = 'hh_est_' # household estimate
            total_pop = 'unitsres' # denominator is total units
        elif denom == 'person': # for variables with total population as the denominator
            est_level = 'pop_est_' # total population estimate
            total_pop = 'bbl_population_estimate' # denominator is total population

        # adding proportion by census tract (for given demo variable) to pop_est_df based on tract ID

        pop_est_df = pop_est_df.merge(demo_df[[var_code, str(census_year) + '_tract_id']], on = str(census_year) + '_tract_id')

        # proportion of the BBL that this demographic holds
        pop_est_df = pop_est_df.rename(columns={var_code: 'prop_' + var_code}) 
        # total number of people in this BBL from this demographic
        pop_est_df[est_level + var_code] = pop_est_df[total_pop] * pop_est_df['prop_' + var_code] 

    return pop_est_df

#

def _calc_proportion_MOE(demo_dict, variance_df, MOE_code, total_pop_code = None, total_house_code = None): 
    
    """
    Calculates the margins of error (MOE) for proportions based on Census Bureau's formula. Helper function for
    `_generate_bbl_variance()`.
    
    Parameters:
    -----------
    demo_dict : dict
        A dictionary where keys are ACS variable codes and values specify whether the variable is 'person' or 'household' level.
        Example for Data Profiles survey: {'DP05_0001E': 'person', 'DP02_0059E': 'household'}.
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
        
    Note:
    -----
    - Find details on the formula used at:
    https://www.census.gov/content/dam/Census/library/publications/2018/acs/acs_general_handbook_2018_ch08.pdf. 
    
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
            return (np.sqrt(under_sqrt) / denom_est_val).round(3)
        else:
            return (np.sqrt(numerator_MOE_val**2 + (numerator_est_val / denom_est_val)**2 * denom_MOE_val**2) / denom_est_val).round(3)

    variance_df[MOE_code] = variance_df.apply(calculate_moe, axis=1) # apply function
    
    variance_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # for any inf values created because of division by 0

    return variance_df

#

def _generate_bbl_variances(survey_key, acs_year, demo_dict, census_api_key, total_pop_code = None, total_house_code = None):

    """
    This function retrieves ACS 5-Year data for specified demographic variables and calculates the estimates' variance in
    proportion form at the census tract level (with total population or households as the denominator). Called in
    `generate_new_estimates()`.
    
    Parameters:
    -----------
    survey_key : str
        Code indicating which ACS 5-Year survey the requested variables come from.
        - Key:
            '1': Detailed Tables
            '2': Data Profiles
            '3': Subject Tables
    acs_year : int/str
        The ACS 5-Year dataset end year (e.g., 2022 for the 2018-2022 ACS 5-Year dataset).
    demo_dict : dict
        Dictionary pairing each demographic variable name with its category ('person' or 'household').
    census_api_key : str
        API key for accessing the U.S. Census Bureau's API.
    total_pop_code : str, optional
        API code for total population. Required if generating person-level estimates.
    total_house_code : str, optional
        API code for total households. Required if generating household-level estimates.

    Returns:
    --------
    DataFrame: A DataFrame containing variances for all specified variables, with columns:
        - '{variable}_variance': Variance of the demographic variable proportion.

    Notes:
    ------
        - Census Tract raw number MOEs are converted to proportions using a census formula
        - Proportion MOEs are converted to variances using the formula: variance = (MOE / 1.645)^2.
    """
    
    # setting census year (the year census tracts are associated with) 
    acs_year = int(acs_year) # ensuring int dtype
    if (acs_year < 2020) and (acs_year >= 2010): # censuses from these years use 2010 census tracts 
        census_year = 2010
    elif acs_year >= 2020: # censuses from these years use 2020 census tracts   
        census_year = 2020
    elif acs_year < 2010: # probably won't come up, but including this as a safeguard
        raise ValueError(f"{acs_year} is not a supported input. Please choose from years 2010 or later.")
        
    # picking which denoms to include
    denom_list = [code for code in (total_pop_code, total_house_code) if code is not None]
    denom_moe_list = [denom_code[:-1] + 'M' for denom_code in denom_list]
        
    var_code_list = list(demo_dict.keys()) + denom_list # list of all variable codes entered in the demo_dict
    MOE_code_list = [var_code[:-1] + 'M' for var_code in var_code_list] # converting to codes that access a variable's MOE (ending in M calls variable's MOE)

    # retrieving the MOE and estimate data by census tract (need this data for calculating MOE of proportion in gen_proportion_MOE)
    variance_df = _pull_raw_census_data(survey_key, acs_year, census_year, var_code_list + MOE_code_list, 'tract', census_api_key)   
    
    for MOE_code in MOE_code_list: # for each code in the list, convert to proportion
        
        if MOE_code not in denom_moe_list: # exclude total population and total households because they are the denominators for the other variables
       
            # turning raw number MOE to MOE of proportion (applies formula for numerator MOE / denominator MOE -> denom either population or households)
            variance_df = _calc_proportion_MOE(demo_dict, variance_df, MOE_code, total_pop_code, total_house_code) 
            
        else: # for denominators, simply divide number MOE by number estimate to get proportion

            # turning raw number MOE to MOE of proportion (total population MOE / total population)
            variance_df[MOE_code] = (variance_df[MOE_code] / variance_df[MOE_code[:-1] + 'E']).round(3)
            variance_df[MOE_code] = variance_df[MOE_code].replace([np.inf, -np.inf], np.nan)

        var_code = MOE_code[:-1] + 'E' # creating column name based on estimate code
        
        variance_df[var_code + '_variance'] = (variance_df[MOE_code] / 1.645) ** 2 # converting MOE to variance
        
    variance_df = variance_df.drop(columns=var_code_list + MOE_code_list) # removing unnecesary columns
        
    return variance_df

#

def _calc_CV(geo_df, var_code):
    
    """
    Calculates the Coefficient of Variation (CV) for a specified variable in the given DataFrame.

    Parameters:
    -----------
        geo_df : pd.DataFrame 
            A DataFrame containing the data for geographic regions. Must include columns for estimates and margins of error.
        var_code : str 
            The variable code representing the estimate column (e.g., '<var_code_base>E'). 
            The function expects the Margin of Error column to follow the naming 
            convention '<var_code_base>M', where <var_code_base> is `var_code[:-1]`.

    Returns:
    --------
    pd.DataFrame: 
        The input DataFrame with an additional column '<var_code_base>V', which contains the calculated CV values.

    Notes:
    ------
        -  CV is calculated as: CV = (Standard Error / Mean) * 100 where the Standard Error is derived from the Margin of Error (MOE) using the formula: Standard Error = MOE / 1.645
        - Infinity values in the CV column (caused by division by zero) are replaced with NaN.
        
    """
    
    var_code_base = var_code[:-1]
    column_name_est = var_code_base + 'E'
    column_name_MOE = var_code_base + 'M'

    # generating coefficient of variation column in geo_df (standard deviation / mean multiplied by 100)
    geo_df[var_code_base + 'V'] = round(100*((geo_df[column_name_MOE] / 1.645) / geo_df[column_name_est]), 2)
    geo_df[var_code_base + 'V'] = geo_df[var_code_base + 'V'].replace(np.inf, np.nan) # converting infinity to NaN (inf comes from estimate aka the denominator being 0)
    
    return geo_df

#

def _get_MOE_and_CV(demo_dict, variance_df, pop_est_df, census_year, geo_df, geo, total_pop_code = None, total_house_code = None, boundary_year = None): 
    
    """
    This function is called by `_estimates_by_geography()` to calculate MOE and CV values for given demographic variables at a
    specified geography level. It uses population estimates and variance data to determine statistical reliability for each
    demographic. Called by `_estimates_by_geography()`.

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
    boundary_year : int
        Year for the geographic boundary (relevant for "councildist"). Options: 2013, 2023.


    Returns:
    --------
        pd.DataFrame:
            The updated `geo_df` with appended MOE and CV columns for each variable in `demo_dict`.

    """
    
    # for councildist requests
    if (boundary_year) and (geo == 'councildist'): boundary_ext = f'_{boundary_year}'
    else: boundary_ext = ''
    
    # picking which denoms to include
    denom_list = [code for code in (total_pop_code, total_house_code) if code is not None]
    
    for var_code in list(demo_dict.keys()) + denom_list: # for all of the variables in the demo_dict
#         if var_code not in denom_list: # excluding denominators

        # accessing denom
        if var_code == total_house_code:
            denom_type = 'household'
        elif var_code == total_pop_code: 
            denom_type = 'person'
        else:
            denom_type = demo_dict.get(var_code) 

        # collecting column names for calculation
        if denom_type == 'household': # will pull values for household-level estimates
            est_level = 'hh_est_' 
            total_pop = 'unitsres' # denominator is total residential units
        elif denom_type == 'person': # will pull correct values for person-level estimates
            est_level = 'pop_est_' 
            total_pop = 'bbl_population_estimate' # denominator is total population

        # following Chris' protocal for converting census tract variances to geo-level variances

        # df that displays the overlap between each geographic region and each census tract 
        # for each overlap, the estimated denominator population and the estimated population of the given demographic
        census_geo_overlap = pop_est_df.groupby([f'{geo}{boundary_ext}', str(census_year) + '_tract_id']).sum()[[total_pop, est_level + var_code]]

        # adding the variance by census tract (in proportion form, with total population/ households being the denominator) to each overlapping geo-tract region
        census_geo_overlap = census_geo_overlap.reset_index()
        census_geo_overlap = census_geo_overlap.merge(variance_df[[var_code + '_variance', str(census_year) + '_tract_id']], on = str(census_year) + '_tract_id')

        # population of each overlapping geo-tract region squared multiplied by the given demographic's variances for that region
        census_geo_overlap['n_squared_x_variance'] = census_geo_overlap[total_pop]**2 * census_geo_overlap[var_code + '_variance']

        # aggregating all values by selected geo
        by_geo = census_geo_overlap.groupby(f'{geo}{boundary_ext}').sum()

        # estimated proportion of the population in each council district that belongs to a given demographic
    #             by_geo['prop_' + var_code] = by_geo[est_level + var_code] / by_geo[total_pop]

        # df of variances by geo region for given demographic variable and chosen geography   
        by_geo[f'{geo}{boundary_ext}_variance'] = by_geo['n_squared_x_variance'] / by_geo[total_pop]**2      

        var_code_base = var_code[:-1] # preparing for naming -> taking first 9 digits, then adding appropriate final letter(s) below
        column_name_MOE = var_code_base + 'M'
        column_name_percent_MOE = var_code_base + 'PM'

        by_geo[column_name_percent_MOE] = round(100*((np.sqrt(by_geo[f'{geo}{boundary_ext}_variance'])) * 1.645),2) # creating MOE as % (square root of variance multiplied by 1.645, then 100)
        by_geo[column_name_MOE] = round((by_geo[column_name_percent_MOE]/100) * by_geo[total_pop]) # MOE as number

        # adding MOE by geo region to geo_df
        geo_df = geo_df.assign(new_col=by_geo[column_name_MOE]).rename(columns={'new_col':column_name_MOE}) # number MOE

        # making MOE null when estimate is 0
        mask = geo_df[var_code_base + 'E'] == 0
        # apply the mask to the desired columns and set those values to NaN
        geo_df.loc[mask, [column_name_MOE]] = np.nan

        # generating coefficient of variation column in geo_df (standard deviation / mean multiplied by 100)
        geo_df = _calc_CV(geo_df, var_code)

    return geo_df

#
 
def _estimates_by_geography(acs_year, demo_dict, geo, pop_est_df, variance_df, total_pop_code=None, total_house_code=None, boundary_year=None):
    
    """
    Aggregates population and household estimates by a specified geography and attaches these values to the corresponding
    geographic DataFrame. Called in `generate_new_estimates()`.

    Parameters:
    ----------
    acs_year : int/str
        The 5-Year ACS end-year to fetch data for (e.g., 2022 for the 2018-2022 ACS).
    demo_dict : dict
        A dictionary where keys are variable codes, and values are either 'person' or 'household', indicating the type of
        denominator used for estimation.
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
    
    """
    
    # setting census year (the year census tracts are associated with) 
    acs_year = int(acs_year) # ensuring int dtype
    if (acs_year < 2020) and (acs_year >= 2010): # censuses from these years use 2010 census tracts 
        census_year = 2010
    elif acs_year >= 2020: # censuses from these years use 2020 census tracts 
        census_year = 2020
    elif acs_year < 2010: # probably won't come up, but including this as a safeguard
        raise ValueError(f"{acs_year} is not a supported input. Please choose from years 2010 or later.")    
    
    # setting boundary year (only applies to councildist)
    boundary_ext = f'_{boundary_year}' if (boundary_year) and (geo == 'councildist') else ''
    
    # setting path
    data_path = files("councilcount").joinpath("data") # setting path
    file_path = f'{data_path}/{geo}{boundary_ext}-boundaries.geojson'
    
    # load GeoJSON file for geographic boundaries
    with open(file_path) as f:
        geo_data = geojson.load(f)

    # create dataframe
    features = geo_data["features"]
    geo_df = pd.json_normalize([feature["properties"] for feature in features])
    # geo_df = pd.read_csv(file_path)
    geo_df = geo_df.set_index(f'{geo}{boundary_ext}')

    # prepare denominators
    denom_list = [code for code in (total_pop_code, total_house_code) if code]

    # process each variable in demo_dict (along with denominators)
    for var_code in list(demo_dict.keys()) + denom_list:
        
        # accessing denom
        if var_code == total_house_code:
            denom_type = 'household'
        elif var_code == total_pop_code: 
            denom_type = 'person'
        else:
            denom_type = demo_dict.get(var_code) 
        
        # gathering column names for calculations
        if denom_type == "household":
            est_level = "hh_est_"
            #total_col = "unitsres" # denominator is residential units
        elif denom_type == "person": 
            est_level = "pop_est_"
            #total_col = "bbl_population_estimate" # denominator is total population

        # aggregating the estimated population by desired geography and adding it to the geo_df
        var_code_base = var_code[:-1]  # preparing for naming -> taking first 9 digits, then adding appropriate final letter(s) below
        aggregated_data = pop_est_df.groupby(f'{geo}{boundary_ext}')[est_level + var_code].sum().round()
        geo_df = geo_df.assign(**{var_code_base + "E": aggregated_data})

    # adding Margin of Error and Coefficient of Variation to geo_df 
    geo_df = _get_MOE_and_CV(demo_dict, variance_df, pop_est_df, census_year, geo_df, geo, total_pop_code, total_house_code, boundary_year)  
        
    # return the final DataFrame
    return geo_df.reset_index() 

######## VIEW AVAILABLE INPUTS  

def available_years():

    """
    Prints the available input years for all package functions that require year variables.

    Parameters:
    ----------
        None

    Returns:
    --------
        None 

    """

    # get the data directory where the data is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # construct the path to the data folder
    data_path = os.path.join(script_dir, "data")

    # find years available for new estimates (also years available for BBL-level population estimates)
    bbl_file_names = [f for f in os.listdir(data_path) if "bbl-population-estimates_" in f]
    bbl_years = sorted([name[25:29] for name in bbl_file_names])
    bbl_acs_years = [f'{int(year)-4}-{year}' for year in bbl_years]
    bbl_acs_list = ', '.join(bbl_acs_years) # for ACS 5-Year surveys

    # print results
    print(f'5-Year ACS Surveys available: {bbl_acs_list}')
    print(f"\nNote: when using `councilcount` functions, only include the end year as the input for the year variable (e.g. '2023' for the 2019-2023 survey).")

    return 

def get_census_api_codes(survey_key, acs_year, census_api_key):
    
    """

    This function pulls from an American Community Survey (ACS) 5-Year data dictionary to show all variable
    codes for a given year, including those beyond the existing CouncilCount database. Each variable code represents a demographic
    estimate provided by the ACS, which can be accessed via an API. Visit
    https://api.census.gov/data/<INSERT ACS YEAR>/acs/acs5.html and click "variables" for the desired survey to view the options in web format.

    Parameters:
    -----------
    survey_key : str
        Code indicating which ACS 5-Year survey the requested variables come from.
        - Key:
            '1': Detailed Tables
            '2': Data Profiles
            '3': Subject Tables
    acs_year : int/str
        The 5-Year ACS end-year to fetch data for (e.g., 2022 for the 2018-2022 ACS).
    census_api_key : str
        API key for accessing the U.S. Census Bureau's API.
        
    Returns:
    --------
        DataFrame: A table with 'variable_code' and 'variable_description' columns. 

    Notes:
    ------
        - This function pulls directly from https://api.census.gov/data/<INSERT ACS YEAR>/acs/acs5/<UNIQUE SURVEY URL>/variables.html.
        - These variable codes may be used as inputs for councilcount functions that generate new estimates, like
        `generate_new_estimates()`.
        - To view the variables that are currently covered by the CouncilCount database, use `get_available_councilcount_codes()`.
        If the desired variable is on this list, you may use `get_councilcount_estimates()` instead of `generate_new_estimates()`.

    """

    # preparing url 

    # define parameters
    base_url = "https://api.census.gov/data"
    dataset = surveys[survey_key] # ACS 5-year dataset
    acs_year = int(acs_year) # consistent dtype
    
    base_url = f'{base_url}/{acs_year}/acs/acs5{dataset}/variables?key={census_api_key}'

    response = requests.get(base_url)
    response.raise_for_status()
    data = response.json()

    acs_dict = {}

    for d in data: # putting all code/ description pairs in a dict

        # all codes have '_' and digits in them and end in E
        # removing any entries that aren't an estimate census codes (must end in 'E', not 'PE')
        # also removing codes for Puerto Rico

        def contains_digit_any(input_string):
            return any(char.isdigit() for char in input_string)
        
        if (('_' in d[0]) and (d[0].endswith('E')) and (contains_digit_any(d[0]))) and ('PR' not in d[0]) and (d[0][-2:] != 'PE'): 

            acs_dict.update({d[0]:d[1]})

    acs_code_df = pd.DataFrame([acs_dict]).melt(var_name="variable_code",
                                                value_name="variable_description").sort_values('variable_code')
    acs_code_df = acs_code_df.reset_index().drop(columns=['index']) # cleaning index
    
    return acs_code_df

#

def get_available_councilcount_codes(acs_year=None):
    
    """
    Retrieve the available American Community Survey (ACS) variable codes that currently exist in the CouncilCount database for a
    given survey year. Each variable code represents a demographic estimate provided by the ACS, which can be accessed via an API.
    Visit https://api.census.gov/data/<INPUT ACS YEAR>/acs/acs5.html and click "variables" for the desired survey to view the options
    in web format.

    Parameters:
    -----------
    acs_year : int/str
        Desired 5-Year ACS year (e.g., for the 2017-2021 5-Year ACS, enter "2021"). If None, the most recent year available will
        be used.

    Returns:
    --------
    pd.DataFrame: 
        Table of available variables with columns for variable code, variable name, denominator code, and denominator name.
        
    Notes:
    ------
        - The "denominator variable" is the denominator population used in percent estimate calculations. 
        - Use desired variable code(s) as the input for `var_codes` in the `get_councilcount_estimates()` function to obtain
        demographic estimates that have already been generated.
        - If the desired variable cannot be found in the DF produced by `available_councilcount_codes()`, use
        `generate_new_estimates()` instead.
        - To view ALL variable codes that can be found in the ACS, use `get_census_api_codes()`. 

    """
    
    if acs_year: acs_year = int(acs_year) # consistent dtype

#     # get the data directory where files are located
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     # construct the path to the data folder
#     data_path = os.path.join(script_dir, "data")

    data_path = files("councilcount").joinpath("data")

    # find all the available years
    csv_names = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    dictionary_csv_names = [name for name in csv_names if "data_dictionary" in name]
    dictionary_years = [int(name[16:20]) for name in dictionary_csv_names]

    # if year is not chosen, set default to the latest year
    if acs_year is None:
        acs_year = max(dictionary_years)

    # error message if the requested year is unavailable
    if acs_year not in dictionary_years:
        dictionary_years_str_list = [str(year) for year in dictionary_years]
        available_years = "\n".join(sorted(dictionary_years_str_list))
        raise ValueError(
            f"This year is not available.\n"
            f"Please choose from the following:\n{available_years}"
        )
    
    # construct the name of the dataset based on the year
    dict_name = f"data_dictionary_{acs_year}.csv"

    # retrieve the data dictionary
    file_path = f'{data_path}/{dict_name}'
    df = pd.read_csv(file_path)

    print(f"Printing data dictionary for the {acs_year} 5-Year ACS")

    return df

######## PULL/ GENERATE ESTIMATES  

def get_bbl_population_estimates(year=None):
    
    """
    Produces a DataFrame containing BBL-level population estimates for a specified year.

    Parameters:
    -----------
    year : int/str
        The desired year for BBL-level estimates. If None, the most recent year available will be used.

    Returns:
    --------
    DataFrame: 
        A table with population estimates by BBL ('bbl_population_estimate' column). 
        
    Notes:
    ------
        - The output includes latitude and longitude columns. This will allow for the aggregation of population numbers
        to various geography levels. Simply convert the table to a GeoDataframe with a geometry column, perform a spatial 
        join with a second GeoDataFrame that contains polygons for the desired geographic regions, and then aggregate population 
        numbers to that level. 
        - Avoid using estimates for individual BBLs; the more aggregation, the less error. 
        - Population numbers were estimated by multiplying the total number of residential units within a BBL by the surrounding census
        tract's housing population density (census tract total population / census tract total residential units).
        
    """

    if year: year = int(year) # consistent dtype

    # get the data directory where the data is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # construct the path to the data folder
    data_path = os.path.join(script_dir, "data")

    # find all available years
    bbl_file_names = [f for f in os.listdir(data_path) if "bbl-population-estimates_" in f]
    bbl_years = [int(name[25:29]) for name in bbl_file_names]

    # if year is not chosen, set default to latest year
    if year is None:
        year = max(bbl_years)

    # error message if unavailable survey year selected
    if year not in bbl_years:
        bbl_years_str_list = [str(year) for year in bbl_years]
        available_years = "\n".join(sorted(bbl_years_str_list))
        raise ValueError(
            f"This year is not available.\n"
            f"Please choose from the following:\n{available_years}"
        )
    
    # construct the name of the dataset based on the year
    bbl_name = f"bbl-population-estimates_{year}.csv"
    
    print(f"Printing BBL population estimates for {year}")

    # retrieve the dataset
    file_path = f'{data_path}/{bbl_name}'
    df = pd.read_csv(file_path)
    
    return df[['borough', 'block', 'lot', 'latitude', 'longitude','bbl_population_estimate']]

#

def generate_new_estimates(survey_key, acs_year, demo_dict, geo, census_api_key, total_pop_code=None, total_house_code=None, boundary_year=None):
        
    """
    Generates demographic estimates, margins of error (MOEs), and coefficients of variation (CVs) for a specified NYC geography. 
    If total_pop_code and/ or total_house_code entered, output columns for these variables will also be included.

    Parameters:
    ----------
        survey_key : str
            Code indicating which ACS 5-Year survey the requested variables come from.
            - Key:
                '1': Detailed Tables
                '2': Data Profiles
                '3': Subject Tables
        acs_year : int/str
            The 5-Year ACS end-year to fetch data for (e.g., 2022 for the 2018-2022 ACS).
        demo_dict : dict
            Dict keys should be the ACS variable codes for desired demographic groups. Dict values should
            specify whether the variable is 'person' or 'household' level. Codes must end in 'E', indicating that they are
            estimate codes. Example for Data Profiles survey: {'DP05_0001E': 'person', 'DP02_0059E': 'household'}. See Notes.
        geo : str
            The geographic level for estimates. Options currently include 'councildist', 'communitydist', 'schooldist',
            'policeprct', 'modzcta', 'nta', 'borough', 'city'.
        census_api_key : str
            User's Census API key.
        total_pop_code : str, optional
            Variable code for total population in ACS survey of interest. Required for person-level estimates. See Notes.
        total_house_code : str, optional
            Variable code for total households in ACS survey of interest. Required for household-level estimates. See Notes.
        boundary_year : int/str, optional
            Boundary year for geography, required if `geo` is 'councildist' (valid values: 2013, 2023).

    Returns:
    --------
        pd.DataFrame: A cleaned DataFrame with demographic estimates, MOEs, and CVs for the specified geography and year.

    Notes:
    ------
        - To explore available variable codes, as well as find the values needed for `total_pop_code` and/ or `total_house_code`,
        use `get_census_api_codes()` or visit https://api.census.gov/data/<INSERT ACS YEAR>/acs/acs5.html and click "variables" 
        for the desired survey to view the options in web format.
        - Variable codes ending in 'E' are number estimates. Those ending in 'M' are number MOEs. Adding 'P' before 'E' or 'M'
        means the values are percents. Codes ending in 'V' are coefficients of variation.
        -  Generates estimates using the 5-Year American Community Survey, Primary Land Use Tax Lot Output, and
        geographic boundary files.
        - Data for geographies available within existing census hierarchy are taken from the ACS. All other data are estimates
        generated by the NYC Council Data Team's methodology. Contact datainfo@council.nyc.gov with questions.
        - If the data you are looking for already exists in the CouncilCount database, please use `get_councilcount_estimates()`
        instead.
        - Geographies fitting into the census hierachy will receive estimates directly from the ACS. In all other cases, estimates generated 
        using the NYCC Data Team's methodology will be provided. 
        - As an exception, pre-2020 ACS NTA requests will be fulfilled using the NYCC Data Team's methodology. 
        This is because all NTA estimates from `councilcount` will be provided along 2020 NTA boundaries 
        (which are directly comprised of 2020 census tracts), and pre-2020 ACS data is provided along 2010 census tract 
        boundaries, making direct aggregation challenging.
        
    """    
    
    # validating inputs

    # consistent dtypes
    if acs_year: acs_year = int(acs_year) 
    if boundary_year: boundary_year = int(boundary_year)
    
    # setting path
    data_path = files("councilcount").joinpath("data") # setting path

    # locate available CSV files
    file_names = os.listdir(data_path)
    
    # record available geos
    geo_file_names = [f for f in file_names if "geographies" in f or "nyc-wide" in f]
    geo_names = list(set([f.split('-')[0] for f in geo_file_names]))
    # cleaning names to allign with input options
    to_remove = ['councildist_2023', 'councildist_2013', 'nyc']
    replacements = ['councildist', 'city']
    geo_names = [g for g in geo_names if g not in to_remove]
    geo_names = geo_names + replacements 

    # record available years
    available_years = sorted(set(int(f.split('_')[-1][:4]) for f in geo_file_names if f.split('_')[-1][:4].isdigit()))

    # ensuring correct geo input
    if geo not in geo_names:
        raise ValueError(f"The geography '{geo}' could not be found. Available options are:\n" + ", ".join(geo_names))
    # ensuring correct acs_year input
    if acs_year not in available_years:
        raise ValueError(f"The ACS year {acs_year} could not be found. Available options are:\n" + ", ".join(map(str, available_years)))
    # ensuring appropriate denominators provided
    if 'person' in demo_dict.values() and total_pop_code is None:
        raise ValueError("Must include total_pop_code for person-level estimates.")
    if 'household' in demo_dict.values() and total_house_code is None:
        raise ValueError("Must include total_house_code for household-level estimates.")
    # include boundary_year when needed    
    if geo == 'councildist':
        if not boundary_year:
            boundary_year = 2023
            warn("`boundary_year` must be set to 2013 or 2023 when `geo` is 'councildist'. Defaulting to 2023.")
        if boundary_year not in [2013, 2023]:
            raise ValueError("Input for boundary_year not recognized. Options include 2013 and 2023")        
    # remove boundary_year when not needed
    if (boundary_year != None) & (geo != 'councildist'): 
        boundary_year = None
        warn("`boundary_year` is only relevant for `geo = councildist`. Ignoring `boundary_year` input.")

    # selections for which estimates must be created using the Data Team's methodology    
    if (geo in ['councildist','schooldist','policeprct','communitydist']) or ((geo in ['nta', 'modzcta']) and (acs_year < 2021)):        
        
        # generating blank BBL-level population estimates df
        blank_pop_est_df = pd.read_csv(f'{data_path}/bbl-population-estimates_{acs_year}.csv')

        # adding columns for BBL-level demographic estimates
        pop_est_df = _generate_bbl_estimates(survey_key, acs_year, demo_dict, blank_pop_est_df, census_api_key, total_pop_code, total_house_code)

        # creating census tract-level variances in order to calculate MOE at the geo-level below
        variance_df = _generate_bbl_variances(survey_key, acs_year, demo_dict, census_api_key, total_pop_code, total_house_code)

        # creating geo-level estimates, MOEs, and CVs
        raw_geo_df = _estimates_by_geography(acs_year, demo_dict, geo, pop_est_df, variance_df, total_pop_code, total_house_code, boundary_year)
      
    # selections for which estimates can be directly taken from the ACS
    elif (geo in ['borough','city']) or ((geo in ['nta', 'modzcta']) and (acs_year >= 2021)):
        
        # setting census year (the year census tracts are associated with) 
        if (acs_year < 2020) and (acs_year >= 2010): # censuses from these years use 2010 census tracts 
            census_year = 2010
        elif acs_year >= 2020: # censuses from these years use 2020 census tracts 
            census_year = 2020
            
        # prepare denominators
        denom_list = [code for code in (total_pop_code, total_house_code) if code]
            
        var_code_list = list(demo_dict.keys()) + denom_list # list of all variable codes entered in the demo_dict + denoms
        MOE_code_list = [var_code[:-1] + 'M' for var_code in var_code_list] # converting to codes that access a variable's MOE 
        
        # pull estimates and MOEs from Census API
        raw_geo_df = _pull_raw_census_data(survey_key, acs_year, census_year, var_code_list + MOE_code_list, geo, census_api_key)
        # add CV
        for var_code in var_code_list: raw_geo_df = _calc_CV(raw_geo_df, var_code)
        
    # cleaning
    cleaned_geo_df = _reorder_columns(raw_geo_df, geo)
    
    return cleaned_geo_df

#

def get_councilcount_estimates(acs_year, geo, var_codes="all", boundary_year=None):
    
    """
    Retrieve demographic estimates by specified geography, ACS year, and boundary year (if applicable). Pulls from the existing 
    database used to support the CouncilCount website.

    Parameters:
    ----------
        acs_year : int/str
            Desired 5-Year ACS year (e.g., "2021" for the 2017-2021 5-Year ACS).
        geo : str)
            Geographic level of aggregation desired. Options include "borough", "communitydist", "councildist", "modzcta", 
            "nta", "policeprct", "schooldist", or "city".
        var_codes : list or str
            List of chosen variable codes selected from the 'estimate_var_codes' column produced by the
            `available_councilcount_codes()` function. Default is "all", which provides estimates for all 
            available variable codes.
        boundary_year : int/str, optional
            Year for the geographic boundary (relevant for "councildist"). Options: 2013, 2023.

    Returns:
    --------
        pandas.DataFrame: 
            Table with estimates for the specified geography, ACS year, and boundary_year (if applicable). 

    Notes:
    ------
        - All variables are taken from 5-Year ACS data dictionaries, which can be found here:
        https://api.census.gov/data/{INSERT YEAR}/acs/acs5.html at the "variables" hyperlinks. 
        - Codes ending with 'E' and 'M' represent numerical estimates and margins of error, respectively, while codes ending with
        'PE' and 'PM' represent percent estimates and margins of error, respectively. Codes ending with 'V' represent coefficients
        of variation. 
        - Data for geographies available within existing census hierarchy are taken from the ACS. All other data are estimates
        generated by the NYC Council Data Team. Contact datainfo@council.nyc.gov with questions.
        - To generate estimates that do not already exist, use `generate_new_estimates()`.
    """
    
    # consistent dtypes
    if acs_year: acs_year = int(acs_year) 
    if boundary_year: boundary_year = int(boundary_year)

    data_path = files("councilcount").joinpath("data")
    
    # locate available CSV files
    file_names = os.listdir(data_path)
    geo_file_names = [f for f in file_names if "geographies" in f or "nyc-wide" in f]
    geo_names = list(set([f.split('-')[0] for f in geo_file_names]))
    # cleaning names to allign with input options
    to_remove = ['councildist_2023', 'councildist_2013', 'nyc']
    replacements = ['councildist', 'city']
    geo_names = [g for g in geo_names if g not in to_remove]
    geo_names = geo_names + replacements 

    # record available years
    available_years = sorted(set(int(f.split('_')[-1][:4]) for f in geo_file_names if f.split('_')[-1][:4].isdigit()))

    def read_geos(geo, boundary_year=None):
        """
        Internal function to read and wrangle geo files.
        """
        # # boundary year information 
        # boundary_year_num = str(boundary_year)[-2:] if boundary_year else ''

        # # preparing to access files with boundary year in name (file name example: councildist-goegraphies_b23_2022.csv)
        add_boundary_year = f'_{boundary_year}' if boundary_year != None else ''

        # building paths
        if geo == 'city':
            file_path = f'{data_path}/nyc-wide_estimates_{acs_year}.csv'
            # geo_df = pd.read_csv(file_path)
        else:
            file_path = f'{data_path}/{geo}{add_boundary_year}-geographies_{acs_year}.csv'#.geojson'

        geo_df = pd.read_csv(file_path)
            
            # with open(file_path) as f:
            #     gj = geojson.load(f)

            # features = gj['features']
            # geo_df = pd.json_normalize([feature['properties'] for feature in features])

        # if list of variable codes requested, subset
        if var_codes == 'all': 
            return geo_df
        
        # if list of variable codes requested, subset
        else: 
            
            # # adding boundary year for accessing column name in geo_df (column name example: 'councildist_2023')
            # boundary_ext = f'_{boundary_year}' if boundary_year else ''
            
            # list of columns for chosen variable(s) if "all" NOT selected
            master_col_list = [f'{geo}{add_boundary_year}'] 
            
            # creating list of desired variables names (for sub-setting final table)
            for var_code in var_codes:  
                
                # check if the variable code is available in the data
                if var_code not in geo_df.columns:
                    raise ValueError(f"Estimates for the variable code {var_code} are not available. Check for any typos.\n"
                                     "View available variable codes using get_available_councilcount_codes(), or input 'all' to view all columns.")
                else:
                    var_code_base = var_code[:-1]
                    var_col_list = [
                        f"{var_code_base}E",  # numeric estimate
                        f"{var_code_base}M",  # numeric MOE
                        f"{var_code_base}PE", # percentage estimate
                        f"{var_code_base}PM", # percentage MOE
                        f"{var_code_base}V"  # coefficient of Variation
                    ]
                    
                    # updating master column list
                    master_col_list.extend(var_col_list)

            # if geo == 'city':
            #     geo_df = geo_df[master_col_list] # adding all desired columns
            # else: 
            #     geo_df = geo_df[master_col_list + ['geometry']] # adding all desired columns + geometry column 
                    
            return geo_df[master_col_list].sort_values(master_col_list) 

    # check input cases
    if acs_year is None:
        raise ValueError("`acs_year` parameter is required. Available options are:\n" +
                         ", ".join(map(str, available_years)))
    elif geo is None:
        raise ValueError("`geo` parameter is required. Available options are:\n" +
                         ", ".join(geo_names))
    elif (geo == "councildist") and ((boundary_year not in [2013, 2023]) | (boundary_year == None)):
        warn("`boundary_year` must be set to 2013 or 2023 when `geo` is 'councildist'. Defaulting to 2023.")
        boundary_year = 2023
        return read_geos(geo, boundary_year)
    elif acs_year not in available_years:
        raise ValueError(f"The ACS year {acs_year} could not be found. Available options are:\n" +
                         ", ".join(map(str, available_years)))
    elif geo not in geo_names:
        raise ValueError(f"The geography '{geo}' could not be found. Available options are:\n" +
                         ", ".join(geo_names))
    elif (geo != "councildist") and (boundary_year is not None):
        warn("`boundary_year` is only relevant for `geo = councildist`. Ignoring `boundary_year` input.")
        return read_geos(geo)
    else:
        return read_geos(geo, boundary_year)
