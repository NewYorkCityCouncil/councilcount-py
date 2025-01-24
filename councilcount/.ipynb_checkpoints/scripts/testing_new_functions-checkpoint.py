#!/usr/bin/env python
# coding: utf-8

# In[1]:


# installing packages

import pandas as pd
import numpy as np
import geopandas as gpd 
import pygris
from pygris.data import get_census
from shapely.ops import unary_union
from collections import OrderedDict
import requests
import geojson
import os
from importlib.resources import files
import councilcount as cc

# In[2]:


# visit https://api.census.gov/data/2021/acs/acs5.html to see other options of surveys to include

# In[3]:


# create that allows for the creation of new BBL df
# give ability to provide your own df but if one not provided, draw from existing options

# In[4]:


import os
from importlib.resources import files
import pandas as pd

def get_bbl_estimates(year=None):
    """
    Produces a dataframe containing BBL-level population estimates for a specified year. 

    Parameters:
    year (str): The desired year for BBL-level estimates. If None, the most recent year available will be used.

    Returns:
    pandas.DataFrame: A table with population estimates by BBL ('bbl_population_estimate' column). 
                      The df includes latitude, longitude, and census tract columns. Use these columns
                      to add any other geographies via .sjoin(). This will allow for the aggregation of 
                      population numbers to any desired geography. Avoid using estimates for individual
                      BBLs; the more aggregation, the less error. Population numbers were estimated by
                      multiplying the 'unitsres' and 'ct_population_density' columns. 'unitsres' specifies
                      the number of residential units present at each BBL, and 'ct_population_density' 
                      represents the division of the total population by the total number of residential
                      units in each census tract.
    """
    if year: year = str(year) # so don't get error if accidentally input wrong dtype

    # # get the data directory where the data is located
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # # construct the path to the data folder
    # data_path = os.path.join(script_dir, "data")
    
    data_path = '/Users/ravram/Desktop/councilcount-py/councilcount/data'#files("councilcount").joinpath("data")

    # find all available years
    csv_names = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    bbl_csv_names = [name for name in csv_names if "bbl-population-estimates_" in name]
    bbl_years = [name[25:29] for name in bbl_csv_names]
    
    # if year is not chosen, set default to latest year
    if year is None:
        year = max(bbl_years)
    
    # construct the name of the dataset based on the year
    bbl_name = f"bbl-population-estimates_{year}.csv"
    
    # error message if unavailable survey year selected
    if year not in bbl_years:
        available_years = "\n".join(bbl_years)
        raise ValueError(
            f"This year is not available.\n"
            f"Please choose from the following:\n{available_years}"
        )
    
    print(f"Printing BBL-level population estimates for {year}")
    
    # retrieve the dataset
    file_path = f'{data_path}/{bbl_name}'
    df = pd.read_csv(file_path)
    
    return df

# In[5]:


def pull_census_api_codes(acs_year, census_api_key):
    
    """

    This function pulls from the ACS 5-Year Data Profiles dictionary to show all available variable codes for a given ACS survey. 
    Each variable code represents a demographic estimate provided by the ACS, which can be accessed via an API.
    Visit https://api.census.gov/data/<acs_year>/acs/acs5/profile/variables.html to view the options in web format.

    Parameters:
    -----------
    acs_year : int
        The 5-Year ACS end-year to fetch data for (e.g., 2022 for the 2018-2022 ACS).
    census_api_key : str
        API key for accessing the U.S. Census Bureau's API.
        
    Returns:
    -----------
        DataFrame: A table with variable_code and variable_description columns. 

    Notes:
        - This function pulls directly from https://api.census.gov/data/{acs_year}/acs/acs5/profile/variables.html.
        - These variable codes may be used as inputs for councilcount functions that generate new estimates.

    """

    # preparing url 
    
    base_url = f'https://api.census.gov/data/{acs_year}/acs/acs5/profile/variables?key={census_api_key}'

    response = requests.get(base_url)
    response.raise_for_status()
    data = response.json()

    acs_dict = {}

    for d in data: # putting all code/ description pairs in a dict

        # removing any entries that aren't an estimate census codes (must end in 'E')
        # also removing codes for Puerto Rico
        
        if ('DP0' in d[0]) and ('PR' not in d[0]) and (d[0][-2:] != 'PE'): 

            acs_dict.update({d[0]:d[1]})

    acs_code_dict = pd.DataFrame([acs_dict]).melt(var_name="variable_code", value_name="variable_description").sort_values('variable_code')
    acs_code_dict = acs_code_dict.reset_index().drop(columns=['index']) # cleaning index
    
    return acs_code_dict

# In[6]:


# pull_census_api_codes(2016, census_api_key)

# In[7]:


# cc.get_ACS_variables(2021).tail(60).sort_values('estimate_var_code')

# In[8]:


acs_year = 2021
census_api_key = '2f42b2b59ea3d5882dd01587b2e25769f000ed56'
geo = 'policeprct'
# demo_dict = {'DP03_0064E':'household', 'DP05_0071E':'person'} 
demo_dict = {'DP03_0045E':'person', 'DP03_0032E':'person'}
total_pop_code = 'DP02_0088E'
total_house_code = 'DP02_0001E'
pop_est_df = get_bbl_estimates(acs_year)

# In[9]:


import requests

def get_census(acs_year, census_year, var_code_list, census_api_key):

    # define parameters
    base_url = "https://api.census.gov/data"
    dataset = "acs/acs5/profile"  # ACS 5-year dataset
    variables = ",".join(var_code_list)  # Concatenate variables into a comma-separated string
    tract = '*' # all tracts
    counties = "005,047,081,085,061" # New York counties
    state = "36"  # New York state

    url = f'{base_url}/{acs_year}/{dataset}?get={variables}&for=tract:{tract}&in=state:{state}&in=county:{counties}&key={census_api_key}'
    response = requests.get(url)

    # check the response
    if response.status_code == 200:
        try:
            data = response.json()  # attempt to parse JSON response
            demo_df = pd.DataFrame(data[1:], columns=data[0]) # first row is the header
            demo_df[var_code_list] = demo_df[var_code_list].astype(int) # setting dtype
            # create unique identifier for each tract (some counties have duplicate census tract names)
            demo_df[f'{census_year}_tract_id'] = demo_df['tract'].astype(int).astype(str) + '-' + demo_df['county'].astype(int).astype(str)
            # dropping unneeded columns
            demo_df = demo_df.drop(columns=['state', 'county', 'tract'])
        except Exception as e:
            print("Error parsing JSON response:", e)
            print("Response text:", response.text)
    else:
        print(f"Error: {response.status_code}")
        print("Response text:", response.text)
        
    return demo_df
        


# In[10]:


### These are all just for council, cd, schooldist, policeprct

# In[11]:


def _gen_percent_estimate(demo_dict, demo_df, var_code, total_pop_code = None, total_house_code = None):

    """

    This function calculates percent estimates for a demographic variable by dividing its population counts by the 
    appropriate denominator (total population or total households) and multiplying by 100. Helper function for _generate_bbl_estimates().

    Parameters:
    -----------
    demo_dict : dict 
        Dictionary pairing each demographic variable name with its category ('person' or 'household').
    demo_df : DataFrame
        DataFrame containing population numbers by census tract for demographic groups.
    var_code : str
        Census API code for the demographic variable.
    total_pop_code : str, optional
        ACS variable code for total population in given ACS year. Must include if generating any person-level estimates. Default is None.
    total_house_code : str, optional
        ACS variable code for total households in given ACS year. Must include if generating any household-level estimates. Default is None.

    Returns:
    -----------
        DataFrame: Updated DataFrame (`demo_df`) with the demographic variable's percent estimates added.

    Notes:
        - Percent estimates are calculated as `(demographic count / denominator) * 100`.
        - Any infinite values resulting from division by zero are replaced with NaN.

    """
    
    denom = demo_dict.get(var_code) # accessing denom
    
    if denom == 'household': # will divide by total households 
        demo_df[var_code] = 100 * demo_df[var_code] / demo_df[total_house_code] # creating percent by tract
    else: # will divide by total population
        demo_df[var_code] = 100 * demo_df[var_code] / demo_df[total_pop_code] 
    
    demo_df.replace([np.inf, -np.inf], np.nan, inplace=True) # for any inf values created because of division by 0
   
    return demo_df

############################

def _generate_bbl_estimates(acs_year, demo_dict, pop_est_df, census_api_key, total_pop_code = None, total_house_code = None):

    """

    This function generates BBL-level (Borough, Block, and Lot) demographic estimates using American Community Survey (ACS) data.
    It integrates census tract-level data with BBL-level data and calculates population or household estimates for given demographic variables.
    Called in generate_new_estimates().

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
        ACS variable code for total population in given ACS year. Must include if generating any person-level estimates. Default is None.
    total_house_code : str, optional
        ACS variable code for total households in given ACS year. Must include if generating any household-level estimates. Default is None.

    Returns:
    --------
    pandas.DataFrame
        An updated DataFrame (`pop_est_df`) with the following:
        - Added columns for proportions (`prop_<variable_code>`) of each demographic variable within census tracts.
        - Estimated BBL-level counts (`pop_est_<variable_code>` or `hh_est_<variable_code>`) for each demographic.

    Notes:
    ------
    - Census tract compatibility is determined by the `acs_year`. Pre-2020 ACS uses 2010 tracts; 2020 and later use 2020 tracts.
    
    """
    
    # error messages for incomplete inputs
    
    if ('person' in list(demo_dict.values())) and (total_pop_code == None):
        
        raise ValueError('Must include input for total_pop_code when generating person-level estimates')
        
    if ('household' in list(demo_dict.values())) and (total_house_code == None):
        
        raise ValueError('Must include input for total_house_code when generating household-level estimates')

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
            
        else:
            raise ValueError("Please input either 'person' or 'household' as the value for each variable code in demo_dict.")

        # adding proportion by census tract (for given demo variable) to pop_est_df based on tract ID

        pop_est_df = pop_est_df.merge(demo_df[[var_code, str(census_year) + '_tract_id']], on = str(census_year) + '_tract_id')

        # proportion of the BBL that this demographic holds
        pop_est_df = pop_est_df.rename(columns={var_code: 'prop_' + var_code}) 
        # total number of people in this BBL from this demographic
        pop_est_df[est_level + var_code] = pop_est_df[total_pop] * pop_est_df['prop_' + var_code] 

    return pop_est_df
    

# In[12]:


k = _generate_bbl_estimates(acs_year, demo_dict, pop_est_df, census_api_key, total_pop_code, total_house_code)

# In[ ]:


def _gen_proportion_MOE(demo_dict, variance_df, MOE_code, total_pop_code = None, total_house_code = None): 
    
    """
    Calculates the MOE for proportions based on Census Bureau's formula. Helper function for _generate_bbl_variance().

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
        ACS variable code for total population in given ACS year. Must include if generating any person-level estimates. Default is None.
    total_house_code : str, optional
        ACS variable code for total households in given ACS year. Must include if generating any household-level estimates. Default is None.

    Returns 
    --------
    pandas.DataFrame
        Updated DataFrame with calculated MOE proportions.
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
    else:
        raise ValueError("Please input either 'person' or 'household' as the value for each variable code in demo_dict.")

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

############################

def _generate_bbl_variances(acs_year, demo_dict, pop_est_df, census_api_key, total_pop_code = None, total_house_code = None):

    """
    This function retrieves ACS 5-Year data for specified demographic variables, calculates the variances at the 
    census tract level, and generates proportion MOEs for demographic estimates (with total population or households 
    as the denominator). Called in generate_new_estimates().
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
    
    # error messages for incomplete inputs
    
    if ('person' in list(demo_dict.values())) and (total_pop_code == None):
        raise ValueError('Must include input for total_pop_code when generating person-level estimates')
    if ('household' in list(demo_dict.values())) and (total_house_code == None):
        raise ValueError('Must include input for total_house_code when generating household-level estimates')

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
       
            # turning raw number to proportion (total population or total households = denominator)
            variance_df = gen_proportion_MOE_hidden(demo_dict, variance_df, MOE_code, total_pop_code, total_house_code) 

        var_code = MOE_code[:-1] + 'E' # creating column name based on estimate code
        
        variance_df[var_code + '_variance'] = (variance_df[MOE_code] / 1.645) ** 2 # converting MOE to variance
        
    variance_df = variance_df.drop(columns=var_code_list + MOE_code_list) # removing unnecesary columns
        
    return variance_df
                    

# In[ ]:


j = _generate_bbl_variances(acs_year, demo_dict, pop_est_df, census_api_key, total_pop_code, total_house_code)


# In[ ]:


def _reorder_columns(geo_df):
    
    """
    
    Output: The inputed dataframe with correct column order (estimate variables are in alphabetical order)
    
    User Inputs: 
    - geo_df: a (geo)dataframe that needs its ACS estimate columns to be organized
    
    """

    variable_col_string = 'DP0' # all estimate columns start with these 3 characters
    
    # separate columns with and without the estimates
    variable_cols = [col for col in geo_df.columns if variable_col_string in col]
    non_variable_cols = [col for col in geo_df.columns if variable_col_string not in col]

    # sort columns with variable estimates
    new_column_order = non_variable_cols + sorted(variable_cols) 
    
    return geo_df.reindex(columns=new_column_order)  # reindex the DataFrame

# In[ ]:


def _get_MOE_and_CV(demo_dict, variance_df, pop_est_df, census_year, geo_df, geo, total_pop_code = None, total_house_code = None): 
    
    '''
    Called by _estimates_by_geography(). 
    '''

    # error messages for incomplete inputs
    
    if ('person' in list(demo_dict.values())) and (total_pop_code == None):
        raise ValueError('Must include input for total_pop_code when generating person-level estimates')
    if ('household' in list(demo_dict.values())) and (total_house_code == None):
        raise ValueError('Must include input for total_house_code when generating household-level estimates')

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
                
            else:
                raise ValueError("Please input either 'person' or 'household' as the value for each variable code in demo_dict.")

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

# In[ ]:


def _estimates_by_geography(acs_year, demo_dict, geo, pop_est_df, variance_df, total_pop_code=None, total_house_code=None, boundary_year=None):
    
    """
    Aggregates population and household estimates by a specified geography and attaches 
    these values to the corresponding geographic DataFrame. Called in generate_new_estimates().

    Parameters:
    ----------
    acs_year : int
        The ACS 5-Year end year for the data.
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
    
    # validate inputs
    if 'person' in demo_dict.values() and total_pop_code is None:
        raise ValueError("Must include total_pop_code for person-level estimates.")
    if 'household' in demo_dict.values() and total_house_code is None:
        raise ValueError("Must include total_house_code for household-level estimates.")
        
    if (geo == 'councildist') and (not boundary_year):
        raise ValueError("Must provide an input for boundary_year when geo = councildist. Options include 2013 and 2023.")

    # setting census year (the year census tracts are associated with) 
    if (acs_year < 2020) and (acs_year >= 2010): # censuses from these years use 2010 census tracts 
        census_year = 2010
    elif acs_year >= 2020: # censuses from these years use 2020 census tracts 
        census_year = 2020

#     # setting path
#     data_path = files("councilcount").joinpath("data") # setting path

    # # locate available CSV files
    # file_names = os.listdir(data_path)
    
    # # record available geos
    # geo_file_names = [f for f in file_names if "boundaries" in f]
    # geo_names = list(set([f.split('-')[0] for f in geo_file_names]))
#     geo_names.remove('nyc')
#     geo_names.append('city')

#     # record available years
#     available_years = sorted(set(int(f.split('_')[-1][:4]) for f in file_names if f.split('_')[-1][:4].isdigit()))

#     # ensuring correct geo input
#     if geo not in geo_names:
#         raise ValueError(f"The geography '{geo}' could not be found. Available options are:\n" + ", ".join(geo_names))
#     # ensuring correct acs_year input
#     if acs_year not in available_years:
#         raise ValueError(f"The ACS year {acs_year} could not be found. Available options are:\n" + ", ".join(map(str, available_years)))
    
    # setting boundary year (only applies to councildist)
    boundary_ext = f'_{boundary_year}' if (boundary_year) and (geo == 'councildist') else ''
    
    # load GeoJSON file for geographic boundaries
    file_path = f"/Users/ravram/Desktop/councilcount-py/councilcount/data/{geo}{boundary_ext}-boundaries.geojson"
#     file_path = f"{data_path}/{geo}-boundaries.geojson"
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
            else:
                raise ValueError(
                    "Each variable code in demo_dict must have a value of 'person' or 'household'."
                )

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


# In[ ]:


def generate_new_estimates(acs_year, demo_dict, geo, census_api_key, total_pop_code=None, total_house_code=None, boundary_year=None):

    '''

    Notes
    -----
        - Variable codes ending in 'E' are number estimates. Those ending in 'M' are number MOEs. Adding
        'P' before 'E' or 'M' means the value is a percent. Codes ending in 'V' are coefficients of variation.  
        - To explore available variable codes use pull_census_api_codes() or visit 
        https://api.census.gov/data/<acs_year>/acs/acs5/profile/variables.html 
    '''        
    
    # validate inputs
    if 'person' in demo_dict.values() and total_pop_code is None:
        raise ValueError("Must include total_pop_code for person-level estimates.")
    if 'household' in demo_dict.values() and total_house_code is None:
        raise ValueError("Must include total_house_code for household-level estimates.")
        
    if (geo == 'councildist') and (not boundary_year):
        raise ValueError("Must provide an input for boundary_year when geo = councildist. Options include 2013 and 2023.")

        # setting path
    data_path = files("councilcount").joinpath("data") # setting path

    # locate available CSV files
    file_names = os.listdir(data_path)
    
    # record available geos
    geo_file_names = [f for f in file_names if "boundaries" in f]
    geo_names = list(set([f.split('-')[0] for f in geo_file_names]))
    geo_names.remove('nyc')
    geo_names.append('city')

    # record available years
    available_years = sorted(set(int(f.split('_')[-1][:4]) for f in file_names if f.split('_')[-1][:4].isdigit()))

    # ensuring correct geo input
    if geo not in geo_names:
        raise ValueError(f"The geography '{geo}' could not be found. Available options are:\n" + ", ".join(geo_names))
    # ensuring correct acs_year input
    if acs_year not in available_years:
        raise ValueError(f"The ACS year {acs_year} could not be found. Available options are:\n" + ", ".join(map(str, available_years)))
        
    # generating blank BBL-level population estimates df
    blank_pop_est_df = get_bbl_estimates(acs_year)
#     blank_pop_est_df = cc.get_bbl_estimates(acs_year)
    
    # adding columns for BBL-level demographic estimates
    pop_est_df = generate_bbl_estimates(acs_year, demo_dict, blank_pop_est_df, census_api_key, total_pop_code, total_house_code)

    # creating census tract-level variances in order to calculate MOE at the geo-level below
    variance_df = generate_bbl_variances(acs_year, demo_dict, pop_est_df, census_api_key, total_pop_code, total_house_code)
    
    # creating geo-level estimates, MOEs, and CVs
    raw_geo_df = estimates_by_geography(acs_year, demo_dict, geo, pop_est_df, variance_df, total_pop_code, total_house_code, boundary_year)
    
    # cleaning
    cleaned_geo_df = reorder_columns(raw_geo_df)
    
    return cleaned_geo_df 

# In[ ]:


g = generate_new_estimates(acs_year, demo_dict, geo, census_api_key, total_pop_code, total_house_code, boundary_year=None)


# In[ ]:


g

# In[ ]:


# g[['DP05_0071E','DP05_0071M']]

# In[ ]:


# cc.get_geo_estimates(acs_year, geo, list(demo_dict.keys())).set_index(geo)[['DP05_0071E','DP05_0071M']]

# In[ ]:


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



# In[ ]:


m = gen_percent_MOE(g, 'DP03_0045M', 'DP03_0032M')

m

# In[ ]:


cc.get_geo_estimates(acs_year, geo, list(demo_dict.keys())).set_index(geo)
