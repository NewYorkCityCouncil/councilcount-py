import requests

def pull_census_api_codes(acs_year, census_api_key):
    
    """

    This function pulls from the American Community Survey (ACS) 5-Year Data Profiles dictionary to show all available variable
    codes for a given year. Each variable code represents a demographic estimate provided by the ACS, which can be accessed
    via an API. Visit https://api.census.gov/data/<acs_year>/acs/acs5/profile/variables.html to view the options in web format.

    Parameters:
    -----------
    acs_year : int
        The 5-Year ACS end-year to fetch data for (e.g., 2022 for the 2018-2022 ACS).
    census_api_key : str
        API key for accessing the U.S. Census Bureau's API.
        
    Returns:
    -----------
        DataFrame: A table with 'variable_code' and 'variable_description' columns. 

    Notes:
        - This function pulls directly from https://api.census.gov/data/{acs_year}/acs/acs5/profile/variables.html.
        - These variable codes may be used as inputs for councilcount functions that generate new estimates, like
        `generate_new_estimates()`.

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