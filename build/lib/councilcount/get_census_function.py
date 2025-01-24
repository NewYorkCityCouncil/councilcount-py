import requests

def get_census(acs_year, census_year, var_code_list, census_api_key):
    
    """
    Fetches American Community Survey (ACS) data from the U.S. Census Bureau API and processes it into a pandas DataFrame.

    Parameters:
    -----------
    acs_year : int 
        The year of the ACS dataset to fetch (e.g., 2019 for 2019 ACS 5-year data).
    census_year : int 
        The decennial census year to associate with the unique identifier for census tracts. Enter '2010' for ACS surveys from
        before 2020 but after 2010. Enter '2020' for surveys 2020 and above.
    var_code_list : list of str
        A list of variable codes to retrieve from the ACS dataset (e.g., ['DP03_0045E', 'DP03_0032E']).
    census_api_key : str
        A valid API key for accessing the U.S. Census Bureau's API.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the requested variables for all tracts within specified New York counties, 
        along with a unique identifier column for each tract.

    Notes:
    ------
    - The unique identifier for each tract is created by concatenating the tract number and county number.
    - If the response is invalid or there is an error in parsing the JSON, the function prints an error message 
      and the response text from the API.

    Example Usage:
    --------------
    acs_year = 2019
    census_year = 2020
    var_code_list = ['DP03_0045E', 'DP03_0032E']
    census_api_key = "your_api_key_here"
    
    df = get_census(acs_year, census_year, var_code_list, census_api_key)
    print(df.head())
    """

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
        
