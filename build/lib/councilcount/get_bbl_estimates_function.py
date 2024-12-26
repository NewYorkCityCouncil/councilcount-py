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
    
    data_path = files("councilcount").joinpath("data")

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
