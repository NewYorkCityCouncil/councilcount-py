import os
from importlib.resources import files
import pandas as pd

def get_ACS_variables(acs_year=None):
    """
    Retrieve the available ACS demographic variables and their codes for a specified survey year.

    Parameters:
    acs_year (str): Desired 5-Year ACS year (e.g., for the 2017-2021 5-Year ACS, enter "2021").
                    If None, the most recent year available will be used.

    Returns:
    pd.DataFrame: Table of available variables with columns for variable code, variable name, 
                  denominator code, and denominator name (the "denominator variable" is the 
                  denominator population in percent estimate calculations). Use desired variable 
                  code(s) as the input for `var_codes` in the get_geo_estimates() function to obtain 
                  demographic estimates.

    Raises:
    ValueError: If the requested year is not available.
    """
    
    if acs_year: acs_year = str(acs_year) # so don't get error if accidentally input wrong dtype

    # # get the data directory where files are located
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # # construct the path to the data folder
    # data_path = os.path.join(script_dir, "data")

    data_path = files("councilcount").joinpath("data")

    # find all the available years
    csv_names = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    dictionary_csv_names = [name for name in csv_names if "data_dictionary" in name]
    dictionary_years = [name[16:20] for name in dictionary_csv_names]

    # if year is not chosen, set default to the latest year
    if acs_year is None:
        acs_year = max(dictionary_years)

    # construct the name of the dataset based on the year
    dict_name = f"data_dictionary_{acs_year}.csv"

    # error message if the requested year is unavailable
    if acs_year not in dictionary_years:
        available_years = "\n".join(dictionary_years)
        raise ValueError(
            f"This year is not available.\n"
            f"Please choose from the following:\n{available_years}"
        )

    print(f"Printing data dictionary for the {acs_year} 5-Year ACS")

    # Retrieve the data dictionary
    file_path = f'{data_path}/{dict_name}'
    df = pd.read_csv(file_path)

    return df
