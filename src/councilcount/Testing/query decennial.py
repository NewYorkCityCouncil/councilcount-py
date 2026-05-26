import pandas as pd
import requests

# This file queries, cleans, aggregates, and saves decennial census data 
# to be used as a ground-truth comparison to councilcount estimates.

######## CONSTANTS
BASE_URL = "https://api.census.gov/data"
DATA_PATH = "/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/data"
API_KEY =  "f15e9a7a298d1c9306f8a2f4a2ca99e1476247fc" # Update as necessary

demo_vars = [
    "P21_002N", "P19_002N", "H4_002N", "H4_004N", "P12_002N", "P12_026N", "P3_008N", "P3_002N", "P3_003N", "P3_004N", "P3_005N",
    "P3_006N", "P3_007N", "P4_003N", "P4_002N", "P5_003N", "P5_004N", "P5_005N", "P5_006N", "P5_007N", "P5_008N", "P5_009N",
    "P1_001N", "H9_002N", "H9_003N", "H9_004N", "H9_005N", "H9_006N", "H9_007N", "H9_008N"]

age_vars = [
    "P12_003N", "P12_027N", "P12_004N", "P12_028N", "P12_005N", "P12_029N", "P12_006N", "P12_030N", "P12_007N", "P12_031N", 
    "P12_008N", "P12_032N", "P12_009N", "P12_033N", "P12_010N", "P12_034N", "P12_011N", "P12_035N", "P12_012N", "P12_036N", 
    "P12_013N", "P12_037N", "P12_014N", "P12_038N", "P12_015N", "P12_039N", "P12_016N", "P12_040N", "P12_017N", "P12_041N",
    "P12_018N", "P12_042N", "P12_019N", "P12_043N", "P12_020N", "P12_044N", "P12_021N", "P12_045N", "P12_022N", "P12_046N", 
    "P12_023N", "P12_047N", "P12_024N", "P12_048N", "P12_025N", "P12_049N"
]

# Congressional Districts predominantly in NYC
congress = [str(num).zfill(2) for num in range(5, 16)]
# NYC Assembly Districts
assembly = [str(num).zfill(3) for num in range(23, 88)]

long_df = []

def _API_pull(url):
    """
    Performs the final steps of a Census API pull.
    Used by _person_to_household() and _pull_single_universe().

    Parameters: url (str)
        The url used to query the Census API.
    Returns: demo_df (pd.DataFrame)
        A dataframe containing the raw census data pulled.
    """
    response = requests.get(url)
    if response.status_code not in [200, 204]:
        raise RuntimeError(
            f"Census API request failed.\n"
            f"Status code: {response.status_code}\n"
            f"Response: {response.text}\n"
        )
        
    data = response.json()
    header = data[0]
        
    demo_df = pd.DataFrame(data[1:], columns=header)
    demo_df = demo_df.drop(columns=["state"])
    return demo_df
#

def _pull_dhc(geo):
    if geo == "congress":
        for_code = f'congressional district:{",".join(congress)}&in=state:36'
        name_var = "congressional district"
    elif geo == "assembly":
        for_code = f'state legislative district (lower chamber):{",".join(assembly)}&in=state:36'
        name_var = "assembly district"
    
    demo = ",".join(demo_vars)
    demo_url = f"{BASE_URL}/2020/dec/dhc?get={demo}&for={for_code}&key={API_KEY}"

    age = ",".join(age_vars)
    age_url = f"{BASE_URL}/2020/dec/dhc?get={age}&for={for_code}&key={API_KEY}"

    urls = [demo_url, age_url]

    # Pull and merge demo and age data
    for url in urls:  
        demo_df = _API_pull(url)
        demo_df = demo_df.apply(pd.to_numeric, errors="ignore")

        if geo == "assembly":
            demo_df = demo_df.rename(columns={"state legislative district (lower chamber)": name_var})
        
        # Order name_var first
        first = demo_df.pop(name_var)
        demo_df.insert(0, name_var, first)

        if url == demo_url:
            # Top-code households at 4+
            demo_df["H9_005N"] = demo_df[["H9_005N", "H9_006N", "H9_007N", "H9_008N"]].sum(axis=1)
            demo_df = demo_df.drop(columns=["H9_006N", "H9_007N", "H9_008N"])
            long_df = demo_df
        else:
            # Aggregate age variables (currently split by gender, and more finely than ACS)
            demo_df["P12_003N"] = demo_df[["P12_003N", "P12_027N"]].sum(axis=1)
            demo_df["P12_004N"] = demo_df[["P12_004N", "P12_028N"]].sum(axis=1)
            demo_df["P12_005N"] = demo_df[["P12_005N", "P12_029N"]].sum(axis=1)
            demo_df["P12_006N"] = demo_df[["P12_006N", "P12_007N", "P12_030N", "P12_031N"]].sum(axis=1)
            demo_df["P12_008N"] = demo_df[["P12_008N", "P12_009N", "P12_010N", "P12_032N", "P12_033N", "P12_034N"]].sum(axis=1)
            demo_df["P12_011N"] = demo_df[["P12_011N", "P12_012N", "P12_035N", "P12_036N"]].sum(axis=1)
            demo_df["P12_013N"] = demo_df[["P12_013N", "P12_014N", "P12_037N", "P12_038N"]].sum(axis=1)
            demo_df["P12_015N"] = demo_df[["P12_015N", "P12_016N", "P12_039N", "P12_040N"]].sum(axis=1)
            demo_df["P12_017N"] = demo_df[["P12_017N", "P12_041N"]].sum(axis=1)
            demo_df["P12_018N"] = demo_df[["P12_018N", "P12_019N", "P12_042N", "P12_043N"]].sum(axis=1)
            demo_df["P12_020N"] = demo_df[["P12_020N", "P12_021N", "P12_022N", "P12_044N", "P12_045N", "P12_046N"]].sum(axis=1)
            demo_df["P12_023N"] = demo_df[["P12_023N", "P12_024N", "P12_047N", "P12_048N"]].sum(axis=1)
            demo_df["P12_025N"] = demo_df[["P12_025N", "P12_049N"]].sum(axis=1)
            demo_df = demo_df.drop(columns=[
                "P12_007N", "P12_009N", "P12_010N", "P12_012N", "P12_014N", "P12_016N", "P12_019N", "P12_021N", "P12_022N", 
                "P12_024N", "P12_027N", "P12_028N", "P12_029N", "P12_030N", "P12_031N", "P12_032N", "P12_033N", "P12_034N", 
                "P12_035N", "P12_036N", "P12_037N", "P12_038N", "P12_039N", "P12_040N", "P12_041N", "P12_042N", "P12_043N", 
                "P12_044N", "P12_045N", "P12_046N", "P12_047N", "P12_048N", "P12_049N"])
            long_df = long_df.merge(demo_df, on=name_var, how="inner")
    
    return long_df

assembly_data = _pull_dhc("assembly")
assembly_data.to_csv("assembly-ground-truth.csv", index = False)

congress_data = _pull_dhc("congress")
congress_data.to_csv("congress-ground-truth.csv", index = False)