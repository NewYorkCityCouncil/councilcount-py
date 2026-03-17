import os
from importlib.resources import files
import pandas as pd
import numpy as np
import geojson
import requests
from warnings import warn
from functools import lru_cache

######## CONSTANTS
BASE_URL = "https://api.census.gov/data"
STATE = "36" # code for New York state
# DATA_PATH = files("councilcount").joinpath("data")
DATA_PATH = "/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/data"
API_KEY =  "f15e9a7a298d1c9306f8a2f4a2ca99e1476247fc" # Update as necessary

######## HELPER FUNCTIONS

def _pums_to_census(PUMS_year):
    """
    Assigns a census year based on the given PUMS year. 
    Used by _get_geo_info(), _person_to_household(), _pull_single_universe(), _pull_census_data(), 
    _generate_bbl_estimates(), _estimates_by_geography(), and generate_new_estimates().
    
    Parameters: PUMS_year (int/str)
        The year of the PUMS dataset to fetch (e.g., 2019 for 2019 PUMS 5-year data).
    Returns: census_year (int)
        The decennial Census year appropriate for the given PUMS year. 2010 PUMS were used up until 2023.
    """
    if PUMS_year < 2010:  # probably won't come up, but including this as a safeguard
        raise ValueError(f"{PUMS_year} is not a supported input. Please choose from years 2010 or later.")

    census_year = 2010 if PUMS_year < 2023 else 2020
    return census_year
#

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
    return demo_df
#

def _get_geo_info(acs_year, level):
    """
    Fetches ACS household and population data from the U.S. Census Bureau API for the given geographic level 
    (PUMA, borough, or city). These household and population estimates and MOEs are crucial for contextualizing 
    PUMs demographic variables and aren't available from PUMS itself.
    Used by _pull_single_universe().
    
    Parameters:
    -----------
    acs_year: int
        Specifies which year of the ACS (2011, 2016, 2021, or 2023) to fetch data for.
    level: str
        The geographic level to fetch household and population data for. Accepted values are "puma", "borough", and "city".
    puma_year: int
        Only required if level == "puma". Specifies which year of PUMAs (2010 or 2020) to fetch data for.
    
    Returns: out (pandas.DataFrame)
        A DataFrame containing the requested household and population estimates for the selected level (pulled from the ACS).
    """
    if level not in {"puma", "borough", "city"}:
        raise ValueError("level must be 'puma', 'borough', or 'city'")

    ACS_BASE = f"{BASE_URL}/{acs_year}/acs/acs5"
    PROFILE_BASE = f"{BASE_URL}/{acs_year}/acs/acs5/profile"

    if level == "puma":
        puma_year = _pums_to_census(acs_year)

        geo_col = f"puma{puma_year}"
        for_key = "public use microdata area"

        file_path = f"{DATA_PATH}/puma{puma_year}-boundaries.geojson"
        with open(file_path) as f:
            df = pd.json_normalize(geojson.load(f)["features"])

        geo_vals = df["properties.puma"].astype(str).str.zfill(5)

        for_clause = f"{for_key}:{','.join(geo_vals)}&in=state:{STATE}"

    elif level == "borough":
        geo_col = "borough"
        for_key = "county"

        name_map = {"005": "The Bronx", "047": "Brooklyn", "061": "Manhattan", "081": "Queens", "085": "Staten Island",}

        for_clause = (f"{for_key}:{','.join(name_map.keys())}&in=state:{STATE}")

    else:  # city
        geo_col = "city"
        for_key = "place"

        for_clause = f"{for_key}:51000&in=state:{STATE}"

    def _acs_pull(base, vars_):
        url = f"{base}?get={','.join(vars_)}&for={for_clause}&key={API_KEY}"
        data = requests.get(url).json()
        return pd.DataFrame(data[1:], columns=data[0])

    pop_df = _acs_pull(ACS_BASE, ["B01003_001E", "B01003_001M"])
    hh_df = _acs_pull(PROFILE_BASE, ["DP02_0001E", "DP02_0001M"])

    out = (pop_df.merge(hh_df, on=[for_key, "state"]))

    if level == "puma":
        out = out.assign(**{geo_col: out[for_key].astype(str).str.zfill(5)})
    elif level == "borough":
        out = out.assign(**{geo_col: out[for_key].map(name_map)})
    else:
        out = out.assign(**{geo_col: "New York City"})

    out = (
        out
        .rename(columns={"B01003_001E": "total_pop_E", "B01003_001M": "total_pop_M", 
                         "DP02_0001E": "total_households_E", "DP02_0001M": "total_households_M"})
        .replace("-555555555", "0") # treats missing MOEs as 0, as the councilcount dashboard does.
        .drop(columns=["state", for_key]))
    
    out = out[[geo_col] + [c for c in out.columns if c != geo_col]]
    total_cols = [c for c in out.columns if c.startswith("total_")]
    out[total_cols] = out[total_cols].astype("Int64")

    # add borough map and city label to PUMA data (important for next function)
    if level == "puma":
        if puma_year == 2020:
            borough_map = {'043': 'Brooklyn', '042': 'The Bronx', '041': 'Manhattan', '044': 'Queens', '045': 'Staten Island'}
        elif puma_year == 2010:
            borough_map = {'040': 'Brooklyn', '037': 'The Bronx', '038': 'Manhattan', '041': 'Queens', '039': 'Staten Island'}
        out["borough"] = out[f'puma{puma_year}'].str[:3].map(borough_map)
        out["city"] = "New York City"

    return out
#

@lru_cache(maxsize=None)
def _get_geo_info_cached(acs_year, level):
    return _get_geo_info(acs_year, level)

# Get lists of all NYC PUMAs to use as constants later
puma_df_20 = _get_geo_info_cached(2023, "puma")
PUMAS_20 = puma_df_20["puma2020"]
puma_df_10 = _get_geo_info_cached(2011, "puma")
PUMAS_10 = puma_df_10["puma2010"]

def _harmonize_variables(df, PUMS_year):
    """
    Harmonizes each variable in df that corresponds to a PUMS variable different from its ACS counterpart.
    Returns a dataframe with harmonized values for these variables. Used by _pull_single_universe().
    
    Parameters:
    -----------
    df: pandas.DataFrame
        A DataFrame containing variables that need to be harmonized.
    PUMS_year: int/str
        The year of the 5-year PUMS survey for which data is being harmonized.
    
    Returns:
    --------
    df: pandas.DataFrame
        A DataFrame containing harmonized variables.
    new_variables: list of str
        Names of new variables created during harmonization
    """
    new_variables = []

    disharmony = [
        "MIG", "LANX", "HFL", "R18", "R65", "FS", "TEL", "DIS", "HICOV", "PRIVCOV", "PUBCOV", "R65", "CIT", "DECADE", "ENG", "NP", 
        "VEH", "JWTRNS", "JWTR", "JWRIP", "TEN", "ESR", "MIL", "RETP", "SSP", "SSIP", "PAP", "POVPIP", "SCHG", "SCHL", "POBP", 
        "WAOB", "LANP", "OCPIP", "GRPIP", "HISP", "HINCP", "NAICSP", "NAICSP07", "AGEP", "MAR", "ADJINC", "TYPEHUGQ", "TYPE"
        ]
    to_harmonize = [col for col in df.columns if col in disharmony]
    
    numeric_cache = {}
    for var in to_harmonize:
        numeric_cache[var] = pd.to_numeric(df[var], errors="coerce")

    # Get ahead of some special cases before entering harmonization loop
    
    # Civilian Noninstitutionalized Population
    civ_ninst = None
    type_col = "TYPEHUGQ" if PUMS_year > 2020 else "TYPE"

    if type_col in df.columns and "ESR" in df.columns:
        civ_ninst = (df[type_col].astype(str) != "2") & (numeric_cache["ESR"] != 4)
    
    computer_sources = {"TABLET", "COMPOTHX", "SMARTPHONE"} & set(df.columns)
    if computer_sources:
        has_computer = df[list(computer_sources)].eq("1").any(axis=1)
        df["COMPUTER"] = np.where(has_computer, "1", "2")

        df.loc[df["TABLET"].eq("0"), "TABLET"] = pd.NA # Clean up these columns, they're not useful
        df.loc[df["COMPOTHX"].eq("0"), "COMPOTHX"] = pd.NA
        df.loc[df["SMARTPHONE"].eq("0"), "SMARTPHONE"] = pd.NA

        new_variables += ["COMPUTER"]

    bband_sources = {"HISPEED", "BROADBND"} & set(df.columns)
    if bband_sources:
        has_bband = df[list(bband_sources)].eq("1").any(axis=1)
        df["BBAND"] = np.where(has_bband, "1", "2")

        df.loc[df["HISPEED"].eq("0"), "HISPEED"] = pd.NA
        df.loc[df["BROADBND"].eq("0"), "BROADBND"] = pd.NA

        new_variables += ["BBAND"]

    if "RAC1P" in df.columns:
        r = pd.to_numeric(df["RAC1P"], errors="coerce")
        df.loc[r.between(4, 5), "RAC1P"] = "3" # American Indian or Alaska Native alone
    

    for var in to_harmonize:
        s = numeric_cache[var]

        if var in ["MIG", "LANX", "HFL"]:
            df.loc[s == 0, var] = pd.NA # Outside the universe we'd like to estimate.
        
        elif var in ["R18", "FS"]:
            df.loc[s != 1, var] = pd.NA # We only estimate these variables at 1

        elif var == "TEL":
            df.loc[s != 2, var] = pd.NA # Only estimate at 2

        elif var in ["DIS", "HICOV", "PRIVCOV", "PUBCOV"] and civ_ninst is not None:
            # Count observations of these variables in people outside of civ_ninst as NA to align with ACS.
            df.loc[~civ_ninst, var] = pd.NA
            if var != "HICOV":
                df.loc[s == 2, var] = pd.NA # HICOV is the only one of these where we estimate "no"s

        elif var == "R65":
            df.loc[s == 2, var] = "1" # No need to distinguish between 1 or multiple 65yos in household
            df.loc[s < 1, var] = pd.NA

        elif var == "CIT":
            df.loc[s == 3, var] = "2" # Born in Puerto Rico, U.S. Island areas, or abroad to American parent(s)

        elif var == "DECADE":
            df.loc[s.between(1, 4), var] = "5" # Before 1990
            df.loc[s == 0, var] = pd.NA # Born in the US: N/A.

        elif var == "ENG":
            df.loc[s.between(3, 4), var] = "2" # Speaks English less than "very well"
            df.loc[s == 0, var] = pd.NA # <5/speaks only English: N/A

        elif var in ["NP", "VEH"]:
            df.loc[s >= 4, var] = "4+" # Top-codes 4+ person households and ownership of 4+ vehicles
            df.loc[s == -1, var] = pd.NA

        elif var in ["JWTRNS", "JWTR"]: # JWTR appears in place of JWTRNS for 2016 and 2011 data
            df.loc[s.between(3, 6), var] = "2" # Public transit (bus, subway, commuter rail, light rail, ferryboat)
            df.loc[s == 0, var] = pd.NA
        
        elif var == "JWRIP":
            df.loc[s.between(3, 10), var] = "2" # 2+ person carpool
            df.loc[s == 0, var] = pd.NA
        # For commute to work, JWRIP1 = Drove, JWRIP2 = Carpool, JWTR(NS)2 = Transit, JWTR(NS)10 = Walked, JWTR(NS)11 = WFH

        elif var == "TEN":
            df.loc[s == 0, var] = pd.NA
            
            # Harmonize homeowner count to only include those with the financial info required for ACS
            if all(col in df.columns for col in ["OCPIP", "HINCP"]):
                mask_owner = ((s == 1) & (df["OCPIP"].isna() | (numeric_cache["HINCP"] <= 0)))
                df.loc[mask_owner, "TEN"] = pd.NA
            # Harmonize renter count symmetrically
            if all(col in df.columns for col in ["GRPIP", "HINCP"]):
                mask_renter = ((s == 3) & (df["GRPIP"].isna() | (numeric_cache["HINCP"] <= 0)))
                df.loc[mask_renter, "TEN"] = pd.NA

        elif var == "ESR":
            df.loc[s == 2, var] = "1" # Civilian employed
            df.loc[s == 5, var] = "4" # In the military
            df.loc[s.isin([0, 6]), var] = pd.NA # <16 (NA) or NILF (made redundant by LBR_FRC2 below)
            
            # "In the labor force" can be calculated as ESR other than "0" or "6".
            lbr = pd.Series(pd.NA, index=df.index)
            lbr[s.between(1, 5)] = "1"
            lbr[s == 6] = "2"
            df["LBR_FRC"] = lbr
            new_variables += ["LBR_FRC"]
            # Unemployment rate can later be calculated as ESR3 / LBR_FRC1.

        elif var == "MIL":
            df.loc[s == 0, var] = pd.NA # <17 years old
            if PUMS_year > 2011:
                df.loc[s == 3, var] = "1" # Counting active military training as "in the armed forces"
            # 2 = "on active duty in the past, but not now" -> "veteran"
            else:
                df.loc[s == 4, var] = "1" # Counting active military training "in the armed forces"
                df.loc[s == 3, var] = "2"
            # 2 = "on active duty in the past, but not now/not during the last 12 months" -> "veteran"

        elif var in ["RETP", "SSP", "SSIP", "PAP"]:
            # Note: PUMs reports these supplementary income types on the person (not the household) level. We fix this later.
            df.loc[s > 0, var] = "1" # Person with supplementary income

        elif var == "POVPIP":
            df.loc[s == -1, var] = pd.NA
            df.loc[s.between(0, 99), var] = "Below 100 percent"
            df.loc[s.between(100, 149), var] = "100 to 149 percent"
            df.loc[s >= 150, var] = "At or above 150 percent"

        elif var == "SCHL":
            if PUMS_year > 2011:
                bins = [
                    (1, 11, "Less than 9th grade"),
                    (12, 15, "9th to 12th grade, no diploma"),
                    (16, 17, "High school graduate"),
                    (18, 19, "Some college, no degree"),
                    (20, 20, "Associate's degree"),
                    (21, 21, "Bachelor's degree"),
                    (22, 24, "Graduate or professional degree"),
                ]
            else:
                bins = [
                    (1, 4, "Less than 9th grade"),
                    (5, 8, "9th to 12th grade, no diploma"),
                    (9, 9, "High school graduate"),
                    (10, 11, "Some college, no degree"),
                    (12, 12, "Associate's degree"),
                    (13, 13, "Bachelor's degree"),
                    (14, 16, "Graduate or professional degree"),
                ]
            for lo, hi, label in bins:
                df.loc[s.between(lo, hi), var] = label
            
            # Restrict sample to 25+ to match ACS
            if "AGEP" in df.columns:
                df.loc[numeric_cache["AGEP"] < 25, "SCHL"] = pd.NA

        elif var == "SCHG":
            df.loc[s == 0, var] = pd.NA
            if PUMS_year > 2011:
                bins = [
                    (3, 10, "Grades 1-8"),
                    (11, 14, "Grades 9-12"),
                    (15, 16, "College or graduate school")
                ]
            else:
                bins = [
                    (3, 4, "Grades 1-8"),
                    (5, 5, "Grades 9-12"),
                    (6, 7, "College or graduate school")
                ]
            for lo, hi, label in bins:
                df.loc[s.between(lo, hi), var] = label
            
        elif var == "POBP":
        # Use POBP for US states and Oceania, as the WAOB categories for these regions are too broad
            bins = [
                (36, 36, "New York"),
                (1, 35, "Diff State"), # Born in a different state or DC
                (37, 56, "Diff State"),
                (501, 527, "OCEANIA") # Foreign Born, Oceania
            ]
            # Default to NA, check for NATIVITY before applying additional harmonization
            df[var] = pd.NA
            has_nativity = "NATIVITY" in df.columns

            for lo, hi, label in bins:
                mask = s.between(lo, hi)
                if lo >= 100 and has_nativity:
                    mask = mask & (df["NATIVITY"].astype(str) == "2")
                df.loc[mask, var] = label

        elif var == "WAOB":
        # Use WAOB for non-Oceania foreign born, as the POBP categories for these regions are too restrictive
            bins = [
                (5, "EUROPE"), # Foreign Born, Europe
                (4, "ASIA"), # Foreign Born, Asia
                (7, "NORTHERN AMERICA"), # Foreign Born, Northern America (Bermuda & Canada, per ACS definition)
                (3, "LATIN AMERICA"), # Foreign Born, Latin America (includes "Americas, Not Specified")
                (6, "AFRICA"), # Foreign Born, Africa
            ]
            
            # Default to NA, check for NATIVITY before applying additional harmonization
            df[var] = pd.NA
            has_nativity = "NATIVITY" in df.columns

            for val, label in bins:
                mask = s == val
                if has_nativity:
                    mask = mask & (df["NATIVITY"].astype(str) == "2")
                df.loc[mask, var] = label
        
        elif var == "LANP":

            if PUMS_year > 2011:
                
                # All the languages ACS considers "other and unspecified"
                other_unspecified = (s.between(1000, 1052) | s.between(1057, 1063) | s.between(1074, 1109) |
                    s.between(1565, 1642) | s.between(3799, 9499) | s.between(9600, 9999))

                conditions = [
                    s == 1200,
                    s.isin([1170, 1055, 1175]),
                    s.between(1110, 1134),
                    s.between(1250, 1278),
                    s == 2575,
                    s.isin([1970, 2000, 2050]),
                    s == 1960,
                    s.between(2910, 2920),
                    s == 4500,
                    s.between(1069, 1564),
                    s.between(1675, 3600),
                    other_unspecified
                ]

                choices = [
                    "Spanish",
                    "French/Haitian/Cajun", 
                    "German/West Germanic",
                    "Russian/Polish/Slavic",
                    "Korean",
                    "Chinese, incl. Mandarin, Cantonese", # excl. Min Nan Chinese (Hokkien)
                    "Vietnamese",
                    "Tagalog, incl. Filipino", # excl. Cebuano and "Other Philippine languages".
                    "Arabic",
                    "Other Indo-European",
                    "Other Asian and Pacific Island",
                    "Other and Unspecified Languages" # Only refers to the census category, not "all languages not listed above"
                ]

                df[var] = np.select(conditions, choices, default=pd.NA)

            else:

                conditions = [
                    s == 625,
                    s.isin([620, 623, 624]),
                    s.between(607, 611),
                    s.between(639, 651),
                    s == 724,
                    s.isin([708, 712, 711]),
                    s == 728,
                    s == 742,
                    s == 777,
                    s == 600,
                    (s.between(601, 678) | (s == 985)),
                    (s.between(684, 695) | (s.between (698, 771)) | s.isin([986, 988])),
                    (s.between(679, 683) | (s.between(696, 697)) | s.between(777, 999))
                ]

                choices = [
                    "Spanish",
                    "French/Haitian/Cajun", # Uses "French Creole" instead of Haitian for 2011 (no code)
                    "German/West Germanic",
                    "Russian/Polish/Slavic",
                    "Korean",
                    "Chinese, incl. Mandarin, Cantonese", 
                    "Vietnamese",
                    "Tagalog, incl. Filipino", # Only Tagalog has its own code, but kept label consistent with other years. Sebuano/Ilocano/Bisayan do appear.
                    "Arabic",
                    "N/A (less than 5 years old/speaks only English)",
                    "Other Indo-European",
                    "Other Asian and Pacific Island",
                    "Other and Unspecified Languages" 
                    # Includes all languages not yet categorized, not just those typically in the census category with this name.
                ]

                df[var] = np.select(conditions, choices, default=pd.NA)


        elif var in ["OCPIP", "GRPIP"]: 
            if var == "OCPIP" and "TEN" in df.columns:
                s.loc[numeric_cache["TEN"] != 1] = 0
            if var == "GRPIP" and "TEN" in df.columns:
                s.loc[numeric_cache["TEN"] != 3] = 0
            
            out = pd.Series(pd.NA, index=df.index)

            # PUMS puts these in integer percentages 
            # Handle lowest levels differently, others the same. 
            if var == "GRPIP": 
                out.loc[s.between(1, 14)] = "<15.0%" 
                out.loc[s.between(15, 19)] = "15.0-19.9%" 
            elif var == "OCPIP": 
                out.loc[s.between(1, 19)] = "<20.0%" 
            
            out.loc[s.between(20, 24)] = "20.0-24.9%" 
            out.loc[s.between(25, 29)] = "25.0-29.9%" 
            out.loc[s.between(30, 34)] = "30.0-34.9%" 
            out.loc[s >= 35] = "35% or more"

            df[var] = out

        elif var == "HISP":
            df[var] = np.select([s == 1, s.between(2, 24)], ["2", "1"], default=pd.NA)

            if "RAC1P" in df.columns:
                not_hispanic = df[var] == "2"

                race_map = {
                    1: "White alone, not Hispanic or Latino",
                    2: "Black or African American alone, not Hispanic or Latino",
                    3: "American Indian or Alaska Native alone, not Hispanic or Latino",
                    6: "Asian alone, not Hispanic or Latino",
                    7: "Native Hawaiian and Other Pacific Islander alone, not Hispanic or Latino",
                    8: "Some Other Race alone, not Hispanic or Latino",
                    9: "Two or More Races, not Hispanic or Latino"
                }

                race_hisp = pd.Series(pd.NA, index=df.index)

                for code, label in race_map.items():
                    race_hisp[not_hispanic & (r == code)] = label

                df["RACE_HISP"] = race_hisp
                new_variables += ["RACE_HISP"]

        elif var == "HINCP":
            # Adjust for inflation
            if "ADJINC" in df.columns:
                s = df[var].astype(float) * numeric_cache["ADJINC"].astype(float)
            else:
                s = df[var].astype(float)

            conditions = [
                s < 10000,
                s.between(10000, 14999),
                s.between(15000, 24999),
                s.between(25000, 34999),
                s.between(35000, 49999),
                s.between(50000, 74999),
                s.between(75000, 99999),
                s.between(100000, 149999),
                s.between(150000, 199999),
                s >= 200000
            ]

            choices = [
                "< $10,000",
                "$10,000-$14,999",
                "$15,000-$24,999",
                "$25,000-$34,999",
                "$35,000-$49,999",
                "$50,000-$74,999",
                "$75,000-$99,999",
                "$100,000-$149,999",
                "$150,000-$199,999",
                ">= $200,000"
            ]

            df[var] = np.select(conditions, choices, default=pd.NA)

        elif var in ["NAICSP", "NAICSP07"]: # Use NAICSP07 for 2011 data. Fortunately, prefixes are consistent across variables.
            col = df[var].astype(str)

            # Universe restriction
            if all(c in df.columns for c in ["AGEP", "ESR"]):
                if PUMS_year > 2011:
                    employed = numeric_cache["ESR"].between(1,2) # NAICSP counts civilian employed pop
                    not_in_universe = (numeric_cache["AGEP"] < 16) | (~employed)
                else:
                    not_in_universe = numeric_cache["AGEP"] < 16 # NAICSP07 does not restrict wrt ESR.
                df.loc[not_in_universe, var] = pd.NA
                valid = ~not_in_universe

            mapping = [
                (("11", "21"), "Agriculture, forestry, fishing and hunting, and mining"),
                (("23",), "Construction"),
                (("3",), "Manufacturing"),
                (("42",), "Wholesale trade"),
                (("44", "45", "4M"), "Retail trade"),
                (("22", "48", "49"), "Transportation and warehousing, and utilities"),
                (("51",), "Information"),
                (("52", "53"), "Finance and insurance, and real estate and rental and leasing"),
                (("54", "55", "56"), "Professional, scientific, and management, and administrative and waste management services"),
                (("61", "62"), "Educational services, and health care and social assistance"),
                (("71", "72"), "Arts, entertainment, and recreation, and accommodation and food services"),
                (("81",), "Other services, except public administration"),
                (("921", "92M", "923", "9281P", "928P"), "Public Administration"),
                (("92811", "N.A.////", "-1", "9920"), pd.NA),
            ]

            for prefixes, label in mapping:
                mask = col.str.startswith(prefixes)
                if all(col in df.columns for col in ["AGEP", "ESR"]):
                    mask = mask & valid
                df.loc[mask, var] = label
        # Doesn't include unemployed or n/a

        elif var == "AGEP":
            # Create broader age categories in their own variables
            df["AGE_CAT"] = np.select([s < 18, s.between(18, 64), s >= 65], ["Under 18", "18 to 64", "65 and Over"], default=pd.NA)
            new_variables += ["AGE_CAT"]

            df["AGE_U18"] = np.where(s < 18, pd.NA, "18 and Over") # Already have an under-18 indicator so set this to NA
            new_variables += ["AGE_U18"]
            
            # Use AGEP to properly harmonize for "never married" (PUMS also puts those under 15 in this category)
            if "MAR" in df.columns:
                age = numeric_cache["AGEP"]
                df.loc[(numeric_cache["MAR"] == 5) & (age < 15), "MAR"] = pd.NA

            # Use broader categories for disability crosstab
            if "DIS" in df.columns:
                w_dis = df["DIS"].eq("1")

                dis_map = {
                    "Under 18": "With a disability, under 18 years",
                    "18 to 64": "With a disability, 18-64 years",
                    "65 and Over": "With a disability, 65 years and over"
                }
                dis_age = pd.Series(pd.NA, index=df.index)

                for code, label in dis_map.items():
                    dis_age[w_dis & (df["AGE_CAT"] == code)] = label

                df["DIS_AGE"] = dis_age
                new_variables += ["DIS_AGE"]

            # Create more specific age categories
            bins = [
                (0, 4, "Under 5 years"),
                (5, 9, "5-9 years"),
                (10, 14, "10-14 years"),
                (15, 19, "15-19 years"),
                (20, 24, "20-24 years"),
                (25, 34, "25-34 years"),
                (35, 44, "35 to 44 years"),
                (45, 54, "45 to 54 years"),
                (55, 59, "55 to 59 years"),
                (60, 64, "60 to 64 years"),
                (65, 74, "65 to 74 years"),
                (75, 84, "75 to 84 years"),
                (85, 200, "85 years and over"),
            ]

            for lo, hi, label in bins:
                df.loc[s.between(lo, hi), var] = label
        
        elif var in ["ADJINC", "TYPEHUGQ", "TYPE"]:
            # These were just for harmonization, no need to aggregate estimates.
            df[var] = pd.NA

    return df, new_variables
#

def _person_to_household(person_df, PUMS_year, hh_vars, census_api_key):
    """
    Convert select person-level PUMS variables to household-level equivalents
    using household weights (WGTP) to match their ACS counterparts. Used by _pull_single_universe().

    Parameters
    ----------
    person_df : pandas.DataFrame
        Harmonized person-level dataframe.
    PUMS_year : int
        PUMS vintage.
    hh_vars : list of str
        Variables to be aggregated.
    census_api_key : str

    Returns
    -------
    hh_df : pandas.DataFrame
        Household-level dataframe with WGTP weights.
    """
    census_year = _pums_to_census(PUMS_year)

    # Drop person-level variables and build HH_ID
    passthrough_cols = (
        [c for c in person_df.columns if c.startswith("puma")] + ['total_pop_E', 'total_pop_M', 
        'total_households_E', 'total_households_M', 'borough', 'city', 'SERIALNO']
    )
    to_keep =  passthrough_cols + hh_vars
    person_df = person_df.loc[:, to_keep].copy()
    
    person_df["HH_ID"] = person_df[f"puma{census_year}"].astype(str) + "_" + person_df["SERIALNO"].astype(str)

    # Aggregate person vars → household vars (special case for PERNP)
    hh_agg = {}

    for var in hh_vars:
        if var == "PERNP":
            # Household has earnings if any person has PERNP > 0
            hh_agg[var] = lambda s: (pd.to_numeric(s, errors="coerce") > 0).any()
        else:
            # Default binary aggregation
            hh_agg[var] = lambda s: (s == "1").any()
    
    for col in passthrough_cols:
        hh_agg[col] = "first"

    hh_vars_df = (person_df
        .groupby("HH_ID", as_index=False)
        .agg(hh_agg)
    )

    if "PERNP" in hh_vars_df.columns:
        hh_vars_df["PERNP"] = np.where(hh_vars_df["PERNP"], 1, pd.NA)

    # Convert other hh_vars to nullable integer flags
    for v in hh_vars:
        if v != "PERNP":
            hh_vars_df[v] = np.where(hh_vars_df[v], 1, pd.NA)

    hh_vars_df = hh_vars_df.astype({v: "Int64" for v in hh_vars})

    # Rename HH variables
    rename_map = {v: f"{v}_hh" for v in hh_vars}

    hh_vars_df = hh_vars_df.rename(columns=rename_map)

    # Pull household weights
    pumas = PUMAS_20 if census_year == 2020 else PUMAS_10
    for_code = f'public use microdata area:{",".join(pumas)}&in=state:{STATE}'

    w1 = ",".join(["SERIALNO"] + ["WGTP"] + [f"WGTP{i}" for i in range(1, 41)])
    w2 = ",".join(["SERIALNO"] + [f"WGTP{i}" for i in range(41, 81)])
    
    urls = [f"{BASE_URL}/{PUMS_year}/acs/acs5/pums?get={w1}&for={for_code}&key={census_api_key}",
        f"{BASE_URL}/{PUMS_year}/acs/acs5/pums?get={w2}&for={for_code}&key={census_api_key}"]

    for url in urls:
        demo_df = _API_pull(url)
        demo_df = demo_df.rename(columns={"public use microdata area": f"puma{census_year}"}).drop(columns=["state"])

        # Ensure pre-2020 PUMA codes are 5-character, zero-padded.
        if PUMS_year < 2020:
            demo_df[f"puma{census_year}"] = (demo_df[f"puma{census_year}"].astype(str).str.zfill(5))
        
        # Create HH_ID and downsize to household-level
        demo_df["HH_ID"] = (demo_df[f"puma{census_year}"].astype(str) + "_" + demo_df["SERIALNO"].astype(str))
        demo_df = demo_df.drop_duplicates(subset="HH_ID")

        if url == urls[0]:
            # Saves first set of weights for later
            hh_wgt = demo_df
        else:
            demo_df = demo_df.drop(columns=[f"puma{census_year}", "SERIALNO"])
            hh_wgt = hh_wgt.merge(demo_df, on="HH_ID", how="inner", validate="1:1")
    drop_cols = ["SERIALNO", f"puma{census_year}"]

    # Merge aggregated vars with household weights
    hh_df = hh_vars_df.merge(hh_wgt.drop(columns=drop_cols), on="HH_ID", how="left", validate="1:1")

    return hh_df
#

# demo_dict supplement for variables created in harmonization
SUPPLEMENTAL_DICT = {"COMPUTER": "household", "BBAND": "household", "RACE_HISP": "person", "AGE_CAT": "person", 
    "AGE_U18": "person", "DIS_AGE": "person", "LBR_FRC": "person", "RETP_hh": "household", "SSP_hh": "household", 
    "SSIP_hh": "household", "PAP_hh": "household", "PERNP_hh": "household"}

def _pull_single_universe(PUMS_year, var_code_list, census_api_key, universe, level):
    """
    Fetches Public Use Microdata Sample (PUMS) data from one universe of the U.S. Census Bureau API, (person or household-level)
    aggregates it, calculates MOE, and processes it into a pandas DataFrame. Used by _pull_census_data().

    Parameters:
    -----------
    PUMS_year : int/str 
        The year of the PUMS dataset to fetch (e.g., 2019 for 2019 PUMS 5-year data).
    var_code_list : list of str
        A list of variable codes to retrieve from the PUMS dataset (e.g., ['SEX', 'AGEP']).
    census_api_key : str
        A valid API key for accessing the U.S. Census Bureau's API.
    universe: str
        The universe ("person" or "household") of data to be pulled.
    level: str
        The level of geographic aggregation desired ("puma", "borough", or "city".)

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the requested variable estimates for the selected universe (pulled directly from the PUMS).
    """
    # Ensure PUMS_year comes in as an int for later comparisons
    PUMS_year = int(PUMS_year)

    # setting census year (the year PUMAs are associated with)
    census_year = _pums_to_census(PUMS_year)

    if universe == "person":
        weight_var = "PWGTP"
        # Add a new variable to determine civilian noninstitutionalized population for person-level harmonization later
        type_col = "TYPEHUGQ" if PUMS_year > 2020 else "TYPE"
        var_code_list.append(type_col)

    elif universe == "household":
        weight_var = "WGTP"
        # Add a new variable to adjust household income for inflation later
        var_code_list.append("ADJINC")
    else:
        raise ValueError("Universe must be 'person' or 'household'")

    final_df = _get_geo_info_cached(PUMS_year, level)
    cc_name = {"puma": f"puma{census_year}", "borough": "borough", "city": "city"}[level]
    group = [cc_name]

    # Collect baseline geo info
    long_df = _get_geo_info_cached(PUMS_year, "puma")

    # Build weight URLs
    pumas = PUMAS_20 if census_year == 2020 else PUMAS_10
    for_code = f'public use microdata area:{",".join(pumas)}&in=state:{STATE}'

    # Collect id variables depending on vintage. CONCAT_ID is used to identify individuals when available. 
    # SERIALNO and SPORDER are still useful for person -> household aggregation.
    id_vars = ["SERIALNO", "SPORDER"]
    if PUMS_year >= 2020:
        id_vars = id_vars + ["CONCAT_ID"]
    
    demo_vars = ",".join(var_code_list + id_vars)
    demo_url = f"{BASE_URL}/{PUMS_year}/acs/acs5/pums?get={demo_vars}&for={for_code}&key={census_api_key}"

    w1 = ",".join(id_vars + [weight_var] + [f"{weight_var}{i}" for i in range(1, 41)])
    w2 = ",".join(id_vars + [f"{weight_var}{i}" for i in range(41, 81)])
    
    urls = [demo_url,
        f"{BASE_URL}/{PUMS_year}/acs/acs5/pums?get={w1}&for={for_code}&key={census_api_key}",
        f"{BASE_URL}/{PUMS_year}/acs/acs5/pums?get={w2}&for={for_code}&key={census_api_key}"]

    # Pull and merge demo data and weights
    for url in urls:  
        demo_df = _API_pull(url)
        demo_df = demo_df.rename(columns={"public use microdata area": f"puma{census_year}"}).drop(columns=["state"])

        # Ensure pre-2020 PUMA codes are 5-character, zero-padded. Also set up a unique id pre-CONCAT_ID.
        if PUMS_year < 2020:
            demo_df[f"puma{census_year}"] = (demo_df[f"puma{census_year}"].astype(str).str.zfill(5))
            demo_df["CONCAT_ID"] = (demo_df[f"puma{census_year}"].astype(str) 
                                    + "_" + demo_df["SERIALNO"].astype(str) 
                                    + demo_df["SPORDER"].astype(str))
            
        if demo_df["CONCAT_ID"].duplicated().any():
            raise ValueError("Duplicate CONCAT_ID detected during PUMS pull")

        if url == demo_url:
            # gathers PUMA-level info for each point estimate
            long_df = long_df.merge(demo_df, on=f"puma{census_year}", how="inner", validate="1:m")
        else:
            demo_df = demo_df.drop(columns=[f"puma{census_year}", "SERIALNO", "SPORDER"])
            long_df = long_df.merge(demo_df, on="CONCAT_ID", how="inner", validate="1:1")
    
    # Harmonize PUMS data so it's comparable to ACS
    long_df, new_vars = _harmonize_variables(long_df, PUMS_year)
    var_code_list += new_vars

    # Collapse to household level if needed
    if universe == "household":
        long_df["HH_ID"] = long_df[f"puma{census_year}"].astype(str) + "_" + long_df["SERIALNO"].astype(str)
        long_df = long_df.drop_duplicates(subset="HH_ID")

    # Aggregate variables from person to household level if needed
    p_to_hh = ["PERNP", "RETP", "SSP", "SSIP", "PAP"]
    requested_p_to_hh = [v for v in var_code_list if v in p_to_hh]
    pums_vars_for_agg = [v for v in var_code_list if v not in requested_p_to_hh]
    hh_df = None

    if requested_p_to_hh:
        hh_df = _person_to_household(long_df, PUMS_year, requested_p_to_hh, census_api_key)
    
    def _aggregate_universe(df, var_code_list, weight_var, final_df):
        z = 1.645
        rep_cols = [f"{weight_var}{i}" for i in range(1, 81)]
        num_cols = [weight_var] + rep_cols

        # Ensure numeric weights (do this once)
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

        # Melt once across all variables
        df_long = df.melt(id_vars=group + num_cols, value_vars=var_code_list, var_name="variable", value_name="level")

        group_cols = group + ["variable", "level"]

        # Point estimates
        point = (df_long.groupby(group_cols, as_index=False)[weight_var].sum().rename(columns={weight_var: "estimate"}))

        # Replicate estimates
        rep_est = (df_long.groupby(group_cols)[rep_cols].sum().reset_index())

        # Merge point estimates
        rep_est = rep_est.merge(point, on=group_cols, validate="1:1")

        # Variance + MOE
        diffs = rep_est[rep_cols].sub(rep_est["estimate"].to_numpy()[:, None])
        variance = (4 / 80) * (diffs ** 2).sum(axis=1)
        moe = z * np.sqrt(variance)

        rep_est["variance"] = variance
        rep_est["moe"] = moe

        # Build wide output
        rep_est["base"] = rep_est["variable"] + rep_est["level"].astype(str)
        rep_est["col_est"] = rep_est["base"] + "_E"
        rep_est["col_var"] = rep_est["base"] + "_Var"
        rep_est["col_moe"] = rep_est["base"] + "_M"

        cc_name = group[0]

        wide = rep_est.set_index(cc_name)[["col_est", "col_var", "col_moe", "estimate", "variance", "moe"]]

        est = wide.pivot(columns="col_est", values="estimate")
        var = wide.pivot(columns="col_var", values="variance")
        moe = wide.pivot(columns="col_moe", values="moe")

        wide_df = pd.concat([est, var, moe], axis=1).reset_index()

        # ---- Merge into final_df ----
        final_df = final_df.merge(wide_df, on=cc_name, how="left", validate="1:1")

        if level == "puma" and {"borough", "city"}.issubset(final_df.columns):
            final_df = final_df.drop(columns=["borough", "city"])

        return final_df
    
    out_df = final_df.copy()

    if universe == "person" or not requested_p_to_hh:
        out_df = _aggregate_universe(long_df, pums_vars_for_agg, weight_var, out_df)

    if requested_p_to_hh:
        out_df = _aggregate_universe(hh_df,[f"{v}_hh" for v in requested_p_to_hh], "WGTP", out_df)

    # Ensure that categories with 0 respondents in a geography (marked NA by earlier merges) are correctly shown as 0
    value_cols = [c for c in out_df.columns if c.endswith(("_E", "_Var", "_M"))]
    out_df[value_cols] = out_df[value_cols].fillna(0)

    return out_df
#

def _pull_census_data(PUMS_year, demo_dict, census_api_key, level):
    """
    Aggregates multiple universes (person, household-level) of PUMS data pulled from _pull_single_universe(), if necessary.
    Used by generate_new_estimates(). 

    Parameters:
    PUMS_year (int/str): The year of the PUMS dataset to fetch (e.g., 2019 for 2019 PUMS 5-year data).
    demo_dict (dict): A dictionary where keys are variable codes, and values are either 
        'person' or 'household', indicating the type of denominator used for estimation.
    census_api_key (str): A valid API key for accessing the U.S. Census Bureau's API.
    level (str): The geographic level to aggregate the census data to. Can be puma, borough, or city.

    Returns:
    pandas.DataFrame: A DataFrame containing the requested variable estimates for the selected geography 
        (pulled directly from the PUMS).
    
    Notes:
    - Raw census data is provided at the person/household level. This function can aggregate it to PUMA, borough, and city.
    Other geographies are introduced later in the pipeline.
    """
    # setting census year (the year PUMAs are associated with)
    census_year = _pums_to_census(PUMS_year)

    # determine a geographic identifier based on level and vintage
    if level == "puma": cc_name = f'puma{census_year}'
    elif level == "borough": cc_name = "borough"
    elif level == "city": cc_name = "city"
    else: raise ValueError("level must be 'puma', 'borough', or 'city'")

    # pull data separately for person and household vars
    person_vars = [v for v, d in demo_dict.items() if d == "person"]
    house_vars  = [v for v, d in demo_dict.items() if d == "household"]

    outputs = []

    if person_vars:
        person_df = _pull_single_universe(PUMS_year, person_vars, census_api_key, "person", level)
        outputs.append(person_df)

    if house_vars:
        house_df = _pull_single_universe(PUMS_year, house_vars, census_api_key, "household", level)
        outputs.append(house_df)

    # merge final outputs, removing repeated columns.
    final_df = outputs[0]
    for df in outputs[1:]:
        shared = set(final_df.columns) & set(df.columns)
        shared -= {cc_name}  # keep join key

        df = df.drop(columns=shared)

        final_df = final_df.merge(df, on=cc_name, how="outer", validate="1:1")

    return final_df
#

# TODO: figure out a cleaner solution for bringing all ID cols to the front (particularly: NTA while avoiding reNTAl jobs). 
def _clean_columns(geo_df, geo):
    """
    Customizes column order and drops unnecessary columns. Used by generate_new_estimates().
    
    Paramaters:
    geo_df (DataFrame): A dataframe that needs its PUMS estimate columns to be organized
    geo (str): A string specifying the geographic region. Options include 'councildist', 'communitydist', 'schooldist',
        policeprct', 'modzcta', 'nta', 'puma', 'borough', 'city'.
        
    Returns:
    pandas.DataFrame: DataFrame with columns organized in alphabetical order of variable codes.
    """
    
    # Drop variance columns
    geo_df = geo_df.loc[:, ~geo_df.columns.str.endswith("_Var")]
    
    # Identify geo columns dynamically
    geo_cols = [col for col in geo_df.columns if col.startswith(f"{geo}")]
    
    # All other columns
    variable_cols = [col for col in geo_df.columns if col not in geo_cols]
    
    # Sort non-geo columns alphabetically
    new_column_order = geo_cols + sorted(variable_cols)
    
    return geo_df.reindex(columns=new_column_order)    
#

def _expand_var_codes(var_code, demo_dict, df, suffix):
    """
    Detects all subsets of var_code in df columns. Returns the unique, sorted list of subset variable names.
    Used by _calc_proportion_estimate, _generate_bbl_estimates, _calc_proportion_MOE, _get_MOE_and_CV, 
    and _estimates_by_geography. 
    
    Parameters:
    var_code (str): The variable (e.g. 'SEX') to be expanded into subsets (e.g. 'SEX1', 'SEX2').
    demo_dict (dict): A dictionary where keys are variable codes, and values are either 
        'person' or 'household', indicating the type of denominator used for estimation.
    df (pd.DataFrame): DataFrame containing variables to expand.
    suffix (str): The variable suffix ("_E", "_M", or "_Var") to detect as subsets.

    Returns:
    list of str: A sorted list of all subsets of var_code found in df, or if there are none, var_code.
    """
    target_cols = []

    if var_code not in demo_dict.keys():
        for col in df.columns:
            if col.endswith(suffix) and col.startswith(var_code):
                target_cols.append(col)
        return sorted(target_cols)
    
    all_prefixes = sorted(demo_dict.keys(), key=len, reverse=True)

    for col in df.columns:
        if not col.endswith(suffix):
            continue

        matched_prefix = None
        for prefix in all_prefixes:
            if col.startswith(prefix):
                matched_prefix = prefix
                break

        if matched_prefix != var_code:
            continue

        # prevent certain created variables from being absorbed by their parents
        next_char_index = len(var_code)
        if len(col) > next_char_index and col[next_char_index] == "_" and "_" not in var_code:
            continue

        target_cols.append(col)
    
    return target_cols
#

def _access_denom(var_code, demo_dict, total_pop_code, total_house_code, denom_only):
    """
    This function determines whether the input variable var_code should be treated as person- or household-level,
    returning commonly used values in later functions as necessary.
    Used in _calc_proportion_estimate, _generate_bbl_estimates, _get_MOE_and_CV, and _estimates_by_geography.

    Parameters:
    var_code (str): Code for the demographic variable in the census API.
    demo_dict (dict): A dictionary where keys are variable codes, and values are either 
        'person' or 'household', indicating the type of denominator used for estimation.
    denom_only (bool): Indicates whether the function should return denom or est_level and total_pop (all 3 are never needed.)
    
    Returns:
    denom (str): Indicates whether variable is person- or household-level.
    est_level (str): Prefix indicating whether estimates derived from var_code will be pop or hh-level.
    total_pop (str): BBL-level column to reference when aggregating estimates.
    """
    if var_code == total_house_code: denom = "household"
    elif var_code == total_pop_code: denom = "person"
    elif var_code in demo_dict.keys(): denom = demo_dict.get(var_code)
    else: denom = SUPPLEMENTAL_DICT.get(var_code)
    
    if denom_only: return denom

    if denom == "household": # for variables with total households as the denominator
        est_level = "hh_est_" # household estimate
        total_pop = "unitsres" # denominator is total units
    elif denom == "person": # for variables with total population as the denominator
        est_level = "pop_est_" # total population estimate
        total_pop = "bbl_population_estimate" # denominator is total population
    
    return est_level, total_pop
#

def _calc_proportion_estimate(demo_dict, demo_df, var_code, total_pop_code = None, total_house_code = None):
    """
    This function calculates proportion estimates for a demographic variable by dividing its population counts by the 
    appropriate denominator (total population or total households). Helper function for _generate_bbl_estimates().

    Parameters:
    demo_dict  (dict): A dictionary where keys are PUMS variable codes and values specify whether the variable is 
        'person' or 'household' level. Example: {'SEX': 'person', 'HINCP': 'household'}.
    demo_df (DataFrame): DataFrame containing population numbers by PUMA for demographic groups.
    var_code (str): Census API code for the demographic variable.
    total_pop_code (str, optional): API code for total population. Required if generating person-level estimates.
    total_house_code (str, optional): API code for total households. Required if generating household-level estimates.

    Returns:
    DataFrame: Updated DataFrame with the demographic variable's percent estimates added.

    Notes:
        - Percent estimates are calculated as (demographic count / denominator).
        - Any infinite values resulting from division by zero are replaced with NaN.
    """
    target_cols = _expand_var_codes(var_code, demo_dict, demo_df, "_E")
    denom = _access_denom(var_code, demo_dict, total_pop_code, total_house_code, True)

    if var_code in [total_house_code, total_pop_code]:
        target_cols = [var_code]
    
    if denom == 'household': denom_col = total_house_code
    elif denom == 'person': denom_col = total_pop_code
    else: raise ValueError("Please ensure that all variables are marked as person- or household-level.")

    for col in target_cols:
        demo_df[col] = (demo_df[col] / demo_df[denom_col]).round(3)

    demo_df.replace([np.inf, -np.inf], np.nan, inplace=True) # for any inf values created because of division by 0
   
    return demo_df
#

def _generate_bbl_estimates(PUMS_year, demo_dict, pop_est_df, demo_df, total_pop_code = None, total_house_code = None):
    """
    This function generates BBL-level (Borough, Block, Lot) demographic estimates using Public Use Microdata Sample (PUMS) data.
    It integrates PUMA-level PUMS data with BBL-level PLUTO data and calculates population or household estimates for given
    demographic variables. Called in generate_new_estimates().

    Parameters:
    PUMS_year (int/str): The 5-Year PUMS end-year to fetch data for (e.g., 2023 for the 2019-2023 PUMS).
    demo_dict  (dict): A dictionary where keys are PUMS variable codes and values specify whether the variable is 
        'person' or 'household' level. Example for Data Profiles survey: {'SEX': 'person', 'HINCP': 'household'}.
    pop_est_df (pandas.DataFrame): A DataFrame containing BBL-level population data. 
        Must include columns 'borough' and 'puma{census_year}' for PUMA identifiers.
    census_api_key (str): API key for accessing the U.S. Census Bureau's API.
    total_pop_code (str, optional): API code for total population. Required if generating person-level estimates.
    total_house_code (str, optional): API code for total households. Required if generating household-level estimates.

    Returns:
    pandas.DataFrame
        An updated DataFrame with the following:
        - Added columns for proportions (prop_<variable_code>) of each demographic variable within PUMAs.
        - Estimated BBL-level counts (pop_est_<variable_code> or hh_est_<variable_code>) for each demographic.

    Notes:
    - PUMA compatibility is determined by the PUMS_year. Pre-2020 PUMS uses 2010 tracts; 2020 and later use 2020 tracts.
    """
    census_year = _pums_to_census(PUMS_year)
    supp_vars = [k for k in SUPPLEMENTAL_DICT if any(col.startswith(k) for col in demo_df.columns)]

    # Variables that require proportion conversion (excludes totals)
    proportion_vars = list(demo_dict.keys()) + supp_vars

    # Ensure total population / household counts are merged
    denom_list = [code for code in (total_pop_code, total_house_code) if code]

    if denom_list:
        pop_est_df = pop_est_df.merge(demo_df[[f"puma{census_year}"] + denom_list], on=f"puma{census_year}", 
            how="left", validate="many_to_one",)

    # Create BBL-level total columns explicitly
    if total_pop_code:
        pop_est_df[total_pop_code] = pop_est_df["bbl_population_estimate"]

    if total_house_code:
        pop_est_df[total_house_code] = pop_est_df["unitsres"]

    # Calculate proportions once per variable
    for var_code in proportion_vars:
        demo_df = _calc_proportion_estimate(demo_dict, demo_df, var_code, total_pop_code, total_house_code)

    # Build BBL-level estimates
    for var_code in proportion_vars:

        var_codes = _expand_var_codes(var_code, demo_dict, demo_df, "_E")
        est_level, total_pop = _access_denom(var_code, demo_dict, total_pop_code, total_house_code, False)

        # Merge PUMA-level proportions into BBL dataframe
        pop_est_df = pop_est_df.merge(demo_df[[f"puma{census_year}"] + var_codes], 
                    on=f"puma{census_year}", how="left", validate="many_to_one")

        for v in var_codes:

            base = v.removesuffix("_E")

            # Rename proportion column explicitly
            prop_col = f"prop_{base}"
            pop_est_df = pop_est_df.rename(columns={v: prop_col})

            # Safety check: proportions should be between 0 and 1
            # (prevents silent explosions)
            if pop_est_df[prop_col].max() > 1:
                raise ValueError(
                    f"{base} appears not to be a proportion. "
                    "Values exceed 1. Check upstream calculation.")

            # Construct BBL-level estimate
            pop_est_df[f"{est_level}{base}"] = pop_est_df[total_pop] * pop_est_df[prop_col]

    return pop_est_df

def _calc_proportion_MOE(demo_dict, variance_df, var_code, total_pop_code = None, total_house_code = None): 
    """
    Calculates the margins of error (MOE) for proportions based on Census Bureau's formula. 
    Helper function for _generate_bbl_variance().
    
    Parameters:
    demo_dict  (dict): A dictionary where keys are PUMS variable codes and values specify whether the variable is 
        'person' or 'household' level. Example for Data Profiles survey: {'SEX': 'person', 'HINCP': 'household'}.
    variance_df (dataframe): DataFrame containing estimates and MOEs pulled from the census API.
    var_code (str): Code for the demographic variable in the census API.
    total_pop_code (str, optional): API code for total population. Required if generating person-level estimates.
    total_house_code (str, optional): API code for total households. Required if generating household-level estimates.

    Returns:
    pandas.DataFrame: Updated DataFrame with calculated proportion MOEs.
        
    Note:
    - Find details on the formula used at:
    https://www.census.gov/content/dam/Census/library/publications/2018/acs/acs_general_handbook_2018_ch08.pdf. 
    """
    # collect MOE columns associated with this variable code
    moe_cols = _expand_var_codes(var_code, demo_dict, variance_df, "_M")

    # determine denominator columns
    level = demo_dict.get(var_code)
    if not level:
        level = SUPPLEMENTAL_DICT.get(var_code)
    if level == 'person':
        denom_est = total_pop_code
        denom_MOE = total_pop_code[:-1] + 'M'
    elif level == 'household':
        denom_est = total_house_code
        denom_MOE = total_house_code[:-1] + 'M'
    else: raise ValueError(f"Please ensure variable {var_code} is marked as person- or household-level.")

    for numerator_MOE in moe_cols:
        numerator_est = numerator_MOE[:-1] + 'E'

        # proportion estimate
        p = variance_df[numerator_est] / variance_df[denom_est]

        # Census MOE formula
        under_sqrt = (variance_df[numerator_MOE] ** 2 - p ** 2 * variance_df[denom_MOE] ** 2)

        variance_df[numerator_MOE] = (np.sqrt(
            np.where(under_sqrt >= 0, under_sqrt, variance_df[numerator_MOE] ** 2 + p ** 2 * variance_df[denom_MOE] ** 2,))
            / variance_df[denom_est])
        
    variance_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return variance_df
#

def _generate_bbl_variances(demo_dict, variance_df, total_pop_code = None, total_house_code = None):
    """
    This function retrieves PUMS 5-Year data for specified demographic variables and calculates the estimates' variance in
    proportion form at the PUMA level (with total population or households as the denominator). Called in
    generate_new_estimates().
    
    Parameters:
    PUMS_year (int/str): The PUMS 5-Year dataset end year (e.g., 2023 for the 2019-2023 PUMS 5-Year dataset).
    demo_dict (dict): A dictionary where keys are variable codes, and values are either 
        'person' or 'household', indicating the type of denominator used for estimation.
    census_api_key (str): API key for accessing the U.S. Census Bureau's API.
    total_pop_code (str, optional): API code for total population. Required if generating person-level estimates.
    total_house_code (str, optional): API code for total households. Required if generating household-level estimates.

    Returns:
    DataFrame: A DataFrame containing variances for all specified variables, with columns:
        - '{variable}_variance': Variance of the demographic variable proportion.

    Notes:
        - PUMA raw number MOEs are converted to proportions using a census formula in _calc_proportion_MOE.
        - Proportion MOEs are converted to variances using the formula: variance = (MOE / 1.645)^2.
    """  

    denom_list = [code for code in (total_pop_code, total_house_code) if code]
    denom_moe_list = [denom_code[:-1] + 'M' for denom_code in denom_list]

    supp_vars = [k for k in SUPPLEMENTAL_DICT if any(col.startswith(k) for col in variance_df.columns)]

    for var_code in list(demo_dict.keys()) + supp_vars + denom_moe_list: # for each code in the list, convert to proportion
        
        if var_code not in denom_moe_list: # exclude total population and total households because they are the denominators for the other variables
            variance_df = _calc_proportion_MOE(demo_dict, variance_df, var_code, total_pop_code, total_house_code)
        else: # for denominators, simply divide number MOE by number estimate to get proportion
            variance_df[var_code] = (variance_df[var_code] / variance_df[var_code[:-1] + 'E']).round(3)
            variance_df[var_code] = variance_df[var_code].replace([np.inf, -np.inf], np.nan)
        
        moe_cols = _expand_var_codes(var_code, demo_dict, variance_df, "_M")

        for moe_col in moe_cols:
            var_col = moe_col.replace("_M", "_E_variance")
            variance_df[var_col] = (variance_df[moe_col] / 1.645) ** 2

    return variance_df
#

def _calc_CV(geo_df, var_code):
    """
    Calculates the Coefficient of Variation (CV) for a specified variable in the given DataFrame. 
    Used by _get_MOE_and_CV() and generate_new_estimates().

    Parameters:
        geo_df (pd.DataFrame): A DataFrame containing the data for geographic regions, 
            including columns for estimates and margins of error.
        var_code (str ): The variable code representing the estimate column (e.g., '<var_code_base>E').

    Returns:
        pd.DataFrame: The input DataFrame with an additional column '<var_code_base>V', which contains the calculated CV values.

    Notes:
        The function expects the Margin of Error (MOE) column to follow the naming convention '<var_code_base>M', 
        where <var_code_base> is `var_code[:-1]`.
        CV is calculated as: CV = (Standard Error / Mean) * 100 
        where the Standard Error is derived from the MOE using the formula: Standard Error = MOE / 1.645
        Infinity values in the CV column (caused by division by zero) are replaced with NaN.    
    """
    est_col = f"{var_code}_E"
    moe_col = f"{var_code}_M"
    cv_col = f"{var_code}_V"

    if est_col not in geo_df.columns or moe_col not in geo_df.columns:
        return pd.Series(index=geo_df.index, dtype=float, name=cv_col)
    
    se = geo_df[moe_col]/1.645

    with np.errstate(divide="ignore", invalid="ignore"):
        cv = round(100 * (se / geo_df[est_col]), 2)
    
    cv = cv.replace([np.inf, -np.inf], np.nan)
    cv[geo_df[est_col] == 0] = np.nan
    cv.name = cv_col
    return cv
#

def _get_MOE_and_CV(demo_dict, variance_df, pop_est_df, census_year, geo_df, geo, total_pop_code = None, total_house_code = None, boundary_year = None): 
    """ 
    This function calculates MOE and CV values for given demographic variables at a specified geography level. 
    It uses population estimates and variance data to determine statistical reliability for each demographic. 
    Called by _estimates_by_geography(). 
    
    Parameters: 
    ----------- 
    demo_dict : dict 
        A dictionary mapping variable codes to their corresponding type ('person' or 'household'). 
    variance_df : pd.DataFrame 
        geo-level DataFrame containing variance information for demographic variables at the PUMA level. 
    pop_est_df : pd.DataFrame
        BBL-level DataFrame with population estimates and columns for geographic regions and PUMAs. 
    census_year : int 
        The census year associated with the data (e.g., 2010 or 2020). 
    geo_df : pd.DataFrame 
        The DataFrame for the specified geography, where calculated values will be appended. 
        For census tract, use f'{census_year}_tract_id'. 
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
        The updated geo_df with appended MOE and CV columns for each variable in demo_dict. 
    """ 

    puma_key = f"puma{census_year}"
    boundary_ext = f"_{boundary_year}" if (boundary_year and geo == "councildist") else ""
    geo_key = f"{geo}{boundary_ext}"

    supp_vars = [k for k in SUPPLEMENTAL_DICT if any(col.startswith(k) for col in variance_df.columns)]

    denom_list = [code for code in (total_pop_code, total_house_code) if code]
    all_base_vars = list(demo_dict.keys()) + supp_vars + denom_list

    # Pre-aggregate BBL → (geo, puma)
    grouped = (pop_est_df
        .groupby([geo_key, puma_key], sort=False)
        .sum(numeric_only=True)
        .reset_index())

    # Bring in PUMA-level proportion variances
    variance_cols = [c for c in variance_df.columns if c.endswith("_E_variance")]
    variance_merge = variance_df[[puma_key] + variance_cols].drop_duplicates()

    grouped = grouped.merge(variance_merge, on=puma_key, how="left")

    new_columns = {}

    # Variance aggregation
    for base_var in all_base_vars:

        est_level, total_pop = _access_denom(base_var, demo_dict, total_pop_code, total_house_code, False,)
        subset_vars = _expand_var_codes(base_var, demo_dict, variance_df, "_E_variance",)

        for var_code_full in subset_vars:

            var_code = var_code_full[:-11]  # remove "_E_variance"
            variance_col = f"{var_code}_E_variance"

            if variance_col not in grouped.columns:
                continue

            # ---------------------------------------------
            # CORRECT FORMULA:
            # Var(sum(N_i * p_i)) = sum(N_i^2 * Var(p_i))
            # ---------------------------------------------

            grouped["_weighted_var"] = (grouped[total_pop] ** 2) * grouped[variance_col]

            geo_variance = grouped.groupby(geo_key)["_weighted_var"].sum()

            se = np.sqrt(geo_variance)
            moe = se * 1.645

            # Align with geo_df index
            moe = moe.reindex(geo_df.index)

            # Null where estimate = 0
            est_col = f"{var_code}_E"
            if est_col in geo_df.columns:
                zero_mask = geo_df[est_col] == 0
                moe[zero_mask] = np.nan

            new_columns[f"{var_code}_M"] = moe

            # Percent MOE
            if total_pop in grouped.columns:
                geo_totals = grouped.groupby(geo_key)[total_pop].sum()
                geo_totals = geo_totals.reindex(geo_df.index)

                with np.errstate(divide="ignore", invalid="ignore"):
                    percent_moe = 100 * (moe / geo_totals)

                percent_moe = percent_moe.round(2)
                percent_moe[geo_totals == 0] = np.nan

                new_columns[f"{var_code}_PM"] = percent_moe

    # Attach MOE columns
    if new_columns:
        geo_df = pd.concat([geo_df, pd.DataFrame(new_columns)], axis=1,)

    # CV calculation
    cv_columns = {}

    for base_var in all_base_vars:
        subset_vars = _expand_var_codes(base_var, demo_dict, variance_df, "_E_variance")

        for var_code_full in subset_vars:

            var_code = var_code_full[:-11]

            if f"{var_code}_E" in geo_df.columns:
                cv_series = _calc_CV(geo_df, var_code)

                if not cv_series.isna().all():
                    cv_columns[cv_series.name] = cv_series

    if cv_columns:
        geo_df = pd.concat([geo_df, pd.DataFrame(cv_columns)], axis=1,)

    return geo_df

def _estimates_by_geography(PUMS_year, demo_dict, geo, pop_est_df, variance_df, total_pop_code=None, total_house_code=None, boundary_year=None):
    """
    Aggregates population and household estimates by a specified geography and attaches these values to the corresponding
    geographic DataFrame. Called in generate_new_estimates().

    Parameters:
    PUMS_year (int/str): The 5-Year PUMS end-year to fetch data for (e.g., 2023 for the 2018-2023 PUMS).
    demo_dict (dict): A dictionary where keys are variable codes, and values are either 
        'person' or 'household', indicating the type of denominator used for estimation.
    geo (str): The geographic level to aggregate by (e.g., "borough", "communitydist").
    pop_est_df (pandas.DataFrame): DataFrame containing demographic estimate data at the BBL level.
    variance_df (pandas.DataFrame): DataFrame containing variance data for the estimates.
    total_pop_code (str, optional): API code for total population. Required if generating person-level estimates.
    total_house_code (str, optional): API code for total households. Required if generating household-level estimates.
    boundary_year (int): Year for the geographic boundary (relevant only for geo = "councildist"). Options: 2013, 2023.
        
    Returns:
    pandas.DataFrame: A DataFrame with aggregated demographic estimates, attached to the specified geography.
    """
    # setting census year (the year PUMAs are associated with)
    census_year = _pums_to_census(PUMS_year)    

    # setting boundary year (only applies to councildist)
    boundary_ext = f'_{boundary_year}' if (boundary_year) and (geo == 'councildist') else ''
    geo_key = f'{geo}{boundary_ext}'
    # setting path
    file_path = f'{DATA_PATH}/{geo_key}-boundaries.geojson'
    
    # load GeoJSON file for geographic boundaries
    with open(file_path) as f:
        geo_data = geojson.load(f)

    # create dataframe
    features = geo_data["features"]
    geo_df = pd.json_normalize([feature["properties"] for feature in features]).set_index(geo_key)

    # prepare supplemental variables and denominators
    supp_vars = [k for k in SUPPLEMENTAL_DICT if any(col.startswith(k) for col in variance_df.columns)]
    denom_list = [code for code in (total_pop_code, total_house_code) if code]
    all_vars = list(demo_dict.keys()) + supp_vars + denom_list

    # Precompute est_level once per variable
    est_level_map = {var: _access_denom(var, demo_dict, total_pop_code, total_house_code, False)[0] for var in all_vars}

    # Precompute expand results once per variable
    expand_cache = {var: _expand_var_codes(var, demo_dict, variance_df, "_E_variance") for var in all_vars}

    # Build aggregation column list
    agg_columns = []
    rename_map = {}

    # Regular demographic + supplemental variables
    for base_var in list(demo_dict.keys()) + supp_vars:
        est_level = est_level_map[base_var]
        subset_vars = expand_cache[base_var]

        for var_code_full in subset_vars:
            var_code = var_code_full[:-11]
            col_name = est_level + var_code

            if col_name in pop_est_df.columns:
                agg_columns.append(col_name)
                rename_map[col_name] = var_code + "_E"

    # Totals (handle explicitly — do NOT expand)
    for total_code in denom_list:
        col_name = total_code + "_E"

        if col_name in pop_est_df.columns:
            agg_columns.append(col_name)
            rename_map[col_name] = total_code + "_E"

    # Remove duplicates while preserving order
    agg_columns = list(dict.fromkeys(agg_columns))

    # Ensure totals exist in pop_est_df for aggregation
    if total_pop_code and total_pop_code not in pop_est_df.columns:
        pop_est_df[total_pop_code] = pop_est_df['bbl_population_estimate']

    if total_house_code and total_house_code not in pop_est_df.columns:
        pop_est_df[total_house_code] = pop_est_df['unitsres']

    # Add totals to aggregation list
    for total_code in denom_list:
        col_name = total_code
        if col_name not in agg_columns and col_name in pop_est_df.columns:
            agg_columns.append(col_name)
            rename_map[col_name] = total_code
    
    # SINGLE GROUPBY
    if agg_columns:
        aggregated_df = pop_est_df.groupby(geo_key)[agg_columns].sum().round().rename(columns=rename_map)
        geo_df = geo_df.join(aggregated_df, how="left")

    # ADD MOE + CV
    geo_df = _get_MOE_and_CV(demo_dict, variance_df, pop_est_df, census_year, 
            geo_df, geo, total_pop_code, total_house_code, boundary_year)

    return geo_df.reset_index()
#

def _calc_ratio(raw_geo_df, numerator, denominator, name):
    """
    Calculate percent estimate, percent MOE, and CV for a ratio.
    Created for unemployment rate, but could extend to % of pop under 200% FPL, etc.
    
    Parameters:
    raw_geo_df (pd.DataFrame): A dataframe containing the estimates from which a ratio will be calculated.
    numerator (str): Column name of numerator estimate (e.g. 'ESR3_E')
    denominator (str): Column name of denominator estimate (e.g. 'LBR_FRC1_E')
    name (str): Output base name (e.g. 'UNEMP')

    Returns:
    DataFrame with new columns added: {name}_PE, {name}_PM, {name}_V
    """
    num_M = numerator[:-1] + "M"
    den_M = denominator[:-1] + "M"

    if not {numerator, denominator, num_M, den_M}.issubset(raw_geo_df.columns):
        return raw_geo_df

    with np.errstate(divide="ignore", invalid="ignore"):
        # proportion
        den = raw_geo_df[denominator].replace({0: np.nan})
        
        p = raw_geo_df[numerator] / den

        raw_geo_df[f"{name}_PE"] = (100 * p).round(2)

        # MOE (ACS ratio formula)
        under_sqrt = raw_geo_df[num_M]**2 - (p**2) * raw_geo_df[den_M]**2
        cond = (under_sqrt >= 0).fillna(False)

        moe_prop = np.sqrt(
            np.where(
                cond,
                under_sqrt,
                raw_geo_df[num_M]**2 + (p**2) * raw_geo_df[den_M]**2
            )
        ) / den

        raw_geo_df[f"{name}_PM"] = (100 * moe_prop).round(2)

        # coefficient of variation
        raw_geo_df[f"{name}_V"] = (
            raw_geo_df[f"{name}_PM"] /
            (1.645 * raw_geo_df[f"{name}_PE"])
        ).round(3)

    raw_geo_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return raw_geo_df


######## VIEW AVAILABLE INPUTS  

def available_years():
    """
    Prints the available input years for all package functions that require year variables.

    Parameters: None
    Returns: None 
    """
    # find years available for new estimates (also years available for BBL-level population estimates)
    bbl_file_names = [f for f in os.listdir(DATA_PATH) if "puma-bbl-population-estimates_" in f]
    bbl_years = sorted([name[30:34] for name in bbl_file_names])
    bbl_PUMS_years = [f'{int(year)-4}-{year}' for year in bbl_years]
    bbl_PUMS_list = ', '.join(bbl_PUMS_years) # for PUMS 5-Year surveys

    # print results
    print(f'5-Year PUMS Surveys available: {bbl_PUMS_list}')
    print(f"\nNote: when using `councilcount` functions, only include the end year as the input for the year variable (e.g. '2023' for the 2019-2023 survey).")

    return 
#

def get_census_api_codes(PUMS_year, census_api_key):
    """
    This function pulls from a PUMS 5-Year data dictionary to show all variable codes for a given year, including those beyond 
    the existing CouncilCount database. Each variable code represents a demographic estimate provided by the PUMS. 
    Weight variables, predicates, and Puerto Rico-exclusive variables are excluded.
    Visit https://api.census.gov/data/<PUMS YEAR>/acs/acs5/pums/variables.html to view the options in web format.

    Parameters:
    PUMS_year (int/str): The 5-Year PUMS end-year to fetch data for (e.g., 2023 for the 2019-2023 PUMS).
    census_api_key (str): API key for accessing the U.S. Census Bureau's API.
        
    Returns:
    DataFrame: A table with 'variable_code' and 'variable_description' columns. 

    Notes:
    - This function pulls directly from https://api.census.gov/data/<PUMS YEAR>/acs/acs5/pums/variables.html.
    - These variable codes may be used as inputs for councilcount functions that generate new estimates, 
    like generate_new_estimates().
    - To view the variables that are currently covered by the CouncilCount database, use `get_available_councilcount_codes()`.
    - If the desired variable is on this list, you may use `get_councilcount_estimates()` instead of `generate_new_estimates()`.
    """
    # define parameters
    PUMS_year = int(PUMS_year) # consistent dtype
    var_codes_url = f'{BASE_URL}/{PUMS_year}/acs/acs5/pums/variables?key={census_api_key}'
    response = requests.get(var_codes_url)
    response.raise_for_status()
    data = response.json()

    PUMS_dict = {}

    for d in data: # putting all code/ description pairs in a dict
        code = d[0]
        desc = d[1]
        predicate = d[2]

        # filter out Puerto Rico variables
        is_pr_code = code.endswith(("PR", "PRP"))
        is_pr_desc = "Puerto Rico" in desc

        # filter out predicate-only variables
        is_predicate_only = predicate is not None

        # filter out weight variables
        is_weight = "WGTP" in code

        if not (is_pr_code or is_pr_desc or is_predicate_only or is_weight):
            PUMS_dict[code] = desc

    PUMS_code_df = (pd.DataFrame([PUMS_dict])
        .melt(var_name="variable_code", value_name="variable_description")
        .sort_values('variable_code')
        .reset_index(drop=True)
        )
    return PUMS_code_df
#

def get_available_councilcount_codes(PUMS_year=None):
    """
    Retrieve the available PUMS variable codes that currently exist in the CouncilCount database for a given survey year. 
    Each variable code represents a demographic estimate provided by the PUMS.
    Visit https://api.census.gov/data/<PUMS YEAR>/acs/acs5/pums/variables.html to view the options in web format.

    Parameters:
    PUMS_year (int/str): Desired 5-Year PUMS year (e.g., for the 2017-2021 5-Year PUMS, enter "2021").
        If None, the most recent year available will be used.

    Returns:
    pd.DataFrame: Table of available variables with columns for variable code, variable name, variable values, and their meanings.
        
    Notes:
        - Use desired variable code(s) as the input for `var_codes` in the get_councilcount_estimates() function to obtain
        demographic estimates that have already been generated.
        - If the desired variable cannot be found in the DF produced by available_councilcount_codes(), use
        generate_new_estimates() instead.
        - To view ALL variable codes that can be found in the PUMS, use get_census_api_codes(). 
    """
    if PUMS_year: PUMS_year = int(PUMS_year) # consistent dtype

    # find all the available years
    txt_names = [f for f in os.listdir(DATA_PATH) if f.endswith(".txt")]
    dictionary_txt_names = [name for name in txt_names if "PUMS_Data_Dictionary" in name]
    dictionary_years = [int(name[26:30]) for name in dictionary_txt_names]

    # if year is not chosen, set default to the latest year
    if PUMS_year is None: PUMS_year = max(dictionary_years)

    # error message if the requested year is unavailable
    if PUMS_year not in dictionary_years:
        dictionary_years_str_list = [str(year) for year in dictionary_years]
        available_years = "\n".join(sorted(dictionary_years_str_list))
        raise ValueError(f"This year is not available.\n"
            f"Please choose from the following:\n{available_years}")
    # construct the name of the dataset based on the year
    dict_name = f"PUMS_Data_Dictionary_{PUMS_year - 4}-{PUMS_year}.txt"

    # retrieve the data dictionary
    file_path = f'{DATA_PATH}/{dict_name}'

    with open(file_path, 'r') as file:
        dict = file.read()

    print(f"Printing data dictionary for the {PUMS_year} 5-Year PUMS")
    print(dict)
#

######## PULL/ GENERATE ESTIMATES  

def get_bbl_population_estimates(year=None):
    """
    Produces a DataFrame containing BBL-level population estimates for a specified year.

    Parameters:
    year (int/str): The desired year for BBL-level estimates. If None, the most recent year available will be used.

    Returns:
    DataFrame: A table with population estimates by BBL ('bbl_population_estimate' column). 
        
    Notes:
        - The output includes latitude and longitude columns. This will allow for the aggregation of population numbers
        to various geography levels. Simply convert the table to a GeoDataframe with a geometry column, perform a spatial 
        join with a second GeoDataFrame that contains polygons for the desired geographic regions, and then aggregate population 
        numbers to that level. 
        - Avoid using estimates for individual BBLs; the more aggregation, the less error. 
        - Population numbers were estimated by multiplying the total number of residential units within a BBL by 
        the surrounding PUMA's housing population density (PUMA total population / PUMA total residential units).
    """
    if year: year = int(year) # consistent dtype

    # find all available years
    bbl_file_names = [f for f in os.listdir(DATA_PATH) if "puma-bbl-population-estimates_" in f]
    bbl_years = [int(name[30:34]) for name in bbl_file_names]

    # if year is not chosen, set default to latest year
    if year is None: year = max(bbl_years)

    # error message if unavailable survey year selected
    if year not in bbl_years:
        bbl_years_str_list = [str(year) for year in bbl_years]
        available_years = "\n".join(sorted(bbl_years_str_list))
        raise ValueError(f"This year is not available.\n"
            f"Please choose from the following:\n{available_years}")
    
    # construct the name of the dataset based on the year
    bbl_name = f"puma-bbl-population-estimates_{year}.csv"
    
    print(f"Printing BBL population estimates for {year}")

    # retrieve the dataset
    file_path = f'{DATA_PATH}/{bbl_name}'
    df = pd.read_csv(file_path)
    
    return df[['borough', 'block', 'lot', 'bbl_population_estimate']]
#

def generate_new_estimates(PUMS_year, demo_dict, geo, census_api_key, total_pop_code=None, total_house_code=None, boundary_year=None):    
    """
    Generates demographic estimates, margins of error (MOEs), and coefficients of variation (CVs) for a specified NYC geography. 
    If total_pop_code and/ or total_house_code entered, output columns for these variables will also be included.

    Parameters:
    ----------
        PUMS_year : int/str
            The 5-Year PUMS end-year to fetch data for (e.g., 2023 for the 2019-2023 PUMS).
        demo_dict : dict
            Dict keys should be the PUMS variable codes for desired demographic groups. Dict values should
            specify whether the variable is 'person' or 'household' level. Codes must end in 'E', indicating that they are
            estimate codes. Example for Data Profiles survey: {'DP05_0001E': 'person', 'DP02_0059E': 'household'}. See Notes.
        geo : str
            The geographic level for estimates. Options currently include 'councildist', 'communitydist', 'schooldist',
            'policeprct', 'modzcta', 'nta', 'puma', 'borough', 'city'.
        census_api_key : str
            User's Census API key.
        total_pop_code : str, optional
            Variable code for total population in PUMS survey of interest. Required for person-level estimates. See Notes.
        total_house_code : str, optional
            Variable code for total households in PUMS survey of interest. Required for household-level estimates. See Notes.
        boundary_year : int/str, optional
            Boundary year for geography, required if `geo` is 'councildist' (valid values: 2013, 2023).

    Returns:
    --------
        pd.DataFrame: A cleaned DataFrame with demographic estimates, MOEs, and CVs for the specified geography and year.

    Notes:
    ------
        - To explore available variable codes, as well as find the values needed for `total_pop_code` and/ or `total_house_code`,
        use `get_census_api_codes()` or visit https://api.census.gov/data/<PUMS YEAR>/acs/acs5/pums/variables.html 
        to view the options in web format.
        - Variable codes ending in 'E' are number estimates. Those ending in 'M' are number MOEs. Adding 'P' before 'E' or 'M'
        means the values are percents. Codes ending in 'V' are coefficients of variation.
        -  Generates estimates using the 5-Year Public Use Microdata Sample, Primary Land Use Tax Lot Output, and
        geographic boundary files.
        - Data for geographies available within existing census hierarchy are taken from the PUMS. All other data are estimates
        generated by the NYC Council Data Team's methodology. Contact datainfo@council.nyc.gov with questions.
        - If the data you are looking for already exists in the CouncilCount database, please use `get_councilcount_estimates()`
        instead.
        - Geographies fitting into the census hierachy will receive estimates directly from the PUMS. 
        In all other cases, estimates generated using the NYCC Data Team's methodology will be provided. 
        - As an exception, pre-2020 PUMS NTA requests will be fulfilled using the NYCC Data Team's methodology. 
        This is because all NTA estimates from `councilcount` will be provided along 2020 NTA boundaries 
        (which are directly comprised of 2020 census tracts), and pre-2020 ACS data is provided along 2010 census tract 
        boundaries, making direct aggregation challenging.
    """    
    # validating inputs

    # consistent dtypes
    if PUMS_year: PUMS_year = int(PUMS_year) 
    if boundary_year: boundary_year = int(boundary_year)

    # setting census year (the year PUMAs are associated with) 
    census_year = _pums_to_census(PUMS_year)
    
    # locate available CSV files
    file_names = os.listdir(DATA_PATH)
    
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
    if geo not in geo_names and geo != "puma":
        raise ValueError(f"The geography '{geo}' could not be found. Available options are:\n" + ", ".join(geo_names))
    # ensuring correct PUMS_year input
    if PUMS_year not in available_years:
        raise ValueError((f"The PUMS year {PUMS_year} could not be found. Available options are:\n" + ", "
            .join(map(str, available_years))))
    # ensuring appropriate denominators provided
    if 'person' in demo_dict.values() and total_pop_code is None:
        raise ValueError("Must include total_pop_code for person-level estimates.")
    if 'household' in demo_dict.values() and total_house_code is None:
        raise ValueError("Must include total_house_code for household-level estimates.")
    if {"RETP", "SSP", "SSIP", "PAP"} & demo_dict.keys() and total_house_code is None:
        raise ValueError("Must include total_house_code for estimates that will be converted person-to-household.")
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
    if (geo in ['councildist','schooldist','policeprct','communitydist', 'nta', 'modzcta']):        
        
        # generating blank BBL-level population estimates df
        blank_pop_est_df = (pd.read_csv(f'{DATA_PATH}/puma-bbl-population-estimates_{PUMS_year}.csv', 
            dtype = {f'puma{census_year}': str}))

        # Pull PUMA-level data
        puma_df = _pull_census_data(PUMS_year, demo_dict, census_api_key, level = "puma")

        # adding columns for BBL-level demographic estimates
        pop_est_df = _generate_bbl_estimates(PUMS_year, demo_dict, blank_pop_est_df, 
            puma_df, total_pop_code, total_house_code)

        # creating PUMA-level variances in order to calculate MOE at the geo-level below
        variance_df = _generate_bbl_variances(demo_dict, puma_df, total_pop_code, total_house_code)

        # creating geo-level estimates, MOEs, and CVs
        raw_geo_df = _estimates_by_geography(PUMS_year, demo_dict, geo, pop_est_df, 
                    variance_df, total_pop_code, total_house_code, boundary_year)
      
    # selections for which estimates can be directly taken from the PUMS
    elif (geo in ['puma', 'borough','city']):
        
        # pull estimates and MOEs from Census API
        raw_geo_df = _pull_census_data(PUMS_year, demo_dict, census_api_key, level = geo)
        supp_vars = [k for k in SUPPLEMENTAL_DICT if any(col.startswith(k) for col in raw_geo_df.columns)]
        denom_list = [code for code in (total_pop_code, total_house_code) if code]

        # Create Percent MOE (_PM)
        for base_var in list(demo_dict.keys()) + supp_vars:
            level_type = demo_dict.get(base_var) or SUPPLEMENTAL_DICT.get(base_var)
            subset_vars = _expand_var_codes(base_var, demo_dict, raw_geo_df, "_M")

            for numerator_MOE in subset_vars:
                numerator_est = numerator_MOE[:-1] + "E"
                base = numerator_MOE[:-2]
                if level_type == 'person':
                    denom_est = total_pop_code
                    denom_MOE = total_pop_code[:-1] + "M"
                elif level_type == 'household':
                    denom_est = total_house_code
                    denom_MOE = total_house_code[:-1] + "M"
                else: continue

                p = raw_geo_df[numerator_est] / raw_geo_df[denom_est]
                under_sqrt = raw_geo_df[numerator_MOE] ** 2 - (p ** 2) * raw_geo_df[denom_MOE] ** 2

                moe_prop = np.sqrt(np.where(under_sqrt >= 0, under_sqrt, 
                    raw_geo_df[numerator_MOE] ** 2 + (p ** 2) * raw_geo_df[denom_MOE] ** 2)) / raw_geo_df[denom_est]

                raw_geo_df[f"{base}_PM"] = (100 * moe_prop).round(2)
        raw_geo_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Create CV
        cv_columns = {}
        for base_var in list(demo_dict.keys()) + supp_vars + denom_list:
            subset_vars = _expand_var_codes(base_var, demo_dict, raw_geo_df, "_E")
            for var_code_full in subset_vars:
                var_code = var_code_full[:-2]
                cv_series = _calc_CV(raw_geo_df, var_code)

                if not cv_series.isna().all():
                    cv_columns[cv_series.name] = cv_series
        if cv_columns:
            raw_geo_df = pd.concat([raw_geo_df, pd.DataFrame(cv_columns)], axis=1)

    # Ensure denominator columns exist for Data Team methodology geos
    if geo in ['councildist','schooldist','policeprct','communitydist', 'nta', 'modzcta']:
        
        # If total_pop_code expected but missing, construct it
        if total_pop_code and total_pop_code not in raw_geo_df.columns:
            if 'bbl_population_estimate' in pop_est_df.columns:
                # This should already have been aggregated inside _estimates_by_geography,
                # but as a safeguard:
                raw_geo_df[total_pop_code] = raw_geo_df.filter(like='pop_est_').sum(axis=1)

        if total_house_code and total_house_code not in raw_geo_df.columns:
            if 'unitsres' in pop_est_df.columns:
                raw_geo_df[total_house_code] = raw_geo_df.filter(like='hh_est_').sum(axis=1)


    # Create Proportion Estimates (_PE)
    supp_vars = [k for k in SUPPLEMENTAL_DICT if any(col.startswith(k) for col in raw_geo_df.columns)]
    for base_var in list(demo_dict.keys()) + supp_vars:
        level_type = demo_dict.get(base_var) or SUPPLEMENTAL_DICT.get(base_var)
        subset_vars = _expand_var_codes(base_var, demo_dict, raw_geo_df, "_E")
        
        for est_col in subset_vars:
            base = est_col[:-2]
            if level_type == 'person':
                denom_col = total_pop_code
            elif level_type == 'household':
                denom_col = total_house_code
            else: continue
            with np.errstate(divide="ignore", invalid="ignore"):
                raw_geo_df[f"{base}_PE"] = (100 * (raw_geo_df[est_col] / raw_geo_df[denom_col])).round(2)
    raw_geo_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Calculate ratios, e.g. unemployment rate
    raw_geo_df = _calc_ratio(raw_geo_df, "ESR3_E", "LBR_FRC1_E", "UNEMP")

    # cleaning
    cleaned_geo_df = _clean_columns(raw_geo_df, geo)
    
    return cleaned_geo_df
#

dict20 = {"SEX": "person", "HISP": "person", "RAC1P": "person", "MAR": "person", "DECADE": "person", "MIG": "person", "SCHG": "person", 
    "SCHL": "person", "ESR": "person", "JWTRNS": "person", "JWRIP": "person", "TYPEHUGQ": "household", "HICOV": "person", 
    "PRIVCOV": "person", "PUBCOV": "person",  "R18": "household", "R65": "household", "VEH": "household", "TEL": "household",
    "HFL": "household", "CIT": "person", "NATIVITY": "person", "POBP": "person", "WAOB": "person", "ENG": "person", "LANX": "person", 
    "LANP": "person", "MIL": "person", "TEN": "household", "BROADBND": "household", "HISPEED": "household", "COMPOTHX": "household", 
    "TABLET": "household", "SMARTPHONE": "household", "NAICSP": "person", "DIS": "person", "HINCP": "household", "PERNP": "person", 
    "RETP": "person", "SSP": "person", "SSIP": "person", "PAP": "person", "FS": "household", "AGEP": "person", "POVPIP": "person", 
    "NP": "household", "OCPIP": "household", "GRPIP": "household"}

dict16 = {"SEX": "person", "HISP": "person", "RAC1P": "person", "MAR": "person", "DECADE": "person", "MIG": "person", 
    "SCHG": "person", "SCHL": "person", "ESR": "person", "JWTR": "person", "JWRIP": "person", "TYPE": "household", 
    "HICOV": "person", "PRIVCOV": "person", "PUBCOV": "person",  "R18": "household", "R65": "household", "VEH": "household", 
    "TEL": "household", "HFL": "household", "CIT": "person", "NATIVITY": "person", "POBP": "person", "WAOB": "person", 
    "ENG": "person", "LANX": "person", "LANP": "person", "MIL": "person", "TEN": "household", "NAICSP": "person", "DIS": "person", 
    "HINCP": "household", "PERNP": "person", "RETP": "person", "SSP": "person", "SSIP": "person", "PAP": "person", "FS": 
    "household", "AGEP": "person", "POVPIP": "person", "NP": "household", "OCPIP": "household", "GRPIP": "household"}

dict11 = {"SEX": "person", "HISP": "person", "RAC1P": "person", "MAR": "person", "DECADE": "person", "MIG": "person", 
    "SCHG": "person", "SCHL": "person", "ESR": "person", "JWTR": "person", "JWRIP": "person", "R18": "household", 
    "R65": "household", "VEH": "household", "TEL": "household", "HFL": "household", "CIT": "person", "NATIVITY": "person", 
    "POBP": "person", "WAOB": "person", "ENG": "person", "LANX": "person", "LANP": "person", "MIL": "person", "TEN": "household", 
    "NAICSP07": "person", "HINCP": "household", "PERNP": "person", "RETP": "person", "SSP": "person", "SSIP": "person", 
    "PAP": "person", "FS": "household", "AGEP": "person", "POVPIP": "person", "NP": "household", "OCPIP": "household", 
    "GRPIP": "household"}

# nyc_wide_estimates_puma_2023 = generate_new_estimates(2023, dict20, "city", API_KEY, "total_pop_E", "total_households_E")
# nyc_wide_estimates_puma_2023.to_csv("nyc_wide_estimates_puma_2023.csv", index = False)
# nyc_wide_estimates_puma_2021 = generate_new_estimates(2021, dict20, "city", API_KEY, "total_pop_E", "total_households_E")
# nyc_wide_estimates_puma_2021.to_csv("nyc_wide_estimates_puma_2021.csv", index = False)
# nyc_wide_estimates_puma_2016 = generate_new_estimates(2016, dict16, "city", API_KEY, "total_pop_E", "total_households_E")
# nyc_wide_estimates_puma_2016.to_csv("nyc_wide_estimates_puma_2016.csv", index = False)
# nyc_wide_estimates_puma_2011 = generate_new_estimates(2011, dict11, "city", API_KEY, "total_pop_E", "total_households_E")
# nyc_wide_estimates_puma_2011.to_csv("nyc_wide_estimates_puma_2011.csv", index = False)

# borough_geographies_puma_2023 = generate_new_estimates(2023, dict20, "borough", API_KEY, "total_pop_E", "total_households_E")
# borough_geographies_puma_2023.to_csv("borough-geographies_puma_2023.csv", index = False)
# borough_geographies_puma_2021 = generate_new_estimates(2021, dict20, "borough", API_KEY, "total_pop_E", "total_households_E")
# borough_geographies_puma_2021.to_csv("borough-geographies_puma_2021.csv", index = False)
# borough_geographies_puma_2016 = generate_new_estimates(2016, dict16, "borough", API_KEY, "total_pop_E", "total_households_E")
# borough_geographies_puma_2016.to_csv("borough-geographies_puma_2016.csv", index = False)
# borough_geographies_puma_2011 = generate_new_estimates(2011, dict11, "borough", API_KEY, "total_pop_E", "total_households_E")
# borough_geographies_puma_2011.to_csv("borough-geographies_puma_2011.csv", index = False)

# puma_geographies_puma_2023 = generate_new_estimates(2023, dict20, "puma", API_KEY, "total_pop_E", "total_households_E")
# puma_geographies_puma_2023.to_csv("puma-geographies_puma_2023.csv", index = False)
# puma_geographies_puma_2021 = generate_new_estimates(2021, dict20, "puma", API_KEY, "total_pop_E", "total_households_E")
# puma_geographies_puma_2021.to_csv("puma-geographies_puma_2021.csv", index = False)
# puma_geographies_puma_2016 = generate_new_estimates(2016, dict16, "puma", API_KEY, "total_pop_E", "total_households_E")
# puma_geographies_puma_2016.to_csv("puma-geographies_puma_2016.csv", index = False)
# puma_geographies_puma_2011 = generate_new_estimates(2011, dict11, "puma", API_KEY, "total_pop_E", "total_households_E")
# puma_geographies_puma_2011.to_csv("puma-geographies_puma_2011.csv", index = False)

# communitydist_geographies_puma_2023 = generate_new_estimates(2023, dict20, "communitydist", API_KEY, "total_pop_E", "total_households_E")
# communitydist_geographies_puma_2023.to_csv("communitydist-geographies_puma_2023.csv", index = False)
# communitydist_geographies_puma_2021 = generate_new_estimates(2021, dict20, "communitydist", API_KEY, "total_pop_E", "total_households_E")
# communitydist_geographies_puma_2021.to_csv("communitydist-geographies_puma_2021.csv", index = False)
# communitydist_geographies_puma_2016 = generate_new_estimates(2016, dict16, "communitydist", API_KEY, "total_pop_E", "total_households_E")
# communitydist_geographies_puma_2016.to_csv("communitydist-geographies_puma_2016.csv", index = False)
# communitydist_geographies_puma_2011 = generate_new_estimates(2011, dict11, "communitydist", API_KEY, "total_pop_E", "total_households_E")
# communitydist_geographies_puma_2011.to_csv("communitydist-geographies_puma_2011.csv", index = False)

# councildist_2023_geographies_puma_2023 = generate_new_estimates(2023, dict20, "councildist", API_KEY, "total_pop_E", "total_households_E", 2023)
# councildist_2023_geographies_puma_2023.to_csv("councildist_2023-geographies_puma_2023.csv", index = False)
# councildist_2023_geographies_puma_2021 = generate_new_estimates(2021, dict20, "councildist", API_KEY, "total_pop_E", "total_households_E", 2023)
# councildist_2023_geographies_puma_2021.to_csv("councildist_2023-geographies_puma_2021.csv", index = False)
# councildist_2023_geographies_puma_2016 = generate_new_estimates(2016, dict16, "councildist", API_KEY, "total_pop_E", "total_households_E", 2023)
# councildist_2023_geographies_puma_2016.to_csv("councildist_2023-geographies_puma_2016.csv", index = False)
# councildist_2023_geographies_puma_2011 = generate_new_estimates(2011, dict11, "councildist", API_KEY, "total_pop_E", "total_households_E", 2023)
# councildist_2023_geographies_puma_2011.to_csv("councildist_2023-geographies_puma_2011.csv", index = False)

# councildist_2013_geographies_puma_2023 = generate_new_estimates(2023, dict20, "councildist", API_KEY, "total_pop_E", "total_households_E", 2013)
# councildist_2013_geographies_puma_2023.to_csv("councildist_2013-geographies_puma_2023.csv", index = False)
# councildist_2013_geographies_puma_2021 = generate_new_estimates(2021, dict20, "councildist", API_KEY, "total_pop_E", "total_households_E", 2013)
# councildist_2013_geographies_puma_2021.to_csv("councildist_2013-geographies_puma_2021.csv", index = False)
# councildist_2013_geographies_puma_2016 = generate_new_estimates(2016, dict16, "councildist", API_KEY, "total_pop_E", "total_households_E", 2013)
# councildist_2013_geographies_puma_2016.to_csv("councildist_2013-geographies_puma_2016.csv", index = False)
# councildist_2013_geographies_puma_2011 = generate_new_estimates(2011, dict11, "councildist", API_KEY, "total_pop_E", "total_households_E", 2013)
# councildist_2013_geographies_puma_2011.to_csv("councildist_2013-geographies_puma_2011.csv", index = False)

# modzcta_geographies_puma_2023 = generate_new_estimates(2023, dict20, "modzcta", API_KEY, "total_pop_E", "total_households_E")
# modzcta_geographies_puma_2023.to_csv("modzcta-geographies_puma_2023.csv", index = False)
# modzcta_geographies_puma_2021 = generate_new_estimates(2021, dict20, "modzcta", API_KEY, "total_pop_E", "total_households_E")
# modzcta_geographies_puma_2021.to_csv("modzcta-geographies_puma_2021.csv", index = False)
# modzcta_geographies_puma_2016 = generate_new_estimates(2016, dict16, "modzcta", API_KEY, "total_pop_E", "total_households_E")
# modzcta_geographies_puma_2016.to_csv("modzcta-geographies_puma_2016.csv", index = False)
# modzcta_geographies_puma_2011 = generate_new_estimates(2011, dict11, "modzcta", API_KEY, "total_pop_E", "total_households_E")
# modzcta_geographies_puma_2011.to_csv("modzcta-geographies_puma_2011.csv", index = False)

# nta_geographies_puma_2023 = generate_new_estimates(2023, dict20, "nta", API_KEY, "total_pop_E", "total_households_E")
# nta_geographies_puma_2023.to_csv("nta-geographies_puma_2023.csv", index = False)
# nta_geographies_puma_2021 = generate_new_estimates(2021, dict20, "nta", API_KEY, "total_pop_E", "total_households_E")
# nta_geographies_puma_2021.to_csv("nta-geographies_puma_2021.csv", index = False)
# nta_geographies_puma_2016 = generate_new_estimates(2016, dict16, "nta", API_KEY, "total_pop_E", "total_households_E")
# nta_geographies_puma_2016.to_csv("nta-geographies_puma_2016.csv", index = False)
# nta_geographies_puma_2011 = generate_new_estimates(2011, dict11, "nta", API_KEY, "total_pop_E", "total_households_E")
# nta_geographies_puma_2011.to_csv("nta-geographies_puma_2011.csv", index = False)

# policeprct_geographies_puma_2023 = generate_new_estimates(2023, dict20, "policeprct", API_KEY, "total_pop_E", "total_households_E")
# policeprct_geographies_puma_2023.to_csv("policeprct-geographies_puma_2023.csv", index = False)
# policeprct_geographies_puma_2021 = generate_new_estimates(2021, dict20, "policeprct", API_KEY, "total_pop_E", "total_households_E")
# policeprct_geographies_puma_2021.to_csv("policeprct-geographies_puma_2021.csv", index = False)
# policeprct_geographies_puma_2016 = generate_new_estimates(2016, dict16, "policeprct", API_KEY, "total_pop_E", "total_households_E")
# policeprct_geographies_puma_2016.to_csv("policeprct-geographies_puma_2016.csv", index = False)
# policeprct_geographies_puma_2011 = generate_new_estimates(2011, dict11, "policeprct", API_KEY, "total_pop_E", "total_households_E")
# policeprct_geographies_puma_2011.to_csv("policeprct-geographies_puma_2011.csv", index = False)

# schooldist_geographies_puma_2023 = generate_new_estimates(2023, dict20, "schooldist", API_KEY, "total_pop_E", "total_households_E")
# schooldist_geographies_puma_2023.to_csv("schooldist-geographies_puma_2023.csv", index = False)
# schooldist_geographies_puma_2021 = generate_new_estimates(2021, dict20, "schooldist", API_KEY, "total_pop_E", "total_households_E")
# schooldist_geographies_puma_2021.to_csv("schooldist-geographies_puma_2021.csv", index = False)
# schooldist_geographies_puma_2016 = generate_new_estimates(2016, dict16, "schooldist", API_KEY, "total_pop_E", "total_households_E")
# schooldist_geographies_puma_2016.to_csv("schooldist-geographies_puma_2016.csv", index = False)
# schooldist_geographies_puma_2011 = generate_new_estimates(2011, dict11, "schooldist", API_KEY, "total_pop_E", "total_households_E")
# schooldist_geographies_puma_2011.to_csv("schooldist-geographies_puma_2011.csv", index = False)

def get_councilcount_estimates(PUMS_year, geo, var_codes="all", boundary_year=None):
    """
    Retrieve demographic estimates by specified geography, PUMS year, and boundary year (if applicable). Pulls from the existing 
    database used to support the CouncilCount website.

    Parameters:
    ----------
        PUMS_year : int/str
            Desired 5-Year PUMS year (e.g., "2021" for the 2017-2021 5-Year PUMS).
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
            Table with estimates for the specified geography, PUMS year, and boundary_year (if applicable). 

    Notes:
    ------
        - All variables are taken from 5-Year PUMS data dictionaries, which can be found here:
        https://api.census.gov/data/{INSERT YEAR}/acs/acs5/pums.html at the "variables" hyperlinks. 
        - Codes ending with 'E' and 'M' represent numerical estimates and margins of error, respectively, while codes ending with
        'PE' and 'PM' represent percent estimates and margins of error, respectively. Codes ending with 'V' represent 
        coefficients of variation. 
        - Data for geographies available within existing census hierarchy are taken from the PUMS. All other data are estimates
        generated by the NYC Council Data Team. Contact datainfo@council.nyc.gov with questions.
        - To generate estimates that do not already exist, use `generate_new_estimates()`.
    """
    # consistent dtypes
    if PUMS_year: PUMS_year = int(PUMS_year) 
    if boundary_year: boundary_year = int(boundary_year)

    if PUMS_year > 2020:
        demo_dict = dict20
    elif PUMS_year == 2016:
        demo_dict = dict16
    elif PUMS_year == 2011:
        demo_dict = dict11
    
    # locate available CSV files
    file_names = os.listdir(DATA_PATH)
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

        # # preparing to access files with boundary year in name (file name example: councildist-goegraphies_b23_2023.csv)
        add_boundary_year = f'_{boundary_year}' if boundary_year != None else ''

        # building paths
        if geo == 'city':
            file_path = f'{DATA_PATH}/nyc_wide_estimates_puma_{PUMS_year}.csv'
        else:
            file_path = f'{DATA_PATH}/{geo}{add_boundary_year}-geographies_puma_{PUMS_year}.csv'#.geojson'

        geo_df = pd.read_csv(file_path)

        # if list of variable codes requested, subset
        if var_codes == 'all': 
            return geo_df
        
        else:
            # identify geography column dynamically
            geo_col = [col for col in geo_df.columns if col.startswith(f"{geo}")]

            if not geo_col:
                raise ValueError("No geography column found in dataset.")
            
            master_col_list = geo_col.copy()

            for var_code in var_codes:
                expanded_codes = _expand_var_codes(var_code, demo_dict,geo_df, "_E")

                if not expanded_codes:
                    raise ValueError(
                        f"No matching expanded variables found for '{var_code}'. "
                        "Check for typos or available codes."
                    )

                for base_code in expanded_codes:
                    base = base_code[:-2]  # remove "_E"

                    suffix_cols = [
                        f"{base}_E",
                        f"{base}_M",
                        f"{base}_PE",
                        f"{base}_PM",
                        f"{base}_V"
                    ]

                    # Only add columns that actually exist
                    existing = [col for col in suffix_cols if col in geo_df.columns]
                    master_col_list.extend(existing)

            # Remove duplicates while preserving order
            master_col_list = list(dict.fromkeys(master_col_list))

            return geo_df[master_col_list] 

    # check input cases
    if PUMS_year is None:
        raise ValueError("`PUMS_year` parameter is required. Available options are:\n" +
                         ", ".join(map(str, available_years)))
    elif geo is None:
        raise ValueError("`geo` parameter is required. Available options are:\n" +
                         ", ".join(geo_names))
    elif (geo == "councildist") and ((boundary_year not in [2013, 2023]) | (boundary_year == None)):
        warn("`boundary_year` must be set to 2013 or 2023 when `geo` is 'councildist'. Defaulting to 2023.")
        boundary_year = 2023
        return read_geos(geo, boundary_year)
    elif PUMS_year not in available_years:
        raise ValueError(f"The PUMS year {PUMS_year} could not be found. Available options are:\n" +
                         ", ".join(map(str, available_years)))
    elif geo not in geo_names:
        raise ValueError(f"The geography '{geo}' could not be found. Available options are:\n" +
                         ", ".join(geo_names))
    elif (geo != "councildist") and (boundary_year is not None):
        warn("`boundary_year` is only relevant for `geo = councildist`. Ignoring `boundary_year` input.")
        return read_geos(geo)
    else:
        return read_geos(geo, boundary_year)
#