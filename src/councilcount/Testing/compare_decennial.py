import os
import pandas as pd
import numpy as np
from importlib.resources import files
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/Testing"
#DATA_PATH = files("councilcount").joinpath("data")

# This file uses Congressional and State Assembly District data from the 2020 decennial census to check whether PUMS-backed and ACS-backed estimates' 90% MOEs align with ground truth.

# Create a dictionary of PUMS variables and their ACS equivalents
pums_to_acs = {
    "R181_": "DP02_0014", "R651_": "DP02_0015",
    "TEN1_": "DP04_0110", "TEN3_": "DP04_0136",
    "SEX1_": "DP05_0002", "SEX2_": "DP05_0003",
    "AGEPUnder 5 years_": "DP05_0005", "AGEP5-9 years_": "DP05_0006", "AGEP10-14 years_": "DP05_0007", "AGEP15-19 years_": "DP05_0008", "AGEP20-24 years_": "DP05_0009", 
    "AGEP25-34 years_": "DP05_0010", "AGEP35 to 44 years_": "DP05_0011", "AGEP45 to 54 years_": "DP05_0012", "AGEP55 to 59 years_": "DP05_0013", "AGEP60 to 64 years_": "DP05_0014", 
    "AGEP65 to 74 years_": "DP05_0015", "AGEP75 to 84 years_": "DP05_0016", "AGEP85 years and over_": "DP05_0017",
    "RAC1P9_": "DP05_0035", "RAC1P1_": "DP05_0037", "RAC1P2_": "DP05_0038", "RAC1P3_": "DP05_0039", "RAC1P6_": "DP05_0044", "RAC1P7_": "DP05_0052", "RAC1P8_": "DP05_0057",
    "HISP1_": "DP05_0071", "HISP2_": "DP05_0076", "RACE_HISPWhite alone, not Hispanic or Latino_": "DP05_0077", "RACE_HISPBlack or African American alone, not Hispanic or Latino_": "DP05_0078", 
    "RACE_HISPAmerican Indian or Alaska Native alone, not Hispanic or Latino_": "DP05_0079", "RACE_HISPAsian alone, not Hispanic or Latino_": "DP05_0080", 
    "RACE_HISPNative Hawaiian and Other Pacific Islander alone, not Hispanic or Latino_": "DP05_0081", "RACE_HISPSome Other Race alone, not Hispanic or Latino_": "DP05_0082", 
    "RACE_HISPTwo or More Races, not Hispanic or Latino_": "DP05_0083",
    
    "total_pop_": "B01001_001",
    "NP1_": "B08201_007", "NP2_": "B08201_013", "NP3_": "B08201_019", "NP4+_": "B08201_025"
}

# Create a dictionary of PUMS variables and their DHC equivalents
pums_to_dhc = {
    "R181_": "P21_002N", "R651_": "P19_002N",
    "TEN1_": "H4_002N", "TEN3_": "H4_004N",
    "SEX1_": "P12_002N", "SEX2_": "P12_026N",
    "AGEPUnder 5 years_": "P12_003N", "AGEP5-9 years_": "P12_004N", "AGEP10-14 years_": "P12_005N", "AGEP15-19 years_": "P12_006N", "AGEP20-24 years_": "P12_008N", 
    "AGEP25-34 years_": "P12_011N", "AGEP35 to 44 years_": "P12_013N", "AGEP45 to 54 years_": "P12_015N", "AGEP55 to 59 years_": "P12_017N", "AGEP60 to 64 years_": "P12_018N", 
    "AGEP65 to 74 years_": "P12_020N", "AGEP75 to 84 years_": "P12_023N", "AGEP85 years and over_": "P12_025N",
    "RAC1P9_": "P3_008N", "RAC1P1_": "P3_002N", "RAC1P2_": "P3_003N", "RAC1P3_": "P3_004N", "RAC1P6_": "P3_005N", "RAC1P7_": "P3_006N", "RAC1P8_": "P3_007N",
    "HISP1_": "P4_003N", "HISP2_": "P4_002N", "RACE_HISPWhite alone, not Hispanic or Latino_": "P5_003N", "RACE_HISPBlack or African American alone, not Hispanic or Latino_": "P5_004N", 
    "RACE_HISPAmerican Indian or Alaska Native alone, not Hispanic or Latino_": "P5_005N", "RACE_HISPAsian alone, not Hispanic or Latino_": "P5_006N", 
    "RACE_HISPNative Hawaiian and Other Pacific Islander alone, not Hispanic or Latino_": "P5_007N", "RACE_HISPSome Other Race alone, not Hispanic or Latino_": "P5_008N", 
    "RACE_HISPTwo or More Races, not Hispanic or Latino_": "P5_009N",
    
    "total_pop_": "P1_001N",
    "NP1_": "H9_002N", "NP2_": "H9_003N", "NP3_": "H9_004N", "NP4+_": "H9_005N"
}

# Use these to back out a dictionary of ACS variables and their DHC equivalents
acs_to_dhc = {
    acs_var: pums_to_dhc[pums_var]
    for pums_var, acs_var in pums_to_acs.items()
    if pums_var in pums_to_dhc
}

def compare_to_dhc(estimate_df, dhc_df, crosswalk, geo_name, source_name="PUMS", id_col=None):
    violations_list = []

    if id_col is None:
        id_col = estimate_df.columns[0]

    merged_df = dhc_df.merge(estimate_df, on=id_col, how="inner")

    if merged_df.empty:
        print(f"No overlapping geographies for {geo_name}")
        return None
    
    # Determine expected estimate columns
    expected_est_cols = []
    for prefix in crosswalk:
        expected_est_cols.append(prefix + "E")

    missing_est_cols = [c for c in expected_est_cols if c not in estimate_df.columns]
    for col in missing_est_cols:
        print(f"{source_name} estimate column missing: {col}")

    # Identify estimate columns
    est_cols = [c for c in estimate_df.columns if c.endswith("E")]
    rows = []

    for col in est_cols:
        prefix = col[:-1]
        if prefix not in crosswalk:
            continue
        
        dhc_col = crosswalk[prefix]
        moe_col = prefix + "M"

        if moe_col not in estimate_df.columns:
            print(f"{source_name} MOE column missing: {moe_col}")
            continue
        if dhc_col not in dhc_df.columns:
            print(f"DHC column missing: {dhc_col}")
            continue
        
        # Previous error categorization method
        # est = merged_df[col]
        # moe = merged_df[moe_col]
        # dhc = merged_df[dhc_col]

        # lower = est - moe
        # upper = est + moe

        # outside_mask = (lower > dhc) | (upper < dhc)

        # if outside_mask.any():
        #     violations = pd.DataFrame({
        #         "Geo": geo_name,
        #         "Geo_ID": merged_df.loc[outside_mask, id_col],
        #         "Source": source_name,
        #         "Column": col,
        #         "Estimate": est[outside_mask],
        #         "MOE": moe[outside_mask],
        #         "DHC_Estimate": dhc[outside_mask],
        #         "Difference": est[outside_mask] - dhc[outside_mask],
        #     })
        #     violations_list.append(violations)
        # if violations_list:
        #     return pd.concat(violations_list, ignore_index=True)
        # else:
        #     return None

        # New error categorization methods
        est = merged_df[col]
        dhc = merged_df[dhc_col]

        smape = (2 * np.abs(est - dhc) / (np.abs(est) + np.abs(dhc))) * 100
        ape = np.abs((est - dhc) / dhc) * 100

        for i in range(len(merged_df)):
            rows.append({
                "Geo": geo_name,
                "Geo_ID": merged_df[id_col].iloc[i],
                "Source": source_name,
                "Column": col,
                "Estimate": est.iloc[i],
                "DHC_Estimate": dhc.iloc[i],
                "APE": ape.iloc[i],
                "SMAPE": smape.iloc[i],
            })

    return pd.DataFrame(rows)

all_violations = []
for geo in ["assembly", "congress"]:
    print(f"\nProcessing {geo}")

    acs_df = pd.read_csv(f"{DATA_PATH}/{geo}-geographies_acs_2020.csv")
    pums_df = pd.read_csv(f"{DATA_PATH}/{geo}-geographies_puma_2020.csv")
    dhc_df = pd.read_csv(f"{DATA_PATH}/{geo}-ground-truth.csv")

    # PUMS vs DHC
    pums_result = compare_to_dhc(
        estimate_df=pums_df,
        dhc_df=dhc_df,
        crosswalk=pums_to_dhc,
        geo_name=geo,
        source_name="PUMS"
    )

    # ACS vs DHC
    acs_result = compare_to_dhc(
        estimate_df=acs_df,
        dhc_df=dhc_df,
        crosswalk=acs_to_dhc,
        geo_name=geo,
        source_name="ACS",
    )

    if pums_result is not None:
        all_violations.append(pums_result)
    if acs_result is not None:
        all_violations.append(acs_result)

# Save!
mismatch = pd.concat(all_violations, ignore_index=True)
mismatch["Year"] = 2020
mismatch.to_csv("/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/Testing/New Ground Truth Comparison.csv", index=False)
