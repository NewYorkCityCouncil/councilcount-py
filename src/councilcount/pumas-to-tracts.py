from importlib.resources import files
import pandas as pd

# This file uses crosswalks of Census Tracts and Public Use Microdata Areas to assign BBL units to PUMAs. It also updates borough identifiers to be compatible with other documents in the pipeline.
path  = "/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/data"

borough_map = {'047': 'BK', '005': 'BX', '061': 'MN', '081': 'QN', '085': 'SI'}

# Prepare crosswalks
def prep_crosswalk(year):
    df = pd.read_csv(f'{path}/{year}_Census_Tract_to_{year}_PUMA.txt', dtype = {'STATEFP': str, 'COUNTYFP': str, 'TRACTCE': int, 'PUMA5CE': str})
    df[f'ct{year}'] = df['TRACTCE'].astype(str) # read in as int and convert to str to strip leading zeroes
    df['PUMA5CE'] = df['PUMA5CE'].str.zfill(5)

    df = df[(df['STATEFP'] == '36') & df['COUNTYFP'].isin(borough_map.keys())] # filter for NYC tracts/PUMAs
    df['borough'] = df['COUNTYFP'].map(borough_map)

    df = df.drop(columns = ['STATEFP', 'COUNTYFP', 'TRACTCE'])
    df = df.rename(columns={'PUMA5CE': f'puma{year}'})
    return df

cw_2010 = prep_crosswalk('2010')
cw_2020 = prep_crosswalk('2020')

# Merge 2010 and 2020 PUMAs into existing population estimates
def merge_with_crosswalk(est_year):
    bbl = pd.read_csv(f"{path}/bbl-population-estimates_{est_year}.csv", dtype={"borough": str, "ct2010": str, "ct2020": str})
    
    merged = bbl.merge(cw_2010, on=["borough", "ct2010"], how="left", validate="many_to_one").merge(cw_2020, on=["borough", "ct2020"], how="left", validate="many_to_one")
    merged["puma2010"] = merged["puma2010"].astype('string').str.strip()
    merged["puma2020"] = merged["puma2020"].astype('string').str.strip()
    merged['borough'] = merged['borough'].replace({'BK': 'Brooklyn', 'BX': 'The Bronx', 'MN': 'Manhattan', 'QN': 'Queens', 'SI': 'Staten Island'})

    return merged

bbl_population = {
    # 2011: merge_with_crosswalk('2011'),
    # 2016: merge_with_crosswalk('2016'),
    2020: merge_with_crosswalk('2020'),
    # 2021: merge_with_crosswalk('2021'),
    # 2023: merge_with_crosswalk('2023'),
}

for year, df in bbl_population.items():
    # Check for missing PUMAs
    puma_cols = [col for col in df.columns if col.startswith("puma")]
    missingdf = df[df[puma_cols].isnull().any(axis=1)]
    print(f"The following BBL estimates from {year} are missing PUMAs:")
    print(missingdf[['borough', 'block', 'lot', '2010_tract_id', 'bbl_population_estimate', 'puma2010', 'puma2020']])
    print(f"This accounts for {sum(missingdf["unitsres"])} households and {sum(missingdf["bbl_population_estimate"])} people.")
# Missing PUMAs only occur outside of the census year assigned to the estimates: only 2020 PUMAs are missing for 2011/16 and only 2010 PUMAs are missing for 2021/22/23.
# This may pose a minor issue for 2021 and 2022 estimates, which entirely or partially rely on 2010 PUMAs. It may be the case that ~700 households and ~1,300-1,500 people are excluded from estimates in these years.
# This accounts for roughly 0.02% of households and people.

# Manually fill in missing PUMAs for populated BBLs in 2020-2022 (this was done by looking up the addresses of the BBLs and comparing to a map of 2010 PUMAs)
# The populated BBLs missing 2010 PUMAs are identical in both years.
manual_fixes = {"23801-81": "04109", "24200-5": "03710"}
bbl_population[2020]["puma2010"] = bbl_population[2020]["puma2010"].fillna(bbl_population[2020]["2010_tract_id"].map(manual_fixes))
# bbl_population[2021]["puma2010"] = bbl_population[2021]["puma2010"].fillna(bbl_population[2021]["2010_tract_id"].map(manual_fixes))

# Make sanity checks, clean, and write new files

for year, df in bbl_population.items():
    
    # Check that no populated BBLs are missing either PUMA (2021, 2022) or both PUMAs (other years)
    populated = (df['unitsres'] > 0) | (df['bbl_population_estimate'] > 0)

    if year in (2021, 2022):
        missing_puma = populated & (df['puma2010'].isna() | df['puma2020'].isna())
    else: 
        missing_puma = populated & (df['puma2010'].isna() & df['puma2020'].isna())
    assert not missing_puma.any()

    assert df['puma2010'].nunique() == 55
    assert df['puma2020'].nunique() == 55

    # Remove leading zero from 2010 PUMAs (no longer in census API)
    df["puma2010"] = df["puma2010"].astype(str).str.lstrip("0")

    df.to_csv(f'{path}/puma-bbl-population-estimates_{year}.csv', index=False)