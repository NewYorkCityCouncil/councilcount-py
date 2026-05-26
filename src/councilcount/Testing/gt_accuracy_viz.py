import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# ==========================================================
# Mismatch visualization script
# Uses:
#   - Mean Absolute Percentage Error
#   - Weighted Absolute Percentage Error
# ==========================================================
mismatch = pd.read_csv("/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/Testing/New Ground Truth Comparison.csv")
mismatch= mismatch.rename(columns={"APE": "Percent_Error"})

# ---------------------
# wape helper
# ---------------------
def wape(df):
    num = np.abs(df["DHC_Estimate"] - df["Estimate"]).sum()
    den = np.abs(df["DHC_Estimate"]).sum()
    return 100 * num / den if den != 0 else np.nan

# ---------------------
# MAPE summary
# ---------------------
def summarize_mape(df):
    return pd.Series({
        "MAPE": df["Percent_Error"].mean()
    })

# =========================
# HEATMAP (collapsed)
# =========================
def create_heatmap(df, title):

    heat = (
        df
        .groupby(["Source", "Geo", "Year"])
        .apply(lambda d: pd.Series({
            "MAPE": d["Percent_Error"].mean()
        }))
        .reset_index()
    )

    heat = heat[heat["Year"] == 2020]

    # overall row
    overall = (
        df[df["Year"] == 2020]
        .groupby("Source")
        .apply(lambda d: pd.Series({
            "MAPE": d["Percent_Error"].mean()
        }))
        .reset_index()
    )
    overall["Geo"] = "overall"

    heat = pd.concat([heat, overall], ignore_index=True)

    mat = (
        heat
        .pivot(index="Source", columns="Geo", values="MAPE")
        [["assembly", "congress", "overall"]]
    )

    mat.columns = ["Assembly Districts", "Congressional Districts", "Overall"]
    mat.index = ["ACS-Backed", "PUMS-Backed"]

    fig, ax = plt.subplots(figsize=(7, 3))

    sns.heatmap(
        mat,
        annot=mat.applymap(lambda x: f"{x:.1f}%"),
        fmt="",
        cmap="RdYlGn_r",
        vmin=10,
        vmax=20,
        linewidths=.5,
        linecolor="white",
        ax=ax
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.show()

create_heatmap(mismatch, "Councilcount MAPE Compared to Decennial Census Ground Truth")

# --------------------
# Map Across Geos
# --------------------

geo_metrics = (
    mismatch[mismatch["Year"] == 2020]
    .groupby(["Source", "Geo", "Geo_ID"])
    .apply(summarize_mape)
    .reset_index()
)

def load_geo_shape(geo):

    if geo == "assembly":
        gdf = gpd.read_file("assembly display.geojson")
        gdf = gdf.rename(columns={"assembly district": "Geo_ID"})

    else:
        gdf = gpd.read_file("congressional display.geojson")
        gdf = gdf.rename(columns={"congressional district": "Geo_ID"})

    gdf["Geo_ID"] = gdf["Geo_ID"].astype(str)

    return gdf


def plot_map(geo_metrics_df, source, geo, title_suffix=""):

    gdf = load_geo_shape(geo)

    df = geo_metrics_df[
        (geo_metrics_df["Source"] == source) &
        (geo_metrics_df["Geo"] == geo)
    ].copy()

    df["Geo_ID"] = df["Geo_ID"].astype(str)
    gdf = gdf.merge(df, on="Geo_ID", how="left")

    fig, ax = plt.subplots(figsize=(8, 6))

    gdf.plot(
        column="MAPE",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=32,
        linewidth=0.35,
        edgecolor="black",
        legend=True,
        ax=ax
    )

    if geo == "assembly":
        ax.set_title(f"{source}-Backed Assembly District MAPE {title_suffix}")
    else:
        ax.set_title(f"{source}-Backed Congressional District MAPE {title_suffix}")
    ax.axis("off")

    plt.show()

for source in ["ACS", "PUMS"]:
    for geo in ["assembly", "congress"]:
        plot_map(geo_metrics, source, geo)

# ----------------------------------
# Most commonly mismatched variables
# ----------------------------------

results = {}

for geo in mismatch["Geo"].unique():
    for source in mismatch["Source"].unique():

        df = mismatch[
            (mismatch["Geo"] == geo) &
            (mismatch["Source"] == source)
        ]

        table = df.groupby("Column").apply(
            lambda x: pd.Series({
                "MAPE": x["Percent_Error"].mean(),
                "WAPE": wape(x)
            })
        )

        # keep worst offenders
        table = table.sort_values("WAPE", ascending=False)
        table = table[(table[["MAPE", "WAPE"]] > 20).any(axis=1)]

        results[(geo, source)] = table

# print results
for (geo, source), table in results.items():
    print(f"\n=== {geo.upper()} | {source} ===")
    print(table.round(2))


# Most commonly mismatched variable categories

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

def categorize_pums(var):
    if var.startswith("TEN"):
        return "Housing Tenure"
    elif var.startswith("SEX"):
        return "Sex"
    elif var.startswith("AGEP"):
        return "Age"
    elif var.startswith("RAC1P"):
        return "Race"
    elif var.startswith("HISP"):
        return "Hispanic Origin"
    elif var.startswith("RACE_HISP"):
        return "Race & Hispanic Origin"
    elif var.startswith("NP"):
        return "Household Size"
    elif var.startswith("R181"):
        return "Under 18 in Household"
    elif var.startswith("R651"):
        return "Over 65 in Household"
    elif var.startswith("total_pop"):
        return "Total Population"
    else:
        return pd.NA
    
acs_to_pums = {v: k for k, v in pums_to_acs.items()}

def categorize_acs(var):
    if var[:-1] not in acs_to_pums:
        return pd.NA
    return categorize_pums(acs_to_pums[var[:-1]])

mismatch["PUMS_Category"] = mismatch["Column"].apply(categorize_pums)
mismatch["ACS_Category"]  = mismatch["Column"].apply(categorize_acs)

pums_results = (
    mismatch[mismatch["Source"] == "PUMS"]
    .groupby(["Geo", "Year", "PUMS_Category"])
    .apply(lambda df: pd.Series({
        "MAPE": df["Percent_Error"].mean(),
        "WAPE": wape(df)
    }))
    .reset_index()
)

acs_results = (
    mismatch[mismatch["Source"] == "ACS"]
    .groupby(["Geo", "Year", "ACS_Category"])
    .apply(lambda df: pd.Series({
        "MAPE": df["Percent_Error"].mean(),
        "WAPE": wape(df)
    }))
    .reset_index()
)

def plot_category_bars(df, geo, title):

    # keep only valid categories
    d = df[df["Geo"] == geo].dropna(subset=["MAPE", "WAPE"]).copy()

    # sort by WAPE descending
    d = d.sort_values("WAPE", ascending=False)

    if "PUMS_Category" in d.columns:
        categories = d["PUMS_Category"]
    else:
        categories = d["ACS_Category"]

    x = np.arange(len(d))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(x - width/2, d["MAPE"], width, label="MAPE")
    ax.bar(x + width/2, d["WAPE"], width, label="WAPE")

    ax.axhline(20, color="red", linestyle="dotted", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")

    ax.set_ylabel("Error (%)")
    if geo == "assembly":
        ax.set_title(f"{title} - Assembly Districts")
    else:
        ax.set_title(f"{title} - Congressional Districts")

    ax.legend()
    plt.tight_layout()
    plt.show()

plot_category_bars(pums_results, "assembly", "PUMS Variable Categories: MAPE vs WAPE")
plot_category_bars(pums_results, "congress", "PUMS Variable Categories: MAPE vs WAPE")
plot_category_bars(acs_results, "assembly", "ACS Variable Categories: MAPE vs WAPE")
plot_category_bars(acs_results, "congress", "ACS Variable Categories: MAPE vs WAPE")

# Clearly race is the biggest detractor of accuracy. Let's make some supplemental figures showing the non-racial predictions.
race_categories = ["Race", "Race & Hispanic Origin"]

mismatch_no_race = mismatch[
    ~mismatch["PUMS_Category"].isin(race_categories).fillna(False) &
    ~mismatch["ACS_Category"].isin(race_categories).fillna(False)
].copy()
# We've removed 35% of all comparisons, the following results are just a thought exercise...

create_heatmap(mismatch_no_race, "MAPE vs DHC (Race Variables Removed)")

geo_metrics_no_race = (
    mismatch_no_race[mismatch_no_race["Year"] == 2020]
    .groupby(["Source", "Geo", "Geo_ID"])
    .apply(summarize_mape)
    .reset_index()
)

for source in ["ACS", "PUMS"]:
    for geo in ["assembly", "congress"]:
        plot_map(geo_metrics_no_race, source, geo, "(Race Variables Removed)")