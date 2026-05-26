import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# This file uses the mismatch summary generated in compare_PUMS_ACS to visualize PUMS/ACS mismatches.

mismatch = pd.read_csv("/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/data/Mismatched Estimates.csv")

# --------------------------
# Heatmap of Accuracy by Geo
# --------------------------

# Number of geographies at each level
G_map = {
    "borough": 5,
    "schooldist": 32,
    "councildist_2013": 51,
    "councildist_2023": 51,
    "communitydist": 71,
    "policeprct": 77,
    "modzcta": 178,
    "nta": 262
}

# Number of variables estimated in each year
L_map = {2023: 176, 2016: 173, 2011: 148}

# full Geo × Year grid
grid = pd.MultiIndex.from_product(
    [G_map.keys(), L_map.keys()],
    names=["Geo", "Year"]
).to_frame(index=False)

errors = mismatch.groupby(["Geo", "Year"]).size().reset_index(name="E")
# merge to include zero-error combos
df = grid.merge(errors, on=["Geo","Year"], how="left").fillna({"E":0})

# attach constants
df["G"] = df["Geo"].map(G_map)
df["L"] = df["Year"].map(L_map)

df["Accuracy"] = ((1 - (df["E"] / (df["G"] * df["L"]))) * 100).round(2)
df["possible"] = df["G"] * df["L"]

geo_totals = df.groupby("Geo").agg(E=("E","sum"), possible=("possible","sum"))
geo_totals["All Years"] = ((1 - geo_totals["E"] / geo_totals["possible"]) * 100).round(2)
geo_totals = geo_totals[["All Years"]]

year_totals = df.groupby("Year").agg(E=("E","sum"), possible=("possible","sum"))
year_totals["Accuracy"] = ((1 - year_totals["E"] / year_totals["possible"]) * 100).round(2)

overall_accuracy = ((1 - df["E"].sum() / df["possible"].sum()) * 100).round(2)

# heatmap table
heatmap_df = df.pivot(index="Geo", columns="Year", values="Accuracy")
heatmap_df = heatmap_df.join(geo_totals)
heatmap_df.loc["All Geographies"] = year_totals["Accuracy"]
heatmap_df.loc["All Geographies","All Years"] = overall_accuracy

totals_row = heatmap_df.loc[["All Geographies"]]
main_rows = heatmap_df.drop(index="All Geographies")
main_rows = main_rows.sort_values("All Years", ascending=False)
heatmap_df_sorted = pd.concat([main_rows, totals_row])

# plot heatmap
sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(
    heatmap_df_sorted,
    annot=heatmap_df_sorted.applymap(lambda x: f"{x:.1f}%"),
    fmt="",
    cmap="RdYlGn",
    linewidths=.5,
    linecolor="white",
    vmin=75,
    vmax=100,
    ax=ax
)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(top=True, bottom=False)

n_rows, n_cols = heatmap_df_sorted.shape
ax.hlines(n_rows-1, *ax.get_xlim(), colors="black", linewidth=2)
ax.vlines(n_cols-1, *ax.get_ylim(), colors="black", linewidth=2)

ax.set_title("Accuracy of PUMS-Backed Estimates Compared to ACS", fontsize=14, pad=12)
ax.set_xlabel("")
ax.set_ylabel("Geography")

plt.tight_layout()
plt.show()

# --------------------------
# Example Comparison
# --------------------------

row = mismatch[
    (mismatch["Geo_ID"] == "33") &
    (mismatch["PUMS_Column"] == "RACE_HISPWhite alone, not Hispanic or Latino_E")
].iloc[0]

pums_est = row["PUMS_Estimate"]
pums_moe = row["PUMS_MOE"]

acs_est = row["ACS_Estimate"]
acs_moe = row["ACS_MOE"]

pums_low  = pums_est - pums_moe
pums_high = pums_est + pums_moe

acs_low  = acs_est - acs_moe
acs_high = acs_est + acs_moe

fig, ax = plt.subplots(figsize=(6,5))

x_positions = [0, 1]

# CI bars
ax.fill_between([x_positions[0]-0.2, x_positions[0]+0.2], pums_low, pums_high, alpha=0.35, label="PUMS 90% CI")
ax.fill_between([x_positions[1]-0.2, x_positions[1]+0.2], acs_low, acs_high, alpha=0.35, label="ACS 90% CI")

# central estimates
ax.scatter(x_positions[0], pums_est, s=80, zorder=3)
ax.scatter(x_positions[1], acs_est, s=80, zorder=3)

# formatting
ax.set_xticks(x_positions)
ax.set_xticklabels(["PUMS", "ACS"])
ax.set_ylabel("Estimate")
ax.set_title("Confidence Interval Comparison\nCD 33 Non-Hispanic White Pop.")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.show()

# ------------------
# Accuracy Over Time
# ------------------

line_df = df[['Geo', 'Year', 'Accuracy']].copy()

overall_acc = df.groupby('Year').apply(
    lambda x: (1 - x['E'].sum() / (x['G'] * x['L']).sum()) * 100
).reset_index(name='Accuracy')
overall_acc['Geo'] = 'All Geos'

line_df = pd.concat([line_df, overall_acc], ignore_index=True)
geo_lines = line_df[line_df['Geo'] != 'All Geos']
overall_line = line_df[line_df['Geo'] == 'All Geos']
geo_order = geo_lines[geo_lines['Year'] == 2023].sort_values('Accuracy', ascending=False)['Geo'].tolist()

plt.figure(figsize=(10,6))

# plot individual geographies in order
sns.lineplot(
    data=geo_lines,
    x='Year',
    y='Accuracy',
    hue='Geo',
    hue_order=geo_order,
    marker='o',
    linewidth=2,
    alpha=0.7,
    palette='tab10'
)

# plot overall accuracy as thick dashed line
sns.lineplot(
    data=overall_line,
    x='Year',
    y='Accuracy',
    color='black',
    marker='o',
    linewidth=3,
    linestyle='--',
    label='All Geos'
)

plt.xticks([2011, 2016, 2023])
plt.title("PUMS vs ACS Accuracy Over Time by Geography")
plt.ylabel("Accuracy (%)")
plt.ylim(75, 100)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(title='Geography', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --------------------
# Accuracy Across Geos
# --------------------

year = 2023
L = L_map[year]

# errors/accuracy per Geo_ID
geo_errors = mismatch[mismatch["Year"] == year].groupby(["Geo", "Geo_ID"]).size().reset_index(name="E")
geo_errors["Accuracy"] = (1 - (geo_errors["E"] / L)) * 100

def plot_accuracy_map(geo):
    gdf = gpd.read_file(f"/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/data/{geo}-boundaries.geojson")
    gdf = gdf.rename(columns={geo: "Geo_ID"})

    geo_df = geo_errors[geo_errors["Geo"] == geo]

    gdf["Geo_ID"] = gdf["Geo_ID"].astype(str)
    geo_df["Geo_ID"] = geo_df["Geo_ID"].astype(str)

    gdf = gdf.merge(geo_df[["Geo_ID","Accuracy"]], on="Geo_ID", how="left")
    gdf["Accuracy"] = gdf["Accuracy"].fillna(100)
    
    fig, ax = plt.subplots(figsize=(8,8))

    gdf.plot(
        column="Accuracy",
        cmap="RdYlGn",
        vmin=75,
        vmax=100,
        linewidth=0.3,
        edgecolor="black",
        legend=True,
        legend_kwds={"label": "Accuracy (%)"},
        ax=ax
    )
    ax.set_title(f"{geo.replace('_',' ').title()} Accuracy (2023)")
    ax.axis("off")

    plt.tight_layout()
    plt.show()

for geo in G_map.keys():
    plot_accuracy_map(geo)

# ----------------------------------
# Most commonly mismatched variables
# ----------------------------------

geo_tables = {
    geo: (
        mismatch[mismatch["Geo"] == geo]
        .groupby("PUMS_Column")
        .size()
        .sort_values(ascending=False)
        .to_frame("Count")
    )
    for geo in mismatch["Geo"].unique()
}
for g in G_map.keys():
    print(g)
    print(geo_tables[g].head(10))