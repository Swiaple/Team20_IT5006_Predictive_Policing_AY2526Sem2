import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pandas.tseries.holiday import USFederalHolidayCalendar

LAT_COL, LON_COL = "Latitude", "Longitude"

st.set_page_config(page_title="Chicago Crime EDA", layout="wide")
st.title("Chicago Crime EDA Dashboard")
st.caption("Artifacts-only deployment: fast, reproducible, GitHub-friendly.")

ART_DIR = "artifacts"


# -----------------------------
# Load artifacts (cached)
# -----------------------------
@st.cache_data
def load_artifacts(base: str = ART_DIR):
    art = {
        "yearly": pd.read_csv(os.path.join(base, "agg_yearly.csv")),
        "monthly": pd.read_csv(os.path.join(base, "agg_monthly.csv")),
        "weekly": pd.read_csv(os.path.join(base, "agg_weekly.csv")),
        "daily": pd.read_csv(os.path.join(base, "agg_daily.csv")),
        "top_types": pd.read_csv(os.path.join(base, "top_types.csv")),
        "hourly_topN": pd.read_csv(os.path.join(base, "hourly_by_type_topN.csv")),
        "yearly_topN": pd.read_csv(os.path.join(base, "yearly_by_type_topN.csv")),
        "arrest_yearly": pd.read_csv(os.path.join(base, "arrest_rate_yearly.csv")),
        "arrest_yearly_topN": pd.read_csv(os.path.join(base, "arrest_rate_yearly_topN.csv")),
        "grid": pd.read_csv(os.path.join(base, "spatial_grid_precomputed.csv")),
    }

    # daily needs datetime
    art["daily"]["Date"] = pd.to_datetime(art["daily"]["Date"], errors="coerce")
    art["daily"] = art["daily"].dropna(subset=["Date"]).sort_values("Date")

    # optional sample points for interactive map
    sample_path = os.path.join(base, "sample_points.csv")
    pts = pd.read_csv(sample_path)
    # ensure numeric
    pts[LAT_COL] = pd.to_numeric(pts[LAT_COL], errors="coerce")
    pts[LON_COL] = pd.to_numeric(pts[LON_COL], errors="coerce")
    pts["Year"] = pd.to_numeric(pts["Year"], errors="coerce")
    pts["Month"] = pd.to_numeric(pts["Month"], errors="coerce")
    pts["Hour"] = pd.to_numeric(pts["Hour"], errors="coerce")
    art["points"] = pts.dropna(subset=[LAT_COL, LON_COL])

    return art


def filter_year(df: pd.DataFrame, year_range):
    if "Year" not in df.columns:
        return df
    return df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]


def mark_holidays_daily(daily_df: pd.DataFrame):
    # daily_df: columns Date, Total_Crimes
    out = daily_df.copy()
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=out["Date"].min(), end=out["Date"].max())
    out["Is_Holiday"] = out["Date"].dt.normalize().isin(holidays)

    out["Is_Holiday_Window"] = False
    for h in holidays:
        window = pd.date_range(start=h - pd.Timedelta(days=2), end=h + pd.Timedelta(days=7))
        out.loc[out["Date"].dt.normalize().isin(window), "Is_Holiday_Window"] = True

    out["Period_Type"] = "Normal Day"
    out.loc[out["Is_Holiday_Window"], "Period_Type"] = "Holiday Window"
    out.loc[out["Is_Holiday"], "Period_Type"] = "Holiday Day"
    return out


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Mode")
    st.write("This deployment version uses precomputed artifacts in `artifacts/`.")
    st.divider()
    st.header("Location map")
    show_points_map = st.checkbox("Enable interactive point density map (needs artifacts/sample_points.csv)", value=True)

art = load_artifacts()

# Basic checks
needed = ["yearly", "monthly", "weekly", "daily", "top_types", "hourly_topN", "yearly_topN", "arrest_yearly", "arrest_yearly_topN", "grid"]
missing_files = [k for k in needed if k not in art or art[k] is None]
if missing_files:
    st.error(f"Missing required artifacts: {missing_files}. Make sure `artifacts/` contains all required CSVs.")
    st.stop()


# -----------------------------
# Pills
# -----------------------------
options = ["Time", "Category", "Location", "Arrest"]
selection = st.pills("Which aspect do you intend to know about?", options, selection_mode="multi")
selected = set(selection)

if not selected:
    st.info("Select at least one pill to start.")
    st.stop()

# Year slider from yearly artifact
year_min = int(art["yearly"]["Year"].min())
year_max = int(art["yearly"]["Year"].max())
year_range = (year_min, year_max)
if "Time" in selected:
    year_range = st.slider("Select Year Range", year_min, year_max, (year_min, year_max))

# Filter key artifacts by year
yearly = filter_year(art["yearly"], year_range)
monthly = art["monthly"]  # month is overall, not year-specific unless you build yearly-month artifact
weekly = art["weekly"]
daily = art["daily"]
hourly_topN = art["hourly_topN"]          # already topN types overall
yearly_topN = filter_year(art["yearly_topN"], year_range)
arrest_yearly = filter_year(art["arrest_yearly"], year_range)
arrest_yearly_topN = filter_year(art["arrest_yearly_topN"], year_range)
grid = art["grid"]
points = art["points"]

# -----------------------------
# Plot helpers
# -----------------------------
def mpl(fig):
    st.pyplot(fig, use_container_width=True)

def plot_year_trend(df_):
    fig = px.line(df_, x="Year", y="Total_Crimes", markers=True, title="Annual Crime Trend")
    st.plotly_chart(fig, use_container_width=True)

def plot_monthly(df_):
    fig = px.bar(df_.sort_values("Month"), x="Month", y="Total_Crimes", color = 'Month', title="Monthly Seasonality")
    st.plotly_chart(fig, use_container_width=True)

def plot_weekly(df_):
    fig = px.bar(df_.sort_values("DayNum"), x="DayOfWeek", y="Total_Crimes", color = 'DayOfWeek', title="Weekly Cycle")
    st.plotly_chart(fig, use_container_width=True)

def plot_top_types(df_):
    fig = px.bar(df_, x="Primary Type", y="Total_Crimes", color="Primary Type", title="Top Crime Types (precomputed)")
    st.plotly_chart(fig, use_container_width=True)

def plot_hourly_by_type(df_):
    fig = px.line(df_, x="Hour", y="Total_Crimes", color="Primary Type", markers=True,
                  title="When do specific crimes happen? (Top types)")
    st.plotly_chart(fig, use_container_width=True)

def plot_structure_over_time(df_):
    # df_: Year, Primary Type, Total_Crimes
    pivot = df_.pivot_table(index="Year", columns="Primary Type", values="Total_Crimes", fill_value=0)
    ratio = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ratio.plot(kind="area", stacked=True, ax=ax, alpha=0.8)
    ax.set_title("Crime Type Structure Over Time (stacked area, top types)")
    ax.set_ylabel("Proportion")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    mpl(fig)

def plot_arrest_rate(df_):
    # df_: Year, Arrest_Rate
    tmp = df_.copy()
    tmp["Arrest_Rate_%"] = tmp["Arrest_Rate"] * 100
    fig = px.line(tmp, x="Year", y="Arrest_Rate_%", markers=True, title="Arrest Rate by Year (%)")
    st.plotly_chart(fig, use_container_width=True)

def plot_arrest_rate_by_type(df_):
    tmp = df_.copy()
    tmp["Arrest_Rate_%"] = tmp["Arrest_Rate"] * 100
    fig = px.line(tmp, x="Year", y="Arrest_Rate_%", color="Primary Type", markers=True,
                  title="Arrest Rate by Year (Top types)")
    st.plotly_chart(fig, use_container_width=True)

def plot_holiday(daily_df):
    dfh = mark_holidays_daily(daily_df)
    # compare daily counts distributions
    comp = dfh.groupby("Period_Type")["Total_Crimes"].mean().reset_index()
    fig = px.bar(comp, x="Period_Type", y="Total_Crimes", title="Mean daily crimes: Holiday vs Normal")
    st.plotly_chart(fig, use_container_width=True)

def plot_moran(grid_df):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    hb = ax.hexbin(grid_df["z_standardized"], grid_df["lag"], gridsize=70, bins="log", mincnt=1, linewidths=0)
    fig.colorbar(hb, ax=ax, shrink=0.9).set_label("log10(count)")
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    I = float(grid_df["Moran_I_overall"].iloc[0]) if "Moran_I_overall" in grid_df.columns else None
    ax.set_title("Moran Scatter" + (f" (I={I:.3f})" if I is not None else ""))
    ax.set_xlabel("Standardized cell count (z)")
    ax.set_ylabel("Spatial lag (rook mean)")
    mpl(fig)

def plot_gistar(grid_df):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(grid_df["lon"], grid_df["lat"], c=grid_df["Gi_cat"], s=6)
    ax.set_title("Gi* Hotspot/Coldspot classes")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    mpl(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.scatter(grid_df["lon"], grid_df["lat"], c=grid_df["Gi_z"], s=6)
    ax2.set_title("Gi* z-scores")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    mpl(fig2)

def plot_location_map(points_df):
    if points_df is None:
        st.info("No sample_points.csv found. Location map disabled.")
        return
    # filter by year & crime filter if possible
    tmp = points_df.copy()
    tmp = tmp[(tmp["Year"] >= year_range[0]) & (tmp["Year"] <= year_range[1])]
    if crime_filter:
        tmp = tmp[tmp["Primary Type"].isin(crime_filter)]
    if tmp.empty:
        st.warning("No points left after filters.")
        return
    fig = px.density_mapbox(
        tmp,
        lat=LAT_COL, lon=LON_COL,
        radius=10,
        center=dict(lat=41.8781, lon=-87.6298),
        zoom=10,
        mapbox_style="carto-positron",
        hover_data=["Primary Type", "Location Description"]
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Rendering logic
# -----------------------------
if len(selected) == 1:
    only = next(iter(selected))

    if only == "Time":
        st.header("Time EDA")
        t1, t2, t3 = st.tabs(["Interactive basics", "Professional add-ons", "Holiday"])

        with t1:
            plot_year_trend(yearly)
            plot_monthly(monthly)
            plot_weekly(weekly)

        with t2:
            plot_hourly_by_type(hourly_topN_f)
            plot_structure_over_time(yearly_topN_f)

        with t3:
            plot_holiday(daily)

    elif only == "Category":
        st.header("Category EDA")
        c1, c2 = st.tabs(["Top types", "Structure & hourly (pro)"])
        with c1:
            plot_top_types(art["top_types"])
        with c2:
            crime_options = sorted(art["top_types"]["Primary Type"].unique().tolist())
            crime_filter = st.multiselect("Crime types (optional, affects type-based charts)", crime_options, default=crime_options[:5])
            if crime_filter:
                hourly_topN_f = hourly_topN[hourly_topN["Primary Type"].isin(crime_filter)]
                yearly_topN_f = yearly_topN[yearly_topN["Primary Type"].isin(crime_filter)]
                arrest_yearly_topN_f = arrest_yearly_topN[arrest_yearly_topN["Primary Type"].isin(crime_filter)]
            else:
                hourly_topN_f, yearly_topN_f, arrest_yearly_topN_f = hourly_topN, yearly_topN, arrest_yearly_topN
            plot_hourly_by_type(hourly_topN_f)
            plot_structure_over_time(yearly_topN_f)

    elif only == "Arrest":
        st.header("Arrest EDA")
        a1, a2 = st.tabs(["Overall arrest rate", "By type (top)"])
        with a1:
            plot_arrest_rate(arrest_yearly)
        with a2:
            crime_options = sorted(art["top_types"]["Primary Type"].unique().tolist())
            crime_filter = st.multiselect("Crime types (optional, affects type-based charts)", crime_options, default=crime_options[:5])
            if crime_filter:
                arrest_yearly_topN_f = arrest_yearly_topN[arrest_yearly_topN["Primary Type"].isin(crime_filter)]
            else:
                arrest_yearly_topN_f = arrest_yearly_topN
            plot_arrest_rate_by_type(arrest_yearly_topN_f)

    elif only == "Location":
        st.header("Location EDA")
        l1, l2 = st.tabs(["Interactive map (optional)", "Professional spatial stats"])
        with l1:
            crime_options = sorted(art["top_types"]["Primary Type"].unique().tolist())
            crime_filter = st.multiselect("Crime types (optional, affects type-based charts)", crime_options, default=crime_options[:5])
            if crime_filter:
                hourly_topN_f = hourly_topN[hourly_topN["Primary Type"].isin(crime_filter)]
                yearly_topN_f = yearly_topN[yearly_topN["Primary Type"].isin(crime_filter)]
                arrest_yearly_topN_f = arrest_yearly_topN[arrest_yearly_topN["Primary Type"].isin(crime_filter)]
            else:
                hourly_topN_f, yearly_topN_f, arrest_yearly_topN_f = hourly_topN, yearly_topN, arrest_yearly_topN
            if show_points_map:
                plot_location_map(points)
            else:
                st.info("Map disabled in sidebar.")
        with l2:
            plot_moran(grid)
            plot_gistar(grid)

elif len(selected) == 2:
    st.header("Combined EDA (2 aspects)")

    if "Time" in selected and "Category" in selected:
        st.subheader("Time × Category")
        crime_options = sorted(art["top_types"]["Primary Type"].unique().tolist())
        crime_filter = st.multiselect("Crime types (optional, affects type-based charts)", crime_options, default=crime_options[:5])
        if crime_filter:
            hourly_topN_f = hourly_topN[hourly_topN["Primary Type"].isin(crime_filter)]
            yearly_topN_f = yearly_topN[yearly_topN["Primary Type"].isin(crime_filter)]
        else:
            hourly_topN_f, yearly_topN_f = hourly_topN, yearly_topN
        plot_hourly_by_type(hourly_topN_f)
        plot_structure_over_time(yearly_topN_f)

    elif "Time" in selected and "Arrest" in selected:
        st.subheader("Time × Arrest")
        plot_arrest_rate(arrest_yearly)
        plot_arrest_rate_by_type(arrest_yearly_topN_f)

    elif "Category" in selected and "Arrest" in selected:
        st.subheader("Category × Arrest")
        # show arrest ranking by type (mean across years in range)
        crime_options = sorted(art["top_types"]["Primary Type"].unique().tolist())
        crime_filter = st.multiselect("Crime types (optional, affects type-based charts)", crime_options, default=crime_options[:5])
        if crime_filter:
            arrest_yearly_topN_f = arrest_yearly_topN[arrest_yearly_topN["Primary Type"].isin(crime_filter)]
        else:
            arrest_yearly_topN_f = arrest_yearly_topN
        tmp = arrest_yearly_topN_f.groupby("Primary Type")["Arrest_Rate"].mean().reset_index()
        tmp["Arrest_Rate_%"] = tmp["Arrest_Rate"] * 100
        fig = px.bar(tmp.sort_values("Arrest_Rate_%", ascending=False),
                     x="Primary Type", y="Arrest_Rate_%", color="Primary Type",
                     title="Mean arrest rate by type (selected years)")
        st.plotly_chart(fig, use_container_width=True)

    elif "Time" in selected and "Location" in selected:
        st.subheader("Time × Location")
        crime_options = sorted(art["top_types"]["Primary Type"].unique().tolist())
        crime_filter = st.multiselect("Crime types (optional, affects type-based charts)", crime_options, default=crime_options[:5])
        if crime_filter:
            hourly_topN_f = hourly_topN[hourly_topN["Primary Type"].isin(crime_filter)]
            yearly_topN_f = yearly_topN[yearly_topN["Primary Type"].isin(crime_filter)]
            arrest_yearly_topN_f = arrest_yearly_topN[arrest_yearly_topN["Primary Type"].isin(crime_filter)]
        else:
            hourly_topN_f, yearly_topN_f, arrest_yearly_topN_f = hourly_topN, yearly_topN, arrest_yearly_topN
        if show_points_map:
            plot_location_map(points)
        plot_moran(grid)

    elif "Category" in selected and "Location" in selected:
        st.subheader("Category × Location")
        crime_options = sorted(art["top_types"]["Primary Type"].unique().tolist())
        crime_filter = st.multiselect("Crime types (optional, affects type-based charts)", crime_options, default=crime_options[:5])
        if crime_filter:
            hourly_topN_f = hourly_topN[hourly_topN["Primary Type"].isin(crime_filter)]
            yearly_topN_f = yearly_topN[yearly_topN["Primary Type"].isin(crime_filter)]
            arrest_yearly_topN_f = arrest_yearly_topN[arrest_yearly_topN["Primary Type"].isin(crime_filter)]
        else:
            hourly_topN_f, yearly_topN_f, arrest_yearly_topN_f = hourly_topN, yearly_topN, arrest_yearly_topN
        if show_points_map:
            plot_location_map(points)
        plot_gistar(grid)

    elif "Arrest" in selected and "Location" in selected:
        st.subheader("Arrest × Location")
        crime_options = sorted(art["top_types"]["Primary Type"].unique().tolist())
        crime_filter = st.multiselect("Crime types (optional, affects type-based charts)", crime_options, default=crime_options[:5])
        if crime_filter:
            hourly_topN_f = hourly_topN[hourly_topN["Primary Type"].isin(crime_filter)]
            yearly_topN_f = yearly_topN[yearly_topN["Primary Type"].isin(crime_filter)]
            arrest_yearly_topN_f = arrest_yearly_topN[arrest_yearly_topN["Primary Type"].isin(crime_filter)]
        else:
            hourly_topN_f, yearly_topN_f, arrest_yearly_topN_f = hourly_topN, yearly_topN, arrest_yearly_topN
        if show_points_map and points is not None:
            tmp = points.copy()
            tmp = tmp[(tmp["Year"] >= year_range[0]) & (tmp["Year"] <= year_range[1])]
            tmp["ArrestLabel"] = tmp["Arrest"].map({True: "Arrested", False: "Not arrested"})
            if crime_filter:
                tmp = tmp[tmp["Primary Type"].isin(crime_filter)]
            fig = px.scatter_mapbox(
                tmp,
                lat=LAT_COL, lon=LON_COL,
                color="ArrestLabel",
                zoom=10,
                mapbox_style="carto-positron",
                hover_data=["Primary Type", "Location Description"]
            )
            st.plotly_chart(fig, use_container_width=True)
        plot_moran(grid)

else:
    st.header("Combined EDA (3+ aspects)")
    st.caption("Overview mode: lightweight plots from artifacts")
    crime_options = sorted(art["top_types"]["Primary Type"].unique().tolist())
    crime_filter = st.multiselect("Crime types (optional, affects type-based charts)", crime_options, default=crime_options[:5])
    if crime_filter:
        hourly_topN_f = hourly_topN[hourly_topN["Primary Type"].isin(crime_filter)]
        yearly_topN_f = yearly_topN[yearly_topN["Primary Type"].isin(crime_filter)]
        arrest_yearly_topN_f = arrest_yearly_topN[arrest_yearly_topN["Primary Type"].isin(crime_filter)]
    else:
        hourly_topN_f, yearly_topN_f, arrest_yearly_topN_f = hourly_topN, yearly_topN, arrest_yearly_topN
    t1, t2 = st.tabs(["Interactive overview", "Professional overview"])
    with t1:
        plot_year_trend(yearly)
        plot_top_types(art["top_types"])
        if show_points_map:
            plot_location_map(points)
        plot_arrest_rate(arrest_yearly)
    with t2:
        plot_structure_over_time(yearly_topN_f)
        plot_moran(grid)
        plot_gistar(grid)
