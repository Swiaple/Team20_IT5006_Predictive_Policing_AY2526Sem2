import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
from pandas.tseries.holiday import USFederalHolidayCalendar
from statsmodels.tsa.stattools import acf

LAT_COL, LON_COL = "Latitude", "Longitude"

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Chicago Crime EDA", layout="wide")
st.title("Chicago Crime EDA Dashboard")
st.caption("Interactive EDA first, followed by professional/statistical plots. Spatial Moran/Gi* loaded from a precomputed grid.")

# -----------------------------
# Data utilities
# -----------------------------
def load_mini_data_csv(csv_path: str) -> pd.DataFrame:
    usecols = ["Date","Primary Type","Arrest","Domestic","Location Description","Latitude","Longitude"]
    df = pd.read_csv(csv_path, usecols=usecols).sample(1000)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")

    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.day_name()
    df["DayNum"] = df.index.dayofweek

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df

def load_big_data_csv(csv_path: str) -> pd.DataFrame:
    usecols = ["Date","Primary Type","Arrest","Domestic","Location Description","Latitude","Longitude"]
    df = pd.read_csv(csv_path, usecols=usecols)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")

    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.day_name()
    df["DayNum"] = df.index.dayofweek

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna(subset=[df.index.name] if df.index.name else None)

    # Core features
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.day_name()
    df["DayNum"] = df.index.dayofweek

    # Clean lat/lon
    if LAT_COL in df.columns and LON_COL in df.columns:
        df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
        df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")

    return df

def load_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return preprocess(df)

def load_spatial_grid(path="spatial_grid_precomputed.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def filter_by_year(df: pd.DataFrame, year_range: tuple[int,int]) -> pd.DataFrame:
    return df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

def mark_holidays(df: pd.DataFrame) -> pd.DataFrame:
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    out = df.copy()
    out["Is_Holiday"] = out.index.normalize().isin(holidays)

    out["Is_Holiday_Window"] = False
    for h in holidays:
        window = pd.date_range(start=h - pd.Timedelta(days=2), end=h + pd.Timedelta(days=7))
        out.loc[out.index.normalize().isin(window), "Is_Holiday_Window"] = True

    out["Period_Type"] = "Normal Day"
    out.loc[out["Is_Holiday_Window"], "Period_Type"] = "Holiday Window"
    out.loc[out["Is_Holiday"], "Period_Type"] = "Holiday Day"
    return out

# -----------------------------
# Sidebar: data source
# -----------------------------
with st.sidebar:
    st.header("Data")
    use_mini = st.checkbox("Use mini test data", value=True)
    st.write('else it would use dataset from 2001 to present, it would take 1-2 minutes to making the graph for different pill chosed')
    st.divider()
    st.header("Chicago 2001 Dataset Path")
    st.caption("Expected file: ~/Crimes_2001.csv")
    data_path = st.text_input("Dataset path", value="/Users/ziyi/Downloads/Crimes_2001.csv")

if use_mini:
    df = load_mini_data_csv(data_path)
else:
    df = load_big_data_csv(data_path)

# sanity checks
required_cols = ["Primary Type", "Arrest", "Location Description"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Your dataset is missing required columns: {missing}")
    st.stop()

# -----------------------------
# Pills: choose aspects
# -----------------------------
options = ["Time", "Category", "Location", "Arrest"]
selection = st.pills("Which aspect do you intend to know about?", options, selection_mode="multi")
selected = set(selection)

if not selected:
    st.info("Select at least one pill to start.")
    st.stop()

# -----------------------------
# Common filters shown when Time is involved
# -----------------------------
year_range = (int(df["Year"].min()), int(df["Year"].max()))
if "Time" in selected:
    year_range = st.slider(
        "Select Year Range",
        int(df["Year"].min()),
        int(df["Year"].max()),
        (int(df["Year"].min()), int(df["Year"].max()))
    )

df_filtered = filter_by_year(df, year_range)

# optional category filter
top_types = df_filtered["Primary Type"].value_counts().nlargest(10).index.tolist()
crime_filter = st.multiselect("Crime types (optional)", options=sorted(df_filtered["Primary Type"].unique()),
                             default=top_types[:5])
if crime_filter:
    df_filtered = df_filtered[df_filtered["Primary Type"].isin(crime_filter)]

# -----------------------------
# Helper: interactive charts (Plotly)
# -----------------------------
def plot_year_trend_interactive(df_):
    yearly = df_.groupby("Year").size().reset_index(name="Total Crimes")
    fig = px.line(yearly, x="Year", y="Total Crimes", markers=True, title="Annual Crime Trend")
    st.plotly_chart(fig, use_container_width=True)

def plot_monthly_interactive(df_):
    monthly = df_.groupby("Month").size().reset_index(name="Total Crimes").sort_values("Month")
    fig = px.bar(monthly, x="Month", y="Total Crimes", color = 'Month', title="Monthly Seasonality")
    st.plotly_chart(fig, use_container_width=True)

def plot_weekly_interactive(df_):
    weekly = (df_.groupby(["DayNum","DayOfWeek"]).size()
              .reset_index(name="Total Crimes").sort_values("DayNum"))
    fig = px.bar(weekly, x="DayOfWeek", y="Total Crimes", color = "DayOfWeek", title="Weekly Cycle")
    st.plotly_chart(fig, use_container_width=True)

def plot_top5_category_interactive(df_):
    counts = df_["Primary Type"].value_counts().nlargest(5).reset_index()
    counts.columns = ["Primary Type", "Total Crimes"]
    fig = px.bar(counts, x="Primary Type", y="Total Crimes", color="Primary Type", title="Top-5 Crime Types")
    st.plotly_chart(fig, use_container_width=True)

def plot_location_density_interactive(df_):
    if LAT_COL not in df_.columns or LON_COL not in df_.columns:
        st.warning("No Latitude/Longitude columns found.")
        return
    tmp = df_.dropna(subset=[LAT_COL, LON_COL])
    if tmp.empty:
        st.warning("No valid latitude/longitude after filtering.")
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

def plot_arrest_rate_interactive(df_):
    ar = df_.groupby("Year")["Arrest"].mean().reset_index()
    ar["Arrest Rate"] = ar["Arrest"] * 100
    fig = px.line(ar, x="Year", y="Arrest Rate", markers=True, title="Arrest Rate by Year (%)")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Helper: professional plots (Matplotlib / statsmodels style)
# -----------------------------
def st_mpl(fig):
    st.pyplot(fig, use_container_width=True)

def plot_acf_professional(df_, mode="raw", lags=60):
    daily = df_.resample("D").size()
    if mode == "diff":
        daily = daily.diff().dropna()
    if daily.empty:
        st.warning("Not enough daily data for ACF.")
        return
    vals = acf(daily.values, nlags=lags, fft=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(vals)), vals)
    ax.axhline(0, linewidth=1)
    ax.set_title("ACF (Daily counts)" + (" after differencing" if mode=="diff" else ""))
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    st_mpl(fig)

def plot_holiday_professional(df_):
    dfh = mark_holidays(df_)
    top5 = dfh["Primary Type"].value_counts().nlargest(5).index
    subset = dfh[dfh["Primary Type"].isin(top5)]
    comp = subset.groupby(["Period_Type","Primary Type"]).size().unstack(fill_value=0)
    comp_pct = comp.div(comp.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    comp_pct.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Crime composition: Holiday vs Normal days (Top-5)")
    ax.set_ylabel("Proportion")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st_mpl(fig)

    # Hourly pulse holiday vs normal
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for p_type in ["Holiday Day", "Normal Day"]:
        data = dfh[dfh["Period_Type"] == p_type]
        hourly = data["Hour"].value_counts(normalize=True).sort_index()
        ax2.plot(hourly.index, hourly.values, linewidth=2.5, label=p_type)
    ax2.set_title("Hourly pulse: Holiday Day vs Normal Day")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Probability density")
    ax2.legend()
    st_mpl(fig2)

def plot_moran_professional(grid_df):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    hb = ax.hexbin(grid_df["z_standardized"], grid_df["lag"], gridsize=70, bins="log", mincnt=1, linewidths=0)
    fig.colorbar(hb, ax=ax, shrink=0.9).set_label("log10(count)")
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_title("Moran Scatter (precomputed grid)")
    ax.set_xlabel("Standardized cell count (z)")
    ax.set_ylabel("Spatial lag (rook mean)")
    st_mpl(fig)

def plot_gistar_professional(grid_df):
    # classification map (scatter as a fast “raster-like” view)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(grid_df["lon"], grid_df["lat"], c=grid_df["Gi_cat"], s=6)
    ax.set_title("Gi* Hotspot/Coldspot classes (precomputed)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st_mpl(fig)

    # Gi* z values
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sc2 = ax2.scatter(grid_df["lon"], grid_df["lat"], c=grid_df["Gi_z"], s=6)
    ax2.set_title("Gi* z-scores (precomputed)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    st_mpl(fig2)

# -----------------------------
# UI rendering
# -----------------------------
# Single-pill pages
if len(selected) == 1:
    only = next(iter(selected))

    if only == "Time":
        st.header("Time EDA")
        t1, t2, t3, t4 = st.tabs(["Interactive basics", "Crime-time (Top types)", "ACF (pro)", "Holiday (pro)"])

        with t1:
            plot_year_trend_interactive(df_filtered)
            plot_monthly_interactive(df_filtered)
            plot_weekly_interactive(df_filtered)

        with t2:
            top5 = df_filtered["Primary Type"].value_counts().nlargest(5).index
            sub = df_filtered[df_filtered["Primary Type"].isin(top5)]
            hourly = sub.groupby(["Hour","Primary Type"]).size().reset_index(name="Total")
            fig = px.line(hourly, x="Hour", y="Total", color="Primary Type", markers=True,
                          title="When do specific crimes happen? (Top-5)")
            st.plotly_chart(fig, use_container_width=True)

        with t3:
            plot_acf_professional(df_filtered, mode="raw", lags=60)
            plot_acf_professional(df_filtered, mode="diff", lags=30)

        with t4:
            plot_holiday_professional(df_filtered)

    elif only == "Category":
        st.header("Category EDA")
        c1, c2 = st.tabs(["Interactive", "Professional (structure)"])
        with c1:
            plot_top5_category_interactive(df_filtered)

        with c2:
            # professional: stacked ratio + absolute counts (Top-5)
            top5 = df_filtered["Primary Type"].value_counts().nlargest(5).index
            df_top5 = df_filtered[df_filtered["Primary Type"].isin(top5)]
            counts = df_top5.groupby(["Year","Primary Type"]).size().unstack(fill_value=0)
            ratio = counts.div(counts.sum(axis=1), axis=0)

            fig, ax = plt.subplots(figsize=(10, 5))
            ratio.plot(kind="area", stacked=True, ax=ax, alpha=0.8)
            ax.set_title("Crime type structure over time (Top-5, stacked area)")
            ax.set_ylabel("Proportion")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            st_mpl(fig)

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            counts.plot(ax=ax2, linewidth=2)
            ax2.set_title("Crime counts by type over time (Top-5)")
            ax2.set_ylabel("Count")
            ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            st_mpl(fig2)

    elif only == "Location":
        st.header("Location EDA")
        plot_location_density_interactive(df_filtered)

    elif only == "Arrest":
        st.header("Arrest EDA")
        a1, a2 = st.tabs(["Interactive", "Professional (Top-5 by type)"])
        with a1:
            plot_arrest_rate_interactive(df_filtered)
        with a2:
            top5 = df_filtered["Primary Type"].value_counts().nlargest(5).index
            df_top5 = df_filtered[df_filtered["Primary Type"].isin(top5)]
            ar = df_top5.groupby(["Year","Primary Type"])["Arrest"].mean().unstack()

            fig, ax = plt.subplots(figsize=(10, 5))
            for ct in top5:
                if ct in ar.columns:
                    ax.plot(ar.index, ar[ct], marker="s", linewidth=2, label=ct)
            ax.set_title("Arrest rate by type (Top-5)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Arrest rate")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            st_mpl(fig)

# Two-pill combinations
elif len(selected) == 2:
    st.header("Combined EDA (2 aspects)")
    s = selected

    # Time + Category
    if "Time" in s and "Category" in s:
        st.subheader("Time × Category")
        top5 = df_filtered["Primary Type"].value_counts().nlargest(5).index
        sub = df_filtered[df_filtered["Primary Type"].isin(top5)]
        hourly = sub.groupby(["Hour","Primary Type"]).size().reset_index(name="Total")
        fig = px.line(hourly, x="Hour", y="Total", color="Primary Type", markers=True,
                      title="Hourly pattern by top crime types")
        st.plotly_chart(fig, use_container_width=True)

        # professional add-on: normalized density
        pivot = sub.groupby(["Hour","Primary Type"]).size().unstack(fill_value=0)
        pivot_norm = pivot.div(pivot.sum(axis=0), axis=1)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        pivot_norm.plot(ax=ax2, linewidth=2)
        ax2.set_title("Professional: normalized hourly density (Top-5)")
        ax2.set_xlabel("Hour")
        ax2.set_ylabel("Relative probability")
        st_mpl(fig2)

    # Time + Location
    elif "Time" in s and "Location" in s:
        st.subheader("Time × Location")
        plot_location_density_interactive(df_filtered)

        # professional add-on: ACF
        st.caption("Professional add-on: daily memory (ACF)")
        plot_acf_professional(df_filtered, mode="raw", lags=60)

    # Category + Location
    elif "Category" in s and "Location" in s:
        st.subheader("Category × Location")
        if LAT_COL in df_filtered.columns and LON_COL in df_filtered.columns:
            tmp = df_filtered.dropna(subset=[LAT_COL, LON_COL])
            top5 = tmp["Primary Type"].value_counts().nlargest(5).index
            tmp = tmp[tmp["Primary Type"].isin(top5)]
            fig = px.scatter_mapbox(
                tmp,
                lat=LAT_COL, lon=LON_COL,
                color="Primary Type",
                zoom=10,
                mapbox_style="carto-positron",
                hover_data=["Location Description"]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Latitude/Longitude found.")

    # Time + Arrest
    elif "Time" in s and "Arrest" in s:
        st.subheader("Time × Arrest")
        plot_arrest_rate_interactive(df_filtered)
        st.caption("Professional add-on: holiday comparison")
        plot_holiday_professional(df_filtered)

    # Category + Arrest
    elif "Category" in s and "Arrest" in s:
        st.subheader("Category × Arrest")
        top5 = df_filtered["Primary Type"].value_counts().nlargest(5).index
        df_top5 = df_filtered[df_filtered["Primary Type"].isin(top5)]
        ar = df_top5.groupby("Primary Type")["Arrest"].mean().sort_values(ascending=False).reset_index()
        ar["Arrest Rate (%)"] = ar["Arrest"] * 100
        fig = px.bar(ar, x="Primary Type", y="Arrest Rate (%)", color="Primary Type",
                     title="Arrest rate by crime type (Top-5)")
        st.plotly_chart(fig, use_container_width=True)

    # Location + Arrest
    elif "Location" in s and "Arrest" in s:
        st.subheader("Location × Arrest")
        if LAT_COL in df_filtered.columns and LON_COL in df_filtered.columns:
            tmp = df_filtered.dropna(subset=[LAT_COL, LON_COL])
            tmp["ArrestLabel"] = tmp["Arrest"].map({True:"Arrested", False:"Not arrested"})
            fig = px.scatter_mapbox(
                tmp,
                lat=LAT_COL, lon=LON_COL,
                color="ArrestLabel",
                zoom=10,
                mapbox_style="carto-positron",
                hover_data=["Primary Type","Location Description"]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Latitude/Longitude found.")

# 3+ pills: show a combined “overview”
else:
    st.header("Combined EDA (3+ aspects)")
    st.caption("Showing a compact overview. For deeper details, select 1–2 pills.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows after filters", f"{len(df_filtered):,}")
    with col2:
        st.metric("Year range", f"{year_range[0]}–{year_range[1]}")
    with col3:
        st.metric("Crime types selected", f"{len(crime_filter)}")

    # Compact multi-view
    t1, t2 = st.tabs(["Interactive overview", "Professional overview"])
    with t1:
        plot_year_trend_interactive(df_filtered)
        plot_top5_category_interactive(df_filtered)
        plot_location_density_interactive(df_filtered)
        plot_arrest_rate_interactive(df_filtered)
    with t2:
        plot_acf_professional(df_filtered, mode="raw", lags=60)
