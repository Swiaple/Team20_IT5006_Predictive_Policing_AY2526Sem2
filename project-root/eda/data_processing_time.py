import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf
from scipy.stats import mannwhitneyu


# 1. data loading and preprocess
def preprocess_crime_data(file_path):
    keep_cols = ['Date', 'Primary Type']
    df = pd.read_csv(file_path, usecols=keep_cols)

    # transform date form
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p')

    # filter 2001-2025 data
    df = df[(df['Date'].dt.year >= 2001) & (df['Date'].dt.year <= 2025)]

    # get the time's feature
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.date
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Year_Week'] = df['Date'].dt.to_period('W').apply(lambda r: r.start_time)

    return df


# 2. analysis
def perform_temporal_analysis(df):
    # A. crime amount series of per day
    daily_counts = df.groupby('Day').size()
    daily_counts.index = pd.to_datetime(daily_counts.index)

    # B. STL (trend and seasonal)
    # period=365 round of year
    res = STL(daily_counts, period=365, robust=True).fit()

    # C. 24x7
    hourly_weekly = df.groupby(['DayOfWeek', 'Hour']).size().unstack()
    # week's days order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hourly_weekly = hourly_weekly.reindex(days_order)

    # D. holiday
    us_holidays = holidays.US()
    weekly_counts = df.groupby('Year_Week').size().reset_index(name='CrimeCount')

    # are some weeks including holidays?
    def is_holiday_week(week_start):
        week_days = pd.date_range(start=week_start, periods=7)
        return any(day in us_holidays for day in week_days)

    weekly_counts['Is_Holiday_Week'] = weekly_counts['Year_Week'].apply(is_holiday_week)

    return daily_counts, res, hourly_weekly, weekly_counts


# 3. visualization
def plot_results(daily_counts, stl_res, hourly_weekly, weekly_counts):
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig = plt.figure(figsize=(18, 20))

    # --- figure1: crime amount and trend by day ---
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(daily_counts.index, daily_counts.values, alpha=0.3, color='gray', label='Daily Count')
    ax1.plot(stl_res.trend.index, stl_res.trend.values, color='red', linewidth=2, label='Long-term Trend (STL)')
    ax1.set_title("Figure 1: Long-term Temporal Evolution and Macro Trend (2001-2025)", loc='left', fontweight='bold')
    ax1.legend()

    # --- figure2: 24x7 ---
    ax2 = plt.subplot(4, 2, 3)
    sns.heatmap(hourly_weekly, cmap="YlGnBu", ax=ax2, cbar_kws={'label': 'Incident Density'})
    ax2.set_title("Figure 2: 24/7 Hourly-Weekly Crime Density", loc='left', fontweight='bold')

    # --- figure3: ACF ---
    ax3 = plt.subplot(4, 2, 4)
    acf_values = acf(daily_counts.dropna(), nlags=50)
    ax3.stem(range(len(acf_values)), acf_values)
    ax3.set_title("Figure 3: Autocorrelation Function (ACF) - Lag analysis", loc='left', fontweight='bold')

    # --- figure4: holiday contract compare ---
    ax4 = plt.subplot(4, 1, 3)
    # normal weeks
    ax4.plot(weekly_counts['Year_Week'], weekly_counts['CrimeCount'], color='blue', alpha=0.5, label='Normal Week')
    # holidays weeks
    h_weeks = weekly_counts[weekly_counts['Is_Holiday_Week']]
    ax4.scatter(h_weeks['Year_Week'], h_weeks['CrimeCount'], color='red', s=40, label='Holiday Week', zorder=5)
    ax4.set_title("Figure 4: Weekly Crime Fluctuations with Holiday Markers", loc='left', fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    plt.show()


# 4. verify and stastical
def statistical_tests(daily_counts, weekly_counts):
    print("--- Statistical Analysis Report ---")

    # A. ADF Test
    adf_res = adfuller(daily_counts.dropna())
    print(f"ADF Statistic: {adf_res[0]:.4f}")
    print(f"p-value: {adf_res[1]:.4e}")
    if adf_res[1] < 0.05:
        print("Result: The series is Stationary (Reject H0)")
    else:
        print("Result: The series is Non-Stationary (Accept H0)")

    # B. Mann-Whitney U Test for holidays
    holiday_counts = weekly_counts[weekly_counts['Is_Holiday_Week'] == True]['CrimeCount']
    normal_counts = weekly_counts[weekly_counts['Is_Holiday_Week'] == False]['CrimeCount']

    u_stat, p_val = mannwhitneyu(holiday_counts, normal_counts)
    print(f"\n--- Holiday Impact Analysis ---")
    print(f"Holiday Week Avg: {holiday_counts.mean():.2f}")
    print(f"Normal Week Avg: {normal_counts.mean():.2f}")
    print(f"Mann-Whitney U Test p-value: {p_val:.4e}")
    if p_val < 0.05:
        print("Conclusion: Public holidays have a SIGNIFICANT impact on crime distribution.")
    else:
        print("Conclusion: No statistically significant difference observed during holidays.")

# run
df = preprocess_crime_data('Crimes_-_2001_to_Present_20260212.csv')
daily, stl, hourly, weekly = perform_temporal_analysis(df)
plot_results(daily, stl, hourly, weekly)
statistical_tests(daily, weekly)