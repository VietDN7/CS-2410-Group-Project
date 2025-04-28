import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Data
df = pd.read_csv("Crime_Data_Cleaned.csv")
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
df['YearMonth'] = df['DATE OCC'].dt.to_period('M')

# Monthly crime counts by area
monthly_crime = df.groupby(['YearMonth', 'AREA NAME']).size().reset_index(name='Crime Count')
pivot_df = monthly_crime.pivot(index='YearMonth', columns='AREA NAME', values='Crime Count').fillna(0)

# Filter out unreliable months
monthly_totals = pivot_df.sum(axis=1)
reliable_months = monthly_totals[monthly_totals > 6000].index
pivot_df = pivot_df.loc[reliable_months]

# Setup
x_vals = np.arange(pivot_df.shape[0])
forecast_all = {}
forecast_next_month = {}
change_from_current_month = {}
change_over_4_months = {}

# Forecasting
for area in pivot_df.columns:
    y_vals = pivot_df[area].values
    if np.count_nonzero(y_vals) < 4:
        continue

    coeffs = np.polyfit(x_vals, y_vals, 2)
    future_months = [len(x_vals) + i for i in range(4)]
    forecast_values = [max(np.polyval(coeffs, m), 0) for m in future_months]

    forecast_all[area] = forecast_values
    forecast_next_month[area] = forecast_values[0]  # Month +1

    # Change for single month prediction
    change_from_current_month[area] = forecast_values[0] - y_vals[-1]

    # Change over 4 months prediction
    change_over_4_months[area] = forecast_values[3] - y_vals[-1]

# --- Plot Top Crime Areas Next Month ---
forecast_series = pd.Series(forecast_next_month).sort_values(ascending=False)
top_areas = forecast_series.head(10)

plt.figure(figsize=(12, 8))
sns.barplot(y=top_areas.index, x=top_areas.values, hue=top_areas.index, palette='flare', legend=False)
plt.title("Predicted Crime in Areas (Next Month)")
plt.xlabel("Predicted Number of Crimes")
plt.ylabel("Area")
plt.tight_layout()
plt.savefig("predicted_hotspots_next_month.png")
plt.show()

# --- Biggest Changes: Single Month ---
change_series = pd.Series(change_from_current_month)

# Sort by absolute change (largest movement up OR down)
biggest_changes_single_month = change_series.reindex(change_series.abs().sort_values(ascending=False).index).head(5)

print("\nTop 5 areas with biggest overall CHANGE from last month to next month:")
print(biggest_changes_single_month)

# --- Forecast Trend 4 Months ---
forecast_df = pd.DataFrame(forecast_all).T
forecast_df.columns = [f"Month +{i+1}" for i in range(4)]

plt.figure(figsize=(14, 8))
for area in forecast_df.index:
    plt.plot(['Month +1', 'Month +2', 'Month +3', 'Month +4'], forecast_df.loc[area][['Month +1', 'Month +2', 'Month +3', 'Month +4']], label=area)

plt.title("Crime Forecast Trends (Next 4 Months)")
plt.xlabel("Month Ahead")
plt.ylabel("Predicted Crimes")
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.savefig("forecast_trend_4months.png")
plt.show()

#Biggest Changes: Over 4 Months
change_4m_series = pd.Series(change_over_4_months)

# Sort by absolute change (largest movement up OR down)
biggest_changes_4_months = change_4m_series.reindex(change_4m_series.abs().sort_values(ascending=False).index).head(5)

print("\nTop 5 areas with biggest overall change from now to 4 months later:")
print(biggest_changes_4_months)
