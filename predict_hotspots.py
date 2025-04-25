import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Getting the data
df = pd.read_csv("Crime_Data_Cleaned.csv")
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
df['YearMonth'] = df['DATE OCC'].dt.to_period('M')

#Monthly crime counts by area
monthly_crime = df.groupby(['YearMonth', 'AREA NAME']).size().reset_index(name='Crime Count')
pivot_df = monthly_crime.pivot(index='YearMonth', columns='AREA NAME', values='Crime Count').fillna(0)

#Filter out unreliable months
#The most recent months seem to have very little data which is making it predict zero for me
monthly_totals = pivot_df.sum(axis=1)
reliable_months = monthly_totals[monthly_totals > 6000].index
pivot_df = pivot_df.loc[reliable_months]

#Prepare for forecasting
x_vals = np.arange(pivot_df.shape[0])
forecast_next_month = {}
forecast_all = {}

for area in pivot_df.columns:
    y_vals = pivot_df[area].values
    if np.count_nonzero(y_vals) < 4:
        continue

    #Polynomial regression of degree 2 (quadratic) 
    #possible for linear graph to not be super straight
    coeffs = np.polyfit(x_vals, y_vals, 2)

    #Predict next 4 months
    future_months = [len(x_vals) + i for i in range(4)]  
    forecast_values = [max(np.polyval(coeffs, m), 0) for m in future_months]

    forecast_all[area] = forecast_values

    #Store only next month (month +1) for bar chart
    forecast_next_month[area] = forecast_values[1]

#Bar Chart: Top 20 forecast for next month
forecast_series = pd.Series(forecast_next_month).sort_values(ascending=False)
top_areas = forecast_series.head(20)

plt.figure(figsize=(12, 8))
sns.barplot(y=top_areas.index, x=top_areas.values, hue=top_areas.index, palette='flare', legend=False)
plt.title("Predicted Crime in Areas (Next Month)")
plt.xlabel("Predicted Number of Crimes")
plt.ylabel("Area")
plt.tight_layout()
plt.savefig("predicted_hotspots_next_month.png")
plt.show()

#Line Graph, forecasting trends for next 4 months
forecast_df = pd.DataFrame(forecast_all).T  # shape: (area, 4 months)
forecast_df.columns = [f"Month +{i+1}" for i in range(4)]

plt.figure(figsize=(14, 8))
for area in forecast_df.index:
    plt.plot(forecast_df.columns, forecast_df.loc[area], label=area)

plt.title("Crime Forecast Trends (Next 4 Months)")
plt.xlabel("Month Ahead")
plt.ylabel("Predicted Crimes")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.savefig("forecast_trend_4months.png")
plt.show()
