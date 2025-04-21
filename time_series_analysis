import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Crime_Data_Cleaned.csv")

# Parse date/time columns
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
df['Hour'] = df['TIME OCC'].astype(str).str.zfill(4).str[:2].astype(int)
df['Year'] = df['DATE OCC'].dt.year
df['Month'] = df['DATE OCC'].dt.month_name()

# Monthly Crime Trends by Year
monthly_crimes = df.groupby(["Year", "Month"]).size().reset_index(name='Count')

month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
monthly_crimes["Month"] = pd.Categorical(monthly_crimes["Month"], categories=month_order, ordered=True)
monthly_crimes.sort_values(["Year", "Month"], inplace=True)

plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_crimes, x="Month", y="Count", hue="Year", marker="o", palette="colorblind")
plt.title("Monthly Crime Trends by Year")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("monthly_crime_trends.png")
plt.show()

# Yearly Crime Trends
yearly_trend = df.groupby('Year').size()

plt.figure(figsize=(10, 5))
sns.barplot(x=yearly_trend.index, y=yearly_trend.values, palette='crest')
plt.title('Total Crimes per Year')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.grid(axis='y', linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig("yearly_crime_trends.png")
plt.show()

# Time of Day Crime Pattern
hourly_crimes = df["Hour"].value_counts().sort_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x=hourly_crimes.index, y=hourly_crimes.values, marker="o", color="#FF7F0E")
plt.title("Crime Frequency by Hour of the Day")
plt.xlabel("Hour (24-Hour Format)")
plt.ylabel("Number of Crimes")
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("hourly_crime_trends.png")
plt.show()

# Trends for Specific Crime Categories
top_crimes = df['Crm Cd Desc'].value_counts().head(5).index
filtered = df[df['Crm Cd Desc'].isin(top_crimes)].copy()
filtered['Month'] = pd.Categorical(filtered['DATE OCC'].dt.month_name(), categories=month_order, ordered=True)
filtered['Year'] = filtered['DATE OCC'].dt.year

monthly_type_trend = (
    filtered.groupby(['Year', 'Month', 'Crm Cd Desc'])
    .size()
    .reset_index(name='Count')
    .sort_values(['Year', 'Month'])
)

plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_type_trend, x="Month", y="Count", hue="Crm Cd Desc", marker="o", palette="Set1")
plt.title("Monthly Trends for Top 5 Crime Types")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45)
plt.grid(True, linestyle='-.', alpha=0.5)
plt.tight_layout()
plt.savefig("monthly_top_crime_types.png")
plt.show()
