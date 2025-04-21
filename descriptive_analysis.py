import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Crime_Data_Cleaned.csv")

# Top 10 Most Common Crimes
top_crimes = df['Crm Cd Desc'].value_counts().head(10)
plt.figure(figsize=(12,6))
sns.barplot(y=top_crimes.index, x=top_crimes.values, palette='viridis')
plt.title('Top 10 Most Common Crimes')
plt.xlabel('Number of Incidents')
plt.ylabel('Crime Type')
plt.tight_layout()
plt.savefig('top_10_crimes.png')
plt.show()

# Crime Counts by Area
area_counts = df['AREA NAME'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(x=area_counts.index, y=area_counts.values, palette='mako')
plt.xticks(rotation=45, ha='right')
plt.title('Crime Counts by Area')
plt.xlabel('Area')
plt.ylabel('Number of Crimes')
plt.tight_layout()
plt.savefig('crime_counts_by_area.png')
plt.show()

# Top Premises for Crimes
top_premises = df['Premis Desc'].value_counts().head(10)
plt.figure(figsize=(12,6))
sns.barplot(y=top_premises.index, x=top_premises.values, palette='rocket')
plt.title('Top 10 Crime Premises')
plt.xlabel('Number of Incidents')
plt.ylabel('Premises Type')
plt.tight_layout()
plt.savefig('top_10_premises.png')
plt.show()
