import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

df = pd.read_csv("Crime_Data_Cleaned.csv")

# Interactive Heatmap of Crime Density
m = folium.Map(location=[34.05, -118.25], zoom_start=10)

heat_data = df[['LAT', 'LON']].values.tolist()
HeatMap(heat_data[:10000], radius=10, blur=15).add_to(m)

m.save("la_crime_heatmap.html")
print("Heatmap saved as la_crime_heatmap.html")

# Heatmaps for Top 5 Crimes
top_crimes = df['Crm Cd Desc'].value_counts().head(5).index

for crime in top_crimes:
    m = folium.Map(location=[34.05, -118.25], zoom_start=11)
    crime_data = df[df['Crm Cd Desc'] == crime][['LAT', 'LON']]
    heat_data = crime_data.values.tolist()

    # Limit to first 10,000 points for performance
    HeatMap(heat_data[:10000], radius=10, blur=12).add_to(m)

    # Clean filename
    filename = f"heatmap_{crime.replace('/', '_').replace(' ', '_')}.html"
    m.save(filename)
    print(f"Saved heatmap for {crime} as {filename}")
