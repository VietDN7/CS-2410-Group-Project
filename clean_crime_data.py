import pandas as pd
import numpy as np

file_path = "Crime_Data_from_2020_to_Present.csv"
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Convert date columns to datetime
df['Date Rptd'] = pd.to_datetime(df['Date Rptd'], errors='coerce')
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')

# Remove rows with missing coordinates
df = df.dropna(subset=['LAT', 'LON'])

# Remove rows where coordinates are clearly invalid (e.g., 0,0 or wildly wrong)
df = df[(df['LAT'] != 0) & (df['LON'] != 0)]
df = df[(df['LAT'].between(33.0, 35.0)) & (df['LON'].between(-119.0, -117.0))]

# Drop empty or mostly-empty columns (customize threshold as needed)
threshold = len(df) * 0.95
df = df.dropna(axis=1, thresh=threshold)

# Strip leading/trailing whitespace in string columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Convert numeric columns that may have been read as object
numeric_cols = ['TIME OCC', 'AREA', 'Vict Age', 'Premis Cd', 'Weapon Used Cd', 'Crm Cd']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with missing essential fields
essential_cols = ['Crm Cd', 'Crm Cd Desc', 'DATE OCC']
df = df.dropna(subset=essential_cols)

# Keep only Part 1 and Part 2 crimes
df = df[df['Part 1-2'].isin([1, 2])]

cleaned_file_path = "Crime_Data_Cleaned.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaning complete. Saved to: {cleaned_file_path}")
