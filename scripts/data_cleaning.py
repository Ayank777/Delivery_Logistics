# ==========================================
# Data Cleaning - Delivery Logistics Dataset
# FINAL VERSION (EDA-Friendly)
# ==========================================

import pandas as pd
import numpy as np
import os

# -------------------------------
# Load Dataset
# -------------------------------
file_path = "data/Delivery_Logistics.csv"

df = pd.read_csv(file_path)

print("✅ Dataset Loaded Successfully\n")

# -------------------------------
# Basic Info
# -------------------------------
print("🔍 Dataset Info:")
print(df.info())

print("\n📊 Statistical Summary:")
print(df.describe())

print("\n📌 Columns:")
print(df.columns)

# -------------------------------
# Remove Duplicates
# -------------------------------
duplicates = df.duplicated().sum()
print(f"\n🔁 Duplicate Rows: {duplicates}")

df = df.drop_duplicates()

# -------------------------------
# Clean Column Names
# -------------------------------
df.columns = df.columns.str.strip().str.replace(' ', '_')

# -------------------------------
# Fix Delay Column (USE ORIGINAL)
# -------------------------------
print("\n🔍 Original delayed values:")
print(df['delayed'].unique())

df['Delayed'] = df['delayed'].astype(str).str.lower().map({
    'yes': 1,
    'no': 0,
    '1': 1,
    '0': 0
})

print("\n📊 Delay Distribution:")
print(df['Delayed'].value_counts())

# -------------------------------
# Clean Time Columns
# -------------------------------
df['delivery_time_hours'] = df['delivery_time_hours'].astype(str).str.extract(r'(\d+\.?\d*)')
df['expected_time_hours'] = df['expected_time_hours'].astype(str).str.extract(r'(\d+\.?\d*)')

df['delivery_time_hours'] = pd.to_numeric(df['delivery_time_hours'], errors='coerce')
df['expected_time_hours'] = pd.to_numeric(df['expected_time_hours'], errors='coerce')

# Drop only required NaNs
df = df.dropna(subset=['delivery_time_hours', 'expected_time_hours'])

# -------------------------------
# CREATE SEASON COLUMN (IMPORTANT)
# -------------------------------
def map_season(weather):
    weather = str(weather).lower()
    
    if 'rain' in weather or 'storm' in weather:
        return 'Monsoon'
    elif 'snow' in weather or 'cold' in weather:
        return 'Winter'
    elif 'sunny' in weather or 'clear' in weather:
        return 'Summer'
    else:
        return 'Other'

df['Season'] = df['weather_condition'].apply(map_season)

print("\n🌦️ Season Distribution:")
print(df['Season'].value_counts())

# -------------------------------
# Remove Outliers (Safe)
# -------------------------------
lower = df['delivery_time_hours'].quantile(0.01)
upper = df['delivery_time_hours'].quantile(0.99)

df = df[
    (df['delivery_time_hours'] >= lower) &
    (df['delivery_time_hours'] <= upper)
]

print("\n📊 After outlier handling:", df.shape)

# -------------------------------
# Create Output Folder
# -------------------------------
os.makedirs("output", exist_ok=True)

# -------------------------------
# Save Cleaned Data
# -------------------------------
output_path = "output/cleaned_data.csv"
df.to_csv(output_path, index=False)

print("\n💾 Cleaned data saved successfully!")
print(f"📁 Location: {output_path}")

# -------------------------------
# Final Dataset Shape
# -------------------------------
print("\n✅ Final Dataset Shape:", df.shape)