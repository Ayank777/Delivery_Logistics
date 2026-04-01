import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("output/cleaned_data.csv")

print("✅ Data Loaded\n")

# -------------------------------
# Delay %
# -------------------------------
print("Delay %:", df['Delayed'].mean()*100)

# -------------------------------
# Partner Performance
# -------------------------------
partner = df.groupby('delivery_partner')['Delayed'].mean()

plt.figure()
partner.plot(kind='bar')
plt.title("Delay Rate by Partner")
plt.show()

# -------------------------------
# Weather Impact
# -------------------------------
weather = df.groupby('weather_condition')['Delayed'].mean()

plt.figure()
weather.plot(kind='bar')
plt.title("Weather vs Delay")
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# Season vs Delay (NEW)
# -------------------------------
season = df.groupby('Season')['Delayed'].mean()

plt.figure()
season.plot(kind='bar')
plt.title("Season vs Delay")
plt.show()

# -------------------------------
# Vehicle vs Season vs Delay
# -------------------------------
pivot = df.pivot_table(
    values='Delayed',
    index='vehicle_type',
    columns='Season',
    aggfunc='mean'
)

pivot.plot(kind='bar')
plt.title("Vehicle vs Season vs Delay")
plt.show()

# -------------------------------
# Distance vs Delay
# -------------------------------
sns.boxplot(x='Delayed', y='distance_km', data=df)
plt.title("Distance vs Delay")
plt.show()

# -------------------------------
# Cost Efficiency
# -------------------------------
df['cost_per_km'] = df['delivery_cost'] / df['distance_km']

sns.histplot(df['cost_per_km'], bins=30, kde=True)
plt.title("Cost per KM")
plt.show()

# -------------------------------
# Correlation
# -------------------------------
corr = df[['distance_km','delivery_time_hours','delivery_cost','Delayed']].corr()

sns.heatmap(corr, annot=True)
plt.title("Correlation")
plt.show()