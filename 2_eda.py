"""
Nigerian Real Estate Price Prediction — Step 2: Exploratory Data Analysis
---------------------------------------------------------------------------
Answers 5 analytical questions with statistics + saves charts to charts/.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
df = pd.read_csv("data/cleaned_data.csv")

def naira(x):
    return f"₦{x:,.0f}"

print("=" * 70)
print("Q1: Which property type has the highest average price?")
print("=" * 70)
q1 = df.groupby("property_type")["price_ngn"].mean().sort_values(ascending=False)
print(q1.apply(naira))

plt.figure(figsize=(8, 5))
q1.sort_values().plot(kind="barh", color="#2b6777")
plt.title("Average Price by Property Type")
plt.xlabel("Average Price (₦)")
plt.tight_layout()
plt.savefig("charts/q1_avg_price_by_property_type.png", dpi=120)
plt.close()

print("\n" + "=" * 70)
print("Q2: Which Abuja location (area) has the highest average property price?")
print("=" * 70)
q2 = df.groupby("area")["price_ngn"].mean().sort_values(ascending=False)
print(q2.apply(naira))

plt.figure(figsize=(9, 6))
q2.sort_values().plot(kind="barh", color="#52ab98")
plt.title("Average Price by Area")
plt.xlabel("Average Price (₦)")
plt.tight_layout()
plt.savefig("charts/q2_avg_price_by_area.png", dpi=120)
plt.close()

print("\n" + "=" * 70)
print("Q3: Does the number of bedrooms affect property price?")
print("=" * 70)
corr_bed_price = df["bedrooms"].corr(df["price_ngn"])
print(f"Correlation between bedrooms and price: {corr_bed_price:.3f}")
print(df.groupby("bedrooms")["price_ngn"].mean().apply(naira))

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="bedrooms", y="price_ngn", alpha=0.6, color="#c1666b")
plt.title(f"Bedrooms vs Price (correlation = {corr_bed_price:.2f})")
plt.ylabel("Price (₦)")
plt.tight_layout()
plt.savefig("charts/q3_bedrooms_vs_price.png", dpi=120)
plt.close()

print("\n" + "=" * 70)
print("Q4: What is the relationship between total rooms and price?")
print("=" * 70)
# (No square-metre data is available in the source dataset, so 'total_rooms' —
# our engineered size proxy — is used in place of property size.)
corr_rooms_price = df["total_rooms"].corr(df["price_ngn"])
print(f"Correlation between total_rooms and price: {corr_rooms_price:.3f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="total_rooms", y="price_ngn", alpha=0.6, color="#d4b483")
plt.title(f"Total Rooms vs Price (correlation = {corr_rooms_price:.2f})")
plt.xlabel("Total Rooms (bedrooms + bathrooms + toilets)")
plt.ylabel("Price (₦)")
plt.tight_layout()
plt.savefig("charts/q4_rooms_vs_price.png", dpi=120)
plt.close()

print("\n" + "=" * 70)
print("Q5: Which property type gives the best price-per-room (most affordable per room)?")
print("=" * 70)
q5 = df.groupby("property_type")["price_per_room"].mean().sort_values()
print(q5.apply(naira))

plt.figure(figsize=(8, 5))
q5.plot(kind="barh", color="#8fb339")
plt.title("Average Price per Room by Property Type (lower = more affordable)")
plt.xlabel("Average Price per Room (₦)")
plt.tight_layout()
plt.savefig("charts/q5_price_per_room_by_type.png", dpi=120)
plt.close()

# Bonus: price distribution by property type (boxplot, from the original brief)
plt.figure(figsize=(9, 6))
order = df.groupby("property_type")["price_ngn"].median().sort_values().index
sns.boxplot(data=df, y="property_type", x="price_ngn", order=order, color="#a8c686")
plt.title("Price Distribution by Property Type")
plt.xlabel("Price (₦)")
plt.tight_layout()
plt.savefig("charts/q6_price_distribution_boxplot.png", dpi=120)
plt.close()

print("\nAll charts saved to charts/")
