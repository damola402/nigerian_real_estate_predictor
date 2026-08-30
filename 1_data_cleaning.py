"""
Nigerian Real Estate Price Prediction — Step 1: Data Cleaning & Feature Engineering
------------------------------------------------------------------------------------
Dataset: PropertyPro Abuja listings (scraped), 481 raw rows.

This script:
1. Loads the raw dataset
2. Cleans it (missing values, duplicates, data types, inconsistent categories)
3. Engineers a new feature: price_per_room
4. Saves a cleaned dataset ready for EDA and modelling

Every cleaning decision is explained in a comment right above the code that does it,
so it's easy to defend during your presentation/defense.
"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv("data/raw_data.csv")
print(f"Raw shape: {df.shape}")
print(df.dtypes)

# ------------------------------------------------------------------
# 2. DROP COLUMNS WE DON'T NEED FOR PRICE PREDICTION
# ------------------------------------------------------------------
# 'description' is a boilerplate safety-tips paragraph repeated on every listing
# (not real property description text), so it carries no predictive value.
# 'pid', 'source_url', 'date_added', 'last_updated', 'address', 'title' are
# identifiers/free text, not model features. We keep 'title' briefly for sanity
# checks, then drop it before modelling.
df = df.drop(columns=["description", "pid", "source_url", "date_added", "last_updated"])

# ------------------------------------------------------------------
# 3. HANDLE LISTING TYPE: SALE vs RENT
# ------------------------------------------------------------------
# The dataset mixes 'sale' listings (prices in the hundreds of millions of naira)
# with 'rent' listings (prices in the low millions per year). Combining both in
# one price-prediction model doesn't make sense — a 4-bedroom duplex "price" of
# ₦4,000,000 could mean cheap-to-buy or expensive-to-rent depending on which it is.
# We restrict this project to SALE listings only, since "Price Predictor" implies
# purchase price. Rent listings are dropped and this is stated clearly in the report.
print("\nListing type breakdown:\n", df["listing_type"].value_counts())
df = df[df["listing_type"] == "sale"].copy()
df = df.drop(columns=["listing_type"])
print(f"\nShape after keeping sale-only listings: {df.shape}")

# ------------------------------------------------------------------
# 4. CHECK FOR DUPLICATES
# ------------------------------------------------------------------
dup_count = df.duplicated(subset=["title", "area", "price_ngn", "bedrooms"]).sum()
print(f"\nDuplicate rows found: {dup_count}")
df = df.drop_duplicates(subset=["title", "area", "price_ngn", "bedrooms"])
print(f"Shape after removing duplicates: {df.shape}")

# ------------------------------------------------------------------
# 5. HANDLE MISSING VALUES: bathrooms, toilets
# ------------------------------------------------------------------
# bathrooms and toilets have a small number of missing values. Since these are
# closely correlated with bedrooms (a property almost always has roughly as many
# bathrooms/toilets as bedrooms), we fill missing values with the median count
# for that specific bedroom size rather than an overall median. This keeps the
# imputed values realistic instead of forcing a single generic number onto every
# missing row.
print("\nMissing values before imputation:\n", df.isnull().sum())

for col in ["bathrooms", "toilets"]:
    df[col] = df.groupby("bedrooms")[col].transform(lambda s: s.fillna(s.median()))
    # fallback: if a bedroom group has NO known value at all, use the overall median
    df[col] = df[col].fillna(df[col].median())

print("\nMissing values after imputation:\n", df.isnull().sum())

# ------------------------------------------------------------------
# 6. FIX DATA TYPES
# ------------------------------------------------------------------
# bathrooms/toilets were read as float because of the missing values above;
# now that they're filled, we convert them to whole numbers (you can't have
# half a bathroom).
df["bathrooms"] = df["bathrooms"].round().astype(int)
df["toilets"] = df["toilets"].round().astype(int)

# ------------------------------------------------------------------
# 7. STANDARDISE CATEGORICAL TEXT (inconsistent naming)
# ------------------------------------------------------------------
# 'area' values like "Wuse-2" vs "Wuse" or trailing/leading spaces and mixed
# capitalisation would be treated as different categories even when they refer
# to the same place. We standardise formatting (strip spaces, consistent title
# case) so the same location isn't split into multiple categories.
df["area"] = df["area"].str.strip().str.title()
df["neighbourhood"] = df["neighbourhood"].str.strip().str.title()
df["property_type"] = df["property_type"].str.strip().str.title()

print("\nCleaned area categories:\n", df["area"].value_counts())
print("\nCleaned property_type categories:\n", df["property_type"].value_counts())

# 'Mansion' appears only once in the whole dataset. A category with a single
# observation can't be learned from and risks landing entirely in either the
# train or test split by chance, so we fold it into the closest matching
# category, 'Detached Duplex' (mansions in this market are effectively large
# detached duplexes), rather than deleting real data.
df["property_type"] = df["property_type"].replace({"Mansion": "Detached Duplex"})

# ------------------------------------------------------------------
# 8. REMOVE BULK / MULTI-UNIT LISTINGS
# ------------------------------------------------------------------
# Some listings are priced for an entire block of units, not a single property
# (e.g. "33 Units Of Spacious Apartment" listed as one row with a 1-bedroom
# spec but a ₦3.5 billion price — that's the price for the whole building, not
# one flat). Left in, these badly distort a *single-property* price predictor,
# since the same bedroom count can appear at wildly different, non-comparable
# price scales depending on whether it's one unit or a block of units. We
# detect and drop these using the word "Units" in the title.
bulk_mask = df["title"].str.contains(r"\bunits?\b", case=False, regex=True)
print(f"\nBulk/multi-unit listings found and removed: {bulk_mask.sum()}")
df = df[~bulk_mask].copy()

# ------------------------------------------------------------------
# 9. OUTLIER CHECK ON PRICE
# ------------------------------------------------------------------
# Real estate prices in Abuja legitimately span a huge range (a small flat vs a
# mansion), so we don't remove outliers just for being large. Instead we only
# drop rows where the price is implausibly low to be a real sale (e.g. below
# ₦1,000,000, which is more likely a data entry error than a real listing) and
# rows above the 99th percentile, which are extreme luxury mansions that would
# distort a simple linear model without adding useful general signal.
print(f"\nPrice range before outlier check: {df['price_ngn'].min():,} - {df['price_ngn'].max():,}")
lower_cutoff = 1_000_000
upper_cutoff = df["price_ngn"].quantile(0.99)
before = len(df)
df = df[(df["price_ngn"] >= lower_cutoff) & (df["price_ngn"] <= upper_cutoff)]
print(f"Removed {before - len(df)} price outlier rows.")
print(f"Price range after outlier check: {df['price_ngn'].min():,} - {df['price_ngn'].max():,}")

# ------------------------------------------------------------------
# 10. FEATURE ENGINEERING — NEW DATA POINT
# ------------------------------------------------------------------
# The raw data has no property size (square metres), so we can't compute
# price-per-sqm as originally planned. Instead we engineer:
#     total_rooms       = bedrooms + bathrooms + toilets
#     price_per_room     = price_ngn / total_rooms
# This gives a normalised "cost per room" figure that lets us compare
# affordability across property types and locations regardless of how big
# the property is — this is our new engineered feature for the project.
df["total_rooms"] = df["bedrooms"] + df["bathrooms"] + df["toilets"]
df["price_per_room"] = (df["price_ngn"] / df["total_rooms"]).round(0)

print("\nSample of engineered feature:")
print(df[["title", "bedrooms", "bathrooms", "toilets", "total_rooms", "price_ngn", "price_per_room"]].head())

# ------------------------------------------------------------------
# 11. SAVE CLEANED DATASET
# ------------------------------------------------------------------
df.to_csv("data/cleaned_data.csv", index=False)
print(f"\nFinal cleaned shape: {df.shape}")
print("Saved to data/cleaned_data.csv")
