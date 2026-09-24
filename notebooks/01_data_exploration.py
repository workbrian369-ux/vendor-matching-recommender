"""Day 1: Data Exploration & Filtering 
Vendor Matching Recommendation System"""

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import ast

# Configuration
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BUSINESS_FILE = RAW_DIR / "yelp_academic_dataset_business.json"
REVIEW_FILE = RAW_DIR / "yelp_academic_dataset_review.json"  # we will use this later

# Filtering decisions (choosen so the project stays fast this week)
TARGET_CITIES = ["Philadelphia", "Tucson", "Tampa", "Indianapolis", "Nashville"]
MIN_REVIEWS = 20 # only keep that have enough signal
MAX_BUSINESSES = 8000 # hard cap so everything runs quickly on a normal laptop

print("=" * 60)
print("Vendor Matching - DAY 1 Data Exploration & Filtering")
print("=" * 60)

# 1. LOAD BUSINESS DATA
print("\n[1/4] Loading business data (Let Batman Load the Data!)...")

businesses = []
with open(BUSINESS_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading business data"):
        businesses.append(json.loads(line))

df_biz = pd.DataFrame(businesses)
print(f"Total business loaded: {len(df_biz):,}")

print("\nColumns available:")
print(df_biz.columns.tolist())

print("\nSample of first 3 businesses:")
print(df_biz[["name", "city", "state", "stars", "review_count", "categories"]].head(3))

# 2. BASIC CLEANING & FILTERING
print("\n[2/4] Basic cleaning & filtering...")

# Keep only currently open businesses
df_biz = df_biz[df_biz["is_open"] == 1].copy()

# Drop rows missing critical fields 
df_biz = df_biz.dropna(subset=["categories", "longitude", "latitude", "city"])

# Keep only the target citites ()this is the biggest size reduction)
df_biz = df_biz[df_biz["city"].isin(TARGET_CITIES)].copy()
print(f"After city filter: {len(df_biz):,}")

# Hard cap so the project stays fast this week 
if len(df_biz) > MAX_BUSINESSES:
    df_biz = df_biz.nlargest(MAX_BUSINESSES, "review_count")
    print(f"Capped at the top {MAX_BUSINESSES} businesses by review count.")

print(f"\nFinal business count will work with: {len(df_biz):,}")
print("\nCity distribution:")
print(df_biz['city'].value_counts())

# 3. CREATE STYLE / BUDGET / LOCATION FEATURES
print("\n[3/4] Creating style, budget and location features...")

def extract_price(attributes):
    """
    Extract price range (1-4) from the attributes dictionary.
    Returns None if the information is missing.
    """
    if pd.isna(attributes):
        return None
    if isinstance(attributes, str):
        try:
            attributes = ast.literal_eval(attributes)
        except Exception:
            return None
    if isinstance(attributes, dict):
        return attributes.get("RestaurantsPriceRange2")
    return None

# Budget Proxy
df_biz["price_range"] = df_biz["attributes"].apply(extract_price)
df_biz["price_range"] = pd.to_numeric(df_biz["price_range"], errors="coerce")

# Style proxy - turn the comma-separated categories string into a clean list
df_biz["categories_list"] = df_biz["categories"].fillna("").apply(
    lambda x: [c.strip() for c in x.split(",") if c.strip()]
)

print("\nPrice range distribution (1= cheap, 4 = expensive):")
print(df_biz["price_range"].value_counts(dropna=False).sort_index())

# Location features already exist: latituted, longitude, city, state

# 4. SAVE THE CLEANED DATA
print("\n[4/4] Batman Is Saving Filterefd business data....")

keep_cols = [
    "business_id", "name", "address", "city", "state", "postal_code",
    "latitude", "longitude", "stars", "review_count", "categories",
    "categories_list", "price_range", "attributes"
]

df_biz_clean = df_biz[keep_cols].copy()

# Save in both formats: CSV and Parquet
df_biz_clean.to_csv(PROCESSED_DIR / "batman_businesses_clean.csv", index=False)
df_biz_clean.to_parquet(PROCESSED_DIR / "batman_businesses_clean.parquet", index=False)

print(f"\nSaved: {PROCESSED_DIR / 'batman_businesses_clean.csv'}")
print(f"Saved: {PROCESSED_DIR / 'batman_businesses_clean.parquet'}")
print("\n Day 1 business filtering complete!")
print("Next step: load a sample of reviews and start building embeddings.")