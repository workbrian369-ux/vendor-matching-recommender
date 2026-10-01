"""
Stage Two: Data Audit
Vendor Matching - Establishing the proof's axioms
"""

import pandas as pd
from pathlib import Path

# configuration
PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "Brucewayne_businesses_for_embeddings.parquet"

print("=" * 60)
print("STAGE TWO: DATA AUDIT")
print("=" * 60)

# 1. LOAD DATA
print("\n[1] Loading Data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Total vendors: {len(df):,}")

# 2. CORE COVERAGE CHECKS
print("\n[2] Coverage of key fields")

total = len(df)

# price
has_price = df["price_range"].notna().sum()
print(f"Vendors with non-null price_range : {has_price:,} ({has_price/total:.1%})")

# Stars 
has_stars = df["stars"].notna().sum()
print(f"Vendors with non-null stars       : {has_stars:,} ({has_stars/total:.1%}) ")

# City
has_city = df["city"].notna().sum()
print(f"Vendors with non-null city        : {has_city:,} ({has_city/total:.1%})")

# Text embedding
# We check embedding if it exists, otherwise fall back to name + categories
if "embedding_text" in df.columns:
    has_text = df["embedding_text"].fillna("").str.len().gt(10).sum()
    text_col = "embedding_text"
else: 
    has_text = (
        df["name"].fillna("").str.len().gt(0) |
        df["categories_list"].notna()
    ).sum()
    text_col = "name + categories"

print(f"Vendors with Usable text ({text_col}): {has_text:,} ({has_text/total:.1%})")

# 3. DISTRIBUTIONS
print("\n[3] Distributions")

print("\nPrice range value counts:")
print(df["price_range"].value_counts(dropna=False).sort_index())

print("\nStars distribution (describe:)")
print(df["stars"].describe())

print("\nTop 10 cities:")
print(df["city"].value_counts().head(10))

print("\nNumber of unique cities:", df["city"].nunique())

# 4. QUICK MISSINGNESS SUMMARY
print("\n[4] Missingness summary (key columns)")
key_cols = ["price_range", "stars", "city", "name", "categories_list"]
if "embedding_text" in df.columns:
    key_cols.append("embedding_text")

print(df[key_cols].isna().mean().sort_values(ascending=False).apply(lambda x: f"{x:.1%}"))

print("\n Audit complete. Paste the key numbers back here.")
