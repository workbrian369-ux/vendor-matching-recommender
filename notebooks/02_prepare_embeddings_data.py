"""
Day 2: Prepare data for embeddings
Vendor Matching Recommendation System
"""

import pandas as pd
import json 
from pathlib import Path
from tqdm import tqdm
import ast

# CONFIGURATION
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BUSINESS_FILE = PROCESSED_DIR / "batman_businesses_clean.parquet"
REVIEWWS_FILE = RAW_DIR / "yelp_academic_dataset_review.json"

# how many reviews per business to keep for embeddings
MAX_REVIEWS_PER_BUSINESS = 15
MAX_TOTAL_REVIEWS = 120_000  # safety limit

print("=" * 60)
print("VENDOR MATCHING - DAY 2: PREPARE DATA FOR EMBEDDINGS")
print("=" * 60)

# 1. LOAD FILTERED BUSINESSES
print("\n[1/5] Loading filtered businesses...")
df_biz = pd.read_parquet(BUSINESS_FILE)
print(f"loaded {len(df_biz):,} businesses")

# Create a set of business IDs for fast filtering reviews
target_business_ids = set(df_biz['business_id'].tolist())
print(f"Target business IDs Ready: {len(target_business_ids):,}")

# 2. LOAD ONLY RELEVANT REVIEWS
print("\n [2/5] Loading reviews for our businesses Bruce Wayne is interested in...")
print("We only keep reviews that belong to the 8000 filtered businesses.")

reviews = []
with open(REVIEWWS_FILE, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Scanning reviews"):
        review = json.loads(line)
        if review['business_id'] in target_business_ids:
            reviews.append({
                "business_id": review['business_id'],
                "stars": review['stars'],
                "text": review['text'],
                "date": review['date']
            })
            if len(reviews) >= MAX_TOTAL_REVIEWS:
                break

df_reviews = pd.DataFrame(reviews)
print(f"\nCollected {len(df_reviews):,} relevantreviews.")

# LIMIT REVIEWS PER BUSINESS
print(f"\n[3/5] limiting to top reviews per business...")

# Keep the most recent reviews for each business
df_reviews["date"] = pd.to_datetime(df_reviews["date"])
df_reviews = df_reviews.sort_values("date", ascending=False)

df_reviews = (
    df_reviews
    .groupby("business_id")
    .head(MAX_REVIEWS_PER_BUSINESS)
    .reset_index(drop=True)
)

print(f"After limiting, {len(df_reviews):,} reviews.")
print(f"Average reviews per business: {len(df_reviews) / len(df_biz):.1f}")

# 4. CREATE TEXT FOR EMBEDINNING
print("\n[4/5] Creating text representation for each business...")

# Aggregate reviews into one block per business
review_texts = (
    df_reviews
    .groupby("business_id")["text"]
    .apply(lambda x: " ".join(x.tolist()))
    .reset_index()
    .rename(columns={"text": "review_text"})
)

# Merge with business data
df = df_biz.merge(review_texts, on="business_id", how="left")
df["review_text"] = df["review_text"].fillna("")

# Create a a rich text field for embedded
# This combines name + categories + reviews (the main signal for "style")
def create_embedding_text(row):
    name = str(row["name"])
    categories = ", ".join(row["categories"]) if isinstance(row["categories"], list) else str(row["categories"])
    reviews = row["review_text"][:2000] # limit length for speed
    return f"{name}. Categories: {categories}. Customer Reviews: {reviews}"

df["embedding_text"] = df.apply(create_embedding_text, axis=1)

print("Example embedding text (first business):")
print(df["embedding_text"].iloc[0][:300] + "...")

# 5. SAVE PREPARED DATA
print("\n[5/5] Bruce Wayne Saving prepared data...")

# keep only useful columns
final_columns = [
    "business_id", "name", "city", "state",
    "latitude", "longitude", "stars", "review_count",
    "price_range", "categories_list", "embedding_text"
]

df_final = df[final_columns].copy()

df_final.to_parquet(PROCESSED_DIR / "Brucewayne_businesses_for_embeddings.parquet", index=False)
df_final.to_csv(PROCESSED_DIR / "Brucewayne_businesses_for_embeddings.csv", index=False)

print(f"\nSaved: {PROCESSED_DIR / 'Brucewayne_businesses_for_embeddings.parquet'}")
print(f"Saved: {PROCESSED_DIR / 'Brucewayne_businesses_for_embeddings.csv'}")
print(f"\nTotal businesses ready: {len(df_final):,}")
print("Day 2 complete. Data is ready for embeddings.")
print("Next: we will generate the actual embeddings.")
