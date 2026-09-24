"""
Day 4: Recommend Function + Simple API
Vendor Matching Recommendation System
"""

import numpy as np 
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import uvicorn

# CONFIGURATION
MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

print("=" * 60)
print("VENDOR MATCHING - DAY 4: RECOMMENDATION ENGINE")
print("=" * 60)

# 1. LOAD EVERYTHING
print("\n[1/3] Loading model, embeddings and index...")

model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = np.load(MODELS_DIR / "Darknight_business_embeddings.npy")
index = faiss.read_index(str(MODELS_DIR / "faiss_index.bin"))
df = pd.read_parquet(PROCESSED_DIR / "Brucewayne_businesses_for_embeddings.parquet")

print(f"Loaded {len(df):,} businesses")
print(f"FAISS index size: {index.ntotal}")

#2. RECOMMENDATION FUNCTION
def recommend(query: str, top_k: int = 10, city: str = None, preferred_price: int = None):
    """
    Improved recommendation function.
    Combines:
    - Semantics similarity (embedding score)
    - Star rating boost
    - Optional price preference
    - Optional city filter
    """
    # Encode the query
    query_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)

    # Search more candidates so we can re-rank
    scores, indices = index.search(query_embedding, top_k * 5)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        row = df.iloc[idx]

        # city filter
        if city and str(row["city"]).lower() != city.lower():
            continue

        # Base semantics score
        semantic_score = float(score)

        # Star boost (normalize stars 1-5 -> 0-1 and give it weight)
        star_boost = (float(row["stars"]) - 1) / 4 * 0.15 # max + 0.15

        # Price preference boost
        price_boost = 0.0
        if preferred_price is not None and pd.notna(row["price_range"]):
            price_diff = abs(row["price_range"] - preferred_price)
            if price_diff == 0:
                price_boost = 0.10
            elif price_diff == 1:
                price_boost = 0.05

        final_score = semantic_score + star_boost + price_boost

        candidates.append({
            "business_id": row["business_id"],
            "name": row["name"],
            "city": row["city"],
            "stars": float(row["stars"]),
            "price_range": float(row["price_range"]) if pd.notna(row["price_range"]) else None,
            "score": round(final_score, 4),
            "categories": row["categories_list"]
        })

    # Sort by final score and return top_k
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]

# 3. TEST THE FUNCTION
print("\n[2/3] Testing recommendation function...")

test_queries = [
    "romantic outdoor wedding venue with garden",
    "affordable italian restaurant for engagement party",
    "modern photography studio"
]

for q in test_queries:
    print(f"\nQuery: '{q}'")
    results = recommend(q, top_k=5)
    for i, r in enumerate(results, 1):
        print(f" {i}. {r['name']} ({r['city']}) - Score: {r['score']:.3f} | Stars: {r['stars']}")

# 4. FASTAPI APP (optional)
app = FastAPI(title="Battsy Vendor Matching API", version="0.1")

class RecommendationResponse(BaseModel):
    business_id: str
    name: str
    city: str
    stars: float
    price_range: float | None
    score: float
    categories: list

@app.get("/recommend", response_model=List[RecommendationResponse])
def get_recommendations(
    query: str = Query(..., description="Kijana Fupi Nono Round! Describe the style, budget or type of vendor you want"),
    top_k: int = Query(default=10, ge=1, le=50),
    city: str = Query(None, description="Optional city filter"),
    preferred_price: int = Query(None, description="Preferred price range (1-4)", ge=1, le=4)
):
    return recommend(
        query=query,
        top_k=top_k, 
        city=city, 
        preferred_price=preferred_price
    )

@app.get("/")
def root():
    return {"message": "Vendor Matching Recommendation API IS running"}

# 5. RUN SERVER (only if this is executed directly)
if __name__ == "__main__":
    print("\n[3/3] Starting FastAPI server...")
    print("Open your browser at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)