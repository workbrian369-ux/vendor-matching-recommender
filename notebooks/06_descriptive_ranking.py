"""
Stage: Express the Proof
Descriptive Vendor Matching - Transport Multi-Component Ranking
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Optional, Dict

# CONFIGURATION -FIXED BY THE PROOF
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

EMBEDDING_MODEL_NAME = "al-MiniLM-L6-v2"

# Pre-declared weights (locked)
WEIGHT_SEMANTIC = 0.50
WEIGHT_BUDGET = 0.30
WEIGHT_QUALITY = 0.20

print("=" * 60)
print("DESCRIPTIVE VENDOR MATCHING - IMPLEMENTATION OF THE PROOF")
print("=" * 60)

# 1. LOAD ARTIFACTS
print("\n[1] Loading data, embeddings and FAISS index...")

df = pd.read_parquet(PROCESSED_DIR / "Brucewayne_businesses_for_embeddings.parquet")
embeddings = np.load(MODELS_DIR / "Darknight_business_embeddings.npy")
index = faiss.read_index(str(MODELS_DIR / "faiss_index.bin"))
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_files_only=False
)

print(f"Loaded {len(df):,} vendors")
print(f"FAISS index size: {index.ntotal}")

# Ensure consistent ordering
assert len(df) == embeddings.shape[0] == index.ntotal,  "Mismatch between data and embeddings"

# 2. HELPER FUNCTIONS (COMPONENT SCORE)
def compute_semantic_scores(query: str, candidate_indices: np.ndarray) -> np.ndarray:
    """Cosine similarity using the pre-built FAISS Index (already L2-normalised)."""
    query_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)

    # Search only among candidates would be ideal, but for simplicity we search globally
    # then map. for exactness we compute directly on the candidate embeddings.
    cand_embs = embeddings[candidate_indices]
    scores = np.dot(cand_embs, query_emb.T).flatten()
    return scores

def compute_budget_scores(preferred_price: Optional[int], price_series: pd.Series) -> np.ndarray:
    """
    Locked rule:
    - If preferred_price is given -> only vendors with known price are scored.
     1.0 exact, 0.6 adjacent, 0.2 otherwise.
    - if no preferred_price -> all get neutral 0.5 (or component is later ignored).
    """
    if preferred_price is None:
        return np.full(len(price_series), 0.5)

    scores = []
    for p in price_series:
        if pd.isna(p):
            scores.append(np.nan)    # will be filtered away
        else:
            diff = abs(p - preferred_price)
            if diff == 0:
                scores.append(1.0)
            elif diff == 1:
                scores.append(0.6)
            else:
                scores.append(0.2)
    return np.array(scores)
    
def compute_quality_scores(stars_series: pd.Series) -> np.ndarray:
    """Simple min-max normalisation of stars (1-5 -> 0-1)."""
    # stars are complete in audit
    return (stars_series.values - 1.0) / 4.0

def make_explanation(row: pd.Series, sem: float, bud: float, qual: float, final: float) -> str:
    """Short transparent explanation."""
    parts = [
        f"Semantic: {sem:.2f}",
        f"Budget: {bud:.2f}" if not np.isnan(bud) else "Budget: N/A",
        f"Quality: {qual:.2f}",
        f"Final: {final:.2f}"
    ]
    return " | ".join(parts)

# 3. MAIN RANKING FUNCTION (THE PROOF EXPRESSED)

def rank_vendors(
        query: str,
        top_k: int = 10,
        city: Optional[str] = None,
        preferred_price: Optional[int] = None
) -> List[Dict]:
    """
    Descriptive matching only.
    Returns ranked list with explicit component scores and explanation.
    faithfull implementation of the proof.
    """

    # Starts with all vendors
    mask = pd.Series(True, index=df.index)

    # Hard filter: city
    if city is not None:
        mask &= df["city"].str.lower() == city.lower()
    
    # Hard filter: budget exclusion rule (Option B)
    if preferred_price is not None:
        mask &= df["price_range"].notna()

    candidate_df = df[mask].copy()
    if len(candidate_df) == 0:
        return []

    candidate_indices = candidate_df.index.to_numpy()

    # Component scores
    semantic = compute_semantic_scores(query, candidate_indices)
    budget = compute_budget_scores(preferred_price, candidate_df["price_range"])
    quality = compute_quality_scores(candidate_df["stars"])

    # Final score (Weights Locked)
    budget = np.asarray(budget, dtype=float)
    budget_clean = np.nan_to_num(budget, nan=0.5)

    final = (
        WEIGHT_SEMANTIC * semantic +
        WEIGHT_BUDGET * budget_clean +
        WEIGHT_QUALITY * quality
    )

    # Assemble results
    results = []
    for i, idx in enumerate(candidate_indices):
        row = df.loc[idx]

        b_val = budget[i]
        b_score_out = None if np.isnan(b_val) else round(float(b_val), 4)

        results.append({
            "business_id": row["business_id"],
            "name": row["name"],
            "city": row["city"],
            "stars": float(row["stars"]),
            "price_range": float(row["price_range"]) if pd.notna(row["price_range"]) else None,
            "semantic_score": round(float(semantic[i]), 4),
            "budget_score": b_score_out,
            "quality_score": round(float(quality[i]), 4),
            "final_score": round(float(final[i]), 4),
            "explanation": make_explanation(
                row, semantic[i], b_val, quality[i], final[i]
            )  
        })

    # Rank by final score
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return results[:top_k]

# 4. Test the implementation
if __name__ == "__main__":
    print("\n[2] Running test queries...")

    test_cases = [
        {
            "query": "romantic outdoor wedding venue",
            "city": None,
            "preferred_price": None
        },
        {
            "query": "affordable italian restaurant",
            "city": "Philadelphia",
            "preferred_price": 2
        },
        {
            "query": "luxury florist",
            "city": None,
            "preferred_price": 4
        }
    ]

    for t in test_cases:
        print(f"\nQuery: '{t['query']}' | City: {t['city']} | Price: {t['preferred_price']}")
        results = rank_vendors(
            query=t["query"],
            top_k=5,
            city=t["city"],
            preferred_price=t["preferred_price"]
        )
        for i, r in enumerate(results, 1):
            print(f" {i}. {r['name']} ({r['city']}) - Final: {r['final_score']:.3f}")
            print(f"    {r['explanation']}")

    print("\n Implementation ready. This is the faithful expression of the locked proof.")
