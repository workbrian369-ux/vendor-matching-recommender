"""
Day 3: Generate Embeddings
Vendor Matching Recommendation System
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss

# CONFIGURATION
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "Brucewayne_businesses_for_embeddings.parquet"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2" # Fast + good quality

print("=" * 60)
print("VENDOR MATCHING - DAY 3: GENERATE EMBEDDINGS")
print("=" * 60)

# 1. LOAD DATA
print("\n[1/4] Loading prepared data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} businesses")

# 2. LOAD EMBEDDINGS MODEL
print(f"\n[2/4] Loading embedding model: {EMBEDDINGS_MODEL}")
print("First time this will download the model, so bruh, be patient...")
model = SentenceTransformer(EMBEDDINGS_MODEL)
print("Model loaded successfully!")

# 3. GENERATE EMBEDDINGS
print("\n[3/4] Generating embeddings...")

texts = df["embedding_text"].tolist()

# generate in batches for better progress tracking
batch_size = 64
embeddings = []

for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
    batch = texts[i : i + batch_size]
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings).astype("float32")
print(f"Embeddings shape: {embeddings.shape}")

# 4. SAVE EMBEDDINGS + BUILD FAISS INDEX
print("\n[4/4] saving embeddings and building FAISS index...")

# Save raw embeddings
np.save(MODELS_DIR / "Darknight_business_embeddings.npy", embeddings)
df[["business_id", "name", "city"]].to_csv(MODELS_DIR / "business_ids.csv", index=False)

# Build FAISS index for fast similarity search
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Save the index
faiss.write_index(index, str(MODELS_DIR / "faiss_index.bin"))

print(f"\nSaved:")
print(f" - {MODELS_DIR / "Darknight_business_embeddings.npy"}")
print(f" - {MODELS_DIR / "faiss_index.bin"}")
print(f" - {MODELS_DIR / "business_ids.csv"}")
print("\n Day 3 complete! Embeddings and FAISS index are ready.")
print("Next: we will build a simple recommendation function + API.")