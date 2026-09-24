# Vendor Matching Recommendation System

Embedding-based vendor recommendation engine as a practical implementation of a real world project (wedding platform vendor matching).

## Problem Statement

A wedding platform needs an intelligent way to recommend relevant vendors (Venues, florist, photographers, etc.) based on a couples preferences around **style**, **budget**, and **location**.

This system solves the problem using **semantic embeddings** + **re-ranking**.

## Solution

- Generate rich text representation of businesses (name + categories + reviews)
- Create emebeddings using `sentence-transformers` (`all-MiniLM-l6-v2`)
- Index embeddings with FAISS for fast similarity search
- Re-rank results using semantic score + star rating + optional price preference
- Serve recommendations through a FASTAPI endpoint

## Features

- Semantic search over thousands of businesses
- Optional city filter
- Optional preferred price range (1-4)
- Star-rating boost in ranking
- Low-latency FAISS retrival
- Interactive API docs at `/docs`

## Tech Stack

- **Python**
- **sentence-transformers**
- **FAISS**
- **FastAPI + Uvicorn**
- **Pandas, Numpy**

## Project Structure

Vendor-matching-recommender/

- notebooks/
    -01_data_exploration.py
    -02_prepare_embeddings_data.py
    -03_generate_embeddings.py
    -04_recommend.py
- requirements.txt
- README.MD

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/workbrian369-ux/vendor-matching-recommender.git
cd vendor-matching-recommender

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows
venv\Scripts\activate
#macOS / Linux
source venv/bin/activate

# 4. Install dependencies
pip instal -r requirements.txt
```

## How to Run

1. Download the Yelp Open Dataset
2. Place the business and review JSON files in data/raw/
3. Run the notebook in order(01-04)
4. Start the API:

```bash
python notebooks/04_recommend.py
# then open your browser:
# http://127.0.0.1:8000/docs
# you can test live queries directly from the interactive Swagger UI.
```

## Example Query

- Q1 - romantic outdoor wedding venue - returns venues & related services
- Q2 - affordable italian restaurant - returns italian restaurants
- Q3 - modern photography studio - returns photographers/studios
- Q4 - Luxury florist for wedding - Returns high-end florist

you can also filter by:

- city (e.g Philadelphia, Nashville)
- preferred_price(1=cheap, 4=expensive)

## Parameters

- Query(required) - text description of what the user wants
- top_k(optional) - number of results (default 10)
- city(optional) - filter by city
- preferred_price(optional) - preferred price range(1-4)

## Future Improvements

- Multi-vector embeddings(separate style / budget / location vectors)
- Learning-to-rank model on top of retrival
- A/B testing framework against a heuristic baseline
- Churn-risk scoring extension for retention
- Production deployment with monitoring and caching

## Notes

This project is a demonstrates end-to-end skills in:

- Recommendation system
- Embeddings-based retrival
- Ranking
- FastAPI
- Practical ML engineering for real client-style problems

## License

This project is for educational and portfolio purposes.
The Yelp dataset has its own usage terms.
