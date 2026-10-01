# Descriptive Vendor Matching System

A transparent, multi-component vendor ranking system under a strict logical workflow.
This project implements a **descriptive matching claim**: given a user's stated preferences, it returns a ranked list of vendors that best match those preferences under explicit, pre-declared definitions. It does **not** claim to predict user preference or prescribe which vendor should be chosen.

---

## Logical Flow

### Descriptive matching claim

> Given a text description of desired style, an optional preference price level (1-4), and an optional city filter, return a ranked list of vendors. Each vendor is accompanied by an explicit match score broken into three components (Semantic, budget, quality) and a short explanantion.

the ranking determines surface order. The final choice remains with the user.

---

## Scoring Componnents (Locked Definitions)

| Component    | Weight  | Definition  |
| -------------|---------| ------------|
| **Semantics**| 0.50    | Cosine similarity between the query embedding and vendors text (name + categories + review) |
| **Budget**   | 0.30    | 1.0 if exact price match, 0.6 if adjacent. Vendors with missing price are excluded when a preferred price is supplied . when no preferred price is given, the component is neutral (0.5). |
| **Quality**  | 0.20    | Min-max normalized star rating (1-5 -> 0-1) |

**Final Score** = weighted sum of the three components.

Weights were fixed before evaluation.

---

## Key Design Rules

- **City** is a hard filter (include/exclude), not a second componet.
- When a preferred price is provided, vendors with missing price information are excluded (option B).
- All definitions and weights are declared in advance remain fixed.
- Every result returns the component scores so the ranking is auditable.

---

## Data Notes & Limitations

- Built on a filtered subset of the Yelp Open Dataset (8,000 businesses across 5 cities).
- Price Coverage: 83.3%. the remaining 16.7% are excluded when a preferred price is supplied.
- Very few vendors have price level 4. Rankings for "Luxury" queries are therefore constrained by data availability.
- This is a descriptive system. It does not claim to predict booking behaviour or user preference.

---

## Tech Stack

- **Python**
- **sentence-transformers**
- **FAISS**
- **FastAPI + Uvicorn**
- **Pandas, Numpy**

## Project Structure

Vendor-matching-recommender/

- data/
    |- raw/  # Place the original Yelp JSON files here (not tracked)
    |- processed/ # Cleaned data  (not tracked)
- models/  # Embedings + FAISS index (not tracked)
- notebooks/
    |- 01_data_exploration.py
    |- 02_prepare_embeddings_data.py
    |- 03_generate_embeddings.py
    |- 04_recommend.py
    |- 05_data_audit.py
    |- 06_descriptive_ranking.py
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
3. Run the notebook in order(01-06)
4. Start the API:

## Run the ranking system only (after data & models exist)

```bash
python notebooks/06_desriptive_ranking.py
```

## Example Query

```text
Query: 'affordable italian restaurant' | City: Philadelphia | Price: 2 
1: Claudio's Speciality Foods (Philadelphia) - Final: 0.785
    Semantic: 0.62 | Budget: 1.00 | Quality: 0.88
...
```

## WorkFlow Discipline

This project was developed under an explicit analytical workflow:

1. **Problem Formulation** - Logical form identified and locked (Descriptive Matching)
2. **Data Audit** - Coverage and limitations established before modelling
3. **Implementation** - Code written to express exacty the logical claim
4. **Verification** - Output checked against the original proof

The goal was not only to produce rankings, but to produce those rankings whose meaning is clear and whose limitations are stated.

## Future Extensions (Out of Scope)

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
