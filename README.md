# Recipe Recommender Project

A personalized Top-N recipe recommendation system using Polars for high-performance data processing and Matrix Factorization for modeling.

## Dataset
https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews

## Project Structure

```
Recipe-Recommender/
├── data/
│   ├── raw/             # Raw parquet files (recipes.parquet, reviews.parquet)
│   └── processed/       # Preprocessed train/test splits
├── src/                 # Source code modules
│   ├── data_loader.py   # Data loading utilities
│   ├── preprocessing.py # Cleaning and splitting logic
│   ├── models.py        # Recommendation model implementations
│   └── evaluation.py    # Metric calculations (Recall@K, NDCG@K)
├── scripts/             # Utility scripts
│   └── run_eda.py       # Exploratory Data Analysis
├── notebooks/
|   └── EDA.ipynb        # EDA in detail
├── main.py              # Main entry point to run the pipeline
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Problem Definition
**Task**: Personalized Top-N recipe recommendation.
**Logic**: Given past interactions, rank recipes so that future relevant interactions appear at the top.
**Relevance**:
- Rating >= 4: Relevant (1)
- Rating <= 2: Negative (0)
- Rating == 3: Dropped

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run EDA**:
   Check the data statistics and distributions.
   ```bash
   python scripts/run_eda.py
   ```

3. **Run Pipeline**:
   Preprocess data, train models, and evaluate on test set.
   ```bash
   python main.py
   ```

## Metrics
Implemented ranking metrics:
- **Recall@K**: Percentage of items the user interacted with in the test set that were recommended in the Top-K.
- **NDCG@K**: Normalized Discounted Cumulative Gain, accounting for the rank position of relevant items.
