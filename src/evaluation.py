import math
import numpy as np
import polars as pl

def get_recall_at_k(actual, predicted, k=10):
    if not actual: return 0.0
    hits = len(set(actual) & set(predicted[:k]))
    return hits / len(actual)

def get_ndcg_at_k(actual, predicted, k=10):
    if not actual: return 0.0
    dcg = sum([1.0 / math.log2(i + 2) for i, p in enumerate(predicted[:k]) if p in actual])
    idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(actual), k))])
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_model(model, test_df, k=10, sample_size=500):
    print(f"Evaluating model performance (k={k}, sample={sample_size})...")
    
    # Group by user and collect list of actual items
    test_grouped = test_df.group_by("AuthorId").agg(pl.col("RecipeId").alias("Actual"))
    
    sample = test_grouped.sample(n=min(sample_size, len(test_grouped)), seed=42).to_dicts()
    
    recalls, ndcgs = [], []
    for row in sample:
        uid = row["AuthorId"]
        actual = row["Actual"]
        predicted = model.recommend(uid, k=k)
        
        recalls.append(get_recall_at_k(actual, predicted, k))
        ndcgs.append(get_ndcg_at_k(actual, predicted, k))
    
    return np.mean(recalls), np.mean(ndcgs)
