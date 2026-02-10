import polars as pl
import numpy as np

def calculate_recall_at_k(recommended_ids, actual_ids, k=10):
    if not actual_ids:
        return 0.0
    recommended_at_k = set(recommended_ids[:k])
    actual_set = set(actual_ids)
    hits = len(recommended_at_k.intersection(actual_set))
    return hits / min(len(actual_set), k)

def calculate_ndcg_at_k(recommended_ids, actual_ids, k=10):
    if not actual_ids:
        return 0.0
    
    actual_set = set(actual_ids)
    dcg = 0.0
    for i, rid in enumerate(recommended_ids[:k]):
        if rid in actual_set:
            dcg += 1.0 / np.log2(i + 2)
            
    idcg = 0.0
    for i in range(min(len(actual_set), k)):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_recommender(model, test_df, k=10, sample_size=None):
    """
    Evaluate a recommender model using Recall@k and NDCG@k
    """
    # Group test data by user
    user_test = test_df.filter(pl.col("Relevance") == 1).group_by("AuthorId").agg(pl.col("RecipeId").alias("actual_ids"))
    
    if sample_size and sample_size < len(user_test):
        user_test = user_test.sample(n=sample_size, seed=42)
        
    recalls = []
    ndcgs = []
    
    for row in user_test.to_dicts():
        uid = row["AuthorId"]
        actual = row["actual_ids"]
        
        # Some models take single UID, some might need history (like ContentBased)
        # We try to pass UID first
        try:
            recs = model.recommend(uid, k=k)
        except TypeError:
            # Fallback for models that might need history or have different signature
            # For simplicity, we assume uid is enough for now
            recs = []
            
        recalls.append(calculate_recall_at_k(recs, actual, k))
        ndcgs.append(calculate_ndcg_at_k(recs, actual, k))
        
    return np.mean(recalls), np.mean(ndcgs)
