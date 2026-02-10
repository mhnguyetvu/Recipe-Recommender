import os
import polars as pl
import pickle
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.models import PopularityModel, ContentBasedModel, LightFMHybridModel, XGBoostRankingModel, LIGHTFM_AVAILABLE
from src.evaluation import evaluate_recommender

def main():
    # 1. Load Data
    print("Loading data...")
    loader = DataLoader(data_dir="data")
    recipes = loader.load_recipes().with_columns(pl.col("RecipeId").cast(pl.Int64))
    reviews = loader.load_reviews().with_columns(pl.col("RecipeId").cast(pl.Int64))

    # 2. Preprocess
    print("Preprocessing...")
    preprocessor = DataPreprocessor()
    processed_reviews = preprocessor.preprocess_reviews(reviews)
    train, val, test = preprocessor.split_data(processed_reviews)
    
    # Ensure types after split
    train = train.with_columns(pl.col("RecipeId").cast(pl.Int64))
    val = val.with_columns(pl.col("RecipeId").cast(pl.Int64))
    test = test.with_columns(pl.col("RecipeId").cast(pl.Int64))

    # 3. Fit Models - SIMPLIFIED FOR DEPLOYMENT
    models_to_eval = {}

    # Popularity - MAIN MODEL FOR DEPLOYMENT
    pop_model = PopularityModel(top_k=200)
    pop_model.fit(train)
    models_to_eval["Popularity"] = pop_model

    # # Content-Based - COMMENTED OUT FOR DEPLOYMENT
    # # Use a subset of recipes for fitting to avoid memory issues if too large
    # # Actually fit on all recipes' content
    # cb_model = ContentBasedModel(top_k=100)
    # cb_model.fit(recipes)
    # # Wrap CB recommend to match evaluate_recommender expected signature
    # # In a real app we'd get user history from train_df
    # user_train_history = train.filter(pl.col("Relevance") == 1).group_by("AuthorId").agg(pl.col("RecipeId").alias("history"))
    # user_history_map = {row["AuthorId"]: row["history"] for row in user_train_history.to_dicts()}
    # 
    # class CBWrapper:
    #     def __init__(self, model, history_map):
    #         self.model = model
    #         self.history_map = history_map
    #     def recommend(self, uid, k=10):
    #         history = self.history_map.get(uid, [])
    #         return self.model.recommend(history, k=k)
    # 
    # models_to_eval["Content-Based"] = CBWrapper(cb_model, user_history_map)

    # # LightFM - COMMENTED OUT FOR DEPLOYMENT
    # if LIGHTFM_AVAILABLE:
    #     lfm_model = LightFMHybridModel(n_components=32)
    #     lfm_model.fit(train, recipes)
    #     models_to_eval["LightFM Hybrid"] = lfm_model
    # else:
    #     print("Skipping LightFM Hybrid (not installed)")

    # 4. Evaluate
    print("\nEvaluating Popularity model...")
    best_recall = -1
    best_model_name = "Popularity"
    
    results = {}
    for name, model in models_to_eval.items():
        print(f"Evaluating {name}...")
        recall, ndcg = evaluate_recommender(model, val, k=10, sample_size=500)
        results[name] = {"Recall@10": recall, "NDCG@10": ndcg}
        print(f"  {name}: Recall@10 = {recall:.4f}, NDCG@10 = {ndcg:.4f}")
        
        if recall > best_recall:
            best_recall = recall
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with Recall@10 = {best_recall:.4f}")

    # # 5. XGBoost Ranking (Stage 2) - COMMENTED OUT FOR DEPLOYMENT
    # # Usually we use candidates from the best stage 1 model
    # print("\nTraining XGBoost Ranker on candidates from", best_model_name)
    # xgb_ranker = XGBoostRankingModel()
    # 
    # # Simple training for demonstration
    # # In reality we'd generate candidates for train set and rank
    # xgb_ranker.fit(train, recipes)
    
    # Save the best model (Popularity only)
    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(models_to_eval[best_model_name], f)
    
    # with open("models/xgb_ranker.pkl", "wb") as f:
    #     pickle.dump(xgb_ranker, f)
        
    print("Models saved to models/ directory.")

if __name__ == "__main__":
    main()
