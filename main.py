import os
import polars as pl
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.models import PopularityModel, SVDModel
from src.evaluation import evaluate_model

def run_pipeline():
    # 1. Load Data
    loader = DataLoader(data_dir="data/raw")
    try:
        recipes = loader.load_recipes()
        reviews = loader.load_reviews()
    except FileNotFoundError:
        print("Error: Raw data parquet files not found in data/raw/. Please ensure they are placed there.")
        return

    # 2. Preprocess
    preprocessor = DataPreprocessor()
    processed_reviews = preprocessor.preprocess_reviews(reviews)
    train, test = preprocessor.split_data(processed_reviews)
    
    # 3. Save Processed Data
    os.makedirs("data/processed", exist_ok=True)
    train.write_parquet("data/processed/train_interactions.parquet")
    test.write_parquet("data/processed/test_interactions.parquet")
    print("Processed data saved to data/processed/")

    # 4. Fit & Evaluate Models
    models = {
        "Popularity": PopularityModel(top_k=100),
        "SVD": SVDModel(n_components=50)
    }
    
    for name, model in models.items():
        model.fit(train)
        recall, ndcg = evaluate_model(model, test, k=10, sample_size=500)
        print(f"[{name}] Recall@10: {recall:.4f} | NDCG@10: {ndcg:.4f}")

if __name__ == "__main__":
    run_pipeline()
