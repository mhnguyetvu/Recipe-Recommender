import polars as pl
import os

class DataLoader:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir

    def load_recipes(self):
        path = os.path.join(self.data_dir, "recipes.parquet")
        return pl.read_parquet(path)

    def load_reviews(self):
        path = os.path.join(self.data_dir, "reviews.parquet")
        return pl.read_parquet(path)

    def load_processed_data(self, processed_dir="data/processed"):
        train_path = os.path.join(processed_dir, "train_interactions.parquet")
        test_path = os.path.join(processed_dir, "test_interactions.parquet")
        return pl.read_parquet(train_path), pl.read_parquet(test_path)
