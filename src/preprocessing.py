import polars as pl
import numpy as np
import pandas as pd
import ast

class DataPreprocessor:
    @staticmethod
    def preprocess_reviews(reviews_df):
        print("Preprocessing reviews...")
        # 1. Cast DateSubmitted to datetime if it's a string
        if reviews_df.schema["DateSubmitted"] == pl.String:
            reviews_df = reviews_df.with_columns(
                pl.col("DateSubmitted").str.to_datetime(strict=False)
            )
        
        # 2. Filter ratings: >=4 relevant, <=2 negative, drop 3
        reviews_df = reviews_df.filter(pl.col("Rating") != 3)
        reviews_df = reviews_df.with_columns(
            pl.when(pl.col("Rating") >= 4).then(1.0).otherwise(0.0).alias("Relevance")
        )
        return reviews_df

    @staticmethod
    def split_data(df, val_ratio=0.1, test_ratio=0.1):
        print(f"Splitting data (val_ratio={val_ratio}, test_ratio={test_ratio})...")
        df = df.sort("DateSubmitted")
        n = len(df)
        test_idx = int(n * (1 - test_ratio))
        val_idx = int(n * (1 - test_ratio - val_ratio))

        train = df[:val_idx, :]
        val = df[val_idx:test_idx, :]
        test = df[test_idx:, :]

        # Keep only users in val and test that are present in train
        train_users = train["AuthorId"].unique()
        val = val.filter(pl.col("AuthorId").is_in(train_users))
        test = test.filter(pl.col("AuthorId").is_in(train_users))

        return train, val, test
    
    @staticmethod
    def create_ingredient_text(x):
        # None / NaN (scalar)
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""

        # Nếu là numpy array -> list
        if isinstance(x, np.ndarray):
            x = x.tolist()

        # Nếu là string -> parse nếu giống list
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    x = ast.literal_eval(s)
                except Exception:
                    return s.lower()
            else:
                return s.lower()

        # Nếu rỗng
        if isinstance(x, (list, tuple)) and len(x) == 0:
            return ""

        # Join list/tuple
        if isinstance(x, (list, tuple)):
            out = []
            for ing in x:
                if ing is None:
                    continue
                if isinstance(ing, float) and np.isnan(ing):
                    continue
                if isinstance(ing, np.ndarray):
                    ing = " ".join(map(str, ing.tolist()))
                s = str(ing).strip().lower()
                if s != "":
                    out.append(s)
            return " ".join(out)

        # Fallback
        return str(x).strip().lower()

