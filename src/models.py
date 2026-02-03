import polars as pl
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

class PopularityModel:
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.popular_items = []

    def fit(self, train_df):
        print("Fitting Popularity Model...")
        self.popular_items = (
            train_df.filter(pl.col("Relevance") == 1)
            .group_by("RecipeId")
            .len()
            .sort("len", descending=True)
            .head(self.top_k)["RecipeId"]
            .to_list()
        )

    def recommend(self, user_id, k=10):
        return self.popular_items[:k]

class SVDModel:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.U = None
        self.V = None
        self.user_map = {}
        self.recipe_map = {}
        self.unique_recipes = []

    def fit(self, train_df):
        print(f"Fitting SVD Model (k={self.n_components})...")
        unique_users = train_df["AuthorId"].unique().to_list()
        self.unique_recipes = train_df["RecipeId"].unique().to_list()
        
        self.user_map = {uid: i for i, uid in enumerate(unique_users)}
        self.recipe_map = {rid: i for i, rid in enumerate(self.unique_recipes)}
        
        train_mapped = train_df.with_columns([
            pl.col("AuthorId").replace(self.user_map).alias("u_idx"),
            pl.col("RecipeId").replace(self.recipe_map).alias("r_idx")
        ])
        
        row = train_mapped["u_idx"].to_numpy()
        col = train_mapped["r_idx"].to_numpy()
        data = train_mapped["Relevance"].to_numpy()
        
        mat = sp.csr_matrix((data, (row, col)), shape=(len(unique_users), len(self.unique_recipes)))
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.U = svd.fit_transform(mat)
        self.V = svd.components_.T
        
        # Candidate pool (top 1000 popular for efficiency)
        self.candidate_ids = (
            train_df.group_by("RecipeId").len().sort("len", descending=True).head(1000)["RecipeId"].to_list()
        )
        self.candidate_indices = [self.recipe_map[r] for r in self.candidate_ids if r in self.recipe_map]
        self.V_cand = self.V[self.candidate_indices]

    def recommend(self, user_id, k=10):
        if user_id not in self.user_map:
            return [] # or return popularity
        u_idx = self.user_map[user_id]
        u_vec = self.U[u_idx]
        scores = u_vec @ self.V_cand.T
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.candidate_ids[i] for i in top_indices]
