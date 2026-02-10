import polars as pl
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os
import pickle

try:
    from lightfm import LightFM
    from lightfm.data import Dataset as LightFMDataset
    LIGHTFM_AVAILABLE = True
except ImportError:
    LIGHTFM_AVAILABLE = False

class PopularityModel:
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.popular_items = []

    def fit(self, train_df):
        print("Fitting Popularity Model...")
        # Assume Relevance column exists (handled by preprocessor)
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

class ContentBasedModel:
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.recipe_ids = []
        self.tfidf_matrix = None
        self.id_to_idx = {}

    def fit(self, recipes_df):
        print("Fitting Content-Based Model...")
        # Create ingredient text robustly
        recipes_df = recipes_df.with_columns(
            pl.col("RecipeIngredientParts").fill_null([])
        ).with_columns(
            pl.col("RecipeIngredientParts").map_elements(
                lambda x: " ".join([str(i) for i in x if i is not None]) if (hasattr(x, "__iter__") and not isinstance(x, (str, bytes))) else str(x if x is not None else ""),
                return_dtype=pl.String
            ).alias("ingredient_text")
        )
        self.recipe_ids = recipes_df["RecipeId"].to_list()
        self.id_to_idx = {rid: i for i, rid in enumerate(self.recipe_ids)}
        self.tfidf_matrix = self.tfidf.fit_transform(recipes_df["ingredient_text"].to_list())

    def recommend(self, user_history_recipe_ids, k=10):
        """Recommend based on user's liked recipes similarity"""
        if not user_history_recipe_ids:
            return []
        
        # Get indices of recipes user has liked
        indices = [self.id_to_idx[rid] for rid in user_history_recipe_ids if rid in self.id_to_idx]
        if not indices:
            return []
        
        # Average vector of liked recipes
        user_vector = self.tfidf_matrix[indices].mean(axis=0)
        user_vector = np.asarray(user_vector) # ensure it is an array
        
        # Note: In real scenarios, we should avoid large dense similarity matrices
        # For simplicity in this example, we calculate similarity to all
        sim_scores = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        # Exclude already liked recipes
        liked_set = set(user_history_recipe_ids)
        
        # Sort and get top-k
        top_indices = np.argsort(sim_scores)[::-1]
        recs = []
        for idx in top_indices:
            rid = self.recipe_ids[idx]
            if rid not in liked_set:
                recs.append(rid)
            if len(recs) == k:
                break
        return recs

class LightFMHybridModel:
    def __init__(self, n_components=30, loss='warp'):
        self.n_components = n_components
        self.loss = loss
        self.model = None
        self.dataset = None
        self.user_to_internal = {}
        self.item_to_internal = {}
        self.internal_to_item = {}

    def fit(self, train_df, recipes_df=None):
        if not LIGHTFM_AVAILABLE:
            print("LightFM not available. Skipping fit...")
            return

        print("Fitting LightFM Hybrid Model...")
        self.dataset = LightFMDataset()
        
        all_users = train_df["AuthorId"].unique().to_list()
        all_items = train_df["RecipeId"].unique().to_list()
        
        # If recipes_df provided, we can add item features
        item_features = None
        if recipes_df is not None:
            # Simple item features: Category
            unique_categories = recipes_df["RecipeCategory"].unique().to_list()
            self.dataset.fit(users=all_users, items=all_items, item_features=unique_categories)
            
            # Map item features
            item_feature_tuples = []
            for row in recipes_df.select(["RecipeId", "RecipeCategory"]).to_dicts():
                item_feature_tuples.append((row["RecipeId"], [row["RecipeCategory"]]))
            item_features = self.dataset.build_item_features(item_feature_tuples)
        else:
            self.dataset.fit(users=all_users, items=all_items)

        self.user_to_internal, _, self.item_to_internal, _ = self.dataset.mapping()
        self.internal_to_item = {v: k for k, v in self.item_to_internal.items()}
        
        (interactions, weights) = self.dataset.build_interactions(
            train_df.select(["AuthorId", "RecipeId", "Relevance"]).to_dicts()
        )
        
        self.model = LightFM(no_components=self.n_components, loss=self.loss, random_state=42)
        self.model.fit(interactions, sample_weight=weights, item_features=item_features, epochs=10, verbose=True)

    def recommend(self, user_id, k=10, item_features=None):
        if not self.model or user_id not in self.user_to_internal:
            return []
        
        u_idx = self.user_to_internal[user_id]
        n_items = len(self.item_to_internal)
        
        scores = self.model.predict(u_idx, np.arange(n_items), item_features=item_features)
        top_indices = np.argsort(-scores)[:k]
        return [self.internal_to_item[i] for i in top_indices]

class XGBoostRankingModel:
    def __init__(self):
        self.model = xgb.XGBRanker(
            objective='rank:pairwise',
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()

    def _extract_features(self, df, recipes_df, user_stats=None):
        # Merge factors/features
        # user_stats could have user_avg_rating, user_count etc.
        # joined = df.join(recipes_df, on="RecipeId")
        # For now, let's assume simple features: Calories, FatContent, ProteinContent
        features = ["Calories", "FatContent", "ProteinContent", "AggregatedRating"]
        X = df.select(features).fill_null(0).to_numpy()
        return self.scaler.fit_transform(X)

    def fit(self, train_df, recipes_df):
        print("Fitting XGBoost Ranking Model...")
        # Ensure consistent types for join
        train_df = train_df.with_columns(pl.col("RecipeId").cast(pl.Int64)).sort("AuthorId")
        recipes_df = recipes_df.with_columns(pl.col("RecipeId").cast(pl.Int64))
        
        groups = train_df.group_by("AuthorId").len().sort("AuthorId")["len"].to_numpy()
        
        # Prepare features
        joined = train_df.join(recipes_df.select(["RecipeId", "Calories", "FatContent", "ProteinContent", "AggregatedRating"]), on="RecipeId", how="left")
        X = self._extract_features(joined, recipes_df)
        y = joined["Relevance"].to_numpy()
        
        self.model.fit(X, y, group=groups)

    def rank(self, user_id, candidates_df, recipes_df):
        """Rank a list of candidate recipes for a specific user"""
        if candidates_df.is_empty():
            return []
        
        # Ensure consistent types
        candidates_df = candidates_df.with_columns(pl.col("RecipeId").cast(pl.Int64))
        recipes_df = recipes_df.with_columns(pl.col("RecipeId").cast(pl.Int64))
        
        # Extract features for candidates
        joined = candidates_df.join(recipes_df.select(["RecipeId", "Calories", "FatContent", "ProteinContent", "AggregatedRating"]), on="RecipeId", how="left")
        X = self._extract_features(joined, recipes_df)
        
        scores = self.model.predict(X)
        joined = joined.with_columns(pl.Series(name="score", values=scores))
        
        ranked_ids = joined.sort("score", descending=True)["RecipeId"].to_list()
        return ranked_ids
