import polars as pl
import os
import sys

# Add project root to path for imports
sys.path.append(os.getcwd())

from src.data_loader import DataLoader

def run_eda():
    loader = DataLoader(data_dir="data/raw")
    try:
        recipes = loader.load_recipes()
        reviews = loader.load_reviews()
    except:
        print("Data not found in data/raw/")
        return

    print("--- Stats ---")
    print(f"Recipes: {len(recipes)}")
    print(f"Reviews: {len(reviews)}")
    
    print("\n--- Ratings Distribution ---")
    print(reviews.group_by("Rating").len().sort("Rating"))
    
    print("\n--- Top Categories ---")
    if "RecipeCategory" in recipes.columns:
        print(recipes.group_by("RecipeCategory").len().sort("len", descending=True).head(10))

if __name__ == "__main__":
    run_eda()
