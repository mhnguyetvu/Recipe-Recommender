from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import polars as pl
import pickle
import os
from src.models import PopularityModel, ContentBasedModel, LIGHTFM_AVAILABLE
from src.data_loader import DataLoader

app = FastAPI(title="Recipe Recommendation System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Global variables for models and data
best_model = None
xgb_ranker = None
recipes_df = None
user_history = None

@app.on_event("startup")
def startup_event():
    global best_model, xgb_ranker, recipes_df, user_history
    
    # Load data
    loader = DataLoader(data_dir="data")
    recipes_df = loader.load_recipes()
    reviews_df = loader.load_reviews()
    
    # Load models
    if os.path.exists("models/best_model.pkl"):
        with open("models/best_model.pkl", "rb") as f:
            best_model = pickle.load(f)
    
    if os.path.exists("models/xgb_ranker.pkl"):
        with open("models/xgb_ranker.pkl", "rb") as f:
            xgb_ranker = pickle.load(f)
            
    # Pre-calculate user history for quick lookup (if needed by Content-Based)
    # Filter only relevant reviews
    relevant_reviews = reviews_df.filter(pl.col("Rating") >= 4)
    user_history = relevant_reviews.group_by("AuthorId").agg(pl.col("RecipeId").alias("history"))
    user_history = {row["AuthorId"]: row["history"] for row in user_history.to_dicts()}

@app.get("/")
def read_root():
    """Serve the web interface"""
    return FileResponse("static/index.html")

@app.get("/api")
def api_info():
    return {"message": "Welcome to the Recipe Recommendation API"}

@app.get("/recommend/{reviewerId}")
def get_recommendations(reviewerId: int, k: int = 10):
    if best_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # reviewerId corresponds to AuthorId
    try:
        # Get base recommendations from best model
        # Some models handle it internally, some need external history
        if hasattr(best_model, "recommend"):
            # Try to pass reviewerId
            try:
                # If it's a wrapper for CB
                recs = best_model.recommend(reviewerId, k=k*2) # get more for ranking
            except TypeError:
                recs = best_model.recommend(reviewerId, k=k*2)
        else:
            recs = []
            
        if not recs:
            # Fallback to popularity if no recs found for user
            # We can use a global popularity model if we have one
            return {"reviewerId": reviewerId, "recommendations": [], "message": "No specific recommendations found."}

        # Optional: Re-rank with XGBoost
        if xgb_ranker and recs:
            candidates_df = recipes_df.filter(pl.col("RecipeId").is_in(recs))
            ranked_recs = xgb_ranker.rank(reviewerId, candidates_df, recipes_df)
            recs = ranked_recs[:k]
        else:
            recs = recs[:k]

        # Get recipe names
        result_df = recipes_df.filter(pl.col("RecipeId").is_in(recs)).select(["RecipeId", "Name", "RecipeCategory"])
        recommendations = result_df.to_dicts()
        
        return {
            "reviewerId": reviewerId,
            "count": len(recommendations),
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2222)
