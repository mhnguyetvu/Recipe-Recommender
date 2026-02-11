from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import polars as pl
import pickle
import os
import time
from src.models import PopularityModel, ContentBasedModel, LIGHTFM_AVAILABLE
from src.data_loader import DataLoader

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Define metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'API request latency', ['method', 'endpoint'])
RECOMMENDATION_COUNT = Counter('recommendations_total', 'Total recommendations served', ['model'])
MODEL_LATENCY = Histogram('model_inference_duration_seconds', 'Model inference latency', ['model'])
ACTIVE_USERS = Gauge('active_users_total', 'Number of active users')
MODEL_PRECISION = Gauge('model_precision', 'Model precision score', ['model'])
MODEL_RECALL = Gauge('model_recall', 'Model recall score', ['model'])

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

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": best_model is not None}

@app.get("/recommend/{reviewerId}")
def get_recommendations(reviewerId: int, k: int = 10, model: str = "best"):
    start_time = time.time()
    status_code = 200
    
    try:
        if best_model is None:
            status_code = 500
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Track active users
        ACTIVE_USERS.inc()
        
        # reviewerId corresponds to AuthorId
        model_start = time.time()
        
        # Get base recommendations from best model
        if hasattr(best_model, "recommend"):
            try:
                recs = best_model.recommend(reviewerId, k=k*2)
            except TypeError:
                recs = best_model.recommend(reviewerId, k=k*2)
        else:
            recs = []
        
        # Track model inference time
        model_duration = time.time() - model_start
        MODEL_LATENCY.labels(model="base").observe(model_duration)
            
        if not recs:
            return {
                "reviewerId": reviewerId, 
                "recommendations": [], 
                "message": "No specific recommendations found."
            }

        # Optional: Re-rank with XGBoost
        if xgb_ranker and recs:
            xgb_start = time.time()
            candidates_df = recipes_df.filter(pl.col("RecipeId").is_in(recs))
            ranked_recs = xgb_ranker.rank(reviewerId, candidates_df, recipes_df)
            recs = ranked_recs[:k]
            xgb_duration = time.time() - xgb_start
            MODEL_LATENCY.labels(model="xgboost").observe(xgb_duration)
            RECOMMENDATION_COUNT.labels(model="xgboost").inc(k)
        else:
            recs = recs[:k]
            RECOMMENDATION_COUNT.labels(model="base").inc(k)

        # Get recipe details
        result_df = recipes_df.filter(pl.col("RecipeId").is_in(recs)).select([
            "RecipeId", "Name", "RecipeCategory"
        ])
        recommendations = result_df.to_dicts()
        
        # Track request metrics
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(method="GET", endpoint="/recommend").observe(duration)
        REQUEST_COUNT.labels(method="GET", endpoint="/recommend", status=status_code).inc()
        
        return {
            "reviewerId": reviewerId,
            "count": len(recommendations),
            "recommendations": recommendations,
            "latency_ms": round(duration * 1000, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        status_code = 500
        REQUEST_COUNT.labels(method="GET", endpoint="/recommend", status=status_code).inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_USERS.dec()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2222)
