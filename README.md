# 🍳 Recipe Recommender System

A production-ready personalized Top-N recipe recommendation system with real-time monitoring using multiple ML algorithms and high-performance data processing.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Monitoring](https://img.shields.io/badge/Monitoring-Grafana%2BPrometheus-orange.svg)](https://grafana.com/)

---

## 📑 Table of Contents

- [Dataset](#-dataset)
- [Problem Definition](#-problem-definition)
- [Quick Start](#-quick-start)
- [Monitoring Stack](#-monitoring-grafana--prometheus)
- [Project Structure](#-project-structure)
- [Models](#-implemented-models)
- [Evaluation](#-evaluation-metrics)
- [API](#-api-endpoints)
- [Development](#-development)
- [Production Deployment](#-production-deployment)

---

## 📊 Dataset

**Source**: [Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)

- **Recipes**: 230K+ recipes with ingredients, nutrition, categories
- **Reviews**: 1.1M+ user ratings and reviews
- **Users**: 270K+ unique users

---

## 🎯 Problem Definition

**Task**: Personalized Top-N recipe recommendation

**Objective**: Given a user's past interactions, rank recipes so that future relevant recipes appear at the top.

**Relevance Criteria**:
- ⭐ Rating ≥ 4: **Relevant** (Positive, label = 1)
- ⭐ Rating ≤ 2: **Negative** (label = 0)
- ⭐ Rating = 3: **Dropped** (neutral, not used)

---

## 🚀 Quick Start

### Option A: Local Development

#### 1. Clone & Install
```bash
git clone <your-repo-url>
cd Recipe-Recommender
pip install -r requirements.txt
```

#### 2. Download Data
Download from [Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) and place in:
```
data/raw/
├── recipes.parquet
└── reviews.parquet
```

#### 3. Run
```bash
# Train models and start API
python main.py --mode all

# Or train only
python main.py --mode train

# Or serve only (requires trained models)
python main.py --mode serve
```

### Option B: Docker (Recommended for Production)

```bash
# Start everything (API + Monitoring)
./start-monitoring.sh

# Or manually
docker-compose up -d
```

**Access**:
- 🌐 **API**: http://localhost:2222
- 📚 **API Docs**: http://localhost:2222/docs
- 📈 **Grafana**: http://localhost:3000 (admin/admin)
- 📊 **Prometheus**: http://localhost:9090

---

## 📊 Monitoring (Grafana + Prometheus)

### Features

✅ **Real-time Metrics**:
- API performance (request rate, latency, error rate)
- Model performance (inference time, recommendations served)
- System health (active users, resource usage)

✅ **Pre-built Dashboard** with 8 panels:
1. API Request Rate
2. API Latency (p95)
3. Model Inference Latency
4. Recommendations Served by Model
5. Active Users
6. Total Requests
7. Error Rate
8. Model Precision/Recall

### Quick Start

```bash
# Start monitoring stack
./start-monitoring.sh

# Generate test traffic
curl http://localhost:2222/recommend/12345?k=10

# View metrics
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
```

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `api_requests_total` | Counter | Total API requests by method, endpoint, status |
| `api_request_duration_seconds` | Histogram | Request latency distribution |
| `recommendations_total` | Counter | Recommendations served by model |
| `model_inference_duration_seconds` | Histogram | Model inference time |
| `active_users_total` | Gauge | Current concurrent users |
| `model_precision` | Gauge | Model precision score |
| `model_recall` | Gauge | Model recall score |

### Useful PromQL Queries

```promql
# Request rate (last 5 min)
rate(api_requests_total[5m])

# API Latency p95
histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))

# Error rate
sum(rate(api_requests_total{status=~"5.."}[5m])) / sum(rate(api_requests_total[5m]))

# Model inference latency
histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m]))
```

### Troubleshooting

**Metrics not showing?**
```bash
# Check API health
curl http://localhost:2222/health

# Check metrics endpoint
curl http://localhost:2222/metrics

# View logs
docker-compose logs -f api
docker-compose logs -f prometheus
```

**Dashboard empty?**
1. Check time range (top-right in Grafana)
2. Verify Prometheus datasource: http://localhost:9090/targets
3. Generate some traffic to create metrics

---

## 🏗️ Project Structure

```
Recipe-Recommender/
├── data/
│   ├── raw/                    # Raw parquet files
│   └── processed/              # Preprocessed train/val/test splits
├── src/                        # Source code
│   ├── data_loader.py          # Data loading utilities
│   ├── preprocessing.py        # Data cleaning & splitting
│   ├── models.py               # Recommendation models
│   ├── evaluation.py           # Metrics (Recall@K, NDCG@K)
│   ├── train_and_evaluate_all.py  # Training pipeline
│   └── api.py                  # FastAPI server with metrics
├── models/                     # Saved trained models
├── monitoring/                 # Monitoring configuration
│   ├── prometheus.yml          # Prometheus config
│   └── grafana/                # Grafana dashboards & provisioning
├── notebooks/
│   └── EDA.ipynb              # Exploratory Data Analysis
├── scripts/
│   └── run_eda.py             # EDA script
├── docker-compose.yml         # Docker services
├── Dockerfile                 # API container
├── start-monitoring.sh        # Quick start script
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🤖 Implemented Models

### 1. **Popularity Baseline** 📊
- Recommends most popular recipes (highest positive ratings)
- **Pros**: Simple, fast, good for cold-start users
- **Cons**: No personalization
- **Use case**: Fallback, new users

### 2. **Content-Based Filtering** 🔍
- Uses TF-IDF on recipe ingredients + nutrition features
- Recommends recipes similar to user's liked recipes
- **Pros**: Explainable, good for new recipes (cold-start items)
- **Cons**: Limited diversity, needs user history
- **Use case**: Users with clear preferences

### 3. **LightFM (Hybrid Collaborative Filtering)** ⚡
- Matrix factorization with WARP loss
- Learns user and item embeddings
- **Pros**: Personalized, can use item features
- **Cons**: Needs sufficient interaction data
- **Use case**: General personalization

### 4. **XGBoost Ranker** 🎯 (Best Performance)
- **Two-stage approach**:
  - **Stage 1**: Generate candidates (Popularity + Content-Based)
  - **Stage 2**: Re-rank with XGBoost using rich features
- **Pros**: Best performance, combines multiple signals
- **Cons**: More complex, slower
- **Use case**: Production recommendations

---

## 📈 Evaluation Metrics

### Recall@K
Percentage of relevant items (that user interacted with in test set) that appear in top-K recommendations.

**Formula**: `Hits in Top-K / Total Relevant Items`

**Example**: User likes 20 recipes in test set, we recommend 10, and 3 are correct → Recall@10 = 3/20 = 15%

### NDCG@K (Normalized Discounted Cumulative Gain)
Accounts for the **position** of relevant items in the ranking. Higher positions get more weight.

**Why it matters**: Finding a relevant item at position 1 is better than at position 10.

### Typical Results

Based on evaluation on validation set (1000 users):

| Model | Precision@10 | Recall@10 | Improvement |
|-------|--------------|-----------|-------------|
| Popularity | 0.0067 | 0.0207 | baseline |
| LightFM | 0.0068 | 0.0211 | +1.5% |
| Content-Based | 0.0072 | 0.0223 | +7.5% |
| **XGBoost** | **0.0085** | **0.0265** | **+26.9%** |

**Note**: Low absolute values are normal for recommendation systems with large item catalogs (230K recipes) and sparse interactions. Focus on **relative improvement**.

---

## 📡 API Endpoints

### GET `/recommend/{user_id}`
Get top-K recommendations for a user.

**Parameters**:
- `user_id` (path): User ID (AuthorId)
- `k` (query, optional): Number of recommendations (default: 10)
- `model` (query, optional): Model to use (default: best)

**Example**:
```bash
curl http://localhost:2222/recommend/12345?k=10
```

**Response**:
```json
{
  "reviewerId": 12345,
  "count": 10,
  "recommendations": [
    {
      "RecipeId": 456,
      "Name": "Chocolate Chip Cookies",
      "RecipeCategory": "Dessert"
    },
    ...
  ],
  "latency_ms": 45.2
}
```

### GET `/health`
Health check endpoint.

```bash
curl http://localhost:2222/health
```

### GET `/metrics`
Prometheus metrics endpoint.

```bash
curl http://localhost:2222/metrics
```

### Interactive Docs
Visit http://localhost:2222/docs for Swagger UI.

---

## 🛠️ Development

### Run EDA
```
jupyter notebook notebooks/EDA.ipynb
```

### Train Individual Models

```python
from src.models import PopularityModel, ContentBasedModel, LightFMHybridModel
from src.data_loader import load_data

# Load data
train_df, val_df, test_df, recipes_df = load_data()

# Train Popularity
pop_model = PopularityModel(top_k=100)
pop_model.fit(train_df)
recommendations = pop_model.recommend(user_id=12345, k=10)

# Train Content-Based
cb_model = ContentBasedModel()
cb_model.fit(recipes_df)
user_history = [101, 202, 303]  # Recipe IDs user liked
recommendations = cb_model.recommend(user_history, k=10)

# Train LightFM
lfm_model = LightFMHybridModel(n_components=64, loss='warp')
lfm_model.fit(train_df, recipes_df)
recommendations = lfm_model.recommend(user_id=12345, k=10)
```

### Evaluate Models

```python
from src.evaluation import evaluate_recommender

recall, ndcg = evaluate_recommender(
    model=pop_model,
    test_df=test_df,
    k=10,
    sample_size=1000
)

print(f"Recall@10: {recall:.4f}")
print(f"NDCG@10: {ndcg:.4f}")
```

---

## 🚢 Production Deployment

### Docker Commands

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f api
docker-compose logs -f prometheus
docker-compose logs -f grafana

# Restart services
docker-compose restart

# Stop services
docker-compose down

# Rebuild
docker-compose build --no-cache
docker-compose up -d

# Check status
docker-compose ps
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:2222/recommend/12345?k=10

# Using curl loop
for i in {1..100}; do
  curl http://localhost:2222/recommend/$((RANDOM % 10000))?k=10 &
done
wait
```

### Production Checklist

- [ ] Change Grafana admin password
- [ ] Set up alerts in Grafana
- [ ] Configure notification channels (Slack, Email)
- [ ] Add resource limits to containers
- [ ] Enable HTTPS for all services
- [ ] Set up log aggregation
- [ ] Configure backup for models and dashboards
- [ ] Set up CI/CD pipeline
- [ ] Load test with realistic traffic
- [ ] Monitor and optimize based on metrics

### Environment Variables

Create `.env` file:
```bash
# API
API_PORT=2222
API_HOST=0.0.0.0

# Grafana
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=your-secure-password

# Prometheus
PROMETHEUS_RETENTION_TIME=15d
```

---

## 📝 Key Features

✅ **Multiple Algorithms**: Popularity, Content-Based, LightFM, XGBoost  
✅ **High Performance**: Polars for fast data processing  
✅ **Production Ready**: FastAPI server with REST API  
✅ **Real-time Monitoring**: Grafana + Prometheus dashboards  
✅ **Comprehensive Evaluation**: Recall@K, NDCG@K metrics  
✅ **Explainable**: Content-based provides similarity explanations  
✅ **Scalable**: Efficient sparse matrix operations  
✅ **Dockerized**: Easy deployment with docker-compose  

---

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### Code Quality
```bash
flake8 src/
black src/
```

### API Tests
```bash
# Health check
curl http://localhost:2222/health

# Get recommendations
curl http://localhost:2222/recommend/12345?k=10

# Check metrics
curl http://localhost:2222/metrics
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **Dataset**: [Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)
- **Libraries**: Polars, LightFM, XGBoost, FastAPI, scikit-learn, Prometheus, Grafana
- **Inspiration**: Modern recommendation systems in production

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 📚 Additional Resources

- [Polars Documentation](https://pola-rs.github.io/polars/)
- [LightFM Documentation](https://making.lyst.com/lightfm/docs/home.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

---

**Built with ❤️ for food lovers and ML enthusiasts**

**Stack**: Python · FastAPI · Polars · LightFM · XGBoost · Docker · Prometheus · Grafana
