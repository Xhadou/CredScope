# CredScope Deployment Guide

Complete guide for deploying CredScope in production or development environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [API Usage](#api-usage)
6. [Dashboard Usage](#dashboard-usage)
7. [Testing](#testing)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- 8GB+ RAM recommended
- Trained model files in `models/` directory

### Install Dependencies

```bash
# Clone repository
git clone <repository-url>
cd CredScope

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Local Development

### 1. Run FastAPI Server

```bash
# From project root
uvicorn src.credscope.api.main:app --reload --host 0.0.0.0 --port 8000
```

**API will be available at:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

### 2. Run Streamlit Dashboard

```bash
# From project root
streamlit run src/credscope/dashboard/app.py
```

**Dashboard will be available at:** http://localhost:8501

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000,
    "DAYS_BIRTH": -15000,
    "DAYS_EMPLOYED": -3000,
    "CODE_GENDER": 1,
    "EXT_SOURCE_1": 0.65,
    "EXT_SOURCE_2": 0.72,
    "EXT_SOURCE_3": 0.58
  }'
```

---

## Docker Deployment

### Build and Run with Docker Compose

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- API: http://localhost:8000
- Dashboard: http://localhost:8501

### Individual Docker Commands

**API:**
```bash
docker build -f Dockerfile.api -t credscope-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro credscope-api
```

**Dashboard:**
```bash
docker build -f Dockerfile.dashboard -t credscope-dashboard .
docker run -p 8501:8501 -v $(pwd)/models:/app/models:ro credscope-dashboard
```

---

## Production Deployment

### Environment Variables

Create a `.env` file:

```bash
# Logging
LOG_LEVEL=INFO

# Model paths
MODELS_DIR=/app/models

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# CORS (adjust for your domain)
CORS_ORIGINS=["https://yourdomain.com"]
```

### Production Docker Compose

```yaml
version: '3.8'

services:
  api:
    image: credscope-api:latest
    restart: always
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=WARNING
      - API_WORKERS=4
    volumes:
      - ./models:/app/models:ro
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  dashboard:
    image: credscope-dashboard:latest
    restart: always
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models:ro
    depends_on:
      - api

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - api
      - dashboard
```

### Nginx Configuration (Reverse Proxy)

```nginx
upstream api {
    server api:8000;
}

upstream dashboard {
    server dashboard:8501;
}

server {
    listen 80;
    server_name yourdomain.com;

    # API
    location /api/ {
        proxy_pass http://api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Dashboard
    location / {
        proxy_pass http://dashboard/;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Scaling with Kubernetes

See `k8s/` directory for Kubernetes manifests (if applicable).

---

## API Usage

### Authentication (if enabled)

```python
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
```

### Python Client Example

```python
import requests

API_URL = "http://localhost:8000"

# Single prediction
applicant = {
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 500000,
    "DAYS_BIRTH": -15000,
    "EXT_SOURCE_1": 0.65,
    "EXT_SOURCE_2": 0.72,
    "EXT_SOURCE_3": 0.58
}

response = requests.post(f"{API_URL}/predict", json=applicant)
result = response.json()

print(f"Decision: {result['decision']}")
print(f"Default Probability: {result['default_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

### Batch Predictions

```python
batch_request = {
    "applicants": [
        applicant1,
        applicant2,
        applicant3
    ]
}

response = requests.post(f"{API_URL}/predict/batch", json=batch_request)
results = response.json()

print(f"Total: {results['total_count']}")
print(f"Approved: {results['approved_count']}")
print(f"Rejected: {results['rejected_count']}")
```

### Explanation Endpoint

```python
response = requests.post(f"{API_URL}/predict/explain", json=applicant)
result = response.json()

print(f"Decision: {result['decision']}")
print("\nTop Contributing Features:")
for feat in result['top_features'][:5]:
    print(f"  {feat['feature']}: {feat['shap_value']:.4f} ({feat['impact']})")
```

---

## Dashboard Usage

### Features

1. **Single Prediction**
   - Input applicant details
   - Get instant decision (APPROVE/REJECT)
   - View default probability and risk level
   - See top contributing features (SHAP)

2. **Batch Predictions**
   - Upload CSV with multiple applicants
   - Process hundreds of applications
   - Download results with predictions
   - View distribution of risk scores

3. **Analytics**
   - Feature importance rankings
   - Model performance metrics
   - Category-wise analysis

### CSV Format for Batch Upload

```csv
AMT_INCOME_TOTAL,AMT_CREDIT,DAYS_BIRTH,EXT_SOURCE_1,EXT_SOURCE_2,EXT_SOURCE_3
180000,500000,-15000,0.65,0.72,0.58
200000,450000,-12000,0.70,0.68,0.62
150000,350000,-18000,0.55,0.60,0.50
```

---

## Testing

### Run Unit Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_api.py -v

# With coverage
pytest tests/ --cov=src.credscope --cov-report=html
```

### Integration Testing

```bash
# Start services
docker-compose up -d

# Wait for services to be ready
sleep 10

# Test API endpoints
curl http://localhost:8000/health

# Test prediction
python tests/integration/test_end_to_end.py

# Cleanup
docker-compose down
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

---

## Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00",
  "models_loaded": true,
  "version": "1.0.0"
}
```

### Logging

Logs are written to stdout/stderr by default.

**View logs:**
```bash
# Docker
docker-compose logs -f api

# Local
# Logs appear in console
```

### Metrics (Optional - Prometheus)

Add Prometheus metrics:

```python
# In main.py
from prometheus_fastapi_instrumentator import Instrumentator

@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app)
```

---

## Troubleshooting

### Models Not Loading

**Error:** `FileNotFoundError: LightGBM model not found`

**Solution:**
1. Ensure model files are in `models/` directory
2. Check file names match expected names
3. Verify file permissions

```bash
ls -l models/
# Should show:
# - lightgbm_model.txt
# - xgboost_model.json
# - catboost_model.cbm
```

### Memory Issues

**Error:** `MemoryError` or container killed

**Solution:**
1. Increase Docker memory limits
2. Reduce batch size
3. Use smaller sample for SHAP explanations

### Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn src.credscope.api.main:app --port 8001
```

### Dashboard Not Loading Models

**Solution:**
1. Check models directory path in dashboard code
2. Ensure models volume is mounted correctly
3. Restart dashboard service

```bash
docker-compose restart dashboard
```

### Prediction Errors

**Error:** `Missing features` or `Prediction failed`

**Solution:**
1. Check input data has all required fields
2. Verify data types match schema
3. Check API logs for detailed error

```bash
docker-compose logs api | tail -50
```

---

## Performance Optimization

### API Optimization

1. **Use Gunicorn for production:**
```bash
gunicorn src.credscope.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. **Enable caching:**
```python
@lru_cache(maxsize=1000)
def cached_prediction(features_hash):
    # Cache frequent predictions
```

3. **Batch predictions for efficiency**

### Dashboard Optimization

1. **Cache model loading:**
Already implemented with `@st.cache_resource`

2. **Limit SHAP calculations:**
Only compute for single predictions, not batch

3. **Use session state:**
Store results in `st.session_state` to avoid recomputation

---

## Security Considerations

1. **Add Authentication:**
   - Implement API key authentication
   - Use OAuth2 for dashboard

2. **Input Validation:**
   - Already implemented with Pydantic
   - Add rate limiting

3. **HTTPS:**
   - Use SSL/TLS in production
   - Configure Nginx with certificates

4. **Environment Variables:**
   - Never commit API keys
   - Use secrets management (AWS Secrets Manager, etc.)

---

## Maintenance

### Updating Models

```bash
# 1. Train new models
python scripts/train_ensemble.py

# 2. Backup old models
mv models models_backup_$(date +%Y%m%d)

# 3. Copy new models
cp -r new_models/* models/

# 4. Restart services
docker-compose restart
```

### Monitoring Model Drift

```python
from src.credscope.evaluation.fairness import audit_model_fairness

# Regular fairness audits
metrics, report = audit_model_fairness(
    y_true, y_pred, y_proba,
    protected_data,
    ['CODE_GENDER', 'AGE_GROUP'],
    output_dir='fairness_audit'
)
```

---

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [repository-url]/docs
- Email: support@credscope.com

---

## License

MIT License - See LICENSE file for details
