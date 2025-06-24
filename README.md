# 🔥 SaaSquatch - Hot Lead Scoring API

This project provides a backend API that scores potential business leads using a machine learning model trained on company data. The goal is to classify and rank leads based on industry, location, and business metrics to identify "hot" prospects.


## 📂 Project Structure
```
SASSQUATCH-IMPROVEMENT/
├── dummy_data/
│ └── companies.csv # Sample company dataset
├── models/
│ ├── preprocessor.joblib # Preprocessing pipeline (scaler + encoder)
│ ├── xgb_lead_model.joblib # XGBoost classifier for hot lead scoring
│ └── xgb_pipeline_model.joblib # (Optional) full pipeline version
├── main.py # 🚀 FastAPI app entrypoint
├── requirements.txt # Python dependencies
├── Procfile # For deployment (e.g. Railway)
├── vercel.json # (Optional) Vercel config
├── README.md
```



## 🚀 API Endpoints

### ✅ `GET /health`

Health check to confirm the server is operational.

**Response:**
```
{
  "status": "ok"
}
```

## 🎯 POST /score-leads
Score and rank potential leads using the trained machine learning model.

Request Body (JSON):
```
{
  "industry": "Technology",
  "location": "California",
  "page": 1,
  "page_size": 5
}
```
| Field      | Type   | Description                    |
| ---------- | ------ | ------------------------------ |
| industry   | string | Target industry (e.g., "Tech") |
| location   | string | Region or state (e.g., "CA")   |
| page       | int    | Page number for pagination     |
| page\_size | int    | Number of results per page     |

Response (Example):
```
{
  "total_results": 42,
  "page": 1,
  "page_size": 5,
  "results": [
    {
      "Name": "Example Corp",
      "Industry": "Technology",
      "Headquarters": "California",
      "Phone": null,
      "hot_lead_score": 0.92,
      "is_hot_lead": 1,
      "Source": "Dummy Source"
    }
  ]
}
```

## ⚙️ Model Information
Model: xgb_lead_model.joblib (XGBoost Classifier)

Preprocessor: preprocessor.joblib (ColumnTransformer)

Features Used:

Log_Revenue

Log_Employees

Revenue growth

One-hot encoded Industry and Headquarters

# ▶️ Running Locally
1. Create a virtual environment
```
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the app
```
uvicorn main:app --reload
```

## 🌐 Deployment
This backend is designed for cloud deployment. The following options are preconfigured:

Railway (via Procfile)

Optional Vercel integration (via vercel.json)

## 🧪 Testing
Using Postman
Import the included Postman collection:
```
SaaSquatch.postman_collection.json
```

Or use curl
```
curl -X POST https://sassquatch-improvement-production.up.railway.app/score-leads \
-H "Content-Type: application/json" \
-d '{
  "industry": "Technology",
  "location": "California",
  "page": 1,
  "page_size": 5
}'
```

## 📄 License
MIT License