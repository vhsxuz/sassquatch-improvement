from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any
import os

# Set base directory relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

# CORS
origins = [
    "http://localhost:3000",  # Add your frontend origin here
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
PROTECTED_COLUMNS = ['Industry', 'Headquarters', 'Name', 'Phone', 'Source']
REQUIRED_MODEL_FEATURES = [
    "Log_Revenue", "Log_Employees", "Revenue growth",
    "Industry", "Headquarters"
]

# Load dataset
DATA_PATH = os.path.join(BASE_DIR, "dummy_data", "companies.csv")
if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"Dummy data file not found at {DATA_PATH}")
dummy_df = pd.read_csv(DATA_PATH)

# Clean dataset
for col in ['Revenue (USD millions)', 'Revenue growth', 'Employees']:
    dummy_df[col] = dummy_df[col].astype(str).str.replace(',', '').str.replace('%', '').astype(float)

# Load model and preprocessor
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_lead_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load ML models: {str(e)}")

# Schemas
class LeadRequest(BaseModel):
    industry: str = Field(..., description="Industry filter")
    location: str = Field(..., description="Location filter")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(10, ge=1, le=100, description="Page size")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/score-leads")
async def score_leads(req: LeadRequest) -> Dict[str, Any]:
    try:
        dummy_df['Industry'] = dummy_df['Industry'].str.strip()
        dummy_df['Headquarters'] = dummy_df['Headquarters'].str.strip()

        filtered = dummy_df[
            dummy_df['Industry'].str.contains(req.industry, case=False, na=False) &
            dummy_df['Headquarters'].str.contains(req.location, case=False, na=False)
        ].copy()

        total_count = len(filtered)
        if total_count == 0:
            return {"status": "success", "count": 0, "leads": [], "page": req.page, "page_size": req.page_size}

        start_idx = (req.page - 1) * req.page_size
        end_idx = start_idx + req.page_size
        paged_data = filtered.iloc[start_idx:end_idx].to_dict(orient="records")

        df = create_guaranteed_dataframe(paged_data, req.industry, req.location)
        df = generate_features(df)
        validate_protected_columns(df)
        validate_for_prediction(df)
        scored_df = predict_and_format(df)

        return {
            "status": "success",
            "count": total_count,
            "page": req.page,
            "page_size": req.page_size,
            "leads": scored_df.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def create_guaranteed_dataframe(data: List[Dict], industry: str, location: str) -> pd.DataFrame:
    df = pd.DataFrame(data)
    for col in PROTECTED_COLUMNS:
        if col not in df.columns:
            df[col] = {
                'Industry': industry,
                'Headquarters': location,
                'Source': 'Dummy CSV'
            }.get(col, None)
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    protected_backup = df[PROTECTED_COLUMNS].copy()
    industry_mapping = {
        'ferry_terminal': 'Transportation',
        'parking_entrance': 'Parking',
        'bicycle': 'Retail',
        'theatre': 'Entertainment',
        'convenience': 'Retail',
        'fast_food': 'Food Service',
        'books': 'Education',
        'post_office': 'Services'
    }
    df['Industry_Cleaned'] = df['Industry'].replace(industry_mapping)
    df = calculate_financials(df)
    for col in PROTECTED_COLUMNS:
        df[col] = protected_backup.get(col, None)
    return df

def calculate_financials(df: pd.DataFrame) -> pd.DataFrame:
    df['Employees'] = df['Employees'].fillna(df['Employees'].median())
    df['Revenue (USD millions)'] = df['Revenue (USD millions)'].fillna(df['Revenue (USD millions)'].median())
    df['Revenue growth'] = df['Revenue growth'].fillna(df['Revenue growth'].median())
    df['Log_Revenue'] = np.log1p(df['Revenue (USD millions)'])
    df['Log_Employees'] = np.log1p(df['Employees'])
    return df

def validate_protected_columns(df: pd.DataFrame) -> None:
    missing = [col for col in PROTECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Protected columns are missing: {missing}")

def validate_for_prediction(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_MODEL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for prediction: {missing}")

def predict_and_format(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()
    X = df[REQUIRED_MODEL_FEATURES].copy()
    try:
        X_transformed = preprocessor.transform(X)
        if hasattr(model, 'predict_proba'):
            result_df['hot_lead_score'] = model.predict_proba(X_transformed)[:, 1]
        else:
            result_df['hot_lead_score'] = model.predict(X_transformed)
        result_df['is_hot_lead'] = (result_df['hot_lead_score'] >= 0.7).astype(int)
        output_cols = [
            'Name', 'Industry', 'Headquarters', 'Phone',
            'Employees', 'Revenue (USD millions)', 'Revenue growth',
            'hot_lead_score', 'is_hot_lead', 'Source'
        ]
        for col in output_cols:
            if col not in result_df.columns:
                result_df[col] = None
        return result_df[output_cols].sort_values('hot_lead_score', ascending=False)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")