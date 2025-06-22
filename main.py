from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any
import os

app = FastAPI()

origins = [
    "http://localhost:3000",  # your frontend origin
    # Add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods like GET, POST, OPTIONS
    allow_headers=["*"],  # allow all headers
)


# Configuration
PROTECTED_COLUMNS = ['Industry', 'Headquarters', 'Name', 'Phone', 'Source']
REQUIRED_MODEL_FEATURES = [
    "Log_Revenue", "Log_Employees", "Revenue growth",
    "Industry", "Headquarters"
]

# Load dummy CSV once at startup
DATA_PATH = "dummy_data/companies.csv"
if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"Dummy data file not found at {DATA_PATH}")
dummy_df = pd.read_csv(DATA_PATH)

# Preprocessing on dummy_df: 
# Ensure columns exist and clean as necessary (you can add more cleaning if needed)
for col in ['Revenue (USD millions)', 'Revenue growth', 'Employees']:
    dummy_df[col] = dummy_df[col].astype(str).str.replace(',', '').str.replace('%', '').astype(float)

# Load model and preprocessor
try:
    model = joblib.load("models/xgb_lead_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise RuntimeError("Failed to load ML models")

class LeadRequest(BaseModel):
    industry: str = Field(..., description="Industry filter (case insensitive substring match)")
    location: str = Field(..., description="Location filter (case insensitive substring match)")
    page: int = Field(1, ge=1, description="Page number (starts at 1)")
    page_size: int = Field(10, ge=1, le=100, description="Number of items per page (max 100)")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/score-leads")
async def score_leads(req: LeadRequest) -> Dict[str, Any]:
    try:
        dummy_df['Industry'] = dummy_df['Industry'].str.strip()
        dummy_df['Headquarters'] = dummy_df['Headquarters'].str.strip()
        # Filter dummy data by industry and location substrings (case-insensitive)
        print(f"Filtering for industry containing: '{req.industry}' and location containing: '{req.location}'")
        filtered = dummy_df[
            dummy_df['Industry'].str.contains(req.industry, case=False, na=False) &
            dummy_df['Headquarters'].str.contains(req.location, case=False, na=False)
        ].copy()
        print(f"Total filtered leads: {len(filtered)}")

        total_count = len(filtered)
        if total_count == 0:
            return {"status": "success", "count": 0, "leads": [], "page": req.page, "page_size": req.page_size}

        # Pagination slicing
        start_idx = (req.page - 1) * req.page_size
        end_idx = start_idx + req.page_size
        paged_data = filtered.iloc[start_idx:end_idx].to_dict(orient="records")

        if not paged_data:
            return {"status": "success", "count": 0, "leads": [], "page": req.page, "page_size": req.page_size}

        # Run pipeline on paged data
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
        print(f"Error in pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def create_guaranteed_dataframe(data: List[Dict], industry: str, location: str) -> pd.DataFrame:
    if not data:
        raise ValueError("Empty business data received")

    df = pd.DataFrame(data)

    for col in PROTECTED_COLUMNS:
        if col not in df.columns:
            if col == 'Industry':
                df[col] = industry
            elif col == 'Headquarters':
                df[col] = location
            elif col == 'Source':
                df[col] = 'Dummy CSV'
            else:
                df[col] = None

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    missing_protected = [col for col in PROTECTED_COLUMNS if col not in df.columns]
    if missing_protected:
        raise ValueError(f"Missing protected columns: {missing_protected}")

    protected_backup = df[PROTECTED_COLUMNS].copy()

    # Industry mapping (optional, can customize or skip)
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

    # Generate derived features for prediction
    df = calculate_financials(df)

    # Restore protected columns exactly
    for col in PROTECTED_COLUMNS:
        if col in protected_backup.columns:
            df[col] = protected_backup[col]
        else:
            df[col] = None

    return df


def calculate_financials(df: pd.DataFrame) -> pd.DataFrame:
    # Use existing Employees, Revenue, Growth from CSV (make sure to handle NaNs)

    # Fill missing Employees with median or a default
    df['Employees'] = df['Employees'].fillna(df['Employees'].median())

    # Revenue (USD millions)
    df['Revenue (USD millions)'] = df['Revenue (USD millions)'].fillna(df['Revenue (USD millions)'].median())

    # Revenue growth
    df['Revenue growth'] = df['Revenue growth'].fillna(df['Revenue growth'].median())

    # Derived logs
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
        print(f"Prediction failed: {str(e)}")
        raise
