from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel, conlist
from typing import List
from joblib import load
from fastapi.middleware.cors import CORSMiddleware

class FraudDetectionRequest(BaseModel):
    data: List[conlist(float)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_bundel = load('model/xgboost.joblib')
    app.model, app.columns = model_bundel
    yield

app = FastAPI(lifespan=lifespan,
              title="Fraud Detection API",
              description="API for detecting fraud using a pre-trained XGBoost model",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(user_input: FraudDetectionRequest):
    data = user_input.data
    preds = app.model.predict(data)
    return {"predictions": preds.tolist()}
