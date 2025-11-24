from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")
MODEL_NAME = "FIR-PREDICTOR-MODEL"
MODEL_VERSION = 3  # <-- FIXED VERSION YOU SELECTED


# --------------------
# FastAPI App
# --------------------
app = FastAPI()


# --------------------
# Input Model
# --------------------
class FIRRequest(BaseModel):
    brief: str


# --------------------
# Load MLflow Model Version
# --------------------
def load_registered_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    print(f"⏳ Loading MLflow Registered Model: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    print("✅ Model loaded successfully from MLflow Registry")
    return model


model = load_registered_model()


# --------------------
# /predict Endpoint
# --------------------
@app.post("/predict")
def predict(data: FIRRequest):
    brief_text = data.brief

    # Model returns concatenated "ACT||SECTION"
    result = model.predict([brief_text])[0]

    act, section = result.split("||")

    return {
        "act": act.strip(),
        "section": section.strip()
    }


@app.get("/")
def root():
    return {"status": "Python Predictor Running", "model_version": MODEL_VERSION}
