from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run
from fastapi.responses import RedirectResponse
from fastapi.responses import Response
import sys

from ml.logger import logging
from ml.exception import TelcoChurnMLException
from ml.pipeline.training_pipeline import TrainingPipeline
from ml.pipeline.prediction import predict


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train", tags=["Train"])
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response("Training pipeline started.")
    except Exception as e:
        raise TelcoChurnMLException(e, sys)
    
@app.post("/predict/{customer_id}")
async def predict_churn(customer_id: str):
    try:
        churn_score = predict(customer_id)
        logging.info(f"Churn score for customer {customer_id}: {churn_score}")
        return {"customer_id": customer_id, "churn_score": churn_score}
        
    except Exception as e:
            raise TelcoChurnMLException(e,sys)
    
if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)   