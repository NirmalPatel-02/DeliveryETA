from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from utils import preprocess_data, model, scaler
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_ajax")
async def predict_ajax(payload: dict):
    df = pd.DataFrame([payload])
    X_processed, meta = preprocess_data(df)
    X_scaled = scaler.transform(X_processed)
    pred = float(model.predict(X_scaled, verbose=0)[0][0])

    speed = "Fast" if pred <= 15 else "Normal" if pred <= 30 else "Slow"

    return JSONResponse({
        "predicted_minutes": round(pred, 2),
        "distance_km": meta["distance_km"],
        "pickup_delay_min": meta["pickup_delay_min"],
        "weather": meta["weather"],
        "traffic": meta["traffic"],
        "speed_category_rule": meta["speed_category"],
        "speed_rating": speed
    })