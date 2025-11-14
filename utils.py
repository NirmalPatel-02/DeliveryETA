import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("model_ann.keras")
preprocessor = joblib.load("preprocessor.pkl")
scaler = joblib.load("scaler.pkl")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def classify_speed(traffic, weather):
    t = str(traffic).lower()
    w = str(weather).lower()
    if t in ("jam", "high") or w in ("stormy", "sandstorms", "fog", "windy"):
        return "Slow"
    if t in ("medium",) or w in ("cloudy",):
        return "Normal"
    return "Fast"

def preprocess_data(df_input):
    df = df_input.copy()

    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors='coerce').fillna(30).astype(int)
    df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors='coerce').fillna(4.5)
    df["Vehicle_condition"] = pd.to_numeric(df["Vehicle_condition"], errors='coerce').fillna(1).astype(int)
    df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors='coerce').fillna(0).astype(int)
    df["Festival"] = df["Festival"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors='coerce')
    df["order_day"] = df["Order_Date"].dt.day.fillna(15).astype(int)
    df["order_month"] = df["Order_Date"].dt.month.fillna(3).astype(int)
    df["order_weekday"] = df["Order_Date"].dt.weekday.fillna(2).astype(int)
    df["is_weekend"] = df["order_weekday"].isin([5, 6]).astype(int)

    def parse_time(t):
        if pd.isna(t): return pd.NaT
        t = str(t).strip()
        if len(t) == 5: t += ":00"
        return pd.to_datetime(t, format="%H:%M:%S", errors='coerce')

    df["Time_Orderd"] = df["Time_Orderd"].apply(parse_time).fillna(pd.Timestamp("15:30:00"))
    df["Time_Order_picked"] = df["Time_Order_picked"].apply(parse_time).fillna(pd.Timestamp("15:40:00"))

    df["order_hour"] = df["Time_Orderd"].dt.hour
    df["order_min"] = df["Time_Orderd"].dt.minute
    df["picked_hour"] = df["Time_Order_picked"].dt.hour
    df["picked_min"] = df["Time_Order_picked"].dt.minute
    df["pickup_delay_min"] = (df["picked_hour"]*60 + df["picked_min"]) - (df["order_hour"]*60 + df["order_min"])
    df["pickup_delay_min"] = df["pickup_delay_min"].clip(lower=0)

    df["distance_km"] = haversine(
        df["Restaurant_latitude"].astype(float),
        df["Restaurant_longitude"].astype(float),
        df["Delivery_location_latitude"].astype(float),
        df["Delivery_location_longitude"].astype(float)
    )

    df["is_rush_hour"] = df["order_hour"].apply(lambda x: 1 if x in [11,12,13,19,20,21] else 0)
    df["is_night"] = df["order_hour"].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)

    meta = {
        "distance_km": round(float(df["distance_km"].iloc[0]), 3),
        "pickup_delay_min": int(df["pickup_delay_min"].iloc[0]),
        "weather": str(df["Weatherconditions"].iloc[0]),
        "traffic": str(df["Road_traffic_density"].iloc[0]),
        "speed_category": classify_speed(df["Road_traffic_density"].iloc[0], df["Weatherconditions"].iloc[0])
    }

    df = df.drop(columns=["Order_Date", "Time_Orderd", "Time_Order_picked"], errors="ignore")

    X = preprocessor.transform(df)
    X_df = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())

    return X_df, meta