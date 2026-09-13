import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip
import json
from pathlib import Path

st.set_page_config(page_title="NYC Airbnb Price Predictor", page_icon="🏙️", layout="wide")

# ---------- Load model + metadata ----------
BASE = Path(__file__).resolve().parent.parent

@st.cache_resource
def load_model():
    with gzip.open(BASE / "models" / "airbnb_price_pipeline_v2.pkl.gz", "rb") as f:
        return joblib.load(f)

@st.cache_data
def load_metadata():
    with open(BASE / "models" / "app_metadata.json") as f:
        meta = json.load(f)
    return meta

model = load_model()
meta = load_metadata()

# Manhattan center for dist calc
MANHATTAN_LAT, MANHATTAN_LON = 40.7580, -73.9855

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ---------- Header ----------
st.title("🏙️ NYC Airbnb Price Predictor")
st.markdown("Enter listing details to get an **estimated nightly price** (USD).")
st.markdown("---")

# ---------- Input form ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location")
    borough = st.selectbox(
        "Borough",
        options=meta["boroughs"],
        index=meta["boroughs"].index("Manhattan") if "Manhattan" in meta["boroughs"] else 0,
    )
    neighbourhood = st.selectbox(
        "Neighbourhood",
        options=meta["neighbourhoods"],
        index=meta["neighbourhoods"].index("Williamsburg") if "Williamsburg" in meta["neighbourhoods"] else 0,
        help="Type to search. 221 neighbourhoods from training data.",
    )
    latitude = st.number_input("Latitude", min_value=40.4, max_value=41.0,
                               value=40.7100, step=0.001, format="%.4f")
    longitude = st.number_input("Longitude", min_value=-74.3, max_value=-73.6,
                                value=-73.9500, step=0.001, format="%.4f")

with col2:
    st.subheader("🏠 Listing details")
    room_type = st.selectbox(
        "Room type",
        options=meta["room_types"],
        index=meta["room_types"].index("Entire home/apt") if "Entire home/apt" in meta["room_types"] else 0,
    )
    minimum_nights = st.slider("Minimum nights", 1, 45, 2)
    number_of_reviews = st.slider("Number of reviews", 0, 300, 10)
    reviews_per_month = st.slider("Reviews per month", 0.0, 10.0, 0.5, step=0.1)
    calculated_host_listings_count = st.slider("Host's total listings", 1, 200, 1)
    availability_365 = st.slider("Availability (days/year)", 0, 365, 180)

st.markdown("---")

# ---------- Predict ----------
if st.button("💵 Estimate Price", type="primary", use_container_width=True):
    dist_km = haversine_km(latitude, longitude, MANHATTAN_LAT, MANHATTAN_LON)
    has_reviews = 1 if number_of_reviews > 0 else 0
    is_multi_host = 1 if calculated_host_listings_count > 1 else 0
    # If no reviews, "days since last review" is unknown -> large sentinel
    # Otherwise assume reviews are somewhat recent
    days_since_last_review = 9999 if has_reviews == 0 else max(1, int(365 / max(reviews_per_month, 0.1)))

    input_df = pd.DataFrame([{
        "neighbourhood_group": borough,
        "neighbourhood": neighbourhood,
        "latitude": latitude,
        "longitude": longitude,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "days_since_last_review": days_since_last_review,
        "has_reviews": has_reviews,
        "dist_to_center_km": dist_km,
        "is_multi_host": is_multi_host,
    }])

    with st.spinner("Predicting..."):
        pred_log = model.predict(input_df)[0]
        pred_price = np.expm1(pred_log)

    st.success(f"### 💰 Estimated nightly price: **${pred_price:,.0f}**")
    st.caption("Statistical estimate from a machine learning model — not a real quote.")

    with st.expander("🔍 See input summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

st.markdown("---")
st.caption("DS605 Lab 4 — Gradient Boosting model · Test R² = 0.626 · MAE ≈ $48")
