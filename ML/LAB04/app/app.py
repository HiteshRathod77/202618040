import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip
from pathlib import Path

# ---------- Page config ----------
st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏙️",
    layout="wide"
)

# ---------- Load model ----------
@st.cache_resource
def load_model():
    # Robust: resolve path relative to this file
    model_path = Path(__file__).resolve().parent.parent / "models" / "airbnb_price_pipeline.pkl.gz"

    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path}")
        st.info("Make sure `models/airbnb_price_pipeline.pkl.gz` exists in your project folder.")
        st.stop()

    with gzip.open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

model = load_model()

# ---------- Header ----------
st.title("🏙️ NYC Airbnb Price Predictor")
st.markdown(
    "Enter the details of an Airbnb listing below and get an "
    "**estimated nightly price** (in USD) based on our tuned Random Forest model."
)
st.markdown("---")

# ---------- Input form ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location")
    neighbourhood_group = st.selectbox(
        "Borough",
        ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    )
    neighbourhood = st.text_input(
        "Neighbourhood",
        value="Williamsburg",
        help="e.g. Williamsburg, Harlem, Midtown, Bushwick..."
    )
    latitude = st.number_input("Latitude", min_value=40.4, max_value=41.0, value=40.71, step=0.001, format="%.4f")
    longitude = st.number_input("Longitude", min_value=-74.3, max_value=-73.6, value=-73.95, step=0.001, format="%.4f")

with col2:
    st.subheader("🏠 Listing details")
    room_type = st.selectbox(
        "Room type",
        ["Entire home/apt", "Private room", "Shared room"]
    )
    minimum_nights = st.slider("Minimum nights", 1, 45, 2)
    number_of_reviews = st.slider("Number of reviews", 0, 300, 10)
    reviews_per_month = st.slider("Reviews per month", 0.0, 10.0, 0.5, step=0.1)
    calculated_host_listings_count = st.slider("Host's total listings", 1, 200, 1)
    availability_365 = st.slider("Availability (days/year)", 0, 365, 180)

st.markdown("---")

# ---------- Predict ----------
if st.button("💵 Estimate Price", type="primary", use_container_width=True):

    input_df = pd.DataFrame([{
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "latitude": latitude,
        "longitude": longitude,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
    }])

    with st.spinner("Predicting..."):
        pred_log = model.predict(input_df)[0]
        pred_price = np.expm1(pred_log)

    st.success(f"### 💰 Estimated nightly price: **${pred_price:,.0f}**")
    st.caption(
        "This is a statistical estimate from a machine learning model, "
        "not a real quote. Actual prices may vary."
    )

    with st.expander("🔍 See input summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

st.markdown("---")
st.caption(
    "Built for DS605 Lab 4 — End-to-End ML Project. "
    "Model: Tuned Random Forest (Test R² = 0.625, MAE = $47.56)."
)
