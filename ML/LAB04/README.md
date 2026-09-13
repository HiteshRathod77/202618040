# NYC Airbnb Price Prediction

**DS605: Fundamentals of Machine Learning — Lab Assignment 4**

**Name:** Hitesh Rathod
**Roll Number:** 202618040

An end-to-end machine learning project that predicts the nightly price of
an Airbnb listing in New York City, with a Streamlit web app for users to
try it out.

**Live app:** https://202618040-f3hfdxwx3devzqhffheajg.streamlit.app/

---

## Objective

Build a complete ML workflow — data cleaning, EDA, feature engineering,
model training, tuning, and deployment — using the Kaggle New York City
Airbnb Open Data (2019).

- Dataset: 48,895 listings × 16 columns
- Target: `price` (nightly rate in USD)

---

## What I found in the data

A few things stood out during EDA:

- Price is heavily right-skewed (skewness ≈ 2.78). Taking `log1p(price)`
  brought it down to ≈ −0.39, which made the models behave much better.
- Room type is the strongest single predictor. Median price for an Entire
  home is around $160, Private room $70, Shared room $45.
- Manhattan listings are roughly twice as expensive as Bronx ones
  ($150 vs $65 median).
- Reviews have essentially no correlation with price (≈ −0.06), which
  surprised me.
- Longitude was the strongest numeric signal (−0.33 correlation with
  log price), which makes sense given Manhattan sits on the west side
  of the city.

Plots and the full analysis are in `notebooks/airbnb_eda.ipynb`.

---

## Cleaning decisions

- Dropped `id`, `host_id`, `host_name`, `name` — no predictive value.
- Dropped 11 listings with `price = 0`.
- Filled missing `reviews_per_month` with 0 (missing here really means
  "no reviews yet", which I confirmed by cross-checking with
  `number_of_reviews`).
- Clipped `price`, `minimum_nights`, and
  `calculated_host_listings_count` at their 99th percentiles to tame
  the extreme outliers.
- Created `price_log = log1p(price)` as the modeling target.

---

## Features and preprocessing

Everything is wrapped in a single `sklearn` Pipeline so training and
inference use identical transformations.

| Group | Columns | Transformation |
|---|---|---|
| Numeric | latitude, longitude, minimum_nights, number_of_reviews, reviews_per_month, calculated_host_listings_count, availability_365, days_since_last_review, dist_to_center_km | Median impute + StandardScaler |
| Categorical | neighbourhood_group, room_type, has_reviews, is_multi_host | Most-frequent impute + OneHot |
| High-cardinality | neighbourhood (221 unique values) | Most-frequent impute + TargetEncoder |

A few engineered features that helped:
- `dist_to_center_km` — Haversine distance to Times Square
- `days_since_last_review` — how stale the listing is
- `has_reviews` — binary flag
- `is_multi_host` — host owns more than one listing

---

## Model comparison

Trained 3 models on the same pipeline, then tuned the top two with
RandomizedSearchCV (20 combinations × 3-fold CV).

| Model | Test R² | MAE ($) | RMSE ($) |
|---|---|---|---|
| Ridge | 0.5609 | 51.88 | 100.06 |
| Random Forest (untuned) | 0.6230 | 47.78 | 94.08 |
| Gradient Boosting (untuned) | 0.6200 | 48.50 | 94.19 |
| **Tuned Gradient Boosting** | **0.6258** | **47.94** | **93.33** |
| Tuned Random Forest | 0.6229 | 47.83 | 94.20 |

**Final model:** Tuned Gradient Boosting
(`n_estimators=200, max_depth=7, learning_rate=0.08, subsample=0.9,
min_samples_leaf=8`)

**Overfitting check:** train R² = 0.733, test R² = 0.626 — gap of 0.107,
which is acceptable for tree ensembles. The untuned GB overfit much
worse, so tuning helped.

**Top 5 features (from the earlier Random Forest permutation):**
1. `room_type = Entire home/apt` — 23.8%
2. `room_type = Private room` — 15.1%
3. `neighbourhood` (target encoded) — 12.3%
4. `longitude` — 11.8%
5. `latitude` — 9.0%

---

## Streamlit app

The app (`app/app.py`) loads the saved pipeline and exposes:

- Borough dropdown (populated from training data)
- Neighbourhood dropdown (all 221 values, searchable)
- Room type dropdown
- Latitude / longitude inputs
- Sliders for nights, reviews, host listings, availability

Everything is passed straight to the pipeline — no manual preprocessing
in the app itself. That way the transformations always match what the
model was trained on.

**Test cases:**

| Input | Output |
|---|---|
| Manhattan / Williamsburg / Entire home | ~$196 |
| Bronx / Mott Haven / Shared room | ~$46 |

Screenshots are in `docs/screenshots/`.

---

## Project structure
