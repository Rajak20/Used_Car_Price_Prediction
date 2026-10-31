import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚘",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "catboost_car_price_model.pkl"

# Keep this reference year consistent with the feature engineering used during training.
REFERENCE_YEAR = 2024

LUXURY_BRANDS = {
    "acura", "alfa romeo", "aston martin", "audi", "bentley", "bmw",
    "cadillac", "ferrari", "genesis", "infiniti", "jaguar", "lamborghini",
    "land rover", "lexus", "lincoln", "maserati", "maybach", "mclaren",
    "mercedes-benz", "mercedes benz", "porsche", "rolls-royce",
    "rolls royce", " tesla", "volvo",
}

@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)

def safe_text(value):
    value = str(value).strip() if value is not None else ""
    return value if value else "missing"

def parse_mileage(value):
    # Accept values like "51,000 mi.", "51000", or "51,000".
    cleaned = re.sub(r"(?i)\s*mi\.?\s*$", "", str(value)).replace(",", "").strip()
    return float(cleaned)

def extract_number(pattern, text, default):
    match = re.search(pattern, str(text), flags=re.IGNORECASE)
    return float(match.group(1)) if match else float(default)

def build_features(
    brand, model, model_year, mileage, fuel_type, engine,
    transmission, accident, clean_title
):
    brand = safe_text(brand)
    model = safe_text(model)
    fuel_type = safe_text(fuel_type)
    engine = safe_text(engine)
    transmission = safe_text(transmission)
    accident = safe_text(accident)
    clean_title = safe_text(clean_title)

    mileage_clean = parse_mileage(mileage)
    car_age = REFERENCE_YEAR - int(model_year)
    # Match the original training feature definition exactly.
    mileage_per_year = mileage_clean / (car_age + 1) if car_age != -1 else mileage_clean
    mileage_squared = mileage_clean ** 2

    engine_liters = extract_number(
        r"(\d+(?:\.\d+)?)\s*(?:L\b|Liter\b)", engine, 3.0
    )
    engine_hp = extract_number(
        r"(\d+(?:\.\d+)?)\s*HP\b", engine, 300.0
    )
    trans_speeds = extract_number(
        r"(\d+)\s*-?\s*Speed\b", transmission, 6.0
    )

    is_luxury = int(brand.lower().strip() in LUXURY_BRANDS)
    trans_lower = transmission.lower()
    is_automatic = int("auto" in trans_lower or "a/t" in trans_lower)

    model_family = model.split()[0] if model and model != "missing" else "missing"

    # Column order MUST match the CatBoost model's training order.
    row = {
        "model_year": int(model_year),
        "milage_clean": float(mileage_clean),
        "car_age": int(car_age),
        "milage_per_year": float(mileage_per_year),
        "milage_squared": float(mileage_squared),
        "engine_liters": float(engine_liters),
        "engine_hp": float(engine_hp),
        "trans_speeds": float(trans_speeds),
        "is_luxury": int(is_luxury),
        "is_automatic": int(is_automatic),
        "brand": brand,
        "model_family": model_family,
        "fuel_type": fuel_type,
        "accident": accident,
        "clean_title": clean_title,
    }
    return pd.DataFrame([row], columns=[
        "model_year", "milage_clean", "car_age", "milage_per_year",
        "milage_squared", "engine_liters", "engine_hp", "trans_speeds",
        "is_luxury", "is_automatic", "brand", "model_family",
        "fuel_type", "accident", "clean_title"
    ])

st.title("🚘 Car Price Predictor")
st.write(
    "Estimate a used car's price in USD using your trained CatBoost regression model. "
    "Enter the vehicle details below."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH.name}`. "
        "Place `catboost_car_price_model.pkl` in the same folder as `app.py`."
    )
    st.stop()

try:
    model_obj = load_model()
except Exception as exc:
    st.error(f"Could not load the model. Check that dependencies and model file are correct. Details: {exc}")
    st.stop()

with st.form("car_details"):
    st.subheader("Vehicle details")
    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.text_input("Brand", placeholder="e.g., Toyota")
        model_name = st.text_input("Model", placeholder="e.g., Camry")
        model_year = st.number_input(
            "Model year", min_value=1980, max_value=2026, value=2020, step=1
        )

    with col2:
        mileage = st.text_input("Mileage", value="51,000 mi.", help="Examples: 51,000 mi. or 51000")
        fuel_type = st.selectbox(
            "Fuel type",
            ["Gasoline", "Hybrid", "Electric", "Diesel", "E85 Flex Fuel", "Plug-In Hybrid", "Other", "missing"]
        )
        engine = st.text_input(
            "Engine description",
            placeholder="e.g., 2.5L 203HP",
            help="Include engine liters and horsepower if available, e.g. 3.0L 300HP."
        )

    with col3:
        transmission = st.text_input(
            "Transmission",
            placeholder="e.g., 8-Speed Automatic",
            help="Include speed count and automatic/manual wording if available."
        )
        accident = st.selectbox(
            "Accident history",
            ["None reported", "At least 1 accident or damage reported", "Unknown", "missing"]
        )
        clean_title = st.selectbox(
            "Clean title",
            ["Yes", "No", "Unknown", "missing"]
        )

    submitted = st.form_submit_button("Predict car price", type="primary", use_container_width=True)

if submitted:
    try:
        if not brand.strip() or not model_name.strip():
            st.error("Please enter both the car brand and model.")
            st.stop()

        features = build_features(
            brand=brand,
            model=model_name,
            model_year=int(model_year),
            mileage=mileage,
            fuel_type=fuel_type,
            engine=engine,
            transmission=transmission,
            accident=accident,
            clean_title=clean_title,
        )

        expected = [
            "model_year", "milage_clean", "car_age", "milage_per_year",
            "milage_squared", "engine_liters", "engine_hp", "trans_speeds",
            "is_luxury", "is_automatic", "brand", "model_family",
            "fuel_type", "accident", "clean_title"
        ]
        if list(features.columns) != expected:
            st.error("Feature order does not match the expected model input.")
            st.stop()

        # CatBoost accepts categorical values as strings; numeric features remain numeric.
        prediction_log = float(np.asarray(model_obj.predict(features)).reshape(-1)[0])

        # Assumption: training target was transformed with np.log1p(y).
        # If training used np.log(y), replace np.expm1(...) with np.exp(...).
        predicted_price = float(np.expm1(prediction_log))

        if not np.isfinite(predicted_price) or predicted_price < 0:
            st.warning(
                "The model returned an unusual value. Verify that the target was log1p-transformed "
                "during training and that preprocessing matches training."
            )
        else:
            st.success("Prediction completed")
            st.metric("Estimated car price", f"${predicted_price:,.0f} USD")
            st.caption(
                "This is a model estimate, not a guaranteed market price. "
                "Results depend on the training data and preprocessing."
            )

            with st.expander("View engineered model inputs"):
                st.dataframe(features, use_container_width=True)
    except ValueError:
        st.error("Mileage must be a number, such as 51,000 mi. Please check the input.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "If the error mentions categorical features, check that the categorical columns "
            "and preprocessing match the model used during training."
        )

with st.sidebar:
    st.header("About")
    st.write("Model: CatBoost Regressor")
    st.write("Inputs: 9 raw vehicle fields")
    st.write("Engineered features: 15")
    st.divider()
    st.caption(
        f"Car age is calculated using reference year {REFERENCE_YEAR}, matching the provided "
        "feature-engineering instructions. Update this only if your training pipeline used a different year."
    )
