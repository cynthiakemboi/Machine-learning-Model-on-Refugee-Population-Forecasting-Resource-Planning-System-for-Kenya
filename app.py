
import streamlit as st
import pandas as pd
import joblib

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Refugee Population Forecasting System",
    page_icon="🌍",
    layout="wide"
)

# =====================================================
# Load Saved Objects
# =====================================================

model = joblib.load("random_forest_refugee_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None

# =====================================================
# Application Title
# =====================================================

st.title("🌍 AI-Powered Refugee Population Forecasting System")

st.write(
    """
    Predict refugee population counts using demographic and
    humanitarian characteristics.
    """
)

# =====================================================
# User Inputs
# =====================================================

origin = st.selectbox(
    "Country of Origin",
    ["SOM","SSD","ETH","COD","BDI","RWA","AFG","SDN"]
)

population_group = st.selectbox(
    "Population Group",
    ["REF","ASY","HST","OOC"]
)

gender = st.selectbox(
    "Gender",
    ["f","m"]
)

age_range = st.selectbox(
    "Age Group",
    ["0-4","5-11","12-17","18-59","60+"]
)

year = st.number_input(
    "Year",
    min_value=2025,
    max_value=2035,
    value=2026
)

origin_has_hrp = st.selectbox(
    "Origin has HRP",
    [0,1]
)

origin_in_gho = st.selectbox(
    "Origin in GHO",
    [0,1]
)

asylum_has_hrp = st.selectbox(
    "Asylum has HRP",
    [0,1]
)

asylum_in_gho = st.selectbox(
    "Asylum in GHO",
    [0,1]
)

# =====================================================
# Prediction
# =====================================================

if st.button("Predict Refugee Population"):

    input_df = pd.DataFrame({

        "origin_location_code":[origin],

        "origin_has_hrp":[origin_has_hrp],

        "origin_in_gho":[origin_in_gho],

        "asylum_has_hrp":[asylum_has_hrp],

        "asylum_in_gho":[asylum_in_gho],

        "population_group":[population_group],

        "gender":[gender],

        "age_range":[age_range],

        "year":[year]

    })

    # Apply saved encoders
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Scale numerical features if required
    if scaler is not None:
        numerical_cols = [
            "year",
            "origin_has_hrp",
            "origin_in_gho",
            "asylum_has_hrp",
            "asylum_in_gho"
        ]

        input_df[numerical_cols] = scaler.transform(
            input_df[numerical_cols]
        )

    prediction = model.predict(input_df)[0]

    prediction = max(0, prediction)

    st.success(
        f"Predicted Refugee Population: {prediction:,.0f}"
    )

    # =================================================
    # Humanitarian Planning Metrics
    # =================================================

    household_size = 5

    daily_food_per_person = 0.45

    shelters = prediction / household_size

    monthly_food = prediction * daily_food_per_person * 30 / 1000

    st.subheader("Operational Planning")

    st.metric(
        "Estimated Households",
        f"{int(shelters):,}"
    )

    st.metric(
        "Monthly Food Requirement",
        f"{monthly_food:.2f} tonnes"
    )
%%writefile requirements.txt

streamlit
pandas
numpy
scikit-learn
joblib
