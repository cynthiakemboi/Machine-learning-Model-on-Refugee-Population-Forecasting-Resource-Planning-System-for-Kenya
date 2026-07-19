#Import libraries.
# Import Streamlit for building the interactive web dashboard.
# PyTorch is used to load and run the deep learning FT-Transformer model.
# Joblib/Pickle load saved machine learning objects such as encoders and scalers.
# JSON is used to read saved model evaluation metrics.
# OS checks whether required files exist.
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import pickle
import json
import os

# Device Configuration

device = torch.device("cpu")
st.set_page_config(
    page_title="Refugee Population Forecasting System",
    page_icon="🌍",
    layout="wide"

)

TRAIN_YEAR_MIN = 2001
TRAIN_YEAR_MAX = 2025  # last year present in Kenya_Refugee.csv

# Recreate the FT-Transformer PyTorch Architecture
class NumericalTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(num_features, embed_dim))
        self.biases = nn.Parameter(torch.Tensor(num_features, embed_dim))
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.biases)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)

class FTTransformer(nn.Module):
    def __init__(self, cat_cardinalities, num_features, embed_dim=32, depth=3, heads=4, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities
        ])
        self.num_tokenizer = NumericalTokenizer(num_features, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=attn_dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        total_tokens = len(cat_cardinalities) + num_features
        self.mlp_head = nn.Sequential(
            nn.Linear(total_tokens * embed_dim, 128),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)

        )
    def forward(self, x_cat, x_num):
        batch_size = x_cat.size(0)
        cat_tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        if cat_tokens:

            cat_tokens = torch.stack(cat_tokens, dim=1)
        num_tokens = self.num_tokenizer(x_num)
        if len(self.cat_embeddings) > 0:
            tokens = torch.cat([cat_tokens, num_tokens], dim=1)
        else:
            tokens = num_tokens
        transformer_out = self.transformer(tokens)
        flat_out = transformer_out.view(batch_size, -1)
        return self.mlp_head(flat_out)

# Safe Helper Utilities for Encoding & Scaling
def get_classes_safely(encoder_obj):
    if hasattr(encoder_obj, 'classes_'):
        return list(encoder_obj.classes_)
    elif hasattr(encoder_obj, 'categories_'):
        return list(encoder_obj.categories_[0])
    elif isinstance(encoder_obj, (list, tuple, np.ndarray)):
        return list(encoder_obj)
    elif isinstance(encoder_obj, dict):
        return list(encoder_obj.keys())
    else:
        return [str(encoder_obj)]

def safe_transform_categorical(encoder_obj, val):
    classes = get_classes_safely(encoder_obj)
    if hasattr(encoder_obj, 'transform'):
        try:
            return int(encoder_obj.transform([val])[0])
        except Exception:
            pass
    try:
        return classes.index(val)
    except Exception:
        return 0

# Age-cohort-specific resource planning profiles (WHO / UNHCR / Sphere-Handbook aligned)
AGE_COHORT_PROFILES = {
    "0-4": {
        "school_age": False,
        "health_kit_type": "Pediatric IMCI Kit (ORS, vaccines, RUTF, growth monitoring)",
        "health_kit_ratio": 1.0,
        "food_ration_kg": 0.35,
        "water_liters": 15,
        "notes": "Priority: malnutrition screening, measles/polio vaccination.",
    },
    "5-11": {
        "school_age": True,
        "health_kit_type": "Pediatric Kit (deworming, boosters, first aid)",
        "health_kit_ratio": 1.0,
        "food_ration_kg": 0.40,
        "water_liters": 15,
        "notes": "Priority: school enrollment, deworming programs.",
    },
    "12-17": {
        "school_age": True,
        "health_kit_type": "Adolescent Kit (MISP reproductive health, MHPSS materials)",
        "health_kit_ratio": 1.0,
        "food_ration_kg": 0.45,
        "water_liters": 15,
        "notes": "Priority: adolescent reproductive health (esp. girls), psychosocial support.",
    },
    "18-59": {
        "school_age": False,
        "health_kit_type": "General Adult Primary Care Kit (+ maternal/safe-delivery kit for women)",
        "health_kit_ratio": 1.0,
        "food_ration_kg": 0.50,
        "water_liters": 15,
        "notes": "Priority: maternal health, livelihoods/employment support.",
    },
    "60+": {
        "school_age": False,
        "health_kit_type": "Geriatric / NCD Kit (hypertension & diabetes mgmt, mobility aids)",
        "health_kit_ratio": 1.0,
        "food_ration_kg": 0.40,
        "water_liters": 18,
        "notes": "Priority: chronic disease continuity of care, mobility, social isolation risk.",
    },
    "all": {
        "school_age": None,
        "health_kit_type": "MIXED COHORT — disaggregate by age before planning kit contents",
        "health_kit_ratio": 1.0,
        "food_ration_kg": 0.45,
        "water_liters": 15,
        "notes": "This selection aggregates all ages; resource breakdown below is indicative only.",

    },

}


# Historical data (for baseline lookup, sanity checks,and the naive-benchmark fallback for model performance)
@st.cache_data
def load_historical_data():
    df = pd.read_csv("Kenya_Refugee.csv")
    df["reference_period_start"] = pd.to_datetime(df["reference_period_start"])
    df["year"] = df["reference_period_start"].dt.year
    return df

def get_historical_baseline(df, origin, population_group, gender, age_range):
    """Most recent recorded population for this exact cohort combination."""
    subset = df[
        (df["origin_location_code"] == origin)
        & (df["population_group"] == population_group)
        & (df["gender"] == gender)
        & (df["age_range"] == age_range)
    ]
    if subset.empty:
        return 0, None, False
    latest = subset.sort_values("year", ascending=False).iloc[0]
    return int(latest["population"]), int(latest["year"]), True

def sanity_check_baseline(df, origin, population_group, gender, age_range, entered_value):
    """Warn if a manually-edited baseline is wildly outside historical range for this cohort."""
    subset = df[
        (df["origin_location_code"] == origin)
        & (df["population_group"] == population_group)
        & (df["gender"] == gender)
        & (df["age_range"] == age_range)
    ]
    if subset.empty:
        return
    hist_max = subset["population"].max()
    if hist_max > 0 and entered_value > hist_max * 3:
        st.warning(
            f"⚠️ Entered baseline ({entered_value:,}) is more than 3x the highest "
            f"historical value ({hist_max:,}) recorded for this cohort. Double-check "
            f"this is intentional before trusting the forecast."
        )

def sanity_check_prediction(df, predicted_pop, origin):
    """Warn if a prediction is far outside all historical values ever seen for this origin."""
    subset = df[df["origin_location_code"] == origin]
    if subset.empty:
        return
    hist_max_all_ages = subset["population"].max()
    if hist_max_all_ages > 0 and predicted_pop > hist_max_all_ages * 2:
        st.warning(
            f"⚠️ Predicted population ({predicted_pop:,}) is more than 2x the "
            f"largest historical figure ever recorded for {origin} in Kenya "
            f"({hist_max_all_ages:,}), across any age band. Treat this forecast "
            f"with caution — it may reflect extrapolation error rather than a "
            f"real trend."
        )

def load_model_metrics():
    """Reads backtested metrics if model_metrics.json exists."""
    if os.path.exists("model_metrics.json"):
        try:
            with open("model_metrics.json") as f:
                return json.load(f), True
        except Exception:
            pass
    return None, False

def naive_baseline_error(df):
    """
    Fallback benchmark ONLY: treats 'last year's value' as this year's prediction
    and measures how wrong that naive approach was historically. This is NOT the
    FT-Transformer's actual performance — used only when model_metrics.json is absent.
    """
    df_sorted = df.sort_values(["origin_location_code", "population_group", "gender", "age_range", "year"])
    df_sorted["naive_pred"] = df_sorted.groupby(
        ["origin_location_code", "population_group", "gender", "age_range"]
    )["population"].shift(1)
    valid = df_sorted.dropna(subset=["naive_pred"])
    mae = np.mean(np.abs(valid["population"] - valid["naive_pred"]))
    rmse = np.sqrt(np.mean((valid["population"] - valid["naive_pred"]) ** 2))
    return mae, rmse

# Safe Asset Loader
@st.cache_resource
def load_assets():
    try:
        raw_encoders = joblib.load("label_encoders.pkl")
    except Exception:
        with open("label_encoders.pkl", "rb") as f:
            raw_encoders = pickle.load(f)
    cols = ['origin_location_code', 'population_group', 'gender', 'age_range']
    label_encoders = {}
    if isinstance(raw_encoders, dict):
        for i, col in enumerate(cols):
            if col in raw_encoders:
                label_encoders[col] = raw_encoders[col]
            elif i in raw_encoders:
                label_encoders[col] = raw_encoders[i]
            elif str(i) in raw_encoders:
                label_encoders[col] = raw_encoders[str(i)]
    elif isinstance(raw_encoders, (list, tuple, np.ndarray)):
        for i, col in enumerate(cols):
            if i < len(raw_encoders):
                label_encoders[col] = raw_encoders[i]
    else:
        for col in cols:
            label_encoders[col] = ["Unknown"]
    try:
        scaler = joblib.load("scaler.pkl")
    except Exception:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    try:
        model_config = joblib.load("model_config.pkl")
    except Exception:
        with open("model_config.pkl", "rb") as f:
            model_config = pickle.load(f)
    state_dict = torch.load("ft_transformer_model.pth", map_location=device)
    max_layer_idx = -1
    for key in state_dict.keys():
        if key.startswith("transformer.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():

                max_layer_idx = max(max_layer_idx, int(parts[2]))
    detected_depth = max_layer_idx + 1 if max_layer_idx != -1 else model_config.get('depth', 3)


    model = FTTransformer(
        cat_cardinalities=model_config['cat_cardinalities'],
        num_features=model_config['num_features'],
        embed_dim=model_config['embed_dim'],
        depth=detected_depth, 
        heads=model_config['heads'],
        attn_dropout=model_config.get('attn_dropout', 0.1),
        ff_dropout=model_config.get('ff_dropout', 0.1)
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, label_encoders, scaler, model_config

# Initialize variables globally to prevent NameErrors in case loading fails
model, label_encoders, scaler, model_config = None, None, None, None
try:
    model, label_encoders, scaler, model_config = load_assets()
except Exception as e:
    st.error(f"⚠️ App Setup Failed: {e}")
    st.stop()
try:
    history_df = load_historical_data()
    history_loaded = True
except Exception as e:
    history_df = None
    history_loaded = False

# Sidebar Navigation & Performance
with st.sidebar:
    st.header("🧠 Model Metadata")
    st.markdown("""
    * **Architecture:** FT-Transformer (Feature Tokenizer Transformer)
    * **Framework:** PyTorch (Deep Learning)
    * **Prediction Target:** Refugee Cohort Population Size
    * **Horizon:** 2026–2030
    """)
    
    st.markdown("---")
    st.header("📈 Model Evaluation")

    metrics, real_metrics_found = load_model_metrics()

    if real_metrics_found:
        st.metric(
            label="R² Score (Variance Explained)",
            value=f"{metrics.get('r2', 0):.3f}"
        )
        st.metric(
            label="Mean Absolute Error (MAE)",
            value=f"{metrics.get('mae', 0):,.1f} individuals"
        )
        st.metric(
            label="Root Mean Squared Error (RMSE)",
            value=f"{metrics.get('rmse', 0):,.1f} individuals",
            help="RMSE penalizes larger prediction deviations heavier than MAE, essential for capacity safety buffer planning."
        )
        st.caption("Backtested on a held-out split during training.")
    elif history_loaded:
        naive_mae, naive_rmse = naive_baseline_error(history_df)
        st.warning(
            "No saved test-set metrics found (`model_metrics.json` missing). "
            "Showing a naive year-over-year benchmark instead — this is NOT the "
            "model's actual error, only a reference point for how volatile these "
            "cohorts are historically."
        )
        st.metric(
            label="Naive benchmark MAE",
            value=f"{naive_mae:,.1f} individuals",
            help="Error from assuming next year's population equals this year's, computed directly from historical records."
        )
        st.metric(
            label="Naive benchmark RMSE",
            value=f"{naive_rmse:,.1f} individuals"
        )
        st.caption(
            "To show the FT-Transformer's real accuracy here instead, save "
            "mae/rmse/r2 from its held-out evaluation to model_metrics.json "
            "at training time."
        )
    else:
        st.info("Historical data and model_metrics.json both unavailable — no performance figures to show.")

# Streamlit Web UI Execution Setup
st.title("🌍 AI-Powered Refugee Population Forecasting System")
st.markdown("""
### **Project Objective**
To provide humanitarian organizations with a proactive tool for estimating localized refugee demographic trends in Kenya and automatically calculating downstream resource allocation.
""")

# Clean, user-friendly status message replacing developer messages
st.info("💡 **System Ready** — Select your forecast parameters below and click **Generate Forecast**.")
st.markdown("---")

#App Tutorial & Quick Terminology Glossary
with st.expander("📖 User Manual & Quick Terminology Glossary", expanded=False):
    t_col1, t_col2 = st.columns([1, 1.2])
    with t_col1:
        st.markdown("""
        ### **How to generate forecasts:**
        1. **Select Demographic Parameters:** Pick the Country of Origin, Population Group, Gender, and Age band you want to forecast.
        2. **Specify Forecast Timeline:** Set the Target Forecast Year (2026–2030) using the slider.
        3. **Refine Geopolitical Controls:** Toggle administrative factors like **HRP** and **GHO** if conditions are changing.
        4. **Set Baseline Population:** The system defaults to **5,000**. Feel free to enter any custom population scale.
        5. **Run Prediction:** Click **🔮 Generate Forecast** to query the deep learning model.
        """)
    with t_col2:
        st.markdown("""
        ### **📌 Terminology Quick Reference**
        * 🔍 **ASY (Asylum Seekers):** Individuals whose requests for international asylum protective status inside Kenya have been officially filed but are still awaiting formal decision/resolution.
        * 📋 **Baseline Population:** The starting count or historically recorded size of this specific demographic cohort. The model uses this baseline to calculate relative future growth or contraction.
        * 🛡️ **HRP (Humanitarian Response Plan):** An active, coordinated UN-led emergency strategy launched inside a crisis country to target and fund life-saving assistance programs.
        * 🌐 **GHO (Global Humanitarian Overview):** A designation indicating whether a country has been prioritized inside the UN's shared global funding appeal due to crisis severity.
        """)
        st.markdown("---")
        st.caption("💡 *You can also hover over the small question mark icons **(?)** next to each input field below to read these descriptions.*")

st.markdown("---")

col1, col2 = st.columns([1, 1.2])

valid_origins = get_classes_safely(label_encoders['origin_location_code'])
valid_pop_groups = get_classes_safely(label_encoders['population_group'])
valid_genders = get_classes_safely(label_encoders['gender'])
valid_age_ranges = get_classes_safely(label_encoders['age_range'])

with col1:
    st.subheader("📋 Demographic Parameters")
    
    origin = st.selectbox(
        "Country of Origin", 
        options=valid_origins,
        help="❓ What are you doing here: Select the country of origin for the displaced population cohort you wish to forecast."
    )
    population_group = st.selectbox(
        "Population Group Type", 
        options=valid_pop_groups,
        help="The administrative classification of the group:\n\n"
             "• ASY: Asylum Seekers (individuals whose request for sanctuary has yet to be processed).\n"
             "• REF: Registered Refugees (individuals officially granted protected status)."
    )
    gender = st.selectbox(
        "Gender Cohort", 
        options=valid_genders,
        help="❓ What are you doing here: Narrow down your population projection parameters to a specific gender segment."
    )
    age_range = st.selectbox(
        "Age Range", 
        options=valid_age_ranges,
        help="❓ What are you doing here: Select an age band cohort. This specific parameter will dynamically dictate the downstream medical, educational, and dietary resource logic profiles."
    )

    st.subheader("⏱️ Forecasting Timeline")
    year = st.slider(
        "Target Forecast Year", 
        min_value=2026, 
        max_value=2030, 
        value=2026,
        help="The future calendar year you wish to project for (2026-2030)."
    )

    if year > TRAIN_YEAR_MAX:
        years_beyond = year - TRAIN_YEAR_MAX
        st.warning(
            f"⚠️ **Extrapolation warning:** the model was trained on data through "
            f"{TRAIN_YEAR_MAX}. {year} is {years_beyond} year(s) beyond that range. "
            f"This is an **extrapolated forecast**, not an in-sample prediction — "
            f"treat it as indicative rather than authoritative, and prefer near-term "
            f"years over longer horizons when precision matters."
        )

    st.subheader("💡 Geopolitical Indicators")
    origin_has_hrp = st.checkbox(
        "Origin country has active Humanitarian Response Plan (HRP)", 
        value=True,
        help="HRP (Humanitarian Response Plan): An active, strategic, and coordinated emergency plan launched by the UN and partners inside the origin country to address critical vulnerabilities."
    )
    origin_in_gho = st.checkbox(
        "Included in Global Humanitarian Overview (GHO)", 
        value=True,
        help="GHO (Global Humanitarian Overview): Indicates if the origin country is officially categorized under the UN's shared global humanitarian funding appeal due to chronic crisis severity."
    )
    asylum_has_hrp = st.checkbox(
        "Kenya has active Humanitarian Response Plan (HRP)", 
        value=True,
        help="❓ What are you doing here: Toggle whether host country operations within Kenya are running under a high-level active strategic HRP response template."
    )
    asylum_in_gho = st.checkbox(
        "Kenya included in Global Humanitarian Overview (GHO)", 
        value=True,
        help="❓ What are you doing here: Indicates if Kenya is tracked inside the current global appeal overview framework."
    )

with col2:
    st.subheader("📊 Dynamic Baseline Lookup")

    # Actually perform the lookup this section is named after — previously
    # baseline_year/lookup_val/is_lookup_verified were referenced below with
    # nothing ever assigning them, which crashed the app with a NameError.
    if history_loaded:
        lookup_val, baseline_year, is_lookup_verified = get_historical_baseline(
            history_df, origin, population_group, gender, age_range
        )
    else:
        lookup_val, baseline_year, is_lookup_verified = 5000, None, False

    baseline_pop = st.number_input(
        "Current Baseline Population (Historical)", 
        min_value=0, 
        max_value=1000000, 
        value=int(lookup_val),
        step=100,
        help="Baseline Population: Represents the starting size of this cohort. The deep learning model processes this baseline alongside indicators to scale its final forecast."
    )

    if is_lookup_verified:
        st.success(f"✅ *Verified database match loaded: {lookup_val:,} individuals (from year {baseline_year}). Feel free to modify.*")
    else:
        st.warning(f"ℹ️ *No historical record matching these exact demographics was found. Populated with default anchor value: {lookup_val}*")

    if history_loaded:
        sanity_check_baseline(history_df, origin, population_group, gender, age_range, baseline_pop)

    st.markdown("---")
    st.subheader("🤖 Model Inference & Resource Forecasting")

    if age_range == "0-4":
        min_age, max_age = 0.0, 4.0
    elif age_range == "5-11":
        min_age, max_age = 5.0, 11.0
    elif age_range == "12-17":
        min_age, max_age = 12.0, 17.0
    elif age_range == "18-59":
        min_age, max_age = 18.0, 59.0
    else:
        min_age, max_age = 60.0, 100.0

    raw_numerical = pd.DataFrame([{
        'origin_has_hrp': 1.0 if origin_has_hrp else 0.0,
        'origin_in_gho': 1.0 if origin_in_gho else 0.0,
        'min_age': min_age,
        'max_age': max_age,
        'population': float(baseline_pop),  
        'year': float(year)
    }])

    raw_categorical = pd.DataFrame([{
        'origin_location_code': origin,
        'population_group': population_group,
        'gender': gender,
        'age_range': age_range
    }])

    if hasattr(scaler, 'feature_names_in_'):
        raw_numerical = raw_numerical[list(scaler.feature_names_in_)]

    if "predicted_pop" not in st.session_state:
        st.session_state.predicted_pop = None

    if st.button("🔮 Generate Forecast"):
        with st.spinner("Generating projections..."):
            try:
                encoded_cat = raw_categorical.copy()
                for col in ['origin_location_code', 'population_group', 'gender', 'age_range']:
                    encoded_cat[col] = safe_transform_categorical(label_encoders[col], raw_categorical.loc[0, col])
                
                if hasattr(scaler, 'transform'):
                    scaled_num = scaler.transform(raw_numerical)
                else:
                    scaled_num = raw_numerical.to_numpy() 

                expected_num_features = model_config.get('num_features', 6)
                if scaled_num.shape[1] > expected_num_features:
                    scaled_num_features = np.delete(scaled_num, 4, axis=1)
                else:
                    scaled_num_features = scaled_num

                tensor_cat = torch.tensor(encoded_cat.values, dtype=torch.long).to(device)
                tensor_num = torch.tensor(scaled_num_features, dtype=torch.float32).to(device)

                with torch.no_grad():
                    pred_raw = model(tensor_cat, tensor_num).item()
                
                if hasattr(scaler, 'inverse_transform') and scaled_num.shape[1] == 6:
                    dummy_row = np.zeros((1, 6))
                    dummy_row[0, 4] = pred_raw
                    inverse_result = scaler.inverse_transform(dummy_row)
                    pred_unscaled = inverse_result[0, 4]
                else:
                    pred_unscaled = pred_raw
                
                st.session_state.predicted_pop = max(0, int(round(pred_unscaled)))
            except Exception as e:
                st.error(f"Prediction Pipeline Failed: {e}")

    if st.session_state.predicted_pop is not None:
        predicted_pop = st.session_state.predicted_pop
        
        st.success("🎉 Forecast Generated Successfully!")
        st.metric(label="👥 Predicted Target Refugee Population Segment", value=f"{predicted_pop:,} individuals")

        if history_loaded:
            sanity_check_prediction(history_df, predicted_pop, origin)

        # Trajectory Growth Bar Comparison Chart Visual 📊
        st.write("📈 **Trajectory Growth Comparison**")
        comparison_df = pd.DataFrame({
            "Stage": [f"Historical Baseline ({baseline_year})", f"AI Forecast ({year})"],
            "Population Size": [baseline_pop, predicted_pop]
        }).set_index("Stage")
        
        st.bar_chart(comparison_df)
        
        st.markdown("---")
        st.subheader("📦 Projected Resource Requirements")
        
        profile = AGE_COHORT_PROFILES.get(age_range, AGE_COHORT_PROFILES["all"])
        
        household_size = 5 
        estimated_households = int(predicted_pop / household_size) if predicted_pop > 0 else 0
        
        daily_food_needed = predicted_pop * profile["food_ration_kg"]
        monthly_food_tonnes = (daily_food_needed * 30.4) / 1000
        
        health_kits = int(predicted_pop * profile["health_kit_ratio"])
        water_liters = predicted_pop * profile["water_liters"]

        if profile["school_age"] is True:
            school_children = predicted_pop
        elif profile["school_age"] is None:
            school_children = None
        else:
            school_children = 0

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(label="🏠 Shelters Needed (Est.)", value=f"{estimated_households:,}")
            if school_children is not None:
                st.metric(label="🎒 School-age Children in Cohort", value=f"{school_children:,}")
            else:
                st.caption("⚠️ School-age figures are hidden for 'all' mixed age bands.")
        with m_col2:
            st.metric(label="🍚 Monthly Food Requirement", value=f"{monthly_food_tonnes:.2f} MT")
            st.metric(label="🚑 Healthcare Kits Required", value=f"{health_kits:,}")
        with m_col3:
            st.metric(label="💧 Daily Clean Water Supply", value=f"{water_liters:,} L")

        st.markdown(f"**Recommended Health Kit Type:** {profile['health_kit_type']}")
        st.caption(profile["notes"])

        st.info("""
        💡 **Strategic Guidance:** These estimates map population counts to standard WHO, WFP, and Sphere Handbook humanitarian indicators to streamline camp deployment planning.
        """)


# Historical Trends Visualization Section (2022-2025 Focus)
if history_loaded:
    st.markdown("---")
    st.subheader("📈 Historical Population Trend Analysis (2022 - 2025)")
    
    # Filter for active historical records from 2022 onwards to clean up empty layout sections
    filtered_df = history_df[
        (history_df["origin_location_code"] == origin) &
        (history_df["gender"] == gender) &
        (history_df["age_range"] == age_range) &
        (history_df["year"] >= 2022)
    ]
    
    if not filtered_df.empty:
        yearly_trend = filtered_df.groupby("year")["population"].sum().reset_index()
        yearly_trend = yearly_trend.sort_values("year")
        
        # Format the year cleanly so it displays without decimal dots on the chart axis
        yearly_trend["year"] = yearly_trend["year"].astype(str)
        chart_data = yearly_trend.set_index("year")
        
        st.line_chart(chart_data, y="population", width="stretch")
        st.caption(f"📉 *Showing recorded historical population timeline from 2022 to 2025 for {gender} cohorts aged {age_range} from {origin}.*")
    else:
        st.info("ℹ️ No historical population records exist in the database from 2022-2025 for this specific parameter combination.")

# Clean, Non-Sticky Footer (Perfect for Mobile Screens)
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 20px 0px; background-color: transparent; font-size: 14px; color: #555555; border-top: 1px solid #e0e0e0; margin-top: 40px; margin-bottom: 20px;">
        🌍 <b>Refugee Population Forecasting & Resource Planning System</b><br>
        Developed as a Data Science Capstone Project by Team <b>XG BOOST BUSTERS</b> © 2026
    </div>
    """,
    unsafe_allow_html=True
)
