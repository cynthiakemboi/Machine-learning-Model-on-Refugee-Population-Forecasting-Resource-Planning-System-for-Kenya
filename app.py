
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import pickle

# =====================================================
# Device Configuration
# =====================================================
device = torch.device("cpu")

st.set_page_config(
    page_title="Refugee Population Forecasting System",
    page_icon="🌍",
    layout="wide"
)

# =====================================================
# Recreate the FT-Transformer PyTorch Architecture
# =====================================================
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

# =====================================================
# Safe Loaders & Diagnostics
# =====================================================
@st.cache_resource
def load_assets():
    # 1. Load label encoders safely
    try:
        with open("label_encoders.pkl", "rb") as f:
            raw_encoders = pickle.load(f)
    except Exception:
        raw_encoders = joblib.load("label_encoders.pkl")
        
    # Standardize label_encoders to a clean dictionary
    label_encoders = {}
    
    if isinstance(raw_encoders, dict):
        label_encoders = raw_encoders
    elif isinstance(raw_encoders, (list, tuple)):
        # If it was saved as a list, map key names by their order
        expected_keys = ['origin_location_code', 'population_group', 'gender', 'age_range']
        for i, encoder in enumerate(raw_encoders):
            if i < len(expected_keys):
                label_encoders[expected_keys[i]] = encoder
    else:
        raise TypeError(f"Loaded encoders are of unsupported type: {type(raw_encoders)}")

    # 2. Load Scaler
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception:
        scaler = joblib.load("scaler.pkl")

    # 3. Load Model Config
    try:
        with open("model_config.pkl", "rb") as f:
            model_config = pickle.load(f)
    except Exception:
        model_config = joblib.load("model_config.pkl")

    state_dict = torch.load("ft_transformer_model.pth", map_location=device)
    
    max_layer_idx = -1
    for key in state_dict.keys():
        if key.startswith("transformer.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                max_layer_idx = max(max_layer_idx, int(parts[2]))
                
    detected_depth = max_layer_idx + 1 if max_layer_idx != -1 else model_config.get('depth', 3)
    
    attn_dropout = model_config.get('attn_dropout', 0.1)
    ff_dropout = model_config.get('ff_dropout', 0.1)
    
    model = FTTransformer(
        cat_cardinalities=model_config['cat_cardinalities'],
        num_features=model_config['num_features'],
        embed_dim=model_config['embed_dim'],
        depth=detected_depth, 
        heads=model_config['heads'],
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, label_encoders, scaler

# Initialize Global App Assets
model = None
label_encoders = None
scaler = None

try:
    model, label_encoders, scaler = load_assets()
    st.success("🤖 FT-Transformer model assets parsed and loaded!")
except Exception as e:
    st.error(f"⚠️ Error preparing encoders: {e}")
    # Run interactive diagnostic display so we can inspect what was inside label_encoders.pkl
    try:
        with open("label_encoders.pkl", "rb") as f:
            raw_data = pickle.load(f)
        st.write("🔧 **Diagnostic Data Inspection of 'label_encoders.pkl':**")
        st.write(f"- Object Type: `{type(raw_data)}`")
        if isinstance(raw_data, dict):
            st.write(f"- Dictionary Keys: `{list(raw_data.keys())}`")
        elif isinstance(raw_data, (list, tuple)):
            st.write(f"- List Length: `{len(raw_data)}` elements")
            for idx, item in enumerate(raw_data):
                st.write(f"  - Element [{idx}] Type: `{type(item)}`")
    except Exception as diagnostic_err:
        st.write(f"Could not run diagnostic check: {diagnostic_err}")

# =====================================================
# App Interface Execution Guard
# =====================================================
if label_encoders is not None and model is not None and scaler is not None:

    st.title("🌍 AI-Powered Refugee Population Forecasting System")
    st.write(
        """
        This dashboard leverages an advanced **Feature Tokenizer Transformer (FT-Transformer)** deep learning network 
        to forecast localized refugee population trends in Kenya.
        """
    )

    col1, col2 = st.columns([1, 1.2])

    # Extract valid options using fallback keys if some aren't present
    try:
        valid_origins = list(label_encoders.get('origin_location_code', list(label_encoders.values())[0]).classes_)
        valid_pop_groups = list(label_encoders.get('population_group', list(label_encoders.values())[1]).classes_)
        valid_genders = list(label_encoders.get('gender', list(label_encoders.values())[2]).classes_)
        valid_age_ranges = list(label_encoders.get('age_range', list(label_encoders.values())[3]).classes_)
    except Exception as parse_err:
        st.error(f"Failed parsing classes from label encoders: {parse_err}")
        st.stop()

    with col1:
        st.subheader("📋 Demographic Parameters")
        
        origin = st.selectbox("Country of Origin", options=valid_origins)
        population_group = st.selectbox("Population Group Type", options=valid_pop_groups)
        gender = st.selectbox("Gender Cohort", options=valid_genders)
        age_range = st.selectbox("Age Range", options=valid_age_ranges)
        
        st.subheader("⏱️ Forecasting Timeline")
        year = st.slider("Target Forecast Year", min_value=2026, max_value=2030, value=2026)

        st.subheader("💡 Geopolitical Indicators")
        origin_has_hrp = st.checkbox("Origin has active Humanitarian Response Plan (HRP)", value=True)
        origin_in_gho = st.checkbox("Included in Global Humanitarian Overview (GHO)", value=True)
        asylum_has_hrp = st.checkbox("Kenya has active Humanitarian Response Plan (HRP)", value=True)
        asylum_in_gho = st.checkbox("Kenya included in Global Humanitarian Overview (GHO)", value=True)

    with col2:
        st.subheader("📊 Model Inference & Resource Forecasting")
        
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
            'population': 0.0,  
            'year': float(year)
        }])

        raw_categorical = pd.DataFrame([{
            'origin_location_code': origin,
            'population_group': population_group,
            'gender': gender,
            'age_range': age_range
        }])

        st.markdown("**Processed Input Vector:**")
        st.write(pd.concat([raw_categorical, raw_numerical.drop(columns=['population'])], axis=1))

        if "predicted_pop" not in st.session_state:
            st.session_state.predicted_pop = None

        if st.button("🔮 Run Deep Learning Inference"):
            with st.spinner("Calculating predictions..."):
                try:
                    # 1. Encode Categoricals safely
                    encoded_cat = raw_categorical.copy()
                    
                    # Target mapping keys or fallbacks based on loaded keys
                    cat_cols = ['origin_location_code', 'population_group', 'gender', 'age_range']
                    for idx, col in enumerate(cat_cols):
                        encoder = label_encoders.get(col, list(label_encoders.values())[idx])
                        encoded_cat[col] = encoder.transform(raw_categorical[col])
                    
                    # 2. Scale Numericals
                    scaled_num = scaler.transform(raw_numerical)
                    scaled_num_features = np.delete(scaled_num, 4, axis=1)

                    # 3. Convert to Tensors
                    tensor_cat = torch.tensor(encoded_cat.values, dtype=torch.long).to(device)
                    tensor_num = torch.tensor(scaled_num_features, dtype=torch.float32).to(device)

                    # 4. Predict
                    with torch.no_grad():
                        pred_raw = model(tensor_cat, tensor_num).item()
                    
                    st.session_state.predicted_pop = max(0, int(round(pred_raw)))
                except Exception as e:
                    st.error(f"Prediction Pipeline Failed: {e}")

        if st.session_state.predicted_pop is not None:
            predicted_pop = st.session_state.predicted_pop
            
            st.success("✅ Prediction Completed!")
            st.metric(
                label="Predicted Refugee Population Segment", 
                value=f"{predicted_pop:,} individuals"
            )

            st.markdown("---")
            st.subheader("📦 Projected Resource Requirements")
            
            household_size = 5 
            daily_ration_kg = 0.45 
            
            estimated_households = int(predicted_pop / household_size)
            daily_food_needed = predicted_pop * daily_ration_kg
            monthly_food_tonnes = (daily_food_needed * 30.4) / 1000
            
            health_kits = predicted_pop
            school_children = int(predicted_pop * 0.42)
            water_liters = predicted_pop * 15

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(label="Estimated Households (Shelters)", value=f"{estimated_households:,}")
                st.metric(label="School-age Children (42% Est.)", value=f"{school_children:,}")
            with m_col2:
                st.metric(label="Monthly Food Target", value=f"{monthly_food_tonnes:.2f} MT")
                st.metric(label="Health Kits Needed", value=f"{health_kits:,}")
            with m_col3:
                st.metric(label="Daily Water Requirement", value=f"{water_liters:,} L")

            st.info("""
            💡 **Strategic Guidance:** These estimates map population counts to standard WHO, WFP, and Sphere Handbook humanitarian indicators to streamline camp deployment planning.
            """)

    st.markdown("---")
    st.caption("""
    🌍 **AI-Powered Refugee Population Forecasting and Humanitarian Resource Planning System for Kenya**
    
    Developed as a Data Science Capstone Project by Team **XG BOOST BUSTERS**.
    """)
else:
    st.info("🕒 Check diagnostic instructions above to resolve the file formats.")
