import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# =====================================================
# Fix 3: Explicitly define CPU support for Cloud environments
# =====================================================
device = torch.device("cpu")

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Refugee Population Forecasting System",
    page_icon="🌍",
    layout="wide"
)

# =====================================================
# Recreate the FT-Transformer PyTorch Architecture
# =====================================================
class NumericalTokenizer(nn.Module):
    """Projections of raw scalars to token embeddings of dimension d"""
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
# Load Saved Weights & Configurations Safely
# =====================================================
@st.cache_resource
def load_assets():
    # Load configuration and scaler filenames
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    model_config = joblib.load("model_config.pkl")
    
    # Safe fallback using .get() to prevent KeyError
    attn_dropout = model_config.get('attn_dropout', 0.1)
    ff_dropout = model_config.get('ff_dropout', 0.1)
    
    # Instantiate the architecture
    model = FTTransformer(
        cat_cardinalities=model_config['cat_cardinalities'],
        num_features=model_config['num_features'],
        embed_dim=model_config['embed_dim'],
        depth=model_config['depth'],
        heads=model_config['heads'],
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout
    )
    
    # Map weights explicitly to CPU device
    state_dict = torch.load("ft_transformer_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, label_encoders, scaler

try:
    model, label_encoders, scaler = load_assets()
    st.success("🤖 State-of-the-art FT-Transformer Model loaded successfully on CPU!")
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.info("Ensure that 'ft_transformer_model.pth', 'model_config.pkl', 'label_encoders.pkl', and 'scaler.pkl' are in your root directory.")

# =====================================================
# Application UI
# =====================================================
st.title("🌍 AI-Powered Refugee Population Forecasting System")
st.write(
    """
    This dashboard leverages an advanced **Feature Tokenizer Transformer (FT-Transformer)** deep learning network 
    to forecast localized refugee population trends in Kenya and translate predicted figures into immediate resource plans.
    """
)

col1, col2 = st.columns([1, 1.2])

# Extract valid categorical options directly from your trained encoders
valid_origins = list(label_encoders['origin_location_code'].classes_)
valid_pop_groups = list(label_encoders['population_group'].classes_)
valid_genders = list(label_encoders['gender'].classes_)
valid_age_ranges = list(label_encoders['age_range'].classes_)

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
    
    # Calculate age boundaries dynamically for scaling consistency
    if age_range == "0-4":
        min_age, max_age = 0.0, 4.0
    elif age_range == "5-11":
        min_age, max_age = 5.0, 11.0
    elif age_range == "12-17":
        min_age, max_age = 12.0, 17.0
    elif age_range == "18-59":
        min_age, max_age = 18.0, 59.0
    else: # "60+"
        min_age, max_age = 60.0, 100.0

    # Build input DataFrame matching Scaler training order
    raw_numerical = pd.DataFrame([{
        'origin_has_hrp': 1.0 if origin_has_hrp else 0.0,
        'origin_in_gho': 1.0 if origin_in_gho else 0.0,
        'min_age': min_age,
        'max_age': max_age,
        'population': 0.0, # Dummy target placeholder
        'year': float(year)
    }])

    # Build categorical inputs
    raw_categorical = pd.DataFrame([{
        'origin_location_code': origin,
        'population_group': population_group,
        'gender': gender,
        'age_range': age_range
    }])

    st.markdown("**Processed Input Vector:**")
    st.write(pd.concat([raw_categorical, raw_numerical.drop(columns=['population'])], axis=1))

    if st.button("🔮 Run Deep Learning Inference"):
        try:
            # 1. Encode Categoricals
            encoded_cat = raw_categorical.copy()
            for col in ['origin_location_code', 'population_group', 'gender', 'age_range']:
                encoded_cat[col] = label_encoders[col].transform(raw_categorical[col])
            
            # 2. Scale Numericals using loaded Scaler
            scaled_num = scaler.transform(raw_numerical)
            # Remove target 'population' index from numeric array to align with feature shape
            scaled_num_features = np.delete(scaled_num, 4, axis=1)

            # 3. Convert to PyTorch Tensors
            tensor_cat = torch.tensor(encoded_cat.values, dtype=torch.long).to(device)
            tensor_num = torch.tensor(scaled_num_features, dtype=torch.float32).to(device)

            # 4. Model Prediction
            with torch.no_grad():
                pred_raw = model(tensor_cat, tensor_num).item()
            
            # Force non-negativity
            predicted_pop = max(0, int(round(pred_raw)))

            # Display Target Forecast Result
            st.metric(
                label="Predicted Refugee Population Segment", 
                value=f"{predicted_pop:,} individuals"
            )

            # =================================================
            # Operational Resource Metrics
            # =================================================
            st.markdown("---")
            st.subheader("📦 Projected Resource Requirements")
            
            # Operational Planning Constants
            household_size = 5 
            daily_ration_kg = 0.45 
            
            # Calculations
            estimated_households = int(predicted_pop / household_size)
            daily_food_needed = predicted_pop * daily_ration_kg
            monthly_food_tonnes = (daily_food_needed * 30.4) / 1000
            
            # Enhanced Indicator Calculations
            health_kits = predicted_pop
            school_children = int(predicted_pop * 0.42)
            water_liters = predicted_pop * 15

            # Grid layout for advanced planning metrics
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

        except Exception as e:
            st.error(f"Prediction Pipeline Failed: {e}")
            st.info("Please verify your input mappings align with scaler configurations.")

# =====================================================
# Application Footer
# =====================================================
st.markdown("---")
st.caption("""
🌍 **AI-Powered Refugee Population Forecasting and Humanitarian Resource Planning System for Kenya**

Built using PyTorch FT-Transformer • Streamlit • CRISP-DM

Developed as a Data Science Capstone Project by Team **XG BOOST BUSTERS**.
"""
)
