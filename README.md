# Kenya Refugee Population Forecasting and Humanitarian Resource Planning

A machine-learning system that forecasts refugee and asylum-seeker populations in Kenya and converts the predictions into indicative humanitarian resource requirements.

## Live Application

**Streamlit App:**  
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gsmt43hnnpxa8rxc293o8f.streamlit.app/)

## Project Overview

Kenya hosts refugees and asylum seekers displaced by conflict, persecution, political instability, and climate-related shocks. 
Humanitarian organizations often rely on current or historical population figures when planning resources. This can make it difficult to prepare for sudden increases in refugee arrivals.

This project uses machine learning to:

- analyze historical refugee-population trends;
- forecast future refugee and asylum-seeker populations;
- identify demographic and geographic patterns; and
- estimate humanitarian needs such as food, shelter, healthcare, and education.

## Business Problem

Changes in refugee populations can create unexpected demand for essential services. Planning is complicated by:

- rapid changes in refugee arrivals;
- different needs across age groups and genders;
- varying population patterns by country of origin;
- delays in humanitarian resource deployment; and
- limited use of historical data for forecasting.

The project provides a data-driven decision-support tool for humanitarian agencies and policymakers.

## Project Objectives

1. Explore refugee-population trends in Kenya.
2. Identify demographic and geographic patterns.
3. Build and compare machine-learning forecasting models.
4. Predict population by year, country of origin, population group, gender, and age range.
5. Estimate humanitarian resource requirements.
6. Deploy the selected model through an interactive Streamlit application.

## Dataset

The project uses the **Kenya Refugee and Asylum Population Dataset** obtained through the Humanitarian Data Exchange Humanitarian API, or **HDX HAPI**.

### Dataset Summary

- **Period covered:** 2001–2025
- **Original records:** 27,664
- **Records after cleaning:** 8,540
- **Target variable:** `population`
- **Asylum location:** Kenya
- **Population groups:** Refugees, asylum seekers, host community, and others of concern

### Main Features

| Feature | Description |
|---|---|
| `origin_location_code` | Refugees’ country of origin |
| `population_group` | Refugee or population category |
| `gender` | Male or female |
| `age_range` | Population age group |
| `year` | Reporting year |
| `origin_has_hrp` | Humanitarian response plan indicator |
| `origin_in_gho` | Global humanitarian overview indicator |
| `population` | Recorded population count |

Aggregate gender and age categories were removed to prevent double counting. Zero-population records and extreme values were retained because they may represent valid observations and real displacement events.

## Project Workflow

The project follows the **CRISP-DM** methodology:

1. Business understanding
2. Data understanding
3. Data cleaning
4. Exploratory data analysis
5. Data preprocessing
6. Model development
7. Model evaluation
8. Deployment

## Key Findings

- Somalia is the largest country of origin for refugees hosted in Kenya.
- South Sudan is the second-largest source population.
- Most refugees originate from neighboring or nearby countries.
- Adults aged 18–59 form the largest age group.
- Children aged 0–17 represent a significant share of the population.
- Male and female refugee populations are relatively balanced.
- Refugee-population trends are nonlinear and influenced by regional conflicts and humanitarian events.
- A small number of countries account for most of the refugee population.

## Machine-Learning Models

The following regression models were developed and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- FT-Transformer

Categorical variables were encoded, while historical data through 2022 was used for training and later records were used for testing.

## Classical Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 2,174.56 | 5,005.00 | 0.429 |
| Decision Tree | 484.93 | **2,282.01** | **0.881** |
| Random Forest | **466.63** | 2,285.70 | 0.881 |
| XGBoost | 548.37 | 2,373.54 | 0.872 |

Random Forest achieved the lowest Mean Absolute Error, while Decision Tree achieved the lowest Root Mean Squared Error.

The deployed prototype uses an **FT-Transformer**, which applies feature tokenization and self-attention to learn relationships between categorical and numerical variables.

## Streamlit Application

The Streamlit application allows users to:

- enter demographic and geographic information;
- select a future forecasting year;
- generate a refugee-population prediction;
- estimate food requirements;
- estimate shelter requirements;
- estimate healthcare requirements; and
- estimate education requirements.

The application is designed to make the model accessible to humanitarian organizations, government agencies, and other non-technical users.

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- PyTorch
- SHAP
- Streamlit
- Joblib

## Limitations

- The dataset does not include real-time conflict, climate, economic, or political indicators.
- Unexpected humanitarian crises may produce patterns that differ from historical data.
- Predictions may be less accurate for countries or demographic groups with limited records.
- The model must be retrained as new data becomes available.
- Resource estimates are based on simplified assumptions and should be validated against official humanitarian standards.
- FT-Transformer evaluation metrics should be calculated on the original population scale before being directly compared with the classical models.

## Future Improvements

- Integrate real-time conflict and climate data.
- Automate data collection and model retraining.
- Add prediction intervals to communicate uncertainty.
- Include interactive maps and downloadable reports.
- Improve resource estimates using location-specific humanitarian standards.
- Monitor model performance and data drift after deployment.

## Contributors

**Team: XG BOOST BUSTERS**

- Cynthia Jemutai
- Stephen Jilani
- Charity Nduati
- Joy Njeru
- Chris Karagu
- Sylvia Wambui
