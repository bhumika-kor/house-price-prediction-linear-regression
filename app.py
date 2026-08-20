import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #050505;
    color: white;
}

/* Page width */
.block-container {
    max-width: 1100px;
    padding-top: 3rem;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #c084fc;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #b8b8b8;
    font-size: 18px;
    margin-bottom: 35px;
}

/* Section title */
.section-title {
    color: #c084fc;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Input labels */
label {
    color: #eeeeee !important;
    font-weight: 600 !important;
}

/* NUMBER INPUT OUTER BOX */
div[data-testid="stNumberInput"] {
    background-color: #111111 !important;
    border-radius: 12px !important;
}

/* NUMBER INPUT INNER BOX */
div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background-color: #111111 !important;
    border: 2px solid #6d28d9 !important;
    border-radius: 10px !important;
}

/* NUMBER INPUT FIELD */
div[data-testid="stNumberInput"] input {
    background-color: #111111 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 17px !important;
    font-weight: 600 !important;
}

/* When clicking inside number box */
div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    background-color: #111111 !important;
    border: 2px solid #a855f7 !important;
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.45) !important;
}

/* Number input +/- buttons */
div[data-testid="stNumberInput"] button {
    background-color: #1f1235 !important;
    color: #c084fc !important;
    border: none !important;
}

/* Prediction button */
div.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    font-size: 18px;
    font-weight: 700;
    color: white;
    background: linear-gradient(
        90deg,
        #6d28d9,
        #9333ea,
        #a855f7
    );
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.35);
}

/* Prediction button hover */
div.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(168, 85, 247, 0.55);
}

/* Footer */
.footer {
    text-align: center;
    color: #777777;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD BEST MODEL
# --------------------------------------------------

model = joblib.load("best_house_price_model.pkl")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🏠 House Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict house prices using Gradient Boosting Regressor</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# HOUSE DETAILS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🏡 Enter House Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

# --------------------------------------------------
# LEFT COLUMN
# --------------------------------------------------

with col1:

    OverallQual = st.number_input(
        "Overall Quality (1-10)",
        min_value=1,
        max_value=10,
        value=5
    )

    GrLivArea = st.number_input(
        "Living Area (sq ft)",
        min_value=334,
        max_value=5642,
        value=1500
    )

    GarageCars = st.number_input(
        "Garage Cars",
        min_value=0,
        max_value=4,
        value=2
    )

    TotalBsmtSF = st.number_input(
        "Total Basement Area (sq ft)",
        min_value=0,
        max_value=6110,
        value=1000
    )

# --------------------------------------------------
# RIGHT COLUMN
# --------------------------------------------------

with col2:

    FullBath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=3,
        value=2
    )

    YearBuilt = st.number_input(
        "Year Built",
        min_value=1872,
        max_value=2010,
        value=2000
    )

    BedroomAbvGr = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=8,
        value=3
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔮 Predict House Price"):

    input_data = pd.DataFrame({
        "OverallQual": [OverallQual],
        "GrLivArea": [GrLivArea],
        "GarageCars": [GarageCars],
        "TotalBsmtSF": [TotalBsmtSF],
        "FullBath": [FullBath],
        "YearBuilt": [YearBuilt],
        "BedroomAbvGr": [BedroomAbvGr]
    })

    prediction = model.predict(input_data)

    price = prediction[0]

    st.subheader("Estimated House Price")
    st.success(f"${price:,.2f}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    '<div class="footer">Powered by Gradient Boosting Regressor • House Price Prediction</div>',
    unsafe_allow_html=True
)
