import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #050505;
    color: white;
}

.block-container {
    max-width: 1100px;
    padding-top: 3rem;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #a855f7, #7c3aed, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #b8b8b8;
    font-size: 18px;
    margin-bottom: 35px;
}

.section-title {
    color: #c084fc;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 15px;
}

label {
    color: #eeeeee !important;
    font-weight: 600 !important;
}

/* INPUT BOX - DARK FILL */
div[data-baseweb="input"] {
    background-color: #1f1f1f !important;
    border: 1px solid #4c1d95 !important;
    border-radius: 10px !important;
}

/* INSIDE INPUT BOX */
div[data-baseweb="input"] > div {
    background-color: #1f1f1f !important;
}

/* NUMBER - WHITE */
div[data-baseweb="input"] input {
    color: white !important;
    -webkit-text-fill-color: white !important;
    background-color: #1f1f1f !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* PLUS AND MINUS BUTTONS */
div[data-baseweb="input"] button {
    color: white !important;
    background-color: #1f1f1f !important;
}

div[data-baseweb="input"] input:focus {
    color: white !important;
    -webkit-text-fill-color: white !important;
}

/* PREDICT BUTTON */
div.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    font-size: 18px;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #6d28d9, #9333ea, #a855f7);
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.35);
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(168, 85, 247, 0.55);
}

/* PREDICTION CARD */
.prediction-card {
    margin-top: 30px;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    background: linear-gradient(135deg, #3b0764, #6d28d9, #9333ea);
    box-shadow: 0 10px 40px rgba(124, 58, 237, 0.4);
}

.prediction-label {
    font-size: 18px;
    color: #e9d5ff;
}

.prediction-price {
    font-size: 38px;
    font-weight: 800;
    color: white;
}

.footer {
    text-align: center;
    color: #777777;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


# Load dataset
df = pd.read_csv("house price.csv")


# Select features
df = df[['OverallQual',
         'GrLivArea',
         'GarageCars',
         'TotalBsmtSF',
         'FullBath',
         'YearBuilt',
         'BedroomAbvGr',
         'SalePrice']]


# Separate features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]


# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)


# Header
st.markdown(
    '<div class="main-title">🏠 House Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict house prices using Linear Regression</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">🏡 Enter House Details</div>',
    unsafe_allow_html=True
)


# Two columns
col1, col2 = st.columns(2)


# Left column
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


# Right column
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


# Prediction button
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

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">
                Estimated House Price
            </div>

            <div class="prediction-price">
                ${prediction[0]:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Footer
st.markdown(
    '<div class="footer">Powered by Linear Regression • House Price Prediction</div>',
    unsafe_allow_html=True
)