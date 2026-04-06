import streamlit as st
import pickle
from datetime import datetime
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# Page config
st.set_page_config(page_title="AI Scrap Predictor", layout="wide")

# Custom CSS (🔥 makes it beautiful)
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
    }
    .main {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .stButton>button {
        background-color: #00c6ff;
        color: black;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("## ♻️ AI Scrap Price Predictor Dashboard")

# Layout
col1, col2 = st.columns([1, 2])

# INPUT SECTION
with col1:
    st.subheader("🔧 Input Parameters")

    material = st.selectbox("Material", ["iron", "plastic", "copper", "aluminum"])
    weight = st.slider("Weight (kg)", 1, 50)
    demand = st.slider("Market Demand", 1, 10)

    predict_btn = st.button("🚀 Predict Price")

# OUTPUT SECTION
with col2:
    st.subheader("📊 Prediction Dashboard")

    if predict_btn:
        material_encoded = encoder.transform([material])[0]

        hour = datetime.now().hour
        day = datetime.now().weekday()

        prediction = model.predict([[material_encoded, weight, demand, hour, day]])
        price = prediction[0]

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Predicted Price", f"₹ {price:.2f}")
        m2.metric("📈 Demand Level", demand)
        m3.metric("⚖️ Weight", f"{weight} kg")

        # Fake trend data (for UI demo)
        trend_data = pd.DataFrame({
            "Price": [price - 50, price - 20, price, price + 10, price + 30]
        })

        st.line_chart(trend_data)

        # Highlight box
        st.success(f"🔥 Final Predicted Price: ₹ {price:.2f}")

    else:
        st.info("👉 Enter values and click Predict")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using AI & Streamlit")