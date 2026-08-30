"""
Nigerian Real Estate Price Prediction — Streamlit App
--------------------------------------------------------
Run with:  streamlit run app.py

Two tabs:
  1. Predict Price — GUI form that predicts a sale price
  2. Data Analysis — the 5 analytical questions with charts
"""

import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Abuja Real Estate Price Predictor", page_icon="🏠", layout="wide")

# ------------------------------------------------------------------
# LOAD DATA + MODEL (cached so it only loads once)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_data.csv")

@st.cache_resource
def load_model():
    with open("model/price_model.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
bundle = load_model()
model = bundle["model"]
model_name = bundle["model_name"]
encoders = bundle["encoders"]
feature_columns = bundle["feature_columns"]

areas = sorted(df["area"].unique())
property_types = sorted(df["property_type"].unique())

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("🏠 Nigerian Real Estate Price Predictor")
st.caption("Machine learning price predictor trained on Abuja property sale listings")

tab1, tab2 = st.tabs(["📈 Predict Price", "📊 Data Analysis"])

# ------------------------------------------------------------------
# TAB 1 — PREDICTION
# ------------------------------------------------------------------
with tab1:
    st.subheader("Enter Property Details")

    col1, col2 = st.columns(2)
    with col1:
        area = st.selectbox("Location (Area)", areas)
        property_type = st.selectbox("Property Type", property_types)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=15, value=4, step=1)
    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=15, value=4, step=1)
        toilets = st.number_input("Toilets", min_value=1, max_value=15, value=5, step=1)

    if st.button("PREDICT PROPERTY PRICE", type="primary", use_container_width=True):
        # Encode inputs using the SAME encoders fit during training.
        try:
            area_enc = encoders["area"].transform([area])[0]
        except ValueError:
            area_enc = -1  # unseen category fallback
        try:
            type_enc = encoders["property_type"].transform([property_type])[0]
        except ValueError:
            type_enc = -1

        input_row = pd.DataFrame(
            [[area_enc, type_enc, bedrooms, bathrooms, toilets]],
            columns=feature_columns,
        )
        prediction = model.predict(input_row)[0]
        prediction = max(prediction, 0)  # price can't be negative

        st.success(f"### Estimated Price: ₦{prediction:,.0f}")
        st.caption(f"Prediction generated using: {model_name}")

        # Give the user some context: how does this compare to similar listings?
        similar = df[(df["area"] == area) & (df["property_type"] == property_type)]
        if len(similar) > 0:
            st.info(
                f"For reference, {len(similar)} similar listing(s) in {area} "
                f"({property_type}) in the dataset averaged "
                f"₦{similar['price_ngn'].mean():,.0f}."
            )
        else:
            st.info(f"No exact matching listings for {property_type} in {area} in the training data — "
                     f"this estimate relies on the model's general pricing patterns.")

    st.divider()
    with st.expander("ℹ️ About this predictor"):
        st.markdown(f"""
        - **Model used:** {model_name} (selected for the best evaluation performance
          between Linear Regression and Decision Tree — see comparison table below)
        - **Trained on:** {len(df)} cleaned sale listings from Abuja, Nigeria
        - **Limitations:** the dataset does not include property size (square metres),
          age of property, or amenities, so predictions reflect location, property
          type, and room counts only.
        """)
        st.dataframe(pd.read_csv("model/model_comparison.csv"), use_container_width=True)

# ------------------------------------------------------------------
# TAB 2 — DATA ANALYSIS
# ------------------------------------------------------------------
with tab2:
    st.subheader("Exploratory Data Analysis")
    st.caption(f"Based on {len(df)} cleaned sale listings")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Q1: Which property type has the highest average price?**")
        q1 = df.groupby("property_type")["price_ngn"].mean().sort_values(ascending=False)
        st.bar_chart(q1)

        st.markdown("**Q3: Does bedroom count affect price?**")
        st.scatter_chart(df, x="bedrooms", y="price_ngn")
        corr = df["bedrooms"].corr(df["price_ngn"])
        st.caption(f"Correlation: {corr:.2f}")

        st.markdown("**Q5: Which property type is most affordable per room?**")
        q5 = df.groupby("property_type")["price_per_room"].mean().sort_values()
        st.bar_chart(q5)

    with c2:
        st.markdown("**Q2: Which area has the highest average price?**")
        q2 = df.groupby("area")["price_ngn"].mean().sort_values(ascending=False)
        st.bar_chart(q2)

        st.markdown("**Q4: Does total room count affect price?**")
        st.scatter_chart(df, x="total_rooms", y="price_ngn")
        corr2 = df["total_rooms"].corr(df["price_ngn"])
        st.caption(f"Correlation: {corr2:.2f}")

    st.divider()
    st.markdown("**Raw cleaned dataset**")
    st.dataframe(df, use_container_width=True)
