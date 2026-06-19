import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="centered")
st.title("🏇 Ascot Live Ranker")
st.write("Enter booklet info to score a runner instantly.")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('lgbm_ranker_model.pkl')
        lookup = pd.read_csv('historical_lookup.csv')
        return model, lookup
    except Exception as e:
        # This keeps the app from crashing while your PC finishes training
        st.info("⏳ App structure is ready! Uploading 'lgbm_ranker_model.pkl' and 'historical_lookup.csv' to GitHub will activate the system.")
        return None, None

model, lookup = load_assets()

st.header("📝 Enter Runner Details")
horse_name = st.text_input("Horse Name (Exact match)").strip()
jockey_name = st.text_input("Jockey Name (Exact match)").strip()

st.subheader("📊 Race Day Booklet Info")
booklet_weight = st.number_input("Carried Weight (lbs)", value=130.0)
booklet_runners = st.number_input("Total Runners in Race (Field Size)", value=10, step=1)

if st.button("🔮 Pull History & Predict Score"):
    if model is None or lookup is None:
        st.error("Model files aren't in your GitHub repository yet.")
    else:
        match = lookup[(lookup['horse'].str.lower() == horse_name.lower()) & 
                       (lookup['jockey'].str.lower() == jockey_name.lower())]
        
        if match.empty:
            st.warning("⚠️ No historical profile found. Using field averages.")
            input_data = lookup.mean(numeric_only=True).to_frame().T
        else:
            st.success("✅ Historical profile loaded!")
            input_data = match.copy()
        
        if 'weight' in input_data.columns:
            input_data['weight'] = booklet_weight
        if 'ran' in input_data.columns:
            input_data['ran'] = booklet_runners
            
        feature_cols = [col for col in input_data.columns if col not in ['horse', 'jockey', 'inverted_rank']]
        final_features = input_data[feature_cols]
        raw_prediction = model.predict(final_features)[0]
        
        st.metric(label="Model Raw Prediction Score", value=f"{raw_prediction:.4f}")
