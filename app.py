import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set mobile-friendly wide configuration layout
st.set_page_config(page_title="Ascot Multi-Horse Ranker", layout="wide")

# =====================================================================
# 1. ASSET LOADING (WITH CACHING AND 3-WAY SPLIT RECONSTRUCTION)
# =====================================================================
@st.cache_resource
def load_production_assets():
    try:
        model = joblib.load('lgbm_ranker_model.pkl')
        
        # Load all three compressed fragments
        part1 = pd.read_csv('historical_lookup_part1.zip')
        part2 = pd.read_csv('historical_lookup_part2.zip')
        part3 = pd.read_csv('historical_lookup_part3.zip')
        
        # Stitch them back into a single master lookup dataframe
        lookup = pd.concat([part1, part2, part3], ignore_index=True)
        return model, lookup
    except Exception as e:
        st.error(f"⚠️ Error loading or reconstructing asset files: {e}")
        return None, None

model, lookup = load_production_assets()

# Pre-process unique sorted lists for the autocomplete dropdowns
if lookup is not None:
    all_horses = sorted(lookup['horse'].dropna().astype(str).str.strip().unique())
    all_jockeys = sorted(lookup['jockey'].dropna().astype(str).str.strip().unique())
    # Grab unique rating bands from the dataset if they exist, fallback if not
    if 'rating_band' in lookup.columns:
        all_rating_bands = sorted(lookup['rating_band'].dropna().astype(str).str.strip().unique())
    else:
        all_rating_bands = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Group 1", "Group 2", "Group 3"]
else:
    all_horses = []
    all_jockeys = []
    all_rating_bands = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]

# =====================================================================
# 2. RACE CONFIGURATION (SIDEBAR REPLACEMENT WITH AUTOCOMPLETE)
# =====================================================================
st.title("🏇 Ascot Multi-Horse Field Ranker")
st.write("Select your runners to review/edit their historical stats, then run the ranking model.")

st.sidebar.header("⚙️ Race Day Configuration")

# Rating Band now uses a searchable selectbox for autocomplete
race_rating_band = st.sidebar.selectbox(
    "Search Rating Band", 
    options=[""] + all_rating_bands, 
    index=1 if len(all_rating_bands) > 0 else 0
).strip()

race_ran = st.sidebar.number_input("Total Runners (ran)", value=8, step=1)

# =====================================================================
# 3. RUNNER SELECTOR SECTION
# =====================================================================
st.header("📋 Field Entry Profile Selection")

if 'num_runners' not in st.session_state:
    st.session_state.num_runners = 3

col_add, col_rem, _ = st.columns([1, 1, 4])
with col_add:
    if st.button("➕ Add Runner Position"):
        st.session_state.num_runners += 1
with col_rem:
    if st.button("❌ Remove Last Position") and st.session_state.num_runners > 2:
        st.session_state.num_runners -= 1

runners_input_list = []
for i in range(st.session_state.num_runners):
    st.markdown(f"**Runner Position #{i+1}**")
    r_col1, r_col2 = st.columns(2)
    
    with r_col1:
        h_name = st.selectbox(f"Search Horse Name", options=[""] + all_horses, index=0, key=f"horse_input_{i}").strip()
    with r_col2:
        j_name = st.selectbox(f"Search Jockey Name", options=[""] + all_jockeys, index=0, key=f"jockey_input_{i}").strip()
        
    runners_input_list.append({"horse": h_name, "jockey": j_name})

# =====================================================================
# 4. INTERACTIVE FEATURE COMPILATION TABLE
# =====================================================================
valid_field_runners = [r for r in runners_input_list if r["horse"] and r["jockey"]]

if len(valid_field_runners) >= 2:
    st.header("📊 Race Feature Matrix Verification")
    st.write("Review the fetched features below. **Any blanks (NaN) are highlighted**—you can type directly into the cells to fill them in before predicting.")
    
    # Define exact tracking feature columns requested
    feature_cols = [
        'horse', 'jockey', 'rating_band', 'ran', 'age', 'wgt', 'or', 'bookie_prob',
        'prev_avg_performance', 'horse_hot_streak', 'jockey_prev_avg_performance',
        'jockey_hot_streak', 'historical_going_performance'
    ]
    
    compiled_rows = []
    for runner in valid_field_runners:
        match = lookup[(lookup['horse'] == runner["horse"]) & (lookup['jockey'] == runner["jockey"])]
        
        if not match.empty:
            row_data = match.copy().iloc[0].to_dict()
        else:
            row_data = {'horse': runner["horse"], 'jockey': runner["jockey"]}
            
        # Inject the global sidebar variables
        row_data['rating_band'] = race_rating_band
        row_data['ran'] = race_ran
        
        # Ensure every requested feature column exists in the row dict (leave empty if missing)
        for col in feature_cols:
            if col not in row_data:
                row_data[col] = np.nan
                
        # Filter down strictly to requested schema keys
        filtered_row = {col: row_data[col] for col in feature_cols}
        compiled_rows.append(filtered_row)
        
    initial_matrix_df = pd.DataFrame(compiled_rows)
    
    # Highlight missing data blocks beautifully for visual auditing
    def highlight_missing(val):
        if pd.isna(val) or val == "":
            return 'background-color: #ffcccc; color: black;'
        return ''
        
    # Render editable table on-screen so missing values can be typed into directly
    edited_matrix_df = st.data_editor(
        initial_matrix_df.style.map(highlight_missing, subset=[c for c in feature_cols if c not in ['horse', 'jockey', 'rating_band', 'ran']]),
        use_container_width=True,
        disabled=["horse", "jockey", "rating_band", "ran"],
        key="feature_matrix_editor"
    )

    # =====================================================================
    # 5. MATH INFERENCE AND FORECAST LEADERBOARD
    # =====================================================================
    if st.button("🔮 Evaluate Competitor Field Ranks"):
        if model is None:
            st.error("🚨 Missing production asset dependencies.")
        else:
            # Copy edited matrix and fill any remaining unaddressed blanks with baseline safety values
            processing_df = edited_matrix_df.copy()
            
            # Simple conversion to handle text inputs as numbers
            numeric_cols = [c for c in feature_cols if c not in ['horse', 'jockey', 'rating_band']]
            for col in numeric_cols:
                processing_df[col] = pd.to_numeric(processing_df[col]).fillna(0.5)
                
            # Align columns perfectly with what your trained LightGBM model specifies
            model_expected_features = model.feature_name_
            
            # Fill missing required training columns with zero-baselines if they aren't part of our editable set
            for col in model_expected_features:
                if col not in processing_df.columns:
                    processing_df[col] = 0.5
                    
            # Force exact layout order required by LightGBM C++ code
            matrix_inference_ready = processing_df[model_expected_features]
            
            # Run prediction using the pure booster engine
            raw_model_predictions = model.booster_.predict(matrix_inference_ready.values)
            
            # Softmax distribution mapping for Win Chance (#1)
            scaled_logits = raw_model_predictions - np.max(raw_model_predictions)
            exponential_values = np.exp(scaled_logits)
            win_probabilities_distribution = exponential_values / np.sum(exponential_values)
            
            # Top-3 Place Probability Proxy Estimation
            placing_scale_factor = min(3.0, len(valid_field_runners))
            top3_probabilities_distribution = np.clip(win_probabilities_distribution * placing_scale_factor, 0.0, 0.99)
            
            # Construct summary display leaderboard frame
            display_names = [f"{r['horse'].upper()} / {r['jockey']}" for _, r in processing_df.iterrows()]
            leaderboard_df = pd.DataFrame({
                "🎯 Predicted Rank": 0,
                "Competitor Setup (Horse / Jockey)": display_names,
                "Win Probability (#1)": [f"{p*100:.1f}%" for p in win_probabilities_distribution],
                "Top 3 Place Probability": [f"{p*100:.1f}%" for p in top3_probabilities_distribution],
                "_sorting_metric": win_probabilities_distribution  
            })
            
            # Sort values cleanly down by highest win percentage
            leaderboard_df = leaderboard_df.sort_values(by="_sorting_metric", ascending=False).reset_index(drop=True)
            leaderboard_df["🎯 Predicted Rank"] = leaderboard_df.index + 1
            leaderboard_df = leaderboard_df.drop(columns=["_sorting_metric"])
            
            st.subheader("🏁 Official Model Forecast Leaderboard")
            st.dataframe(leaderboard_df, use_container_width=True)
else:
    st.info("💡 Please pick at least 2 autocomplete runner pairs above to generate the interactive feature spreadsheet.")
