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
    # Get unique names, strip whitespace, drop NaNs, and sort alphabetically
    all_horses = sorted(lookup['horse'].dropna().astype(str).str.strip().unique())
    all_jockeys = sorted(lookup['jockey'].dropna().astype(str).str.strip().unique())
else:
    all_horses = []
    all_jockeys = []

# =====================================================================
# 2. APP CONFIGURATION & APP HEADER
# =====================================================================
st.title("🏇 Ascot Multi-Horse Field Ranker")
st.write("Type a few letters of a horse or jockey name to search, select to autocomplete, and calculate field rankings.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Race Conditions")

if lookup is not None:
    going_cols = [col for col in lookup.columns if col.startswith('prev_') and col.endswith('_performance')]
    unique_goings_list = [col.replace('prev_', '').replace('_performance', '') for col in going_cols]
    if not unique_goings_list:
        unique_goings_list = ["Good", "Good to Firm", "Good to Soft", "Soft", "Heavy"]
else:
    unique_goings_list = ["Good", "Good to Firm", "Good to Soft", "Soft", "Heavy"]

today_going = st.sidebar.selectbox("Track Going Status", unique_goings_list)

# =====================================================================
# 3. DYNAMIC MULTI-HORSE FIELDS GENERATOR WITH AUTOCOMPLETE
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
    r_col1, r_col2, r_col3 = st.columns(3)
    
    with r_col1:
        # Searchable selectbox for horses with an empty default state
        h_name = st.selectbox(
            f"Search Horse Name", 
            options=[""] + all_horses, 
            index=0, 
            key=f"horse_input_{i}"
        ).strip()
    with r_col2:
        # Searchable selectbox for jockeys with an empty default state
        j_name = st.selectbox(
            f"Search Jockey Name", 
            options=[""] + all_jockeys, 
            index=0, 
            key=f"jockey_input_{i}"
        ).strip()
    with r_col3:
        wgt_lbs = st.number_input(f"Carried Weight (lbs)", value=130.0, step=0.5, key=f"weight_input_{i}")
        
    runners_input_list.append({"horse": h_name, "jockey": j_name, "weight": wgt_lbs})

# =====================================================================
# 4. MATH INFERENCE AND FORECAST LEADERBOARD
# =====================================================================
if st.button("🔮 Evaluate Competitor Field Ranks"):
    # Filter down to fields that are actually filled out
    valid_field_runners = [r for r in runners_input_list if r["horse"] and r["jockey"]]
    field_size = len(valid_field_runners)
    
    if field_size < 2:
        st.error("🚨 Evaluation requires a minimum of 2 valid profile entries to map structural rank probability fields.")
    elif model is None or lookup is None:
        st.error("🚨 Missing production asset dependencies. Please verify your file uploads on GitHub.")
    else:
        field_feature_rows = []
        display_names_list = []
        
        for runner in valid_field_runners:
            match = lookup[(lookup['horse'].str.lower() == runner["horse"].lower()) & 
                           (lookup['jockey'].str.lower() == runner["jockey"].lower())]
            
            if match.empty:
                input_row = lookup.mean(numeric_only=True).to_frame().T
                input_row['horse'] = runner["horse"]
                input_row['jockey'] = runner["jockey"]
                input_row['historical_going_performance'] = input_row.get('prev_avg_performance', 0.5)
            else:
                input_row = match.copy()
                target_col = f"prev_{today_going}_performance"
                if target_col in input_row.columns:
                    input_row['historical_going_performance'] = input_row[target_col].values[0]
                else:
                    input_row['historical_going_performance'] = input_row.get('prev_avg_performance', 0.5)
            
            input_row['weight'] = runner["weight"]
            input_row['ran'] = field_size
            
            field_feature_rows.append(input_row)
            display_names_list.append(f"{runner['horse'].upper()} / {runner['jockey']}")
            
        race_day_matrix = pd.concat(field_feature_rows, ignore_index=True)
        
        going_cols_to_drop = [f"prev_{g}_performance" for g in unique_goings_list]
        drop_cols_list = ['horse', 'jockey', 'inverted_rank'] + going_cols_to_drop
        
        feature_columns_final = [col for col in race_day_matrix.columns if col not in drop_cols_list]
        matrix_inference_ready = race_day_matrix[feature_columns_final]
        
        raw_model_predictions = model.booster_.predict(matrix_inference_ready.values)
        
        scaled_logits = raw_model_predictions - np.max(raw_model_predictions)
        exponential_values = np.exp(scaled_logits)
        win_probabilities_distribution = exponential_values / np.sum(exponential_values)
        
        placing_scale_factor = min(3.0, field_size)
        top3_probabilities_distribution = np.clip(win_probabilities_distribution * placing_scale_factor, 0.0, 0.99)
        
        leaderboard_df = pd.DataFrame({
            "🎯 Predicted Rank": 0,
            "Competitor Setup (Horse / Jockey)": display_names_list,
            "Win Probability (#1)": [f"{p*100:.1f}%" for p in win_probabilities_distribution],
            "Top 3 Place Probability": [f"{p*100:.1f}%" for p in top3_probabilities_distribution],
            "_sorting_metric": win_probabilities_distribution  
        })
        
        leaderboard_df = leaderboard_df.sort_values(by="_sorting_metric", ascending=False).reset_index(drop=True)
        leaderboard_df["🎯 Predicted Rank"] = leaderboard_df.index + 1
        leaderboard_df = leaderboard_df.drop(columns=["_sorting_metric"])
        
        st.subheader("🏁 Official Model Forecast Leaderboard")
        st.dataframe(leaderboard_df, use_container_width=True)
