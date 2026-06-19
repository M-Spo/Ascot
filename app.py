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
    
    # Extract historical going types dynamically from dataset column names
    going_cols = [col for col in lookup.columns if col.startswith('prev_') and col.endswith('_performance')]
    unique_goings_list = [col.replace('prev_', '').replace('_performance', '') for col in going_cols]
    if not unique_goings_list:
        unique_goings_list = ["Good", "Good to Firm", "Good to Soft", "Soft", "Heavy"]
        
    if 'rating_band' in lookup.columns:
        all_rating_bands = sorted(lookup['rating_band'].dropna().astype(str).str.strip().unique())
    else:
        all_rating_bands = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]
else:
    all_horses = []
    all_jockeys = []
    all_rating_bands = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]
    unique_goings_list = ["Good", "Good to Firm", "Good to Soft", "Soft", "Heavy"]

# =====================================================================
# 2. RACE CONFIGURATION (SIDEBAR REPLACEMENT WITH AUTOCOMPLETE)
# =====================================================================
st.sidebar.header("⚙️ Race Day Configuration")

# Track Going is back to calculate 'historical_going_performance'
today_going = st.sidebar.selectbox("Track Going Status", unique_goings_list)

# Rating Band autocomplete
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
    st.write("Review features below. **Blanks are highlighted in red**—type overrides or missing values directly into cells.")
    
    feature_cols = [
        'horse', 'jockey', 'rating_band', 'ran', 'age', 'wgt', 'or', 'bookie_prob',
        'prev_avg_performance', 'horse_hot_streak', 'jockey_prev_avg_performance',
        'jockey_hot_streak', 'historical_going_performance'
    ]
    
    compiled_rows = []
    for runner in valid_field_runners:
        horse_match = lookup[lookup['horse'] == runner["horse"]]
        jockey_match = lookup[lookup['jockey'] == runner["jockey"]]
        
        row_data = {'horse': runner["horse"], 'jockey': runner["jockey"]}
        
        # Pull horse records
        if not horse_match.empty:
            h_row = horse_match.iloc[0]
            for col in ['age', 'wgt', 'or', 'prev_avg_performance', 'horse_hot_streak']:
                if col in h_row.index:
                    row_data[col] = h_row[col]
            
            # Use the Track Going selection to dynamically set historical_going_performance
            target_going_col = f"prev_{today_going}_performance"
            
            if target_going_col in h_row.index and pd.notna(h_row[target_going_col]):
                row_data['historical_going_performance'] = h_row[target_going_col]
            else:
                row_data['historical_going_performance'] = h_row.get('prev_avg_performance', np.nan)
                    
        # Pull jockey records
        if not jockey_match.empty:
            j_row = jockey_match.iloc[0]
            for col in ['jockey_prev_avg_performance', 'jockey_hot_streak']:
                if col in j_row.index:
                    row_data[col] = j_row[col]
                    
        # Global inputs
        row_data['rating_band'] = race_rating_band
        row_data['ran'] = race_ran
        row_data['bookie_prob'] = np.nan
            
        for col in feature_cols:
            if col not in row_data:
                row_data[col] = np.nan
                
        filtered_row = {col: row_data[col] for col in feature_cols}
        compiled_rows.append(filtered_row)
        
    initial_matrix_df = pd.DataFrame(compiled_rows)
    
    def highlight_missing(val):
        if pd.isna(val) or val == "" or val is None:
            return 'background-color: #ffcccc; color: black;'
        return ''
        
    styled_df = initial_matrix_df.style.map(
        highlight_missing, 
        subset=[c for c in feature_cols if c not in ['horse', 'jockey', 'rating_band', 'ran']]
    )
    
    editor_key = f"matrix_editor_{today_going}_{race_rating_band}_{len(valid_field_runners)}"
    
    edited_matrix_df = st.data_editor(
        styled_df,
        use_container_width=True,
        disabled=["horse", "jockey", "rating_band", "ran"],
        key=editor_key
    )

    # =====================================================================
    # 5. MATH INFERENCE AND FORECAST LEADERBOARD
    # =====================================================================
    if st.button("🔮 Evaluate Competitor Field Ranks"):
        if model is None:
            st.error("🚨 Missing production asset dependencies.")
        else:
            processing_df = edited_matrix_df.copy()
            
            numeric_cols = [c for c in feature_cols if c not in ['horse', 'jockey', 'rating_band']]
            for col in numeric_cols:
                processing_df[col] = pd.to_numeric(processing_df[col]).fillna(0.5)
                
            model_expected_features = model.feature_name_
            
            for col in model_expected_features:
                if col not in processing_df.columns:
                    processing_df[col] = 0.5
                    
            matrix_inference_ready = processing_df[model_expected_features].copy()
            
            # Formats categories and objects securely for scikit-learn/LGBM
            for col in matrix_inference_ready.columns:
                if matrix_inference_ready[col].dtype == 'object' or isinstance(matrix_inference_ready[col].iloc[0], str):
                    matrix_inference_ready[col] = matrix_inference_ready[col].astype('category')
                else:
                    matrix_inference_ready[col] = pd.to_numeric(matrix_inference_ready[col], errors='coerce').fillna(0.5)
            
            raw_model_predictions = model.booster_.predict(matrix_inference_ready)
            
            scaled_logits = raw_model_predictions - np.max(raw_model_predictions)
            exponential_values = np.exp(scaled_logits)
            win_probabilities_distribution = exponential_values / np.sum(exponential_values)
            
            placing_scale_factor = min(3.0, len(valid_field_runners))
            top3_probabilities_distribution = np.clip(win_probabilities_distribution * placing_scale_factor, 0.0, 0.99)
            
            display_names = [f"{r['horse'].upper()} / {r['jockey']}" for _, r in processing_df.iterrows()]
            leaderboard_df = pd.DataFrame({
                "🎯 Predicted Rank": 0,
                "Competitor Setup (Horse / Jockey)": display_names,
                "Win Probability (#1)": [f"{p*100:.1f}%" for p in win_probabilities_distribution],
                "Top 3 Place Probability": [f"{p*100:.1f}%" for p in top3_probabilities_distribution],
                "_sorting_metric": win_probabilities_distribution  
            })
            
            leaderboard_df = leaderboard_df.sort_values(by="_sorting_metric", ascending=False).reset_index(drop=True)
            leaderboard_df["🎯 Predicted Rank"] = leaderboard_df.index + 1
            leaderboard_df = leaderboard_df.drop(columns=["_sorting_metric"])
            
            st.subheader("🏁 Official Model Forecast Leaderboard")
            st.dataframe(leaderboard_df, use_container_width=True)

else:
    st.info("💡 Please pick at least 2 autocomplete runner pairs above to generate the interactive feature spreadsheet.")

# =====================================================================
# 6. FRACTION ODDS TO PROBABILITY PERCENTAGE CONVERTER
# =====================================================================
st.markdown("---")
st.header("🧮 Bookie Fraction Converter")
st.write("Type in the bookmaker fractional odds (e.g., **4 / 1** or **13 / 8**) to quickly get the probability value for your table.")

calc_col1, calc_col2, calc_col3 = st.columns([2, 1, 3])

with calc_col1:
    numerator = st.number_input("Numerator (Top Number)", value=4, min_value=1, step=1, key="calc_num")
with calc_col2:
    st.markdown("<h3 style='text-align: center; margin-top: 25px;'>/</h3>", unsafe_allow_html=True)
with calc_col3:
    denominator = st.number_input("Denominator (Bottom Number)", value=1, min_value=1, step=1, key="calc_den")

# Probability math execution
implied_prob_decimal = denominator / (numerator + denominator)
implied_prob_percentage = implied_prob_decimal * 100

st.metric(
    label="⚡ Target Value for 'bookie_prob' Column", 
    value=f"{implied_prob_decimal:.3f}  ({implied_prob_percentage:.1f}%)",
    help="Plug the decimal value directly into the red 'bookie_prob' row cells above."
)
