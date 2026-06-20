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
    lookup['horse'] = lookup['horse'].astype(str).str.strip()
    lookup['jockey'] = lookup['jockey'].astype(str).str.strip()

    all_horses = sorted(lookup['horse'].dropna().unique())
    all_jockeys = sorted(lookup['jockey'].dropna().unique())
    
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
# 3. RUNNER SELECTOR SECTION (WITH CUSTOM ENTRY SUPPORT)
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

# Precompute lowercase variants for faster searching
all_horses_lc = [(h, h.lower()) for h in all_horses]
all_jockeys_lc = [(j, j.lower()) for j in all_jockeys]

runners_input_list = []
for i in range(st.session_state.num_runners):
    st.markdown(f"**Runner Position #{i+1}**")
    r_col1, r_col2 = st.columns(2)
    
    with r_col1:
        h_search = st.text_input("Type Horse Name to Filter/Add Custom", key=f"horse_search_{i}").strip()
        filtered_horses = [h for h, hl in all_horses_lc if h_search.lower() in hl][:20] if h_search else []
        h_sel = st.selectbox(f"Select Horse Match #{i+1}", options=[""] + filtered_horses, key=f"horse_input_{i}")
        # Fall back to the raw string typed if nothing is selected in dropdown
        h_name = h_sel.strip() if h_sel else h_search

    with r_col2:
        j_search = st.text_input("Type Jockey Name to Filter/Add Custom", key=f"jockey_search_{i}").strip()
        filtered_jockeys = [j for j, jl in all_jockeys_lc if j_search.lower() in jl][:20] if j_search else []
        j_sel = st.selectbox(f"Select Jockey Match #{i+1}", options=[""] + filtered_jockeys, key=f"jockey_input_{i}")
        # Fall back to the raw string typed if nothing is selected in dropdown
        j_name = j_sel.strip() if j_sel else j_search
        
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
        horse_match = lookup[lookup['horse'] == runner["horse"]] if lookup is not None else pd.DataFrame()
        jockey_match = lookup[lookup['jockey'] == runner["jockey"]] if lookup is not None else pd.DataFrame()
        
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
    
    def highlight
