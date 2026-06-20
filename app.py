import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Set page layout configuration
st.set_page_config(page_title="Ascot Predictive Engine", layout="wide")

# =====================================================================
# 1. INITIAL SYSTEM PATH INITIALIZATION & STORAGE VALIDATION
# =====================================================================
DB_DIR = "competitor_databases"
MODEL_PATH = "lgbm_ranker_model.pkl"

@st.cache_resource
def load_predictive_assets():
    """Safely handles importing the core serialized LightGBM runtime pipeline."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

@st.cache_data
def load_competitor_records():
    """Initializes and transforms compressed lookup registers into reference indices."""
    horse_path = os.path.join(DB_DIR, "horse_lookup.csv")
    jockey_path = os.path.join(DB_DIR, "jockey_lookup.csv")
    
    h_df = pd.read_csv(horse_path).set_index("horse") if os.path.exists(horse_path) else pd.DataFrame()
    j_df = pd.read_csv(jockey_path).set_index("jockey") if os.path.exists(jockey_path) else pd.DataFrame()
    
    return h_df, j_df

model = load_predictive_assets()
horse_lookup, jockey_lookup = load_competitor_records()

# =====================================================================
# 2. RUNTIME APPLICATION STATE SEEDING
# =====================================================================
if "matrix_ready" not in st.session_state:
    st.session_state.matrix_ready = False
if "matrix_df" not in st.session_state:
    st.session_state.matrix_df = pd.DataFrame()

# =====================================================================
# 3. INTERACTIVE CONTROL PANEL (TRACKSIDE ENVIRONMENTS)
# =====================================================================
st.title("🏇 Trackside Live Evaluation Framework")
st.sidebar.header("📋 Environment Specifications")

today_going = st.sidebar.selectbox("Current Track Surface Going", ["Good", "Firm", "Soft", "Heavy", "Yielding"])
race_rating_band = st.sidebar.text_input("Race Class / Rating Band", "Class 4 - 80-95")
race_ran = st.sidebar.number_input("Total Declared Field Size (Ran)", min_value=2, max_value=40, value=10, step=1)

# =====================================================================
# 4. ACTIVE ENTRANT COMPILATION REGISTRY
# =====================================================================
st.header("🎟️ Active Competitor Roster")
st.markdown("Populate the active race card dropdown selectors below to pull historical profiling.")

# Fallback structures for select values if files are unpopulated
horse_pool = sorted(horse_lookup.index.tolist()) if not horse_lookup.empty else ["Add Horse Manually"]
jockey_pool = sorted(jockey_lookup.index.tolist()) if not jockey_lookup.empty else ["Add Jockey Manually"]

runners_input_list = []
slots_to_generate = race_ran

col1, col2 = st.columns(2)
for idx in range(slots_to_generate):
    # Dynamically alternate placements between layout panels for dense optimization
    target_col = col1 if idx % 2 == 0 else col2
    with target_col:
        st.markdown(f"**Gate Position Slot #{idx + 1}**")
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            h_sel = st.selectbox(f"Select Horse #{idx+1}", [""] + horse_pool, key=f"h_slot_{idx}")
        with sub_c2:
            j_sel = st.selectbox(f"Select Jockey #{idx+1}", [""] + jockey_pool, key=f"j_slot_{idx}")
        runners_input_list.append({"horse": h_sel, "jockey": j_sel})

# =====================================================================
# 5. CODE BLOCK: FEATURE MATRIX COMPILATION ENGINE
# =====================================================================
st.markdown("---")
if st.button("🚀 Build Feature Matrix"):

    valid = [r for r in runners_input_list if r["horse"] and r["jockey"]]

    if len(valid) < 2:
        st.error("❗ Minimum Requirement Mismatch: Please fully assign at least two (2) complete Horse-Jockey pairs.")
        st.stop()

    rows = []

    for r in valid:
        h, j = r["horse"], r["jockey"]
        
        # Enforce structural creation of all baseline evaluation slots.
        # This guarantees user-input spaces materialize explicitly for modification.
        row = {
            "horse": h,
            "jockey": j,
            "rating_band": race_rating_band,
            "ran": race_ran,
            "age": np.nan,         
            "wgt": np.nan,         
            "or": np.nan,          
            "bookie_prob": np.nan, 
            "prev_avg_performance": np.nan,
            "horse_hot_streak": np.nan,
            "historical_going_performance": np.nan,
            "jockey_prev_avg_performance": np.nan,
            "jockey_hot_streak": np.nan
        }

        # Parse historical database indexes if matched
        if h in horse_lookup.index:
            hr = horse_lookup.loc[h]
            
            if "prev_avg_performance" in hr:
                row["prev_avg_performance"] = hr["prev_avg_performance"]
            if "horse_hot_streak" in hr:
                row["horse_hot_streak"] = hr["horse_hot_streak"]

            # Ground-state dynamic lookup fallbacks for track variance
            go_col = f"prev_{today_going.lower()}_performance"
            if go_col in hr and pd.notna(hr[go_col]):
                row["historical_going_performance"] = hr[go_col]
            elif "prev_avg_performance" in hr and pd.notna(hr["prev_avg_performance"]):
                row["historical_going_performance"] = hr["prev_avg_performance"]

        if j in jockey_lookup.index:
            jr = jockey_lookup.loc[j]
            
            if "jockey_prev_avg_performance" in jr:
                row["jockey_prev_avg_performance"] = jr["jockey_prev_avg_performance"]
            if "jockey_hot_streak" in jr:
                row["jockey_hot_streak"] = jr["jockey_hot_streak"]

        rows.append(row)

    st.session_state.matrix_df = pd.DataFrame(rows)
    st.session_state.matrix_ready = True
    st.success("🏁 Vector Matrix compilation finalized. Manual overrides unlocked below.")

# =====================================================================
# 6. CODE BLOCK: USER CONTEXT DISPLAY & RE-ENTRY INTERFACES
# =====================================================================
if st.session_state.matrix_ready:
    st.header("📊 Trackside Feature Matrix")
    st.info("Double-click empty white cells to manually supply Live Age, Carry Weight, OR, or Bookmaker Odds info.")

    # Divert widget output to a separate frame object to safeguard runtime baseline and prevent focus resets
    edited_matrix = st.data_editor(
        st.session_state.matrix_df,
        hide_index=True,
        use_container_width=True,
        disabled=[
            "horse", "jockey", "rating_band", "ran", 
            "prev_avg_performance", "horse_hot_streak", 
            "historical_going_performance", "jockey_prev_avg_performance", 
            "jockey_hot_streak"
        ],
        column_config={
            "wgt": "added_wgt(lbs)",
            "bookie_prob": "bookie_win"
        },
        key="main_race_matrix_editor"
    )

    # =====================================================================
    # 7. MODEL RUNTIME EVALUATION & INFERENCE PIPELINE
    # =====================================================================
    st.markdown("---")
    if st.button("🔮 Evaluate Competitor Field Ranks"):
        if model is None:
            st.error(f"❌ Missing Core Pipeline Dependency: No trained '{MODEL_PATH}' was detected on root storage.")
            st.stop()

        with st.spinner("Executing structural feature extraction and model inference..."):
            # Work downstream directly using our modified cache reference frame
            X = edited_matrix.copy()
            
            # Fetch properties matching what the model was trained on
            try:
                model_features = model.feature_name_
            except AttributeError:
                # Fallback variant parsing sequence for variant pipelines
                model_features = model.booster_.feature_name()

            # Align inputs perfectly to avoid missing or extra variable mismatches
            for col in model_features:
                if col not in X.columns:
                    X[col] = np.nan

            # Subset frame down exactly to target structures
            X = X[model_features]

            # Re-verify and format matching categorical dtypes for LightGBM
            for c in X.columns:
                if X[c].dtype == "object" or isinstance(X[c].dtype, pd.CategoricalDtype):
                    X[c] = X[c].astype("category")
                else:
                    # Enforce solid numerical boundaries over remaining fields
                    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

            # Compute raw margins using LightGBM core booster runtime structures
            raw_scores = model.booster_.predict(X)
            
            # Apply stable Softmax normalizations to raw outputs to yield probabilities
            exp_scores = np.exp(raw_scores - np.max(raw_scores))
            probabilities = exp_scores / np.sum(exp_scores)

            # Map array solutions directly back into context visualization frames
            output_df = pd.DataFrame({
                "Competitor (Horse)": edited_matrix["horse"],
                "Assigned Jockey": edited_matrix["jockey"],
                "Model Structural Probability": [f"{p*100:.2f}%" for p in probabilities],
                "Raw Score Metric": raw_scores
            }).sort_values(by="Raw Score Metric", ascending=False).reset_index(drop=True)

            # Render final analytics tracking frame back to client
            st.header("🏆 Calculated Performance Output Metrics")
            st.dataframe(output_df, use_container_width=True)
            st.balloons()
