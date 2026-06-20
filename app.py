import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Ascot Multi-Horse Ranker", layout="wide")

# =====================================================================
# 1. LOAD ASSETS
# =====================================================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("lgbm_ranker_model.pkl")

        part1 = pd.read_csv("historical_lookup_part1.zip")
        part2 = pd.read_csv("historical_lookup_part2.zip")
        part3 = pd.read_csv("historical_lookup_part3.zip")

        lookup = pd.concat([part1, part2, part3], ignore_index=True)

        return model, lookup

    except Exception as e:
        st.error(f"Asset load failed: {e}")
        return None, None


model, lookup = load_assets()

# =====================================================================
# 2. SAFE LOOKUP PREP
# =====================================================================
if lookup is not None:

    lookup["horse"] = lookup["horse"].astype(str).str.strip()
    lookup["jockey"] = lookup["jockey"].astype(str).str.strip()

    all_horses = sorted(lookup["horse"].dropna().unique()) if "horse" in lookup.columns else []
    all_jockeys = sorted(lookup["jockey"].dropna().unique()) if "jockey" in lookup.columns else []

    horse_lookup = lookup.drop_duplicates("horse").set_index("horse")
    jockey_lookup = lookup.drop_duplicates("jockey").set_index("jockey")

    going_cols = [c for c in lookup.columns if c.startswith("prev_") and c.endswith("_performance")]
    goings = [c.replace("prev_", "").replace("_performance", "") for c in going_cols]

    if not goings:
        goings = ["Good", "Good to Firm", "Good to Soft", "Soft", "Heavy"]

    if "rating_band" in lookup.columns:
        rating_options = [""] + sorted(lookup["rating_band"].dropna().astype(str).unique())
    else:
        rating_options = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]

else:
    all_horses, all_jockeys = [], []
    horse_lookup = pd.DataFrame()
    jockey_lookup = pd.DataFrame()
    goings = ["Good", "Good to Firm", "Good to Soft", "Soft", "Heavy"]
    rating_options = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]

# =====================================================================
# 3. SIDEBAR
# =====================================================================
st.sidebar.header("⚙️ Race Setup")

today_going = st.sidebar.selectbox("Going", goings)
race_ran = st.sidebar.number_input("Total Runners", value=8, step=1)
race_rating_band = st.sidebar.selectbox("Rating Band", rating_options)

# =====================================================================
# 4. SESSION STATE
# =====================================================================
if "num_runners" not in st.session_state:
    st.session_state.num_runners = 3

if "matrix_ready" not in st.session_state:
    st.session_state.matrix_ready = False

if "matrix_df" not in st.session_state:
    st.session_state.matrix_df = None

# =====================================================================
# 5. RUNNER SELECTOR (FIXED KEY LIFECYCLE FOR MOBILE)
# =====================================================================
st.header("📋 Field Entry Profile Selection")

col_add, col_rem, _ = st.columns([1, 1, 4])

with col_add:
    if st.button("➕ Add Runner Position"):
        st.session_state.num_runners += 1

with col_rem:
    if st.button("❌ Remove Last Position") and st.session_state.num_runners > 2:
        st.session_state.num_runners -= 1

# Precompute lowercase lookup lists for speed
all_horses_lc = [(h, h.lower()) for h in all_horses]
all_jockeys_lc = [(j, j.lower()) for j in all_jockeys]

runners_input_list = []

for i in range(st.session_state.num_runners):

    st.markdown(f"**Runner Position #{i+1}**")
    r_col1, r_col2 = st.columns(2)

    # ============================
    # HORSE SEARCH
    # ============================
    with r_col1:
        h_search = st.text_input("Type Horse Name to Filter", key=f"horse_search_{i}").strip().lower()
        
        # Keep structural setup stable: update options dynamically without changing widget paths
        if len(h_search) >= 2:
            filtered_horses = [h for h, hl in all_horses_lc if h_search in hl][:20]
        else:
            filtered_horses = []
            
        h_name = st.selectbox("Select Horse Match", [""] + filtered_horses, key=f"horse_input_{i}")

    # ============================
    # JOCKEY SEARCH
    # ============================
    with r_col2:
        j_search = st.text_input("Type Jockey Name to Filter", key=f"jockey_search_{i}").strip().lower()
        
        if len(j_search) >= 2:
            filtered_jockeys = [j for j, jl in all_jockeys_lc if j_search in jl][:20]
        else:
            filtered_jockeys = []
            
        j_name = st.selectbox("Select Jockey Match", [""] + filtered_jockeys, key=f"jockey_input_{i}")

    runners_input_list.append({"horse": h_name, "jockey": j_name})

# =====================================================================
# 6. BUILD MATRIX (GUARANTEED COLUMNS FOR MANUAL INPUT)
# =====================================================================
if st.button("🚀 Build Feature Matrix"):

    valid = [r for r in runners_input_list if r["horse"] and r["jockey"]]

    if len(valid) < 2:
        st.error("Need at least 2 valid runners fully selected from dropdowns.")
        st.stop()

    rows = []

    for r in valid:
        h, j = r["horse"], r["jockey"]
        
        # ⚡ FIX: Explicitly pre-define every single column header so they 
        # are guaranteed to appear in a clean, predictable trackside order.
        row = {
            "horse": h,
            "jockey": j,
            "rating_band": race_rating_band,
            "ran": race_ran,
            "age": np.nan,         # Blank slot waiting for your manual input
            "wgt": np.nan,         # Blank slot waiting for your manual input
            "or": np.nan,          # Blank slot waiting for your manual input
            "bookie_prob": np.nan, # Blank slot waiting for your manual input
            "prev_avg_performance": np.nan,
            "horse_hot_streak": np.nan,
            "historical_going_performance": np.nan,
            "jockey_prev_avg_performance": np.nan,
            "jockey_hot_streak": np.nan
        }

        # Pull historical values from database if the competitor exists
        if h in horse_lookup.index:
            hr = horse_lookup.loc[h]
            
            if "prev_avg_performance" in hr:
                row["prev_avg_performance"] = hr["prev_avg_performance"]
            if "horse_hot_streak" in hr:
                row["horse_hot_streak"] = hr["horse_hot_streak"]

            # Robust going lookup fallback logic
            go_col = f"prev_{today_going}_performance"
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
    st.success("Feature matrix built successfully.")

# =====================================================================
# 7. MATRIX DISPLAY
# =====================================================================
if st.session_state.matrix_ready:

    st.header("📊 Feature Matrix")

    # ⚡ FIXED: Assign data output to edited_df to prevent data drop-out on first keypress
    edited_df = st.data_editor(
        st.session_state.matrix_df,
        use_container_width=True,
        disabled=["horse", "jockey", "rating_band", "ran"],
        column_config={
            "wgt": "carry_wgt(lbs)",
            "bookie_prob": "bookie_win"
        },
        key="main_race_matrix_editor"
    )

    # =================================================================
    # 8. MODEL EVALUATION
    # =================================================================
    if st.button("🔮 Evaluate Competitor Field Ranks"):

        if model is None:
            st.error("Model not loaded")
            st.stop()

        df = edited_df.copy()
        features = model.feature_name_

        for c in features:
            if c not in df.columns:
                df[c] = 0.5

        X = df[features].copy()

        # Get the exact list of categorical columns LightGBM expects
        expected_cats = getattr(model, "pandas_categorical", [])
        if expected_cats is None:
            expected_cats = []

        for c in X.columns:
            if c in expected_cats:
                # Force it to category only if LightGBM expects it
                X[c] = X[c].astype("category")
            else:
                # Everything else must be strictly numerical
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.5)

        raw = model.booster_.predict(X)

        exp_vals = np.exp(raw - np.max(raw))
        win_probs = exp_vals / np.sum(exp_vals)

        placing_scale_factor = min(3.0, len(df))
        top3_probs = np.clip(win_probs * placing_scale_factor, 0.0, 0.99)

        df["Win Probability"] = [f"{p*100:.1f}%" for p in win_probs]
        df["Top 3 Probability"] = [f"{p*100:.1f}%" for p in top3_probs]

        df = df.assign(_p=win_probs).sort_values("_p", ascending=False)
        df["Rank"] = range(1, len(df) + 1)
        df = df.drop(columns=["_p"])

        st.subheader("🏁 Leaderboard")
        st.dataframe(df, use_container_width=True)

# =====================================================================
# 9. BOOKIE ODDS CONVERTER
# =====================================================================
st.markdown("---")
st.header("🧮 Bookie Fraction Converter")

st.write("Convert fractional odds into implied probability for bookie_prob feature.")

col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    num = st.number_input("Numerator", value=4, min_value=1, step=1)

with col2:
    st.markdown("<h3 style='text-align:center;'>/</h3>", unsafe_allow_html=True)

with col3:
    den = st.number_input("Denominator", value=1, min_value=1, step=1)

implied = den / (num + den)

st.metric("Implied Probability", f"{implied:.4f}")
st.caption(f"{implied*100:.2f}%")
