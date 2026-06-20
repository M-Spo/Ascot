import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Ascot Multi-Horse Ranker", layout="wide")

# =====================================================================
# 1. LOAD ASSETS (CACHE)
# =====================================================================
@st.cache_resource
def load_production_assets():
    try:
        model = joblib.load('lgbm_ranker_model.pkl')

        part1 = pd.read_csv('historical_lookup_part1.zip')
        part2 = pd.read_csv('historical_lookup_part2.zip')
        part3 = pd.read_csv('historical_lookup_part3.zip')

        lookup = pd.concat([part1, part2, part3], ignore_index=True)
        return model, lookup
    except Exception as e:
        st.error(f"Asset load error: {e}")
        return None, None


model, lookup = load_production_assets()

# =====================================================================
# PREPROCESS LOOKUPS (RUN ONCE)
# =====================================================================
if lookup is not None:
    lookup['horse'] = lookup['horse'].astype(str).str.strip()
    lookup['jockey'] = lookup['jockey'].astype(str).str.strip()

    all_horses = sorted(lookup['horse'].dropna().unique())
    all_jockeys = sorted(lookup['jockey'].dropna().unique())

    horse_fast_lookup = lookup.drop_duplicates('horse').set_index('horse')
    jockey_fast_lookup = lookup.drop_duplicates('jockey').set_index('jockey')

    going_cols = [c for c in lookup.columns if c.startswith('prev_') and c.endswith('_performance')]
    unique_goings_list = [c.replace('prev_', '').replace('_performance', '') for c in going_cols]

    if not unique_goings_list:
        unique_goings_list = ["Good", "Good to Firm", "Good to Soft", "Soft", "Heavy"]
else:
    all_horses, all_jockeys = [], []
    horse_fast_lookup = pd.DataFrame()
    jockey_fast_lookup = pd.DataFrame()
    unique_goings_list = ["Good", "Good to Firm", "Good to Soft", "Soft", "Heavy"]

# =====================================================================
# SIDEBAR CONFIG
# =====================================================================
st.sidebar.header("⚙️ Race Config")

today_going = st.sidebar.selectbox("Going", unique_goings_list)
race_ran = st.sidebar.number_input("Runners", value=8, step=1)

# rating band optional
race_rating_band = st.sidebar.selectbox("Rating Band", [""] + sorted(lookup['rating_band'].dropna().astype(str).unique()) if lookup is not None else [""])

# =====================================================================
# SESSION STATE INIT
# =====================================================================
if "runners" not in st.session_state:
    st.session_state.runners = [{"horse": "", "jockey": ""}]

if "matrix_ready" not in st.session_state:
    st.session_state.matrix_ready = False

if "matrix_df" not in st.session_state:
    st.session_state.matrix_df = None

# =====================================================================
# RUNNER INPUT (LIGHTWEIGHT ONLY)
# =====================================================================
st.header("📋 Enter Runners (No Processing Yet)")

col_add, col_rem = st.columns(2)

with col_add:
    if st.button("➕ Add Runner"):
        st.session_state.runners.append({"horse": "", "jockey": ""})

with col_rem:
    if st.button("➖ Remove Runner") and len(st.session_state.runners) > 1:
        st.session_state.runners.pop()

# input UI only
for i, r in enumerate(st.session_state.runners):
    c1, c2 = st.columns(2)

    with c1:
        r["horse"] = st.selectbox(
            "Horse",
            [""] + all_horses,
            key=f"h_{i}"
        ).strip()

    with c2:
        r["jockey"] = st.selectbox(
            "Jockey",
            [""] + all_jockeys,
            key=f"j_{i}"
        ).strip()

# =====================================================================
# STEP 1 BUTTON → BUILD MATRIX ONLY
# =====================================================================
if st.button("🚀 Build Feature Matrix"):
    valid = [r for r in st.session_state.runners if r["horse"] and r["jockey"]]

    if len(valid) < 2:
        st.error("Need at least 2 valid runners")
    else:
        rows = []

        for runner in valid:
            h, j = runner["horse"], runner["jockey"]
            row = {"horse": h, "jockey": j}

            if h in horse_fast_lookup.index:
                hrow = horse_fast_lookup.loc[h]

                for col in ['age', 'wgt', 'or', 'prev_avg_performance', 'horse_hot_streak']:
                    if col in hrow:
                        row[col] = hrow[col]

                go_col = f"prev_{today_going}_performance"
                row["historical_going_performance"] = (
                    hrow.get(go_col, hrow.get("prev_avg_performance", np.nan))
                )

            if j in jockey_fast_lookup.index:
                jrow = jockey_fast_lookup.loc[j]
                for col in ['jockey_prev_avg_performance', 'jockey_hot_streak']:
                    if col in jrow:
                        row[col] = jrow[col]

            row["rating_band"] = race_rating_band
            row["ran"] = race_ran
            row["bookie_prob"] = np.nan

            rows.append(row)

        df = pd.DataFrame(rows)
        st.session_state.matrix_df = df
        st.session_state.matrix_ready = True

        st.success("Matrix built successfully")

# =====================================================================
# SHOW MATRIX ONLY AFTER BUILD
# =====================================================================
if st.session_state.matrix_ready:

    st.header("📊 Feature Matrix")

    edited = st.data_editor(
        st.session_state.matrix_df,
        use_container_width=True
    )

    st.session_state.matrix_df = edited

    # =================================================================
    # STEP 2 BUTTON → MODEL RUN
    # =================================================================
    if st.button("🔮 Evaluate Field"):

        if model is None:
            st.error("Model missing")
        else:
            df = st.session_state.matrix_df.copy()

            model_features = model.feature_name_

            for c in model_features:
                if c not in df.columns:
                    df[c] = 0.5

            X = df[model_features].copy()

            for c in X.columns:
                if X[c].dtype == "object":
                    X[c] = X[c].astype("category")
                else:
                    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.5)

            preds = model.booster_.predict(X)

            probs = np.exp(preds - np.max(preds))
            probs = probs / np.sum(probs)

            df["Win %"] = [f"{p*100:.1f}%" for p in probs]

            df = df.sort_values(probs, ascending=False)

            df["Rank"] = range(1, len(df) + 1)

            st.subheader("🏁 Leaderboard")
            st.dataframe(df)

else:
    st.info("Build matrix first before running model.")
