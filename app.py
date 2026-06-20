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
# 3. SIDEBAR CONFIG
# =====================================================================
st.sidebar.header("⚙️ Race Setup")

today_going = st.sidebar.selectbox("Going", goings)
race_ran = st.sidebar.number_input("Total Runners", value=8, step=1)
race_rating_band = st.sidebar.selectbox("Rating Band", rating_options)

# =====================================================================
# 4. SESSION STATE
# =====================================================================
if "runners" not in st.session_state:
    st.session_state.runners = [{"horse": "", "jockey": ""}]

if "matrix_ready" not in st.session_state:
    st.session_state.matrix_ready = False

if "matrix_df" not in st.session_state:
    st.session_state.matrix_df = None

# =====================================================================
# 5. INPUT SECTION (NO PROCESSING)
# =====================================================================
st.header("📋 Enter Runners")

c1, c2 = st.columns(2)

with c1:
    if st.button("➕ Add Runner"):
        st.session_state.runners.append({"horse": "", "jockey": ""})

with c2:
    if st.button("➖ Remove Runner"):
        if len(st.session_state.runners) > 1:
            st.session_state.runners.pop()

for i, r in enumerate(st.session_state.runners):
    col1, col2 = st.columns(2)

    with col1:
        r["horse"] = st.selectbox(
            "Horse",
            [""] + all_horses,
            key=f"h_{i}"
        ).strip()

    with col2:
        r["jockey"] = st.selectbox(
            "Jockey",
            [""] + all_jockeys,
            key=f"j_{i}"
        ).strip()

# =====================================================================
# 6. BUILD MATRIX (ONLY ON CLICK)
# =====================================================================
if st.button("🚀 Build Feature Matrix"):

    valid = [r for r in st.session_state.runners if r["horse"] and r["jockey"]]

    if len(valid) < 2:
        st.error("Need at least 2 valid runners")
        st.stop()

    rows = []

    for r in valid:
        h, j = r["horse"], r["jockey"]
        row = {"horse": h, "jockey": j}

        # horse features
        if h in horse_lookup.index:
            hr = horse_lookup.loc[h]

            for col in ["age", "wgt", "or", "prev_avg_performance", "horse_hot_streak"]:
                if col in hr:
                    row[col] = hr[col]

            go_col = f"prev_{today_going}_performance"
            row["historical_going_performance"] = (
                hr.get(go_col, hr.get("prev_avg_performance", np.nan))
            )

        # jockey features
        if j in jockey_lookup.index:
            jr = jockey_lookup.loc[j]

            for col in ["jockey_prev_avg_performance", "jockey_hot_streak"]:
                if col in jr:
                    row[col] = jr[col]

        # globals
        row["rating_band"] = race_rating_band
        row["ran"] = race_ran
        row["bookie_prob"] = np.nan

        rows.append(row)

    st.session_state.matrix_df = pd.DataFrame(rows)
    st.session_state.matrix_ready = True

    st.success("Feature matrix built")

# =====================================================================
# 7. MATRIX DISPLAY
# =====================================================================
if st.session_state.matrix_ready:

    st.header("📊 Feature Matrix")

    st.session_state.matrix_df = st.data_editor(
        st.session_state.matrix_df,
        use_container_width=True
    )

    # =================================================================
    # 8. MODEL EVALUATION (YOUR ODDS LOGIC KEPT)
    # =================================================================
    if st.button("🔮 Evaluate Competitor Field Ranks"):

        if model is None:
            st.error("Model not loaded")
            st.stop()

        df = st.session_state.matrix_df.copy()

        features = model.feature_name_

        for c in features:
            if c not in df.columns:
                df[c] = 0.5

        X = df[features].copy()

        for c in X.columns:
            if X[c].dtype == "object":
                X[c] = X[c].astype("category")
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.5)

        # ============================
        # YOUR MODEL LOGIC (UNCHANGED)
        # ============================
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

else:
    st.info("Build feature matrix first.")

# =====================================================================
# 9. BOOKIE ODDS CONVERTER (RESTORED)
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
