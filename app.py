from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


DATA_PATH = Path(__file__).resolve().parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"
DROP_COLUMNS = ["customerID"]

st.set_page_config(
    page_title="Churn Club · Customer Retention",
    page_icon="🪩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #181728;
        --muted: #777487;
        --paper: #f7f6f2;
        --lilac: #d8c8ff;
        --acid: #dbff71;
        --pink: #ff9cb7;
        --line: rgba(24, 23, 40, .12);
    }
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background:
            radial-gradient(ellipse at 84% 4%, rgba(216,200,255,.48), transparent 27rem),
            radial-gradient(ellipse at 4% 25%, rgba(219,255,113,.24), transparent 24rem),
            var(--paper);
    }
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"],
    footer { display: none !important; }
    .block-container { max-width: 1320px; padding: 2rem 3rem 3rem; }
    .topline {
        display:flex; align-items:center; justify-content:space-between;
        padding: .15rem 0 1.35rem; border-bottom:1px solid var(--line);
        font: 500 .7rem 'DM Mono', monospace; text-transform:uppercase;
        letter-spacing:.12em; color:var(--muted);
    }
    .brand { display:flex; align-items:center; gap:.65rem; color:var(--ink); }
    .brand-mark {
        display:grid; place-items:center; width:2rem; height:2rem;
        border-radius:50%; background:var(--acid); font-size:1.05rem;
    }
    .live-dot {
        display:inline-block; width:.48rem; height:.48rem; border-radius:50%;
        background:#78be75; margin-right:.45rem; box-shadow:0 0 0 4px #78be7522;
        animation:pulse 1.8s infinite;
    }
    .hero {
        display:grid; grid-template-columns:1fr auto; align-items:center;
        gap:2rem; padding:3.35rem 0 2rem;
    }
    .eyebrow {
        font:500 .72rem 'DM Mono',monospace; letter-spacing:.14em;
        text-transform:uppercase; color:#7564a5; margin-bottom:.9rem;
    }
    .hero h1 {
        font-size:clamp(3.5rem,7.5vw,6.8rem); line-height:.88;
        letter-spacing:-.085em; margin:0; font-weight:700; color:var(--ink);
    }
    .hero h1 span { color:#8c71d7; }
    .hero p {
        font-size:1.02rem; line-height:1.65; color:var(--muted);
        max-width:39rem; margin:1.2rem 0 0;
    }
    .mascot {
        width:9.3rem; height:9.3rem; display:grid; place-items:center;
        border:1px solid #18172820; border-radius:2.5rem;
        background:linear-gradient(145deg,#e5d9ff,#d3c0ff);
        font-size:5rem; transform:rotate(7deg);
        box-shadow:10px 12px 0 #181728;
        animation:floaty 3.2s ease-in-out infinite;
    }
    .ticker {
        overflow:hidden; white-space:nowrap; border-block:1px solid var(--line);
        padding:.75rem 0; color:var(--muted);
        font:500 .66rem 'DM Mono',monospace; letter-spacing:.12em;
        text-transform:uppercase;
    }
    .ticker span { display:inline-block; animation:marquee 27s linear infinite; }
    .ticker b { color:#8c71d7; padding:0 1.4rem; }
    .stats { margin:1.5rem 0 2rem; }
    [data-testid="stMetric"] {
        padding:1rem 1.15rem; background:#fff; border:1px solid var(--line);
        border-radius:1rem; box-shadow:0 4px 16px #18172808;
    }
    [data-testid="stMetricLabel"] {
        font:500 .65rem 'DM Mono',monospace; text-transform:uppercase;
        letter-spacing:.08em; color:var(--muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-size:1.65rem; font-weight:700; letter-spacing:-.05em;
        color:var(--ink) !important;
    }
    .section-heading {
        font-size:1.55rem; font-weight:700; letter-spacing:-.055em;
        margin:0 0 .25rem; color:var(--ink);
    }
    .section-note { color:var(--muted); font-size:.9rem; margin-bottom:1rem; }
    [data-testid="stForm"] {
        background:#fff; border:1px solid var(--line); border-radius:1.2rem;
        padding:1.35rem 1.4rem .7rem; box-shadow:0 8px 28px #18172808;
    }
    [data-testid="stWidgetLabel"] p {
        font-size:.78rem; font-weight:600; color:#454256;
    }
    div[data-baseweb="select"] > div, .stNumberInput input {
        border-radius:.7rem !important; background:#fbfaf8 !important;
    }
    .stSlider [data-baseweb="slider"] { padding-top:.3rem; }
    .stButton button, [data-testid="stFormSubmitButton"] button {
        border:0; border-radius:.75rem; background:var(--ink); color:#fff;
        font-weight:700; min-height:3rem; transition:transform .18s, box-shadow .18s;
    }
    .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
        color:#fff; transform:translateY(-2px); box-shadow:0 7px 0 var(--lilac);
    }
    .result-card {
        position:relative; overflow:hidden; min-height:26rem;
        padding:1.6rem; border-radius:1.2rem; background:var(--ink); color:#fff;
        box-shadow:0 8px 28px #18172820;
    }
    .result-card:after {
        content:""; position:absolute; width:15rem; height:15rem; border-radius:50%;
        right:-5rem; top:-7rem; background:#d8c8ff22; filter:blur(2px);
    }
    .result-kicker {
        font:500 .68rem 'DM Mono',monospace; color:#cbb9ff;
        letter-spacing:.12em; text-transform:uppercase;
    }
    .result-score {
        position:relative; z-index:1; margin:.65rem 0 0;
        font-size:clamp(3.8rem,6vw,5.8rem); font-weight:700;
        letter-spacing:-.09em; line-height:1;
    }
    .result-score small { font-size:1.3rem; letter-spacing:0; color:#bcb8ca; }
    .meter {
        height:.55rem; overflow:hidden; border-radius:2rem;
        background:#ffffff25; margin:1.25rem 0 1.4rem;
    }
    .meter span {
        display:block; height:100%; border-radius:2rem;
        background:linear-gradient(90deg,var(--acid),var(--pink));
        animation:fillin .9s ease-out both;
    }
    .risk-pill {
        display:inline-flex; align-items:center; gap:.4rem;
        border-radius:2rem; padding:.45rem .75rem;
        color:var(--ink); background:var(--acid);
        font:500 .68rem 'DM Mono',monospace; text-transform:uppercase;
    }
    .result-copy { color:#c5c2d0; font-size:.91rem; line-height:1.6; margin-top:1rem; }
    .empty-planet {
        display:grid; place-items:center; height:7rem;
        border-radius:1rem; background:#ffffff0d; font-size:3.5rem;
        animation:floaty 3.2s ease-in-out infinite;
    }
    .empty-title { font-size:1.35rem; font-weight:700; margin:1.2rem 0 .3rem; }
    .feature-card {
        background:#fff; border:1px solid var(--line); border-radius:1.2rem;
        padding:1.4rem 1.5rem; margin-top:1.5rem;
    }
    .feature-row {
        display:grid; grid-template-columns:2rem minmax(9rem,1fr) 3fr 3rem;
        align-items:center; gap:.8rem; margin:.85rem 0;
        font-size:.82rem; color:#454256;
    }
    .feature-rank { color:#aaa6b6; font:500 .68rem 'DM Mono',monospace; }
    .feature-track { height:.42rem; border-radius:2rem; background:#f0eef5; overflow:hidden; }
    .feature-track span {
        display:block; height:100%; border-radius:2rem;
        background:linear-gradient(90deg,#d8c8ff,#9c7be9);
    }
    .feature-score { text-align:right; font:500 .68rem 'DM Mono',monospace; color:var(--muted); }
    .fineprint {
        margin-top:1.5rem; color:var(--muted); font-size:.74rem;
        line-height:1.6; border-top:1px solid var(--line); padding-top:1rem;
    }
    @keyframes floaty { 0%,100%{transform:translateY(0) rotate(7deg)} 50%{transform:translateY(-9px) rotate(3deg)} }
    @keyframes pulse { 50%{box-shadow:0 0 0 7px #78be7500} }
    @keyframes marquee { to{transform:translateX(-50%)} }
    @keyframes fillin { from{width:0} }
    @media(max-width:700px) {
        .block-container { padding:1rem 1rem 2rem; }
        .hero { padding:2.5rem 0 1.4rem; }
        .mascot { width:5rem; height:5rem; border-radius:1.5rem; font-size:2.8rem; }
        .hero p { font-size:.9rem; }
        .feature-row { grid-template-columns:1.5rem minmax(6rem,1fr) 2fr 2.5rem; gap:.5rem; }
    }
    @media(prefers-reduced-motion:reduce) {
        *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def train_model(data_path: str, modified_at: float):
    del modified_at
    frame = pd.read_csv(data_path)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame = frame.dropna(subset=["TotalCharges"]).copy()
    frame[TARGET] = frame[TARGET].map({"No": 0, "Yes": 1})
    frame = frame.drop(columns=DROP_COLUMNS)

    features = frame.drop(columns=[TARGET])
    target = frame[TARGET]
    categorical_columns = [
        column
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    encoders = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        features[column] = encoder.fit_transform(features[column])
        encoders[column] = encoder

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    x_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(
        x_train, y_train
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train_resampled, y_train_resampled)
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    return model, encoders, features.columns.tolist(), auc, frame


if not DATA_PATH.is_file():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

model, encoders, feature_names, auc, training_data = train_model(
    str(DATA_PATH), DATA_PATH.stat().st_mtime
)

st.markdown(
    """
    <div class="topline">
      <div class="brand"><span class="brand-mark">🪩</span> CHURN CLUB / CUSTOMER RETENTION LAB</div>
      <div><span class="live-dot"></span> MODEL ONLINE · VIBES MONITORED</div>
    </div>
    <div class="hero">
      <div>
        <div class="eyebrow">Your customers called. They want better commitment.</div>
        <h1>Keep the good<br>ones <span>close.</span></h1>
        <p>A tiny crystal ball for customer loyalty. Enter a few details, get a churn-risk read, and maybe save someone from a dramatic exit.</p>
      </div>
      <div class="mascot" aria-label="A disco ball mascot">🪩</div>
    </div>
    <div class="ticker"><span>
      BREAKING: CUSTOMER THINKS ABOUT LEAVING <b>✳</b>
      WE HAVE A SPREADSHEET AND A DREAM <b>✳</b>
      NO CUSTOMERS WERE GUILT-TRIPPED IN THIS PREDICTION <b>✳</b>
      BREAKING: CUSTOMER THINKS ABOUT LEAVING <b>✳</b>
      WE HAVE A SPREADSHEET AND A DREAM <b>✳</b>
      NO CUSTOMERS WERE GUILT-TRIPPED IN THIS PREDICTION <b>✳</b>
    </span></div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(3)
metric_columns[0].metric("Customers in the dataset", f"{len(training_data):,}")
metric_columns[1].metric("Current heartbreak rate", f"{training_data[TARGET].mean():.1%}")
metric_columns[2].metric("Model's report card · AUC", f"{auc:.3f}")

predictor_column, result_column = st.columns([1.05, 0.95], gap="large")

with predictor_column:
    st.markdown(
        '<div class="section-heading">Build your customer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-note">The details are real. The tiny disco ball is emotionally supportive.</div>',
        unsafe_allow_html=True,
    )
    with st.form("customer_details"):
        customer = {}
        labels = {
            "gender": "Gender",
            "SeniorCitizen": "Senior citizen",
            "Partner": "Has a partner",
            "Dependents": "Has dependents",
            "tenure": "Tenure (months)",
            "PhoneService": "Phone service",
            "MultipleLines": "Multiple lines",
            "InternetService": "Internet service",
            "OnlineSecurity": "Online security",
            "OnlineBackup": "Online backup",
            "DeviceProtection": "Device protection",
            "TechSupport": "Tech support",
            "StreamingTV": "Streaming TV",
            "StreamingMovies": "Streaming movies",
            "Contract": "Contract type",
            "PaperlessBilling": "Paperless billing",
            "PaymentMethod": "Payment method",
            "MonthlyCharges": "Monthly charges ($)",
            "TotalCharges": "Total charges ($)",
        }
        preferred_values = {
            "gender": "Female",
            "Partner": "No",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
        }
        left_inputs, right_inputs = st.columns(2)
        for position, column in enumerate(feature_names):
            parent = left_inputs if position % 2 == 0 else right_inputs
            with parent:
                label = labels.get(column, column)
                if column in encoders:
                    classes = encoders[column].classes_.tolist()
                    default = preferred_values.get(column)
                    if default not in classes:
                        default = classes[0]
                    customer[column] = st.selectbox(
                        label,
                        classes,
                        index=classes.index(default),
                    )
                elif column == "SeniorCitizen":
                    customer[column] = st.selectbox(
                        label,
                        [0, 1],
                        format_func=lambda value: "Yes" if value else "No",
                    )
                elif column == "tenure":
                    customer[column] = st.slider(label, 0, 72, 12)
                elif column == "MonthlyCharges":
                    customer[column] = st.number_input(
                        label,
                        min_value=0.0,
                        max_value=200.0,
                        value=65.0,
                        step=1.0,
                    )
                elif column == "TotalCharges":
                    customer[column] = st.number_input(
                        label,
                        min_value=0.0,
                        max_value=10000.0,
                        value=780.0,
                        step=10.0,
                    )

        submitted = st.form_submit_button(
            "🪩  Scan for impending plot twists",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_frame = pd.DataFrame([customer], columns=feature_names)
        for column, encoder in encoders.items():
            input_frame[column] = encoder.transform(input_frame[column])
        st.session_state["prediction"] = float(
            model.predict_proba(input_frame)[0, 1]
        )

with result_column:
    st.markdown(
        '<div class="section-heading">The loyalty forecast</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-note">Crystal ball sold separately. Model included.</div>',
        unsafe_allow_html=True,
    )
    churn_probability = st.session_state.get("prediction")
    if churn_probability is None:
        st.markdown(
            """
            <div class="result-card">
              <div class="result-kicker">LIVE FROM THE CUSTOMER-VERSE</div>
              <div class="empty-planet">🛸</div>
              <div class="empty-title">No one's ghosted us. Yet.</div>
              <div class="result-copy">Fill in the customer details and run a scan.
              Our highly scientific disco ball is standing by.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        high_risk = churn_probability >= 0.5
        status = "Might be a runner" if high_risk else "Seems pretty loyal"
        status_icon = "🚨" if high_risk else "💚"
        copy = (
            "This one may be browsing the exit. Consider a friendly check-in "
            "(and perhaps don't open with 'we need to talk')."
            if high_risk
            else "Looking good! Keep delivering the good stuff. No grand gestures necessary."
        )
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-kicker">THE LOYALTY FORECAST IS IN</div>
              <div class="result-score">{churn_probability:.1%}<small> chance of churn</small></div>
              <div class="meter"><span style="width:{churn_probability:.1%}"></span></div>
              <div class="risk-pill">{status_icon} &nbsp;{escape(status)}</div>
              <div class="result-copy">{escape(copy)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

importance = pd.DataFrame(
    {"Feature": feature_names, "Importance": model.feature_importances_}
).sort_values("Importance", ascending=False)
top_features = importance.head(8).reset_index(drop=True)
max_importance = float(top_features["Importance"].max())

st.markdown(
    '<div class="feature-card"><div class="section-heading">What makes the model side-eye?</div>'
    '<div class="section-note">Top signals it learned from historical customer behavior.</div>',
    unsafe_allow_html=True,
)
for rank, row in top_features.iterrows():
    percent = float(row["Importance"]) / max_importance * 100 if max_importance else 0
    st.markdown(
        f"""
        <div class="feature-row">
          <div class="feature-rank">0{rank + 1}</div>
          <div>{escape(str(row['Feature']))}</div>
          <div class="feature-track"><span style="width:{percent:.1f}%"></span></div>
          <div class="feature-score">{float(row['Importance']):.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="fineprint">
      A prediction is a nudge to investigate, not a crystal-ball guarantee.
      This is a churn-risk model, not a real-world offer-response predictor;
      its sibling uplift notebook uses simulated treatment data.
      Made with data, care, and a suspicious number of disco balls.
    </div>
    """,
    unsafe_allow_html=True,
)
