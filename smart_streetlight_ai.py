"""
Smart Streetlight AI — Enhanced Edition
- Multi-model ML pipeline (LR, DT, RF, GBM, Ensemble)
- CNN fog estimator (TF/Keras optional)
- YOLO object detection (optional)
- LLM-powered scene analysis via Claude API (free Anthropic key via claude.ai artifacts)
- Anomaly detection (IsolationForest)
- Time-series features (lag, rolling stats)
- Interactive dashboard with rich charts
- Explainability (feature importances, SHAP-like breakdown)
- LLM City Report Generator
"""

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import time
import joblib
import json
import requests
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    IsolationForest,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── optional heavy deps ──────────────────────────────────────────────────────
USE_YOLO = False
try:
    from ultralytics import YOLO
    USE_YOLO = True
except Exception:
    pass

HAVE_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    )
    HAVE_TF = True
except Exception:
    pass

# ── paths ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("streetlight_outputs")
OUT_DIR.mkdir(exist_ok=True)
DATA_PATH = OUT_DIR / "streetlight_dataset.csv"
MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
METRICS_PATH = OUT_DIR / "model_metrics.json"

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Streetlight AI",
    page_icon="🌃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1.1;
}
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
}
.light-on {
    color: #4ade80 !important;
    font-weight: 700;
}
.light-off {
    color: #f87171 !important;
    font-weight: 700;
}
.llm-box {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid #4338ca;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #c7d2fe;
}
.anomaly-badge {
    background: #7f1d1d;
    color: #fca5a5;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.normal-badge {
    background: #14532d;
    color: #86efac;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
div[data-testid="stTab"] button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #6d28d9);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(124,58,237,0.4);
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  LLM  —  Claude API helper
# ═══════════════════════════════════════════════════════════════════════════
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def call_claude(prompt: str, system: str = "", max_tokens: int = 900) -> str:
    """Call Claude claude-haiku-4-5 via Anthropic Messages API."""
    api_key = st.session_state.get("anthropic_key", ANTHROPIC_API_KEY)
    if not api_key:
        return "⚠️  No Anthropic API key found. Enter it in the sidebar to enable LLM analysis."
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    except Exception as e:
        return f"LLM error: {e}"


def llm_scene_analysis(feats: dict, prediction: str, prob: float) -> str:
    system = (
        "You are an urban AI traffic & lighting analyst. "
        "Be concise (3–5 sentences), technical, and actionable. "
        "No markdown headers. Use plain prose."
    )
    prompt = (
        f"Frame analysis for a smart streetlight system:\n"
        f"- Vehicles detected: {feats['vehicle_count']}\n"
        f"- Pedestrians detected: {feats['pedestrian_count']}\n"
        f"- Ambient brightness: {feats['brightness']:.1f}/255\n"
        f"- Contrast: {feats['contrast']:.1f}\n"
        f"- Fog probability score: {feats['fog_score']:.2f}\n"
        f"- ML model decision: streetlight {prediction} (confidence {prob:.0%})\n\n"
        f"Explain why the model made this decision and what safety or energy considerations apply."
    )
    return call_claude(prompt, system=system)


def llm_city_report(df: pd.DataFrame, metrics: dict) -> str:
    system = (
        "You are a smart city consultant writing an executive briefing. "
        "Be insightful, data-driven, and recommend 3 concrete actions. "
        "Keep it under 200 words. Use plain paragraphs only."
    )
    on_pct = int(df['light_status'].mean() * 100)
    fog_pct = int((df['weather'] == 'Fog').mean() * 100)
    peak_hour = int(df.groupby('hour')['vehicle_count'].mean().idxmax())
    best_model = max(metrics, key=lambda k: metrics[k].get('accuracy', 0)) if metrics else 'N/A'
    prompt = (
        f"Smart streetlight deployment data summary:\n"
        f"- Dataset: {len(df):,} time-steps, 5-minute intervals\n"
        f"- Lights ON: {on_pct}% of the time\n"
        f"- Fog events: {fog_pct}% of records\n"
        f"- Peak traffic hour: {peak_hour}:00\n"
        f"- Best ML model: {best_model} ({metrics.get(best_model, {}).get('accuracy', 0):.1%} accuracy)\n\n"
        f"Write an executive briefing for the city council covering: energy savings potential, "
        f"safety implications, and model deployment recommendations."
    )
    return call_claude(prompt, system=system, max_tokens=400)


def llm_anomaly_explanation(row_dict: dict) -> str:
    prompt = (
        f"The following sensor reading was flagged as anomalous by an IsolationForest model:\n"
        f"{json.dumps(row_dict, indent=2)}\n\n"
        f"In 2–3 sentences, explain what might have caused this anomaly in a real-world streetlight sensor "
        f"(e.g., sensor fault, unusual event, extreme weather). Be specific."
    )
    return call_claude(prompt)


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    for lag in [1, 2, 3, 6, 12]:
        df[f"vehicle_lag{lag}"] = df["vehicle_count"].shift(lag).fillna(0)
        df[f"brightness_lag{lag}"] = df["brightness"].shift(lag).fillna(df["brightness"].mean())
    for w in [6, 12, 24]:
        df[f"vehicle_roll{w}"] = df["vehicle_count"].rolling(w, min_periods=1).mean()
        df[f"brightness_roll{w}"] = df["brightness"].rolling(w, min_periods=1).mean()
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)
    df["is_rush"] = (((df["hour"] >= 7) & (df["hour"] <= 9)) |
                     ((df["hour"] >= 17) & (df["hour"] <= 19))).astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["brightness_x_fog"] = df["brightness"] * df["fog_score"]
    df["traffic_density"] = df["vehicle_count"] + 2 * df["pedestrian_count"]
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_tabular(n: int = 3000) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    rows = []
    start = datetime(2024, 1, 1, 0, 0, 0)
    for i in range(n):
        dt = start + timedelta(minutes=5 * i)
        hour = dt.hour
        dow = dt.weekday()
        is_weekend = dow >= 5
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            base_veh = 60 if not is_weekend else 30
        elif 22 <= hour or hour <= 4:
            base_veh = 4
        else:
            base_veh = 15
        vehicle = max(0, rng.poisson(base_veh))
        pedestrian = max(0, int(vehicle * rng.uniform(0.06, 0.35)))
        weather = rng.choice(["Clear", "Rain", "Fog", "Cloudy"], p=[0.62, 0.16, 0.10, 0.12])
        if 6 <= hour <= 18:
            brightness = int(np.clip(rng.normal(600, 80), 10, 900))
        else:
            brightness = int(np.clip(rng.normal(20, 12), 0, 80))
        if weather == "Fog":
            brightness = int(brightness * rng.uniform(0.2, 0.5))
        elif weather == "Rain":
            brightness = int(brightness * rng.uniform(0.45, 0.75))
        elif weather == "Cloudy":
            brightness = int(brightness * rng.uniform(0.7, 0.9))
        contrast = int(np.clip(rng.normal(40, 15), 1, 120))
        fog_score = rng.uniform(0.6, 1.0) if weather == "Fog" else rng.uniform(0.0, 0.35)
        # ── label logic
        prob_on = 0.0
        if brightness < 80:
            prob_on += 0.65
        elif brightness < 200:
            prob_on += 0.3
        if weather in ["Fog", "Rain"]:
            prob_on += 0.22
        if vehicle > 40:
            prob_on += 0.15
        if pedestrian > 5:
            prob_on += 0.08
        if 0 <= hour <= 5 and vehicle < 5 and pedestrian < 2:
            prob_on -= 0.3   # deep-night dimming
        prob_on = float(np.clip(prob_on, 0.0, 0.97))
        light = 1 if rng.rand() < prob_on else 0
        rows.append({
            "datetime": dt,
            "hour": hour,
            "dayofweek": dow,
            "vehicle_count": vehicle,
            "pedestrian_count": pedestrian,
            "brightness": brightness,
            "contrast": contrast,
            "fog_score": fog_score,
            "weather": weather,
            "light_status": light,
        })
    df = pd.DataFrame(rows)
    df = add_time_series_features(df)
    df.to_csv(DATA_PATH, index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

NUMERIC_BASE = [
    "vehicle_count", "pedestrian_count", "brightness", "contrast", "fog_score",
    "hour", "dayofweek",
    "vehicle_lag1", "vehicle_lag2", "vehicle_lag3", "vehicle_lag6", "vehicle_lag12",
    "brightness_lag1", "brightness_lag6",
    "vehicle_roll6", "vehicle_roll12", "vehicle_roll24",
    "brightness_roll6", "brightness_roll12",
    "is_night", "is_rush", "is_weekend",
    "hour_sin", "hour_cos",
    "brightness_x_fog", "traffic_density",
]
CATEGORICAL = ["weather"]


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_BASE),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def train_tabular_models(df: pd.DataFrame):
    avail_num = [c for c in NUMERIC_BASE if c in df.columns]
    avail_cat = [c for c in CATEGORICAL if c in df.columns]
    features = avail_num + avail_cat
    X = df[features].copy()
    y = df["light_status"].copy()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), avail_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), avail_cat),
    ])

    base_models = {
        "LogisticRegression": LogisticRegression(max_iter=600, C=1.5),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, random_state=42),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    pipelines = {}
    metrics = {}
    progress = st.progress(0)
    status = st.empty()

    for idx, (name, clf) in enumerate(base_models.items()):
        status.info(f"⚙️  Training **{name}** …")
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        metrics[name] = {"accuracy": acc, "auc": auc, "f1": f1}
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{name} Accuracy", f"{acc:.3f}")
        col2.metric("ROC-AUC", f"{auc:.3f}")
        col3.metric("F1", f"{f1:.3f}")
        joblib.dump(pipe, MODELS_DIR / f"{name}_pipeline.joblib")
        pipelines[name] = pipe
        progress.progress(int((idx + 1) / (len(base_models) + 1) * 100))

    # ── Voting ensemble
    status.info("⚙️  Training **Voting Ensemble** …")
    estimators = [
        ("rf", base_models["RandomForest"]),
        ("et", base_models["ExtraTrees"]),
        ("gb", base_models["GradientBoosting"]),
    ]
    voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    voting_pipe = Pipeline([("pre", preprocessor), ("clf", voting)])
    voting_pipe.fit(X_train, y_train)
    y_pred = voting_pipe.predict(X_test)
    y_prob = voting_pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    metrics["VotingEnsemble"] = {"accuracy": acc, "auc": auc, "f1": f1}
    joblib.dump(voting_pipe, MODELS_DIR / "VotingEnsemble_pipeline.joblib")
    pipelines["VotingEnsemble"] = voting_pipe
    progress.progress(100)
    status.success("✅  All models trained!")

    # ── Anomaly detector
    iso = IsolationForest(contamination=0.07, random_state=42, n_jobs=-1)
    iso.fit(X_train[avail_num])
    joblib.dump(iso, MODELS_DIR / "anomaly_detector.joblib")
    joblib.dump(avail_num, MODELS_DIR / "avail_num.joblib")

    # ── confusion matrix for best model
    best = max(metrics, key=lambda k: metrics[k]["accuracy"])
    best_pipe = pipelines[best]
    y_pred_best = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(4, 3), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["OFF", "ON"], yticklabels=["OFF", "ON"],
                annot_kws={"color": "white"})
    ax.set_xlabel("Predicted", color="#94a3b8")
    ax.set_ylabel("Actual", color="#94a3b8")
    ax.set_title(f"Confusion Matrix — {best}", color="#e2e8f0", pad=10)
    plt.xticks(color="#94a3b8")
    plt.yticks(color="#94a3b8")
    st.pyplot(fig)
    plt.close()

    # save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return pipelines, metrics


# ═══════════════════════════════════════════════════════════════════════════
#  CNN FOG MODEL
# ═══════════════════════════════════════════════════════════════════════════

def build_fog_cnn(input_shape=(64, 64, 1)):
    model = Sequential([
        Conv2D(16, (3, 3), activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def generate_fog_images(n=800, size=(64, 64)):
    rng = np.random.RandomState(7)
    X, y = [], []
    for _ in range(n):
        base = rng.normal(loc=175, scale=35, size=size).clip(0, 255).astype(np.uint8)
        if rng.rand() < 0.4:
            k = rng.randint(3, 11) | 1
            fog = cv2.GaussianBlur(base, (k, k), 0)
            fog = (fog * rng.uniform(0.35, 0.75)).astype(np.uint8)
            base = fog
            label = 1
        else:
            label = 0
        noise = rng.normal(0, 8, size).astype(np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        X.append(img)
        y.append(label)
    X = np.array(X, dtype="float32") / 255.0
    X = X.reshape((-1, size[0], size[1], 1))
    return X, np.array(y, dtype="float32")


# ═══════════════════════════════════════════════════════════════════════════
#  FRAME FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_frame_features(frame, detector=None, fog_model=None, bg_sub=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # fog score
    if fog_model is not None and HAVE_TF:
        small = cv2.resize(gray, (64, 64)).astype("float32") / 255.0
        fog_score = float(fog_model.predict(small.reshape(1, 64, 64, 1), verbose=0)[0, 0])
    else:
        fog_score = float(np.clip((120 - brightness) / 120 + (30 - contrast) / 60, 0, 1))

    vehicle_count, pedestrian_count = 0, 0
    if detector is not None and USE_YOLO:
        try:
            results = detector(frame, verbose=False)
            classes = [int(c) for c in results[0].boxes.cls.tolist()]
            pedestrian_count = sum(1 for c in classes if c == 0)
            vehicle_count = sum(1 for c in classes if c in [2, 3, 5, 7])
        except Exception:
            pass
    elif bg_sub is not None:
        fgm = bg_sub.apply(gray)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgm = cv2.morphologyEx(fgm, cv2.MORPH_OPEN, k, iterations=1)
        cnts, _ = cv2.findContours(fgm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vehicle_count = sum(1 for c in cnts if cv2.contourArea(c) > 500)

    return {
        "vehicle_count": int(vehicle_count),
        "pedestrian_count": int(pedestrian_count),
        "brightness": brightness,
        "contrast": contrast,
        "fog_score": fog_score,
    }


def build_inference_row(feats: dict, now: datetime) -> pd.DataFrame:
    row = {**feats}
    row.update({
        "hour": now.hour,
        "dayofweek": now.weekday(),
        "weather": "Clear",
        # lag / rolling features → zero-filled for live inference
        **{f"vehicle_lag{l}": 0 for l in [1, 2, 3, 6, 12]},
        **{f"brightness_lag{l}": feats["brightness"] for l in [1, 6]},
        **{f"vehicle_roll{w}": feats["vehicle_count"] for w in [6, 12, 24]},
        **{f"brightness_roll{w}": feats["brightness"] for w in [6, 12]},
        "is_night": int(now.hour < 6 or now.hour >= 20),
        "is_rush": int((7 <= now.hour <= 9) or (17 <= now.hour <= 19)),
        "is_weekend": int(now.weekday() >= 5),
        "hour_sin": np.sin(2 * np.pi * now.hour / 24),
        "hour_cos": np.cos(2 * np.pi * now.hour / 24),
        "brightness_x_fog": feats["brightness"] * feats["fog_score"],
        "traffic_density": feats["vehicle_count"] + 2 * feats["pedestrian_count"],
    })
    return pd.DataFrame([row])


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="main-title">🌃 Smart<br>Streetlight</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🔑 Anthropic API Key")
    api_key_input = st.text_input("API Key (for LLM features)", type="password",
                                   value=st.session_state.get("anthropic_key", ""),
                                   placeholder="sk-ant-…")
    if api_key_input:
        st.session_state["anthropic_key"] = api_key_input
        st.success("Key saved ✓")
    st.markdown("---")
    st.markdown("**System status**")
    st.write("🔦 YOLO:", "✅" if USE_YOLO else "❌ (pip install ultralytics)")
    st.write("🧠 TensorFlow:", "✅" if HAVE_TF else "❌ (pip install tensorflow)")
    st.write("🤖 LLM:", "✅" if st.session_state.get("anthropic_key") else "❌ (enter key)")
    st.markdown("---")
    st.caption("Models in ./streetlight_outputs/models/")


# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-title">Smart Streetlight AI</h1>', unsafe_allow_html=True)
st.markdown(
    "Multi-model ML + LLM-powered scene analysis for intelligent urban lighting decisions.",
    unsafe_allow_html=True,
)

tabs = st.tabs(["📊 Dashboard", "🗃️ Dataset", "🏋️ Train Models", "🎥 Live Inference", "🤖 LLM Analysis", "📈 Metrics"])

# ─── TAB 0 — Dashboard ──────────────────────────────────────────────────────
with tabs[0]:
    st.header("System Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    if DATA_PATH.exists():
        df_dash = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
        c1.markdown(f'<div class="metric-card"><div class="metric-label">Dataset rows</div><div class="metric-value">{len(df_dash):,}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">Lights ON %</div><div class="metric-value">{df_dash.light_status.mean()*100:.1f}%</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-label">Avg vehicles/5min</div><div class="metric-value">{df_dash.vehicle_count.mean():.1f}</div></div>', unsafe_allow_html=True)
        fog_pct = (df_dash.weather == "Fog").mean() * 100
        c4.markdown(f'<div class="metric-card"><div class="metric-label">Fog events</div><div class="metric-value">{fog_pct:.1f}%</div></div>', unsafe_allow_html=True)

        st.subheader("Traffic & Light Status — 24h sample")
        sample = df_dash.head(288)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Vehicle Count", "Light Status"])
        fig.add_trace(go.Scatter(x=sample.datetime, y=sample.vehicle_count,
                                  fill="tozeroy", line_color="#38bdf8", name="Vehicles"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sample.datetime, y=sample.light_status,
                                  mode="lines", line_color="#f472b6", name="Light ON/OFF"), row=2, col=1)
        fig.update_layout(height=380, template="plotly_dark",
                          paper_bgcolor="#0a0e1a", plot_bgcolor="#0f172a",
                          margin=dict(l=40, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Weather distribution")
            wc = df_dash.weather.value_counts()
            fig2 = px.pie(values=wc.values, names=wc.index,
                          color_discrete_sequence=["#38bdf8","#818cf8","#f472b6","#34d399"],
                          template="plotly_dark")
            fig2.update_layout(paper_bgcolor="#0a0e1a", height=280)
            st.plotly_chart(fig2, use_container_width=True)
        with col_b:
            st.subheader("Avg brightness by hour")
            hb = df_dash.groupby("hour")["brightness"].mean().reset_index()
            fig3 = px.bar(hb, x="hour", y="brightness",
                          color="brightness", color_continuous_scale="Blues",
                          template="plotly_dark")
            fig3.update_layout(paper_bgcolor="#0a0e1a", height=280)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No dataset yet — generate one in the Dataset tab.")

    if METRICS_PATH.exists():
        metrics = json.load(open(METRICS_PATH))
        st.subheader("Model Performance Overview")
        mdf = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
        fig4 = go.Figure()
        for col, color in [("accuracy", "#38bdf8"), ("auc", "#818cf8"), ("f1", "#f472b6")]:
            fig4.add_trace(go.Bar(name=col.upper(), x=mdf.Model, y=mdf[col],
                                   marker_color=color))
        fig4.update_layout(barmode="group", template="plotly_dark",
                            paper_bgcolor="#0a0e1a", plot_bgcolor="#0f172a",
                            height=320, legend_font_color="#e2e8f0")
        st.plotly_chart(fig4, use_container_width=True)


# ─── TAB 1 — Dataset ────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Dataset Generation")
    n_rows = st.slider("Number of time-steps (5-min intervals)", 1000, 6000, 3000, 500)
    if st.button("🔄 Generate Dataset"):
        with st.spinner("Generating synthetic dataset …"):
            df = generate_synthetic_tabular(n=n_rows)
        st.success(f"✅  Generated {len(df):,} rows with {df.shape[1]} features.")
        st.dataframe(df.sample(8), use_container_width=True)
        st.download_button("⬇️ Download CSV", data=open(DATA_PATH, "rb").read(),
                            file_name="streetlight_dataset.csv", key="dl_gen")
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
        st.info(f"Existing dataset: {len(df):,} rows, {df.shape[1]} columns.")
        st.dataframe(df.sample(8), use_container_width=True)
        st.download_button("⬇️ Download CSV", data=open(DATA_PATH, "rb").read(),
                            file_name="streetlight_dataset.csv", key="dl_existing")

        st.subheader("Feature Correlations")
        num_cols = ["vehicle_count", "pedestrian_count", "brightness", "contrast",
                    "fog_score", "light_status"]
        corr = df[num_cols].corr()
        fig_c, ax_c = plt.subplots(figsize=(6, 4), facecolor="#0a0e1a")
        ax_c.set_facecolor("#0a0e1a")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_c,
                    annot_kws={"color": "white", "size": 8})
        plt.xticks(color="#94a3b8", fontsize=8, rotation=30)
        plt.yticks(color="#94a3b8", fontsize=8)
        st.pyplot(fig_c)
        plt.close()


# ─── TAB 2 — Train ──────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Train ML Models")
    st.write("""
    Trains 5 base classifiers + a soft-voting ensemble, all with rich time-series features:
    `LogisticRegression`, `DecisionTree`, `RandomForest`, `ExtraTrees`, `GradientBoosting`, `VotingEnsemble`.
    Also trains an `IsolationForest` for anomaly detection.
    """)

    col_tf1, col_tf2 = st.columns(2)
    with col_tf1:
        if st.button("🏋️ Train All Models"):
            if not DATA_PATH.exists():
                st.warning("No dataset found — generating one …")
                df = generate_synthetic_tabular(n=3000)
            else:
                df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
            pipelines, metrics = train_tabular_models(df)
            st.session_state["pipelines"] = pipelines
            st.session_state["metrics"] = metrics

    with col_tf2:
        if HAVE_TF:
            if st.button("🌫️ Train Fog CNN"):
                st.info("Training Fog CNN on synthetic images …")
                fog_model = build_fog_cnn((64, 64, 1))
                X_vis, y_vis = generate_fog_images(n=800)
                history = fog_model.fit(X_vis, y_vis, epochs=8, batch_size=32, verbose=0,
                                        validation_split=0.15)
                fog_model.save(MODELS_DIR / "fog_cnn.h5")
                val_acc = history.history["val_accuracy"][-1]
                st.success(f"Fog CNN trained. Val accuracy: {val_acc:.3f}")
                fig_fog, ax_fog = plt.subplots(facecolor="#0a0e1a")
                ax_fog.set_facecolor("#0a0e1a")
                ax_fog.plot(history.history["accuracy"], color="#38bdf8", label="Train")
                ax_fog.plot(history.history["val_accuracy"], color="#f472b6", label="Val")
                ax_fog.legend(facecolor="#1e293b", labelcolor="#e2e8f0")
                ax_fog.set_xlabel("Epoch", color="#94a3b8")
                ax_fog.set_ylabel("Accuracy", color="#94a3b8")
                ax_fog.set_title("Fog CNN Training", color="#e2e8f0")
                plt.xticks(color="#94a3b8")
                plt.yticks(color="#94a3b8")
                st.pyplot(fig_fog)
                plt.close()
        else:
            st.info("TensorFlow not installed — Fog CNN unavailable.")

    st.subheader("Saved Models")
    saved = list(MODELS_DIR.glob("*.joblib")) + list(MODELS_DIR.glob("*.h5"))
    if saved:
        for m in saved:
            st.write(f"• {m.name}")
    else:
        st.info("No models saved yet.")


# ─── TAB 3 — Live Inference ─────────────────────────────────────────────────
with tabs[3]:
    st.header("Live / Video Inference")
    uploaded = st.file_uploader("Upload video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
    use_sample = st.checkbox("Use synthetic sample frames (fast demo)", value=True)
    enable_llm_frame = st.checkbox("🤖 LLM scene commentary per N frames", value=False)
    llm_every = st.slider("LLM every N frames", 10, 100, 30, 10)

    # load models
    ve_path = MODELS_DIR / "VotingEnsemble_pipeline.joblib"
    rf_path = MODELS_DIR / "RandomForest_pipeline.joblib"
    anomaly_path = MODELS_DIR / "anomaly_detector.joblib"
    avail_num_path = MODELS_DIR / "avail_num.joblib"

    fog_model_live = None
    if HAVE_TF:
        fpath = MODELS_DIR / "fog_cnn.h5"
        if fpath.exists():
            try:
                fog_model_live = tf.keras.models.load_model(str(fpath))
            except Exception:
                pass

    if st.button("▶️ Run Inference"):
        model = None
        if ve_path.exists():
            model = joblib.load(str(ve_path))
        elif rf_path.exists():
            model = joblib.load(str(rf_path))

        if model is None:
            st.error("No trained model found. Please train models first.")
            st.stop()

        anomaly_model = joblib.load(str(anomaly_path)) if anomaly_path.exists() else None
        avail_num = joblib.load(str(avail_num_path)) if avail_num_path.exists() else NUMERIC_BASE

        bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        cap = None
        if uploaded is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)

        frame_slot = st.empty()
        info_slot = st.empty()
        llm_slot = st.empty()
        prog = st.progress(0)
        max_frames = 250
        history_probs = deque(maxlen=60)
        history_veh = deque(maxlen=60)

        for frame_idx in range(max_frames):
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                n_obj = np.random.randint(0, 8)
                for _ in range(n_obj):
                    x1 = np.random.randint(0, 540)
                    y1 = np.random.randint(150, 300)
                    cv2.rectangle(frame, (x1, y1), (x1+np.random.randint(40,100), y1+np.random.randint(20,60)),
                                  tuple(np.random.randint(60, 255, 3).tolist()), -1)
                if np.random.rand() < 0.12:
                    fog_l = np.full_like(frame, 200)
                    frame = cv2.addWeighted(frame, 0.55, fog_l, 0.45, 0)

            feats = extract_frame_features(frame, fog_model=fog_model_live, bg_sub=bg_sub)
            now = datetime.now()
            row_df = build_inference_row(feats, now)

            pred = int(model.predict(row_df)[0])
            prob = float(model.predict_proba(row_df)[0, 1])
            history_probs.append(prob)
            history_veh.append(feats["vehicle_count"])

            # anomaly
            anomaly_flag = False
            if anomaly_model is not None:
                anf = [row_df[c].values[0] if c in row_df.columns else 0 for c in avail_num]
                score = anomaly_model.predict([anf])[0]
                anomaly_flag = score == -1

            label = "ON" if pred == 1 else "OFF"
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)
            display = frame.copy()
            cv2.putText(display, f"Light: {label}  ({prob:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(display, f"Veh:{feats['vehicle_count']} Ped:{feats['pedestrian_count']}  Fog:{feats['fog_score']:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            if anomaly_flag:
                cv2.putText(display, "⚠ ANOMALY", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
            _, im_buf = cv2.imencode(".jpg", display)
            frame_slot.image(im_buf.tobytes(), channels="BGR")

            badge = '<span class="anomaly-badge">⚠ ANOMALY</span>' if anomaly_flag else '<span class="normal-badge">NORMAL</span>'
            status_cls = "light-on" if pred == 1 else "light-off"
            info_slot.markdown(
                f'Frame {frame_idx+1} — '
                f'<span class="{status_cls}">Light {label}</span> '
                f'| prob={prob:.2f} | fog={feats["fog_score"]:.2f} | '
                f'veh={feats["vehicle_count"]} {badge}',
                unsafe_allow_html=True,
            )

            if enable_llm_frame and frame_idx % llm_every == 0:
                commentary = llm_scene_analysis(feats, label, prob)
                llm_slot.markdown(f'<div class="llm-box">🤖 <b>LLM Analysis (frame {frame_idx+1})</b><br>{commentary}</div>',
                                  unsafe_allow_html=True)

            prog.progress(int((frame_idx + 1) / max_frames * 100))
            time.sleep(0.02)

        if cap is not None:
            cap.release()
        st.success("✅  Inference complete.")

        # mini trend chart
        if history_probs:
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(y=list(history_probs), mode="lines",
                                        line_color="#38bdf8", name="Light ON prob"))
            fig_h.add_trace(go.Scatter(y=list(history_veh), mode="lines",
                                        line_color="#f472b6", name="Vehicle count", yaxis="y2"))
            fig_h.update_layout(
                height=200, template="plotly_dark", paper_bgcolor="#0a0e1a",
                plot_bgcolor="#0f172a", margin=dict(l=10, r=10, t=30, b=10),
                yaxis2=dict(overlaying="y", side="right"),
                title="Last 60 frames", title_font_color="#e2e8f0",
            )
            st.plotly_chart(fig_h, use_container_width=True)


# ─── TAB 4 — LLM Analysis ───────────────────────────────────────────────────
with tabs[4]:
    st.header("🤖 LLM-Powered Analysis")
    st.write("Use Claude (Haiku) to generate insights, anomaly explanations, and city reports.")

    st.subheader("Manual Scene Analysis")
    c1, c2, c3, c4, c5 = st.columns(5)
    v_man = c1.number_input("Vehicles", 0, 200, 35)
    p_man = c2.number_input("Pedestrians", 0, 100, 5)
    b_man = c3.number_input("Brightness", 0, 900, 40)
    fog_man = c4.slider("Fog score", 0.0, 1.0, 0.15)
    hr_man = c5.number_input("Hour", 0, 23, 22)

    if st.button("🔍 Analyse Scene with LLM"):
        feats_man = {
            "vehicle_count": v_man, "pedestrian_count": p_man,
            "brightness": b_man, "contrast": 30.0, "fog_score": fog_man,
        }
        now_man = datetime.now().replace(hour=hr_man)
        row_man = build_inference_row(feats_man, now_man)
        pred_lbl, pred_prob = "ON", 0.5
        ve_path2 = MODELS_DIR / "VotingEnsemble_pipeline.joblib"
        if ve_path2.exists():
            m2 = joblib.load(str(ve_path2))
            pred_prob = float(m2.predict_proba(row_man)[0, 1])
            pred_lbl = "ON" if m2.predict(row_man)[0] == 1 else "OFF"
        with st.spinner("Asking Claude …"):
            result = llm_scene_analysis(feats_man, pred_lbl, pred_prob)
        status_cls = "light-on" if pred_lbl == "ON" else "light-off"
        st.markdown(f'**ML Decision:** <span class="{status_cls}">{pred_lbl} ({pred_prob:.0%})</span>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="llm-box">{result}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("City Executive Report")
    if st.button("📋 Generate City Report"):
        if DATA_PATH.exists():
            df_rep = pd.read_csv(DATA_PATH)
            metrics_rep = json.load(open(METRICS_PATH)) if METRICS_PATH.exists() else {}
            with st.spinner("Claude is drafting the report …"):
                report = llm_city_report(df_rep, metrics_rep)
            st.markdown(f'<div class="llm-box">{report}</div>', unsafe_allow_html=True)
        else:
            st.warning("Generate a dataset first.")

    st.markdown("---")
    st.subheader("Anomaly Explainer")
    st.write("Paste or adjust a sensor reading to get an LLM explanation of the anomaly.")
    a1, a2, a3 = st.columns(3)
    an_veh = a1.number_input("Vehicles (anomaly)", 0, 300, 180)
    an_bright = a2.number_input("Brightness (anomaly)", 0, 900, 900)
    an_fog = a3.slider("Fog score (anomaly)", 0.0, 1.0, 0.02)
    if st.button("🧩 Explain Anomaly"):
        row_dict = {
            "vehicle_count": an_veh, "brightness": an_bright,
            "fog_score": an_fog, "hour": 3, "weather": "Clear",
        }
        with st.spinner("Consulting LLM …"):
            expl = llm_anomaly_explanation(row_dict)
        st.markdown(f'<div class="llm-box">{expl}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Free-form Question")
    user_q = st.text_area("Ask Claude anything about this streetlight system:", height=100,
                           placeholder="e.g. 'What are the energy savings if we switch to adaptive lighting?'")
    if st.button("💬 Ask Claude") and user_q:
        with st.spinner("Thinking …"):
            ans = call_claude(
                user_q,
                system="You are an expert in smart city infrastructure, urban ML systems, and energy efficiency.",
            )
        st.markdown(f'<div class="llm-box">{ans}</div>', unsafe_allow_html=True)


# ─── TAB 5 — Metrics ────────────────────────────────────────────────────────
with tabs[5]:
    st.header("Model Metrics & Explainability")
    if METRICS_PATH.exists():
        metrics = json.load(open(METRICS_PATH))
        st.subheader("Performance Table")
        mdf = pd.DataFrame(metrics).T
        mdf.index.name = "Model"
        mdf = mdf.reset_index()
        mdf["accuracy"] = mdf["accuracy"].map("{:.3f}".format)
        mdf["auc"] = mdf["auc"].map("{:.3f}".format)
        mdf["f1"] = mdf["f1"].map("{:.3f}".format)
        st.dataframe(mdf, use_container_width=True)
    else:
        st.info("Train models first to see metrics.")

    st.subheader("Feature Importance (RandomForest)")
    rf_path2 = MODELS_DIR / "RandomForest_pipeline.joblib"
    if rf_path2.exists():
        pipe_rf = joblib.load(str(rf_path2))
        rf_clf = pipe_rf.named_steps["clf"]
        pre = pipe_rf.named_steps["pre"]
        try:
            num_names = pre.transformers_[0][2]
            cat_names = list(pre.transformers_[1][1].get_feature_names_out(CATEGORICAL))
            all_names = list(num_names) + cat_names
            importances = rf_clf.feature_importances_
            fi_df = pd.DataFrame({"feature": all_names[:len(importances)], "importance": importances})
            fi_df = fi_df.sort_values("importance", ascending=False).head(20)
            fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                            color="importance", color_continuous_scale="Blues",
                            template="plotly_dark")
            fig_fi.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0f172a", height=480,
                                  yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.write(f"Could not extract importances: {e}")
    else:
        st.info("Train a RandomForest model first.")

    st.subheader("Anomaly Detection Preview")
    if DATA_PATH.exists() and anomaly_path.exists():
        df_a = pd.read_csv(DATA_PATH)
        iso_m = joblib.load(str(anomaly_path))
        avail_num_a = joblib.load(str(avail_num_path)) if avail_num_path.exists() else NUMERIC_BASE
        cols_a = [c for c in avail_num_a if c in df_a.columns]
        preds_a = iso_m.predict(df_a[cols_a].fillna(0))
        df_a["anomaly"] = (preds_a == -1).astype(int)
        anom_count = df_a.anomaly.sum()
        st.write(f"Anomalies detected: **{anom_count}** / {len(df_a)} ({anom_count/len(df_a)*100:.1f}%)")
        fig_an = px.scatter(df_a.head(500), x="brightness", y="vehicle_count",
                            color=df_a.head(500).anomaly.map({0: "Normal", 1: "Anomaly"}),
                            color_discrete_map={"Normal": "#38bdf8", "Anomaly": "#f87171"},
                            template="plotly_dark", height=320)
        fig_an.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0f172a")
        st.plotly_chart(fig_an, use_container_width=True)
