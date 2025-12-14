
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import time
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#yolo
USE_YOLO = False
try:
    from ultralytics import YOLO
    USE_YOLO = True
except Exception:
    USE_YOLO = False

# checking for fog
HAVE_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    HAVE_TF = True
except Exception:
    HAVE_TF = False

OUT_DIR = Path("streamlit_outputs")
OUT_DIR.mkdir(exist_ok=True)

DATA_PATH = OUT_DIR / "streamlit_streetlight_dataset.csv"
MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Smart Streetlight AI", layout="wide")


# exractor

def compute_brightness_contrast(gray):
    return float(np.mean(gray)), float(np.std(gray))

def build_fog_cnn(input_shape=(64,64,1)):
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(32, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def generate_synthetic_visibility_images(n=800, size=(64,64)):
    X = []
    y = []
    for i in range(n):
        base = np.random.normal(loc=180, scale=30, size=size).clip(0,255).astype(np.uint8)
        label = 0
        if np.random.rand() < 0.4:
            k = np.random.randint(3,9)
            fog = cv2.GaussianBlur(base, (k|1, k|1), 0)
            fog = (fog * np.random.uniform(0.4,0.8)).astype(np.uint8)
            base = fog
            label = 1
        noise = np.random.normal(0,10,size).astype(np.int16)
        img = np.clip(base.astype(np.int16)+noise,0,255).astype(np.uint8)
        X.append(img)
        y.append(label)
    X = np.array(X).astype('float32') / 255.0
    X = X.reshape((-1, size[0], size[1], 1))
    y = np.array(y).astype('float32')
    return X, y
# fake generator
def extract_frame_features(frame, detector=None, fog_model=None, bg_subtractor=None):
    h,w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness, contrast = compute_brightness_contrast(gray)
    fog_score = 0.0
    vehicle_count = 0
    pedestrian_count = 0

    if fog_model is not None and HAVE_TF:
        small = cv2.resize(gray, (64,64)).astype('float32')/255.0
        small = small.reshape((1,64,64,1))
        fog_score = float(fog_model.predict(small, verbose=0)[0,0])
    else:
        fog_score = float(max(0.0, min(1.0, (120 - brightness)/120.0 + (30 - contrast)/60.0)))

    if detector is not None and USE_YOLO:
        try:
            results = detector(frame, verbose=False)
            boxes = results[0].boxes
            classes = [int(c) for c in boxes.cls.tolist()]
            pedestrian_count = sum(1 for c in classes if c==0)
            vehicle_count = sum(1 for c in classes if c in [2,3,5,7])
        except Exception:
            vehicle_count = 0
            pedestrian_count = 0
    else:
        if bg_subtractor is not None:
            fgmask = bg_subtractor.apply(gray)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:
                    count += 1
            vehicle_count = count
            pedestrian_count = 0
    return {'vehicle_count': int(vehicle_count),
            'pedestrian_count': int(pedestrian_count),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'fog_score': float(fog_score)}

def generate_synthetic_tabular(n=2000):
    rng = np.random.RandomState(42)
    rows = []
    start = datetime(2024,1,1,0,0,0)
    for i in range(n):
        dt = start + pd.Timedelta(minutes=5*i)
        hour = dt.hour
        dayofweek = dt.weekday()
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            vehicle = rng.poisson(60)
        else:
            vehicle = rng.poisson(12)
        pedestrian = int(vehicle * rng.uniform(0.08,0.35))
        weather = rng.choice(['Clear','Rain','Fog','Cloudy'], p=[0.68,0.16,0.08,0.08])
        if 6 <= hour <= 18:
            brightness = int(max(10, rng.normal(600,80)))
        else:
            brightness = int(max(0, rng.normal(20,10)))
        if weather == 'Fog':
            brightness = int(brightness * rng.uniform(0.25,0.5))
        if weather == 'Rain':
            brightness = int(brightness * rng.uniform(0.5,0.8))
        contrast = int(max(1, rng.normal(40,15)))
        fog_score = 1.0 if weather=='Fog' else rng.uniform(0,0.4)
        prob_on = 0.0
        if brightness < 80: prob_on += 0.6
        if weather in ['Fog','Rain']: prob_on += 0.2
        if vehicle > 40: prob_on += 0.12
        if pedestrian > 5: prob_on += 0.05
        if 0 <= hour <= 5 and (vehicle < 5 and pedestrian < 2): prob_on -= 0.25
        prob_on = max(0.0, min(0.98, prob_on))
        light = 1 if rng.rand() < prob_on else 0
        rows.append({
            'datetime': dt,
            'hour': hour,
            'dayofweek': dayofweek,
            'vehicle_count': vehicle,
            'pedestrian_count': pedestrian,
            'brightness': brightness,
            'contrast': contrast,
            'fog_score': fog_score,
            'weather': weather,
            'light_status': light
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    return df

def train_tabular_models(df):
    features = ['vehicle_count','pedestrian_count','brightness','contrast','fog_score','hour','dayofweek','weather']
    X = df[features].copy()
    y = df['light_status'].copy()
    numeric = ['vehicle_count','pedestrian_count','brightness','contrast','fog_score','hour','dayofweek']
    categorical = ['weather']
    preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
])

    models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'DecisionTree': DecisionTreeClassifier(max_depth=8, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=150, random_state=42)
    }
    pipelines = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    for name, clf in models.items():
        pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
        st.info(f"Training {name} ...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        st.write(f"**{name} accuracy:**", accuracy_score(y_test, y_pred))
        st.text(classification_report(y_test, y_pred))
        joblib.dump(pipe, MODELS_DIR / f"{name}_pipeline.joblib")
        pipelines[name] = pipe
    estimators = []
    if 'RandomForest' in models:
        estimators.append(('rf', models['RandomForest']))
    if 'GradientBoosting' in models:
        estimators.append(('gb', models['GradientBoosting']))
    voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    voting_pipe = Pipeline([('pre', preprocessor), ('clf', voting)])
    st.info("Training Voting Ensemble ...")
    voting_pipe.fit(X_train, y_train)
    y_pred = voting_pipe.predict(X_test)
    st.write("**Voting Ensemble accuracy:**", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))
    joblib.dump(voting_pipe, MODELS_DIR / "VotingEnsemble_pipeline.joblib")
    pipelines['VotingEnsemble'] = voting_pipe
    return pipelines

st.title("Smart Streetlight — Streamlit Demo")
st.markdown("Real-time camera + CNN fog estimator + multiple ML algorithms for ON/OFF prediction.")

tabs = st.tabs(["Overview", "Train Models", "Run Inference", "Models & Data"])

with tabs[0]:
    st.header("Overview")
    st.write("""
    This app extracts features from video frames (vehicle counts, pedestrian counts, brightness, fog score),
    builds a synthetic dataset, trains multiple models, and runs inference to predict whether a streetlight should
    be ON or OFF. For best demo, upload a short street video or use your webcam (via file upload).
    """)
    st.write("YOLO available:" , USE_YOLO, "  |  TensorFlow available:", HAVE_TF)

with tabs[1]:
    st.header("Generate Dataset & Train Models")
    st.write("1) Generate synthetic dataset (tabular) with camera-like features.")
    if st.button("Generate Dataset (2000 rows)"):
        df = generate_synthetic_tabular(n=2000)
        st.success("Dataset generated and saved.")
        st.dataframe(df.sample(6))
        st.download_button(
    "Download dataset CSV",
    data=open(DATA_PATH, 'rb').read(),
    file_name=str(DATA_PATH.name),
    key="download_dataset_train")

    st.write("---")
    st.write("2) Train models on dataset")
    if st.button("Train Models (this may take a few minutes)"):
        if not DATA_PATH.exists():
            st.warning("Dataset not found — generating one automatically.")
            df = generate_synthetic_tabular(n=2000)
        else:
            df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
        pipelines = train_tabular_models(df)
        st.success("Training complete. Models saved to ./streamlit_outputs/models")
        st.write(list(MODELS_DIR.glob("*_pipeline.joblib")))

with tabs[2]:
    st.header("Run Inference on Video / Webcam")
    st.write("Upload a short video (mp4) or use the sample video. The app will process the video and show frame-level predictions.")
    uploaded_file = st.file_uploader("Upload video (mp4) — or leave empty to process sample frames", type=['mp4','avi','mov'])
    sample_mode = st.checkbox("Use sample generated frames instead of video (fast demo)", value=False)
    fog_model = None
    if HAVE_TF:
        if st.button("Train tiny Fog CNN (quick)"):
            st.info("Training fog CNN on synthetic images...")
            fog_model = build_fog_cnn((64,64,1))
            X_vis, y_vis = generate_synthetic_visibility_images(n=600, size=(64,64))
            fog_model.fit(X_vis, y_vis, epochs=6, batch_size=32, verbose=1)
            fog_model.save(MODELS_DIR / "fog_cnn.h5")
            st.success("Fog CNN trained and saved.")
        else:
            fpath = MODELS_DIR / "fog_cnn.h5"
            if fpath.exists() and HAVE_TF:
                try:
                    fog_model = tf.keras.models.load_model(str(fpath))
                    st.info("Loaded pre-trained fog CNN.")
                except Exception:
                    fog_model = None

    detector = None
    if USE_YOLO:
        if st.button("Load YOLOv8 model"):
            try:
                detector = YOLO('yolov8n.pt')
                st.success("YOLOv8 loaded.")
            except Exception as e:
                st.error("Could not load YOLOv8: " + str(e))
                detector = None
        else:
            try:
                detector = YOLO('yolov8n.pt')
            except Exception:
                detector = None

    st.write("Background subtractor will be used if YOLO not loaded.")
    run_btn = st.button("Run Inference on Video")
    if run_btn:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        pipelines = {}
        for f in MODELS_DIR.glob("*_pipeline.joblib"):
            try:
                pipelines[f.stem] = joblib.load(f)
            except Exception:
                pass
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        elif sample_mode:
            cap = None
        else:
            st.info("No video uploaded — will show sample frames. For webcam, upload a small recorded file.")
            cap = None

        progress_bar = st.progress(0)
        frame_slot = st.empty()
        max_frames = 300
        frame_idx = 0
        while True:
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                frame = np.zeros((360,640,3), dtype=np.uint8)
                for i in range(np.random.randint(0,8)):
                    x1 = np.random.randint(0,540)
                    y1 = np.random.randint(200,320)
                    x2 = x1 + np.random.randint(30,100)
                    y2 = y1 + np.random.randint(20,60)
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(np.random.randint(50,255),np.random.randint(50,255),np.random.randint(50,255)), -1)
                if np.random.rand() < 0.1:
                    fog_layer = np.full_like(frame, 200)
                    frame = cv2.addWeighted(frame, 0.6, fog_layer, 0.4, 0)
            feats = extract_frame_features(frame, detector=detector, fog_model=fog_model, bg_subtractor=cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True))
            now = datetime.now()
            row = pd.DataFrame([{
                'vehicle_count': feats['vehicle_count'],
                'pedestrian_count': feats['pedestrian_count'],
                'brightness': feats['brightness'],
                'contrast': feats['contrast'],
                'fog_score': feats['fog_score'],
                'hour': now.hour,
                'dayofweek': now.weekday(),
                'weather': 'Clear'
            }])
            pred = 0
            pred_prob = 0.0
            ve_path = MODELS_DIR / 'VotingEnsemble_pipeline.joblib'
            rf_path = MODELS_DIR / 'RandomForest_pipeline.joblib'
            if ve_path.exists():
                model = joblib.load(str(ve_path))
                pred_prob = float(model.predict_proba(row)[0,1])
                pred = int(model.predict(row)[0])
            elif rf_path.exists():
                model = joblib.load(str(rf_path))
                pred_prob = float(model.predict_proba(row)[0,1])
                pred = int(model.predict(row)[0])
            label = "ON" if pred==1 else "OFF"
            color = (0,255,0) if pred==1 else (0,0,255)
            display = frame.copy()
            cv2.putText(display, f"Pred: {label} ({pred_prob:.2f})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(display, f"Veh:{feats['vehicle_count']} Ped:{feats['pedestrian_count']}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            _, im_buf = cv2.imencode('.jpg', display)
            frame_slot.image(im_buf.tobytes(), channels="BGR")
            frame_idx += 1
            progress_bar.progress(min(100, int(frame_idx / max_frames * 100)))
            if frame_idx >= max_frames:
                break
        if cap is not None:
            cap.release()
        st.success("Inference run complete.")

with tabs[3]:
    st.header("Models & Data")
    st.write("Download trained models or dataset if available.")
    if DATA_PATH.exists():
        st.download_button(
    "Download dataset CSV",
    data=open(DATA_PATH, 'rb').read(),
    file_name=str(DATA_PATH.name),
    key="download_dataset_models"
)

    else:
        st.info("No dataset generated yet.")
    models_list = list(MODELS_DIR.glob("*_pipeline.joblib")) + list(MODELS_DIR.glob("*.h5"))
    if models_list:
        for m in models_list:
            st.write(m.name)
            st.download_button(f"Download {m.name}", data=open(m,'rb').read(), file_name=m.name)
    else:
        st.info("No models saved yet. Train models in the Train tab.")
