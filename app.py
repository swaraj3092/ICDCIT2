import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# ---------------- CONFIG ----------------
DATA_PATH = "energy.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "energy_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.npy")

EMISSION_FACTOR = 0.82
SAFE_LIMIT_KWH = 5.0
SAFE_LIMIT_W = SAFE_LIMIT_KWH * 1000

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_all():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = np.load(META_PATH, allow_pickle=True).item()

    return df, model, scaler, meta

df, model, scaler, meta = load_all()

feature_cols = meta["feature_cols"]
error_mean = meta["error_mean"]
error_std = meta["error_std"]
feature_importances = meta["feature_importances"]

if "idx" not in st.session_state:
    st.session_state.idx = len(df) - 200

# ---------------- FUNCTIONS ----------------
def compute_status(current, predicted):
    ratio = predicted / SAFE_LIMIT_W * 100

    delta = predicted - current
    if delta > 50:
        base = "Reduce AC & dim lights"
    elif delta < -50:
        base = "Shift heavy loads"
    else:
        base = "Stable – monitor"

    if ratio >= 90:
        return ratio, "CRITICAL", base
    elif ratio >= 75:
        return ratio, "WARNING", base
    else:
        return ratio, "NORMAL", base

def explain(row):
    fi = np.array(feature_importances)
    top = np.argsort(fi)[::-1][:3]
    txt = []
    for i in top:
        f = feature_cols[i]
        txt.append(f"{f}≈{round(float(row[f]),2)}")
    return "Prediction driven by: " + ", ".join(txt)

# ---------------- UI ----------------
st.set_page_config("Smart Energy Coach", layout="wide")
st.title("⚡ Explainable AI Smart Energy Coach")

row = df.iloc[st.session_state.idx]
ts = row["date"]
current_load = float(row["Appliances"])

row_feat = row.copy()
row_feat["hour"] = ts.hour
row_feat["minute"] = ts.minute
row_feat["dayofweek"] = ts.dayofweek

X = row_feat[feature_cols].values.reshape(1,-1)
pred = float(model.predict(scaler.transform(X))[0])

co2 = (pred/1000)*EMISSION_FACTOR

future = df.iloc[st.session_state.idx+6]["Appliances"]
error = abs(future - pred)
z = (error-error_mean)/error_std if error_std>0 else 0

ratio, status, action = compute_status(current_load, pred)

# ---------------- DASHBOARD ----------------
c1,c2,c3 = st.columns(3)

c1.metric("Current Load (W)", round(current_load,2))
c2.metric("Predicted Load (W)", round(pred,2))
c3.metric("CO₂ Next Hour (kg)", round(co2,3))

st.progress(min(int(ratio),100))

if status=="CRITICAL":
    st.error("CRITICAL – Transformer Overload Risk")
elif status=="WARNING":
    st.warning("WARNING – High Load")
else:
    st.success("NORMAL")

st.write("### Suggested Action")
st.info(action)

st.write("### Explanation")
st.write(explain(row_feat))

st.write("### Anomaly Score (Z)")
st.metric("Z Score", round(z,2))

# ---------------- WHAT IF ----------------
st.divider()
st.subheader("🔮 What-if Simulator")

dt = st.slider("Δ AC Setpoint (°C)",-2,2,0)
lf = st.slider("Lights Level (%)",50,100,100)/100
shift = st.checkbox("Shift Heavy Loads")

wf = row_feat.copy()

for t in ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]:
    if t in wf:
        wf[t]+=dt

if "lights" in wf:
    wf["lights"]*=lf

if shift:
    wf["Appliances"]*=0.8

X2 = wf[feature_cols].values.reshape(1,-1)
pred2 = float(model.predict(scaler.transform(X2))[0])

save = ((pred-pred2)/pred)*100 if pred>0 else 0

st.metric("Baseline (W)",round(pred,2))
st.metric("What-if (W)",round(pred2,2))
st.metric("Saving %",round(save,2))

# ---------------- GRAPH ----------------
st.divider()
st.subheader("📈 Load Trend")

hist = df.iloc[st.session_state.idx-24:st.session_state.idx]
st.line_chart(hist["Appliances"])

# ---------------- AUTO REFRESH ----------------
if st.button("🔄 Refresh"):
    st.session_state.idx+=1
    st.rerun()
