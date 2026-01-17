# app.py
from flask import Flask, jsonify, render_template_string, request
import pandas as pd
import numpy as np
import joblib
import os

DATA_PATH = "energy.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "energy_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.npy")

# Example grid emission factor (kg CO2 per kWh) – tune for your region
EMISSION_FACTOR_KG_PER_KWH = 0.82  # [web:33][web:35]

# Transformer safe limit in kWh for the next hour (demo: ~60 kWh)
SAFE_LIMIT_KWH = 5.0  # corresponds to about 60 kW over 1 hour [web:87][web:90]

# Convert safe limit to W for internal load ratio calculation
SAFE_LIMIT_W = SAFE_LIMIT_KWH * 1000.0

app = Flask(__name__)

# load data
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# load model and meta
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
meta = np.load(META_PATH, allow_pickle=True).item()

feature_cols = meta["feature_cols"]
error_mean = meta["error_mean"]
error_std = meta["error_std"]
feature_importances = meta["feature_importances"]

# simulate streaming
current_index = len(df) - 200

# in-memory contact info
user_contact = {
    "email": None,
    "phone": None
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Explainable Smart Energy Coach</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .card { border: 1px solid #ccc; padding: 16px; margin-bottom: 16px; border-radius: 6px; }
        .label { font-weight: bold; }
        .section-title { font-size: 18px; margin-top: 0; }
        input[type=range] { width: 200px; }
        .row { display: flex; gap: 20px; flex-wrap: wrap; }
        .col { flex: 1 1 320px; }
        .status-normal { color: green; font-weight: bold; }
        .status-warning { color: orange; font-weight: bold; }
        .status-critical { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Explainable AI-based Smart Energy Coach</h1>

    <div class="row">
        <div class="col">
            <div class="card">
                <h2 class="section-title">Live Status</h2>
                <p><span class="label">Timestamp:</span> <span id="ts"></span></p>
                <p><span class="label">Current Appliances Load (W):</span> <span id="current"></span></p>
                <p><span class="label">Predicted Load 1 Hour Ahead (W):</span> <span id="pred"></span></p>
                <p><span class="label">Predicted Next Hour CO₂:</span>
                   <span id="next_hour_co2"></span> kg</p>
                <p><span class="label">Transformer Safe Limit (Next Hour):</span>
                   <span id="safe_limit_kwh"></span> kWh</p>
                <p><span class="label">Transformer Loading (Next Hour):</span>
                   <span id="load_ratio"></span> %</p>
                <p><span class="label">Status:</span>
                   <span id="status_text" class="status-normal"></span></p>
                <p><span class="label">Anomaly Score (z):</span> <span id="anom"></span></p>
                <p><span class="label">Suggested Action:</span> <span id="action"></span></p>
                <p><span class="label">Why this suggestion?</span></p>
                <p id="explanation"></p>
            </div>
        </div>

        <div class="col">
            <div class="card">
                <h2 class="section-title">What-if Simulator (1 Hour Ahead)</h2>
                <p>
                    <span class="label">Δ AC Setpoint (°C):</span><br/>
                    <input id="delta_temp" type="range" min="-2" max="2" step="1" value="0"
                           oninput="document.getElementById('delta_temp_val').innerText=this.value">
                    <span id="delta_temp_val">0</span>
                </p>
                <p>
                    <span class="label">Lights Level (% of current):</span><br/>
                    <input id="light_factor" type="range" min="50" max="100" step="10" value="100"
                           oninput="document.getElementById('light_factor_val').innerText=this.value">
                    <span id="light_factor_val">100</span>%
                </p>
                <p>
                    <label>
                        <input id="shift_heavy" type="checkbox" />
                        Shift flexible heavy loads to low-demand period
                    </label>
                </p>
                <button onclick="runWhatIf()">Run What-if</button>

                <h3>What-if Result (1 Hour Ahead)</h3>
                <p><span class="label">Baseline Predicted Load (W):</span> <span id="wf_base"></span></p>
                <p><span class="label">What-if Predicted Load (W):</span> <span id="wf_new"></span></p>
                <p><span class="label">Expected Saving (%):</span> <span id="wf_save"></span></p>
                <p><span class="label">Coach Comment:</span> <span id="wf_comment"></span></p>
            </div>
        </div>
    </div>

    <div class="row">
        <div class="col">
            <div class="card">
                <h2 class="section-title">Notification Settings</h2>
                <p><span class="label">Email (Facility Manager / Warden):</span><br>
                   <input id="email_input" type="email" placeholder="user@example.com" style="width: 100%;">
                </p>
                <p><span class="label">Phone (optional):</span><br>
                   <input id="phone_input" type="text" placeholder="+91XXXXXXXXXX" style="width: 100%;">
                </p>
                <button onclick="saveContact()">Save Contact</button>
                <p id="contact_status"></p>
            </div>
        </div>
    </div>

    <script>
        async function refresh() {
            const res = await fetch("/api/latest");
            const data = await res.json();

            document.getElementById("ts").innerText = data.timestamp;
            document.getElementById("current").innerText = data.current_load.toFixed(2);
            document.getElementById("pred").innerText = data.predicted_load.toFixed(2);
            document.getElementById("next_hour_co2").innerText = data.next_hour_co2_kg.toFixed(3);
            document.getElementById("safe_limit_kwh").innerText = data.safe_limit_kwh.toFixed(1);
            document.getElementById("load_ratio").innerText = data.load_ratio_percent.toFixed(1);

            const statusText = document.getElementById("status_text");
            statusText.innerText = data.status_label;
            statusText.className = "";
            if (data.status_level === "normal") {
                statusText.classList.add("status-normal");
            } else if (data.status_level === "warning") {
                statusText.classList.add("status-warning");
            } else if (data.status_level === "critical") {
                statusText.classList.add("status-critical");
            }

            document.getElementById("anom").innerText = data.anomaly_score.toFixed(2);
            document.getElementById("action").innerText = data.suggested_action;
            document.getElementById("explanation").innerText = data.explanation;
        }

        async function runWhatIf() {
            const dt = parseInt(document.getElementById("delta_temp").value);
            const lf = parseInt(document.getElementById("light_factor").value) / 100.0;
            const sh = document.getElementById("shift_heavy").checked;

            const res = await fetch("/api/what_if", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    delta_temp: dt,
                    light_factor: lf,
                    shift_heavy: sh
                })
            });
            const data = await res.json();

            document.getElementById("wf_base").innerText = data.baseline_pred.toFixed(2);
            document.getElementById("wf_new").innerText = data.whatif_pred.toFixed(2);
            document.getElementById("wf_save").innerText = data.saving_percent.toFixed(2);
            document.getElementById("wf_comment").innerText = data.comment;
        }

        async function saveContact() {
            const email = document.getElementById("email_input").value;
            const phone = document.getElementById("phone_input").value;

            const res = await fetch("/api/set_contact", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email: email, phone: phone })
            });
            const data = await res.json();
            document.getElementById("contact_status").innerText =
                "Saved. Email: " + (data.email || "None") + ", Phone: " + (data.phone || "None");
        }

        refresh();
        setInterval(refresh, 5000);
    </script>
</body>
</html>
"""

def compute_status_and_action(current, predicted):
    """
    Decide transformer status and suggested action based on 1-hour ahead load.
    """
    load_ratio = 0.0
    if SAFE_LIMIT_W > 0:
        load_ratio = predicted / SAFE_LIMIT_W * 100.0  # predicted W vs safe W

    # Base suggested action from load comparison
    delta = predicted - current
    if delta > 50:
        base_action = "High upcoming 1-hour load: Dim lights and increase AC setpoint by 1°C in common areas."
    elif delta < -50:
        base_action = "1-hour load expected to drop: You may shift some flexible tasks (laundry, heating) to this period."
    else:
        base_action = "1-hour load stable: Maintain current settings; monitor anomalies."

    # Status levels based on load ratio
    if load_ratio >= 90:
        status_level = "critical"
        status_label = "CRITICAL – Risk of transformer overload"
        action = "CRITICAL: Forecasted 1-hour energy is close to or above the safe transformer limit. Immediately reduce non-essential loads and postpone heavy equipment." \
                 + " " + base_action
    elif load_ratio >= 75:
        status_level = "warning"
        status_label = "WARNING – High loading"
        action = "WARNING: Transformer loading will be high in the next hour. Start applying energy-saving actions now (setpoint increase, dim lights, shift flexible tasks)." \
                 + " " + base_action
    else:
        status_level = "normal"
        status_label = "NORMAL"
        action = base_action

    return load_ratio, status_level, status_label, action

def build_explanation(row, importances):
    fi = np.array(importances)
    idx_sorted = np.argsort(fi)[::-1]
    top_k = 3
    idx_top = idx_sorted[:top_k]
    parts = []
    for idx in idx_top:
        fname = feature_cols[idx]
        fval = row[fname] if fname in row else None
        parts.append(f"{fname}≈{round(float(fval),2) if fval is not None else 'NA'}")
    if not parts:
        return "Model uses several sensor and time features; this time step is dominated by normal patterns."
    return ("1-hour forecast driven mainly by " +
            ", ".join(parts) +
            ". Adjusting these conditions has the strongest effect on load.")

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/latest")
def latest():
    global current_index
    # need 6 steps ahead for anomaly (1 hour)
    if current_index >= len(df) - 7:
        current_index = len(df) - 200

    row = df.iloc[current_index]
    ts = row["date"]
    current_load = float(row["Appliances"])

    hour = ts.hour
    minute = ts.minute
    dayofweek = ts.dayofweek

    feat_row = row.copy()
    feat_row["hour"] = hour
    feat_row["minute"] = minute
    feat_row["dayofweek"] = dayofweek

    X = feat_row[feature_cols].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    predicted = float(model.predict(X_scaled)[0])  # 1-hour ahead average W [web:38]

    # 1-hour energy and CO2 (energy not displayed, only CO2)
    next_hour_energy_kwh = predicted / 1000.0
    next_hour_co2_kg = next_hour_energy_kwh * EMISSION_FACTOR_KG_PER_KWH  # [web:33][web:35]

    # anomaly: compare to actual Appliances 1 hour later (6 steps)
    future_row = df.iloc[current_index + 6]
    true_future = float(future_row["Appliances"])
    error = abs(true_future - predicted)
    if error_std > 0:
        z = (error - error_mean) / error_std
    else:
        z = 0.0
    anomaly_score = float(z)

    # transformer status + suggested action
    load_ratio, status_level, status_label, action = compute_status_and_action(current_load, predicted)

    explanation = build_explanation(feat_row, feature_importances)

    current_index += 1

    return jsonify({
        "timestamp": ts.isoformat(),
        "current_load": current_load,
        "predicted_load": predicted,
        "next_hour_co2_kg": next_hour_co2_kg,
        "safe_limit_kwh": SAFE_LIMIT_KWH,
        "load_ratio_percent": load_ratio,
        "status_level": status_level,
        "status_label": status_label,
        "anomaly_score": anomaly_score,
        "suggested_action": action,
        "explanation": explanation
    })

@app.route("/api/what_if", methods=["POST"])
def what_if():
    """
    What-if simulator for 1-hour ahead forecast.
    """
    global current_index
    data = request.get_json(force=True)
    delta_temp = float(data.get("delta_temp", 0.0))
    light_factor = float(data.get("light_factor", 1.0))
    shift_heavy = bool(data.get("shift_heavy", False))

    idx = max(current_index - 1, 0)
    row = df.iloc[idx]
    ts = row["date"]

    hour = ts.hour
    minute = ts.minute
    dayofweek = ts.dayofweek

    feat_row = row.copy()
    feat_row["hour"] = hour
    feat_row["minute"] = minute
    feat_row["dayofweek"] = dayofweek

    # baseline prediction (1-hour ahead W)
    X_base = feat_row[feature_cols].values.reshape(1, -1)
    X_base_scaled = scaler.transform(X_base)
    baseline_pred = float(model.predict(X_base_scaled)[0])

    # modified features for what-if
    wf_row = feat_row.copy()

    for c in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]:
        if c in wf_row:
            wf_row[c] = wf_row[c] + delta_temp

    if "lights" in wf_row:
        wf_row["lights"] = wf_row["lights"] * light_factor

    if shift_heavy and "Appliances" in wf_row:
        wf_row["Appliances"] = max(0.0, wf_row["Appliances"] * 0.8)

    X_wf = wf_row[feature_cols].values.reshape(1, -1)
    X_wf_scaled = scaler.transform(X_wf)
    whatif_pred = float(model.predict(X_wf_scaled)[0])

    if baseline_pred > 0:
        saving_percent = (baseline_pred - whatif_pred) / baseline_pred * 100.0
    else:
        saving_percent = 0.0

    if saving_percent > 10:
        comment = "This change has a strong impact on reducing upcoming 1-hour load and emissions."
    elif saving_percent > 3:
        comment = "Moderate saving: good for fine-tuning schedules and comfort settings."
    else:
        comment = "Very small effect now; try larger setpoint change or more dimming."

    return jsonify({
        "baseline_pred": baseline_pred,
        "whatif_pred": whatif_pred,
        "saving_percent": saving_percent,
        "comment": comment
    })

@app.route("/api/set_contact", methods=["POST"])
def set_contact():
    data = request.get_json(force=True)
    email = data.get("email")
    phone = data.get("phone")
    if email:
        user_contact["email"] = email
    if phone:
        user_contact["phone"] = phone
    return jsonify({
        "status": "ok",
        "email": user_contact["email"],
        "phone": user_contact["phone"]
    })

if __name__ == "__main__":
    app.run(debug=True)