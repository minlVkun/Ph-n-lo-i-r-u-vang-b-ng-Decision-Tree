# app.py
import os
from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import pickle
from sklearn.datasets import load_wine

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev_secret_key")

# Load metadata (feature names & class names)
wine = load_wine()
FEATURE_NAMES = wine.feature_names
CLASS_NAMES = wine.target_names.tolist()

# Load model & scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "wine_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = None
scaler = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Loaded model.")
except Exception as e:
    print("Could not load wine_model.pkl:", e)

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("Loaded scaler.")
except Exception as e:
    print("Could not load scaler.pkl:", e)
    scaler = None

@app.route("/", methods=["GET"])
def index():
    # provide default sample values (mean of features)
    sample = {name: "" for name in FEATURE_NAMES}
    return render_template("index.html", features=FEATURE_NAMES, sample=sample)

def parse_inputs(form):
    vals = []
    for name in FEATURE_NAMES:
        v = form.get(name)
        if v is None or v.strip() == "":
            return None, f"Thiếu giá trị cho {name}"
        v = v.replace(",", ".")
        try:
            fv = float(v)
        except ValueError:
            return None, f"Giá trị cho {name} không hợp lệ: {v}"
        vals.append(fv)
    arr = np.array(vals).reshape(1, -1)
    return arr, None

@app.route("/predict", methods=["POST"])
def predict():
    values = []
    for f in FEATURE_NAMES:
        v = request.form[f]
        try:
            v = float(v)
        except ValueError:
            flash(f"Giá trị '{f}' không hợp lệ!", "danger")
            return redirect(url_for("index"))
        values.append(v)

    arr = np.array(values).reshape(1, -1)

    # Scale dữ liệu
    if scaler is not None:
        arr_proc = scaler.transform(arr)
    else:
        arr_proc = arr

    # Dự đoán
    pred_idx = int(model.predict(arr_proc)[0])  # FIX
    predicted_class = CLASS_NAMES[pred_idx]     # FIX

    return render_template("result.html",
                           label=predicted_class,
                           inputs=values)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
