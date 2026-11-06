import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")


@st.cache_resource
def load_artifacts():
    return joblib.load("bean_model.joblib")


art = load_artifacts()
model, scaler, le = art["model"], art["scaler"], art["label_encoder"]
features, bounds, right_skew = art["features"], art["bounds"], art["right_skew"]


def fix_skew(data):
    data = data.copy()
    for c in right_skew:
        data[c] = np.log1p(data[c])
    for c in ["Eccentricity", "Extent"]:
        data[c] = data[c] ** 3
    for c in ["Solidity", "ShapeFactor4"]:
        data[c] = -np.log(1.001 - data[c].clip(upper=1.0))
    return data


st.title("Dry Bean Type Classifier")
st.write("Enter the physical measurements of a bean (from the camera system) and the model predicts its type: "
         + ", ".join(le.classes_) + ".")

example = st.selectbox("Start from an example bean", ["Dataset median"] + list(le.classes_))
defaults = art["median"] if example == "Dataset median" else art["class_means"][example]

st.subheader("Bean measurements")
cols = st.columns(4)
values = {}
for i, f in enumerate(features):
    lo, hi = float(art["min"][f]), float(art["max"][f])
    with cols[i % 4]:
        values[f] = st.number_input(f, min_value=lo, max_value=hi, value=float(defaults[f]),
                                    step=(hi - lo) / 200, format="%.6f", key=f"{f}_{example}")

if st.button("Predict bean type", type="primary"):
    x = pd.DataFrame([values])[features]
    for c in features:
        x[c] = x[c].clip(*bounds[c])
    x = fix_skew(x)
    xs = scaler.transform(x)
    label = le.inverse_transform(model.predict(xs))[0]
    st.success(f"Predicted bean type: **{label}**")
    if hasattr(model, "predict_proba"):
        proba = pd.Series(model.predict_proba(xs)[0], index=le.classes_).sort_values(ascending=False)
        st.caption(f"Confidence: {proba.iloc[0] * 100:.1f}%")
        st.bar_chart(proba)
