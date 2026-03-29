import streamlit as st
import numpy as np
import pickle
import time

# -------------------------
# 1. Load your trained model
# -------------------------
# Make sure you train a model that uses these 7 features:
# px, py, pz, rx, ry, rz, depth
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------
# 2. App state (simulate Unity _isSending and _hasValidValues)
# -------------------------
if "is_sending" not in st.session_state:
    st.session_state.is_sending = False
if "last_result" not in st.session_state:
    st.session_state.last_result = ""

# -------------------------
# 3. App title
# -------------------------
st.title("🧠 NeuroPath AI - ETV Success Predictor (Unity-style)")
st.write("Input relative position, rotation, and depth to predict success probability.")

# -------------------------
# 4. Real-time inputs (like TMP_Text fields)
# -------------------------
st.subheader("Relative Position")
px = st.slider("px", -10.0, 10.0, 0.0, step=0.01)
py = st.slider("py", -10.0, 10.0, 0.0, step=0.01)
pz = st.slider("pz", -10.0, 10.0, 0.0, step=0.01)

st.subheader("Relative Rotation (Euler)")
rx = st.slider("rx", -180.0, 180.0, 0.0, step=0.1)
ry = st.slider("ry", -180.0, 180.0, 0.0, step=0.1)
rz = st.slider("rz", -180.0, 180.0, 0.0, step=0.1)

st.subheader("Depth along B-forward")
depth = st.slider("depth", 0.0, 20.0, 5.0, step=0.01)

# Show real-time “text fields”
st.text(f"Position (px, py, pz): ({px:.3f}, {py:.3f}, {pz:.3f})")
st.text(f"Rotation (rx, ry, rz): ({rx:.3f}, {ry:.3f}, {rz:.3f})")
st.text(f"Depth: {depth:.3f}")

# -------------------------
# 5. Predict button (like OnSubmit)
# -------------------------
def predict():
    st.session_state.is_sending = True
    st.session_state.last_result = "Sending..."
    st.rerun()  # update UI immediately

    # simulate network/model delay
    time.sleep(0.5)

    # build input array exactly in order: px, py, pz, rx, ry, rz, depth
    input_data = np.array([[px, py, pz, rx, ry, rz, depth]])
    prob = model.predict_proba(input_data)[0][1]

    st.session_state.last_result = f"Predicted Success Probability: {prob:.3f}"
    st.session_state.is_sending = False

if st.button("Predict Success Probability", disabled=st.session_state.is_sending):
    predict()

# -------------------------
# 6. Show result
# -------------------------
st.subheader("Prediction Result")
st.text(st.session_state.last_result)

# Optional tip
st.info("This app mirrors the Unity client logic: only uses relative position, rotation, and depth.")