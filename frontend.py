import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Session State
for key, default in {
    "message": "",
    "predictions": None,
    "last_updated": None,
    "threshold": 0.7,
    "example_index_disaster": 0,
    "example_index_nondisaster": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Example Messages
disaster_examples = [
    "URGENT: Wildfire approaching residential area. Evacuate now!",
    "Earthquake reported. Emergency response needed.",
    "Flooding in downtown. Rescue boats deployed.",
    "Injured people found after landslide. Requesting medical aid.",
    "Storm has knocked down power lines. Danger in the area."
]
nondisaster_examples = [
    "Had an amazing trip to the mountains!",
    "Movie night with friends was so fun.",
    "Going grocery shopping later today.",
    "Just finished reading a great novel.",
    "Looking forward to the weekend getaway."
]

st.set_page_config(page_title="Disaster Classifier", page_icon="🚨", layout="centered")

# Styling
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: #e6e6e6;
}
.block-container {
    max-width: 1000px;
    padding: 2rem;
}
.hero-banner {
    background: linear-gradient(135deg, #1a1f2e 0%, #2a3f5f 100%);
    color: #fff;
    padding: 1.5rem 1rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 2rem;
}
.hero-banner h2 {
    font-size: 1.8rem;
    font-weight: 700;
}
.stTextArea textarea {
    font-size: 1rem;
    background-color: #2d3748;
    color: #e6e6e6;
}
.stButton>button {
    border-radius: 6px;
    padding: .5rem 1.2rem;
    font-weight: 600;
}
.stButton>button:has-text("Predict") {
    background-color: #dc2626 !important;
    color: white !important;
    border: none;
    font-weight: bold;
    box-shadow: 0 0 0 3px rgba(255, 0, 0, 0.2);
}
.suggestion-title {
    color: #d1d5db;
    font-size: 14px;
    margin: 10px 0 5px 0;
}
.col-separator {
    border-left: 1px solid #334155;
    height: 100%;
    padding-left: 1rem;
}
.footer {
    text-align: center;
    font-size: .85rem;
    color: #6b7280;
    margin-top: 2rem;
    padding: 1.5rem 0;
    border-top: 1px solid #4a5568;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="hero-banner">
        <h2>Disaster Message Classifier</h2>
        <p>Classify disaster-related messages into actionable categories with confidence scores</p>
    </div>
""", unsafe_allow_html=True)

# Message Input
st.session_state.message = st.text_area(
    "✍️ Enter a disaster-related message:",
    value=st.session_state.message,
    height=150,
    placeholder="e.g., 'Flood warning issued for downtown area'",
    key="message_input",
)

# Layout
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<p class="suggestion-title">Try these examples:</p>', unsafe_allow_html=True)
    
    if st.button("🌪️ Disaster Example", use_container_width=True):
        idx = st.session_state.example_index_disaster
        st.session_state.message = disaster_examples[idx]
        st.session_state.example_index_disaster = (idx + 1) % len(disaster_examples)
        st.rerun()

    if st.button("☀️ Non-Disaster Example", use_container_width=True):
        idx = st.session_state.example_index_nondisaster
        st.session_state.message = nondisaster_examples[idx]
        st.session_state.example_index_nondisaster = (idx + 1) % len(nondisaster_examples)
        st.rerun()

with right_col:
    st.markdown('<div class="col-separator">', unsafe_allow_html=True)
    st.markdown("#### 🎚️ Confidence & Actions")
    st.session_state.threshold = st.slider("Confidence Threshold", 0.0, 1.0, st.session_state.threshold, 0.01)

    c1, c2 = st.columns(2)
    with c1:
        clear_button = st.button("🧹 Clear", use_container_width=True)
    with c2:
        predict_button = st.button("🚨 Predict", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Clear Input
if clear_button:
    st.session_state.message = ""
    st.session_state.predictions = None
    st.session_state.last_updated = None
    st.rerun()

# Prediction
if predict_button and st.session_state.message:
    with st.spinner("🔄 Running analysis, please wait..."):
        try:
            response = requests.post(
                "http://127.0.0.1:9000/predict",
                json={"message": st.session_state.message},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            all_predictions = []
            for model_name, predictions in result.items():
                if isinstance(predictions, list):
                    df = pd.DataFrame(predictions)
                    if "confidence" in df.columns:
                        category_col = next((col for col in ["label", "category", "prediction", "class"]
                                             if col in df.columns), None)
                        if category_col:
                            df = df[[category_col, "confidence"]].copy()
                            df.rename(columns={category_col: "category"}, inplace=True)
                            df["model"] = model_name
                            df["confidence"] = df["confidence"].astype(float).round(3)
                            all_predictions.append(df)

            if all_predictions:
                st.session_state.predictions = pd.concat(all_predictions, ignore_index=True)
                st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                st.session_state.predictions = None
                st.info("No valid predictions were returned.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Results
if st.session_state.predictions is not None:
    filtered_df = st.session_state.predictions[
        (st.session_state.predictions["confidence"] >= st.session_state.threshold) &
        (st.session_state.predictions["category"].str.lower() != "related")
    ].sort_values(by="confidence", ascending=False)

    st.markdown("### 📊 Prediction Results")
    if not filtered_df.empty:
        top_category = filtered_df.iloc[0]["category"]
        avg_confidence = filtered_df["confidence"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(filtered_df))
        col2.metric("Average Confidence", f"{avg_confidence:.1%}")
        col3.metric("Top Category", top_category)  # 🔥 No emoji

        filtered_df["confidence_pct"] = (filtered_df["confidence"] * 100).round(1)

        st.dataframe(
            filtered_df[["model", "category", "confidence_pct"]],
            column_config={
                "model": "Model",
                "category": "Category",
                "confidence_pct": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
            hide_index=True,
            use_container_width=True
        )
        st.caption(f"Last updated: {st.session_state.last_updated}")



    else:
        st.info("No categories met the confidence threshold.")
    st.markdown(f"""
        <div class="footer">
            <p>Disaster Classifier © {datetime.now().year}</p>
        </div>
        """, unsafe_allow_html=True)