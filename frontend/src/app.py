import streamlit as st
import requests
import os
import json

# Configuration
API_URL = os.getenv("API_URL", "http://api-serving:8000")

# Page Config
st.set_page_config(
    page_title="Amazon Sentiment Analyzer",
    page_icon="📦",
    layout="centered"
)

# Title and Description
st.title("Amazon Sentiment Analyzer 📦")
st.markdown("Enter a product review below to analyze its sentiment.")

# Input Area
review_text = st.text_area("Enter your review here...", height=150)

# Session State Initialization
if 'prediction_id' not in st.session_state:
    st.session_state.prediction_id = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# Analyze Button
if st.button("Analyze Sentiment", type="primary"):
    if review_text:
        try:
            with st.spinner("Analyzing..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"review": review_text}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.prediction_id = data['prediction_id']
                    st.session_state.prediction_result = data
                    st.session_state.feedback_submitted = False # Reset feedback state
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Is it running?")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text to analyze.")

# Display Results
if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    sentiment = result['sentiment']
    probability = result['probability']
    
    # Determine color and icon
    if sentiment == "POSITIVE":
        color = "green"
        icon = "😊"
    else:
        color = "red"
        icon = "😠"
        
    # Metric Card
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sentiment", f"{icon} {sentiment}")
    with col2:
        st.metric("Confidence", f"{probability:.2%}")
        
    # Feedback Section
    if not st.session_state.feedback_submitted:
        st.markdown("### Was this prediction correct?")
        col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 4])
        
        with col_fb1:
            if st.button("👍 Yes"):
                correct_sentiment = 1 if sentiment == "POSITIVE" else 0
                try:
                    requests.post(
                        f"{API_URL}/feedback",
                        json={
                            "prediction_id": st.session_state.prediction_id,
                            "correct_sentiment": correct_sentiment
                        }
                    )
                    st.session_state.feedback_submitted = True
                    st.success("Thanks! Feedback recorded for model retraining.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to send feedback: {e}")

        with col_fb2:
            if st.button("👎 No"):
                # Invert sentiment
                correct_sentiment = 0 if sentiment == "POSITIVE" else 1
                try:
                    requests.post(
                        f"{API_URL}/feedback",
                        json={
                            "prediction_id": st.session_state.prediction_id,
                            "correct_sentiment": correct_sentiment
                        }
                    )
                    st.session_state.feedback_submitted = True
                    st.success("Thanks! Feedback recorded for model retraining.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to send feedback: {e}")
    else:
        st.info("Feedback received. Thank you!")
