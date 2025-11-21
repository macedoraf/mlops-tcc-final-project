import streamlit as st
import requests
import os

# Configuration
API_URL = os.getenv("API_URL", "http://api-serving:8000")

st.set_page_config(page_title="Sentiment Analysis App", page_icon="🤖")

st.title("🤖 Sentiment Analysis Classifier")
st.markdown("Enter a review below to analyze its sentiment.")

# Input
review_text = st.text_area("Review Text", height=150, placeholder="Type your review here...")

if st.button("Analyze Sentiment"):
    if not review_text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            with st.spinner("Analyzing..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"review": review_text, "language": "pt"}
                )
                
            if response.status_code == 200:
                result = response.json()
                sentiment = result.get("sentiment")
                probability = result.get("probability")
                version = result.get("model_version")
                
                # Display results
                st.success("Analysis Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sentiment", sentiment)
                
                with col2:
                    st.metric("Confidence", f"{probability:.2%}")
                    
                with col3:
                    st.metric("Model Version", version)
                
                # Visual indicator
                if sentiment == "Positive":
                    st.balloons()
                
                st.json(result)
                
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to backend at {API_URL}. Is it running?")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("This app demonstrates an End-to-End MLOps pipeline.")
st.sidebar.markdown(f"**Backend URL:** `{API_URL}`")

if st.sidebar.button("Check API Health"):
    try:
        health = requests.get(f"{API_URL}/health").json()
        st.sidebar.success(f"Status: {health.get('status')}")
        st.sidebar.json(health)
    except Exception as e:
        st.sidebar.error("API Offline")
