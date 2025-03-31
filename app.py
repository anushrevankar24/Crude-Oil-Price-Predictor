import streamlit as st
import numpy as np
import pandas as pd
import joblib
import config
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import calendar

st.set_page_config(**config.PAGE_CONFIG)
st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)

# Load dataset
DATASET_PATH = "wti-dataset.csv"
df = pd.read_csv(DATASET_PATH, thousands=',', index_col='Date', parse_dates=['Date'], date_format="%Y-%m-%d")
df.sort_index(inplace=True)
num_cols = df.columns.tolist()
data = df[num_cols].values

# Load scaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Define constants
N_INPUT = 12  # Fixed lookback period
LSTM_MODEL_PATH = "lstm_model.h5"
XGB_MODEL_PATH = "lstm_multi_xgb_model.pkl"

# Load models
lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
xgb_model = joblib.load(XGB_MODEL_PATH)

# Create LSTM feature extractor
feature_extractor = Model(inputs=lstm_model.input, outputs=lstm_model.get_layer("lstm2").output)

def get_latest_sequence(data, n_input):
    """Extract the last `n_input` steps from the dataset."""
    return np.array([data[-n_input:]])

def predict(n_output):
    """Make predictions using the LSTM and XGBoost models, limited to `n_output` steps."""
    latest_sequence = get_latest_sequence(scaled_data, N_INPUT)
    lstm_features = feature_extractor.predict(latest_sequence)
    xgb_predictions = xgb_model.predict(lstm_features)
    
    # Ensure predictions are limited to 5 steps max
    max_predictions = 5  
    xgb_predictions = xgb_predictions[:max_predictions]  # Always take only the first 5
    
    # Prepare scaled predictions
    scaled_predictions = np.zeros((max_predictions, data.shape[1]))
    scaled_predictions[:, 0] = xgb_predictions.reshape(-1)
    
    # Inverse transform
    final_predictions = scaler.inverse_transform(scaled_predictions)[:, 0]
    
    return final_predictions[:n_output]  # Return only first `n_output` values

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "About", "Methodology", "Results"])

if page == "Home":
    st.markdown('<div class="main-title">Department of Information Technology</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">National Institute of Technology Karnataka, Surathkal</div>', unsafe_allow_html=True)
    st.markdown('<div class="header">Deep Learning IT353 Course Project Title : Predict WTI Crude Spot Oil Price</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-sub-title">By <strong>Anush Revankar</strong> and <strong>Aishini Bhattacharjee</strong> under the guidance of Prof. <a href="https://infotech.nitk.ac.in/faculty/jaidhar-c-d" target="_blank">Jaidhar C.D</a></div>', unsafe_allow_html=True)
    
    cols = st.columns([1, 2, 1])  
    
    with cols[1]:
            n_output = st.number_input("Enter the number of months for prediction (1-5):", min_value=1, max_value=5, value=5, step=1)

            if st.button("Predict"):
                predictions = predict(n_output)
                
                current_date = datetime.today()

                # Create prediction HTML content
                prediction_content = "<div class='prediction-container'><h3>Predictions (in dollars per barrel):</h3>"
                for i, pred in enumerate(predictions, start=1):
                    # Calculate future month
                    future_month = (current_date.month + i) % 12
                    future_year = current_date.year + ((current_date.month + i - 1) // 12)

                    # Handle case where future_month becomes 0 (i.e., December + 1 should be January)
                    future_month = 12 if future_month == 0 else future_month

                    month_name = calendar.month_name[future_month]
                    
                    prediction_content += f"<p class='prediction-text'><strong>{month_name} {future_year}:</strong> ${pred:.2f}</p>"
                    
                prediction_content += "</div>"

                # Display the container
                st.markdown(prediction_content, unsafe_allow_html=True)

elif page == "About":
    st.header("About the Project")
    st.write("This project predicts WTI Crude Oil Spot Prices using Deep Learning and Machine Learning models...")
    
elif page == "Methodology":
    st.header("Methodology")
    st.write("The prediction model uses LSTM for feature extraction and XGBoost for final price prediction...")
    
elif page == "Results":
    st.header("Results")
    st.write("The results showcase the accuracy and effectiveness of the hybrid LSTM-XGBoost model in predicting oil prices...")