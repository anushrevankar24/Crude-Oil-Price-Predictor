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
max_predictions = 5  

# Create LSTM feature extractor
feature_extractor = Model(inputs=lstm_model.input, outputs=lstm_model.get_layer("lstm2").output)

def get_latest_sequence(data, n_input):
    return np.array([data[-n_input:]])

def predict(n_output):
    latest_sequence = get_latest_sequence(scaled_data, N_INPUT)
    lstm_features = feature_extractor.predict(latest_sequence)
    xgb_predictions = xgb_model.predict(lstm_features)
    xgb_predictions = xgb_predictions[:max_predictions]  # Always take only the first 5
    
    # Prepare scaled predictions
    scaled_predictions = np.zeros((max_predictions, data.shape[1]))
    scaled_predictions[:, 0] = xgb_predictions.reshape(-1)
    
    # Inverse transform
    final_predictions = scaler.inverse_transform(scaled_predictions)[:, 0]
     
    return final_predictions[:n_output]  

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "About", "Dataset", "Results","Tech Stack"])

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
                

# About Page
if page == "About":
    st.markdown('<div class="title">About the Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section"><p class="content">'
        'The <b>WTI Crude Oil Price Predictor</b> is a hybrid deep learning and machine learning model that forecasts crude oil prices for up to five months. '
        'Developed at the Department of Information Technology, NITK Surathkal, the model integrates <b>LSTM networks</b> and <b>XGBoost regression</b> to provide accurate predictions. '
        'An interactive web app built using <b>Streamlit</b> enables easy visualization and user-friendly forecasting.</p></div>',
        unsafe_allow_html=True,
    )

elif page == "Dataset":
    st.markdown('<div class="title">Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section"><p class="content">'
        'The model is trained on historical WTI crude oil price data, including economic indicators such as:</p>'
        '<ul class="content">'
        '<li><b>WTI Spot Prices</b> (Target variable)</li>'
        '<li><b>US Dollar Index</b></li>'
        '<li><b>Gold Prices</b></li>'
        '<li><b>Index of Global Economic Activity</b></li>'
        '<li><b>10-Year US Bond Yield</b></li>'
        '</ul>'
        '<p class="content">Data is preprocessed with <b>MinMaxScaler</b> and structured using a <b>sliding window approach</b> to enhance model training.</p></div>',
        unsafe_allow_html=True,
    )

# Results Page
elif page == "Results":
    st.markdown('<div class="title">Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section"><p class="content">'
        'The hybrid LSTM-XGBoost model achieves strong forecasting performance, balancing temporal sequence modeling with robust regression.</p>'
        '<div class="metric-box">R² Score: 0.5319</div>'
        '<div class="metric-box">MAE: 0.0808</div>'
        '<div class="metric-box">MSE: 0.0099</div>'
        '<div class="metric-box">RMSE: 0.0997</div>'
        '<div class="metric-box">MAPE: 26.79%</div>'
        '<p class="content">The model’s competitive performance reflects its ability to capture trends despite the volatility of oil prices.</p></div>',
        unsafe_allow_html=True,
    )

# Tech Stack Page
elif page == "Tech Stack":
    st.markdown('<div class="title">Tech Stack</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section"><p class="content">'
        '<b>Machine Learning & Deep Learning:</b>'
        '<ul class="content">'
        '<li><b>TensorFlow/Keras</b> – LSTM-based feature extraction</li>'
        '<li><b>XGBoost</b> – Multi-output regression</li>'
        '<li><b>scikit-learn</b> – Data preprocessing & evaluation</li>'
        '</ul>'
        '<b>Data Processing:</b>'
        '<ul class="content">'
        '<li><b>Pandas & NumPy</b> – Data manipulation</li>'
        '<li><b>MinMaxScaler</b> – Feature scaling</li>'
        '</ul>'
        '<b>Web Application:</b>'
        '<ul class="content">'
        '<li><b>Streamlit</b> – UI and visualization</li>'
        '<li><b>Matplotlib/Seaborn</b> – Graphing and charts</li>'
        '</ul>'
        '<p class="content">This stack ensures efficient forecasting, seamless interaction, and insightful visualization.</p></div>',
        unsafe_allow_html=True,
    )
                

