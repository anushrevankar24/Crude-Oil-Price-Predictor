# WTI Crude Oil Price Predictor

A deep learning and machine learning hybrid model for predicting West Texas Intermediate (WTI) crude oil spot prices developed at the Department of Information Technology, National Institute of Technology Karnataka, Surathkal.

## Project Overview

This application provides monthly forecasts of WTI crude oil spot prices using a hybrid deep learning approach that combines LSTM (Long Short-Term Memory) networks with XGBoost regression. The model can predict prices for up to 5 months into the future.

## 🔍 Features

- Monthly WTI crude oil price forecasting for 1-5 months ahead
- Interactive web application built with Streamlit
- Hybrid model combining neural networks with gradient boosting
- Responsive UI with intuitive controls

## 🌐 Live Deployment

The application is deployed and accessible online at:

🔗 **[Live Demo: WTI Crude Oil Price Predictor](https://crudeoilpricepredictor.streamlit.app/)**

This Streamlit-based web application allows users to interactively forecast crude oil prices based on historical data. Users can select the forecast period (1-5 months) and visualize predicted trends.


## 📊 Dataset

The project uses historical WTI crude oil price data stored in `wti-dataset.csv`. The data includes:
- Time-series price data indexed by date
- US Dollar Index
- Gold Price (USD)
- Index of Global Economic Activity
- 10-Year Bond Yield for the USA

## 🧠 Methodology

### Model Architecture

The prediction system employs a two-stage hybrid approach:

1. **Feature Extraction with LSTM**: 
   - Bidirectional LSTM followed by standard LSTM layers
   - Captures temporal patterns and dependencies in the time series data
   - Uses a fixed lookback period of 12 months

2. **Prediction with XGBoost**:
   - Takes features extracted by the LSTM as input
   - Trained to predict up to 5 months of future prices
   - Hyperparameter-optimized using GridSearchCV

### Implementation Details

- The LSTM model uses two stacked layers (one bidirectional) with dropout for regularization
- Feature extraction comes from the second LSTM layer with 100 units
- XGBoost is implemented as a MultiOutputRegressor to handle multi-step forecasting

## 📈 Model Performance

The hybrid LSTM-XGBoost model was evaluated using several metrics:

| Metric | Value |
|--------|-------|
| MAE    | 0.0808 |
| MSE    | 0.0099 |
| RMSE   | 0.0997 |
| R²     | 0.5319 |
| MAPE   | 26.79% |

### Significance of Results

- **R² Score of 0.5319**: The model explains approximately 53% of the variance in future oil prices, which is noteworthy given the inherent volatility and unpredictability of oil markets.
- **RMSE of 0.0997**: On normalized data, this indicates reasonable prediction accuracy.
- **MAPE of 26.79%**: While this percentage error may seem high, it's actually competitive for multi-step commodity price forecasting, which is notoriously difficult.
- **Cross-Validation MSE of 0.0225**: The low cross-validation MSE suggests the model generalizes well to unseen data.

The best hyperparameters found through GridSearchCV were:
- Learning rate: 0.05
- Maximum depth: 7
- Number of estimators: 300
- Subsample ratio: 1.0

## 🚀 Getting Started

### Prerequisites

Required Python packages are listed in `requirements.txt` and include:

```
joblib==1.3.2
keras==3.9.1
numpy==1.26.4
pandas==2.2.2
pydantic==2.9.2
pydantic_core==2.23.4
python-dateutil==2.8.2
python-decouple==3.8
python-dotenv==1.0.1
scikit-learn==1.6.1
streamlit==1.44.0
tensorflow==2.17.0
xgboost==3.0.0
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anushrevankar24/Crude-Oil-Price-Predictor.git
   cd Crude-Oil-Price-Predictor
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 🖥 Application Interface

The application features a user-friendly interface with four main sections:
- **Home**: Enter the number of months for prediction (1-5) and view results
- **About**: Information about the project goals and background
- **Methodology**: Details about the technical approach used
- **Results**: Performance evaluation and model comparison


## 👥 Contributors

This project was developed by **Anush Revankar** and **Aishini Bhattacharjee**



