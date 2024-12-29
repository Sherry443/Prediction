# Install required libraries (Uncomment when running locally)
# !pip install streamlit
# !pip install xgboost
# !pip install pandas
# !pip install scikit-learn
# !pip install tensorflow

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# ===========================
# App Title
# ===========================
st.title("1X Multiplier Prediction Bot")
st.write("Predict multiplier values using Neural Networks.")

# ===========================
# File Upload
# ===========================
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # ===========================
    # Preprocessing
    # ===========================
    if "Time" in data.columns:
        data = data.drop("Time", axis=1)

    # Define features and target
    X = data[['Number of players', 'Total bets']]
    y = data['Multiplier']

    # ===========================
    # Input Recent Data Points
    # ===========================
    st.sidebar.header("Add Recent Data")
    st.sidebar.write("Enter 5 recent data points:")
    recent_values = []
    for i in range(5):
        st.sidebar.write(f"Data Point {i + 1}:")
        num_players = st.sidebar.number_input(f"Number of players ({i+1})", value=0.0, step=1.0, key=f"players_{i}")
        total_bets = st.sidebar.number_input(f"Total bets ({i+1})", value=0.0, step=1.0, key=f"bets_{i}")
        multiplier = st.sidebar.number_input(f"Multiplier ({i+1})", value=0.0, step=0.1, key=f"multiplier_{i}")
        recent_values.append([num_players, total_bets, multiplier])

    # Merge recent data with the dataset
    recent_df = pd.DataFrame(recent_values, columns=['Number of players', 'Total bets', 'Multiplier'])
    data = pd.concat([data, recent_df], ignore_index=True)

    # Split data
    X = data[['Number of players', 'Total bets']]
    y = data['Multiplier']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ===========================
    # Model Training
    # ===========================
    # XGBoost
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Neural Network
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nn_model = Sequential([
        Dense(128, input_dim=2, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer
    ])
    nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

    # ===========================
    # Predictions
    # ===========================
    st.header("Make a Prediction for 1XBET")
    new_num_players = st.number_input("Enter Number of Players:", value=0.0, step=1.0)
    new_total_bets = st.number_input("Enter Total Bets:", value=0.0, step=1.0)

    if st.button("Predict Multiplier"):
        new_data = pd.DataFrame([[new_num_players, new_total_bets]], columns=['Number of players', 'Total bets'])
        new_data_scaled = scaler.transform(new_data)

        # XGBoost Prediction
        y_new_pred_xgb = xgb_model.predict(new_data)[0]

        # Neural Network Prediction
        y_new_pred_nn = nn_model.predict(new_data_scaled)[0][0]

        # Display Results
        st.write(f"**1st Prediction:** {y_new_pred_xgb:.2f}")
        st.write(f"**2nd Prediction:** {y_new_pred_nn:.2f}")

    # ===========================
    # Model Evaluation
    # ===========================
    # st.header("Model Performance")

    # XGBoost
    y_pred_xgb = xgb_model.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    # Neural Network
    y_pred_nn = nn_model.predict(X_test_scaled)
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)

    # st.write("### XGBoost")
    # st.write(f"Mean Squared Error: {mse_xgb:.2f}")
    # st.write(f"R² Score: {r2_xgb:.2f}")

    # st.write("### Neural Network")
    # st.write(f"Mean Squared Error: {mse_nn:.2f}")
    # st.write(f"R² Score: {r2_nn:.2f}")

