# google_stock_predictor.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yahoo_fin import stock_info as si
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
import datetime
from time import sleep

# Set up Streamlit title and description
st.title("📈 Google Stock Price Prediction")
st.markdown("This app uses RNN, LSTM, and Hybrid models to predict Google (GOOGL) stock prices based on historical data.")

# User input: Select date range and number of days to predict
start_date = st.date_input('📅 Start Date', value=pd.to_datetime('2015-01-01'))
end_date = st.date_input('📅 End Date', value=pd.to_datetime('2024-01-01'))
predict_days = st.number_input("🔮 Number of Days to Predict", min_value=1, max_value=100, value=30)

# User input: Select model
model_option = st.selectbox("🧠 Select Model", ("RNN", "LSTM", "Hybrid (LSTM + RNN)"))

# Validate dates
if start_date >= end_date:
    st.error("❌ Start date must be earlier than end date.")
    st.stop()

# Fetch Google stock data using yahoo_fin with retry
@st.cache_data
def get_yahoo_data_retry(ticker, start_date, end_date, retries=3):
    for _ in range(retries):
        try:
            df = si.get_data(ticker, start_date=start_date, end_date=end_date)
            if not df.empty:
                return df
        except:
            sleep(1)
    return pd.DataFrame()

df = get_yahoo_data_retry("GOOGL", start_date, end_date)

if df.empty:
    st.error("❌ Failed to fetch data from Yahoo Finance. Please try again later or change the date range.")
    st.stop()

df = df[['close']]
df.columns = ['Close']
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

st.write("✅ Data sample (last 5 rows):", df.tail())

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Prepare dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])
        y.append(data[i])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and train model
@st.cache_resource
def build_and_train_model(model_option, X_train, y_train, X_test, y_test):
    model = Sequential()
    if model_option == "RNN":
        model.add(SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(SimpleRNN(50))
    elif model_option == "LSTM":
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50))
    else:
        model.add(SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=0)
    return model, history

model, history = build_and_train_model(model_option, X_train, y_train, X_test, y_test)

# Predict and inverse
y_pred_scaled = model.predict(X_test)
y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_true = scaler.inverse_transform(y_pred_scaled)

# Plot predictions
st.subheader(f"{model_option} vs Actual Stock Prices")
fig, ax = plt.subplots(figsize=(12, 5))
test_dates = df.index[-len(y_test):]
ax.plot(test_dates, y_test_true, label='Actual Price', color='black')
ax.plot(test_dates, y_pred_true, label=f'{model_option} Prediction', color='blue')
ax.set_title(f'{model_option} vs Actual - Google Stock Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Plot training vs validation loss
st.subheader("📉 Training vs Validation Loss")
fig2, ax2 = plt.subplots()
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
st.pyplot(fig2)

# Evaluate model
def evaluate(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

def calculate_accuracy(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    max_price = np.max(y_true)
    accuracy = 100 - ((rmse / max_price) * 100)
    return round(accuracy, 2)

evaluation = evaluate(y_test_true, y_pred_true)
accuracy = calculate_accuracy(y_test_true, y_pred_true)

st.write(f"📊 Evaluation for {model_option} Model:", evaluation)
st.write(f"🎯 Accuracy: {accuracy}%")

# Predict future prices
def predict_future(model, last_sequence, predict_days):
    predictions = []
    current_input = last_sequence.copy()
    for _ in range(predict_days):
        next_pred = model.predict(current_input.reshape(1, time_step, 1))[0][0]
        predictions.append(next_pred)
        current_input = np.append(current_input[1:], [[next_pred]], axis=0)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

future_input = scaled_data[-time_step:]
future_predictions = predict_future(model, future_input, predict_days)
future_dates = pd.date_range(df.index[-1] + datetime.timedelta(1), periods=predict_days)

# Plot future predictions
st.subheader("📈 Future Price Prediction")
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(future_dates, future_predictions, label='Future Predictions', color='green')
ax3.set_title(f"{model_option} Forecast for Next {predict_days} Days")
ax3.set_xlabel('Date')
ax3.set_ylabel('Price (USD)')
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)

# Export predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions.flatten()})
st.download_button("📁 Download Predictions as CSV", data=future_df.to_csv(index=False), file_name="future_predictions.csv", mime='text/csv')

# Reset button
if st.button("🔄 Reset App"):
    st.experimental_rerun()
