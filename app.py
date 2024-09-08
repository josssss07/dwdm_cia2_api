import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dense
from textblob import TextBlob
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def get_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df[['Close']]

def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

def lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=30, batch_size=32)
    return model

def random_forest_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def svm_model(X_train, y_train):
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    return model

def get_sentiment(symbol):
    stock = yf.Ticker(symbol)
    news = stock.news
    if news:
        recent_news = news[0]['title']
        sentiment = TextBlob(recent_news).sentiment.polarity
        if sentiment > 0:
            return "Positive", sentiment
        elif sentiment < 0:
            return "Negative", sentiment
        else:
            return "Neutral", sentiment
    else:
        return "No recent news", 0

def generate_chart(data, prediction):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-30:], data['Close'].values[-30:], label='Actual')
    plt.plot(data.index[-1:], prediction, 'ro', label='Prediction')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    return chart_url

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['symbol']
        model_type = request.form['model_type']
        
        data = get_stock_data(symbol, '2020-01-01', '2023-12-31')
        X, y, scaler = prepare_data(data)
        X_train, y_train = X[:-1], y[:-1]
        
        if model_type == 'lstm':
            model = lstm_model(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train)
            prediction = model.predict(X[-1].reshape((1, X.shape[1], 1)))
        elif model_type == 'random_forest':
            model = random_forest_model(X_train, y_train)
            prediction = model.predict(X[-1].reshape(1, -1))
        else:  # SVM
            model = svm_model(X_train, y_train)
            prediction = model.predict(X[-1].reshape(1, -1))
        
        prediction = scaler.inverse_transform(prediction.reshape(1, -1))[0, 0]
        last_price = data['Close'].values[-1]
        
        sentiment, sentiment_score = get_sentiment(symbol)
        chart_url = generate_chart(data, prediction)
        
        if prediction > last_price:
            signal = "Buy"
        elif prediction < last_price:
            signal = "Sell"
        else:
            signal = "Hold"
        
        return render_template('result.html', 
                               symbol=symbol, 
                               prediction=prediction, 
                               last_price=last_price,
                               sentiment=sentiment,
                               sentiment_score=sentiment_score,
                               signal=signal,
                               chart_url=chart_url)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)