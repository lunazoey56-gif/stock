from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import os

app = Flask(__name__)
CORS(app)

class StockAnalyzer:
    def __init__(self):
        self.model = LinearRegression()
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for stock analysis"""
        df = df.copy()
        
        # Simple Moving Average (5-day)
        df['sma_5'] = df['close'].rolling(window=5).mean()
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Price vs Moving Average
        df['price_vs_sma'] = (df['close'] - df['sma_5']) / df['sma_5']
        
        return df.dropna()
    
    def prepare_features(self, df, lookback_days=5):
        """Prepare features for machine learning model"""
        df = self.calculate_technical_indicators(df)
        
        features = []
        targets = []
        
        for i in range(lookback_days, len(df)-1):
            recent_data = df.iloc[i-lookback_days:i]
            
            feature_set = [
                recent_data['price_change'].mean(),
                recent_data['volume_change'].mean(),
                recent_data['price_vs_sma'].mean(),
                df.iloc[i]['close']
            ]
            
            target_price = df.iloc[i+1]['close']
            
            features.append(feature_set)
            targets.append(target_price)
        
        return np.array(features), np.array(targets)
    
    def train_and_predict(self, df):
        """Train ML model and make prediction"""
        try:
            features, targets = self.prepare_features(df)
            
            if len(features) < 10:
                return None, "Not enough data for reliable prediction"
            
            # Split data (80% train, 20% test)
            split_idx = int(0.8 * len(features))
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = targets[:split_idx], targets[split_idx:]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Test model accuracy
            predictions = self.model.predict(X_test)
            accuracy = mean_absolute_error(y_test, predictions)
            
            # Predict tomorrow's price
            latest_features = features[-1].reshape(1, -1)
            tomorrow_prediction = self.model.predict(latest_features)[0]
            current_price = df['close'].iloc[-1]
            
            # Calculate predicted change
            predicted_change = ((tomorrow_prediction - current_price) / current_price) * 100
            
            result = {
                'current_price': float(round(current_price, 2)),
                'predicted_price': float(round(tomorrow_prediction, 2)),
                'predicted_change_percent': float(round(predicted_change, 2)),
                'model_accuracy': float(round(accuracy, 2)),
                'direction': 'UP' if predicted_change > 0 else 'DOWN'
            }
            
            return result, "Success"
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def analyze_trend(self, df):
        """Analyze stock trend over last 10 days"""
        recent_prices = df['close'].tail(10)
        
        # Calculate 10-day price change
        price_change = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100
        
        # Classify trend
        if price_change > 5:
            trend = "STRONG UPTREND 📈"
        elif price_change > 2:
            trend = "UPTREND ↗️"
        elif price_change > -2:
            trend = "SIDEWAYS ↔️"
        elif price_change > -5:
            trend = "DOWNTREND ↘️"
        else:
            trend = "STRONG DOWNTREND 📉"
        
        # Volume analysis
        volume_trend = 'HIGH' if df['volume'].iloc[-1] > df['volume'].mean() else 'NORMAL'
        
        return {
            'trend': trend,
            'price_change_10d': float(round(price_change, 2)),
            'current_volume': int(df['volume'].iloc[-1]),
            'volume_trend': volume_trend
        }
    
    def get_trading_recommendation(self, prediction_data, trend_data):
        """Generate BUY/SELL/HOLD recommendation"""
        pred_change = prediction_data['predicted_change_percent']
        accuracy = prediction_data['model_accuracy']
        trend = trend_data['trend']
        
        # Calculate confidence (1-10)
        confidence = max(1, min(10, 10 - (accuracy / prediction_data['current_price'] * 50)))
        
        # Recommendation logic
        if pred_change > 1.5 and confidence > 7 and "UPTREND" in trend:
            return "🟢 STRONG BUY", int(confidence)
        elif pred_change > 0.5 and confidence > 5:
            return "🟢 BUY", int(confidence)
        elif pred_change < -2.0 and confidence > 7 and "DOWNTREND" in trend:
            return "🔴 STRONG SELL", int(confidence)
        elif pred_change < -1.0 and confidence > 5:
            return "🔴 SELL", int(confidence)
        else:
            return "🟡 HOLD", int(confidence)

@app.route('/')
def home():
    """Home endpoint - API status"""
    return jsonify({
        "message": "🤖 Stock Prediction API is Live!", 
        "status": "active",
        "endpoints": {
            "analyze_stock": "/analyze/<symbol>",
            "example": "/analyze/AAPL"
        }
    })

@app.route('/analyze/<symbol>')
def analyze_stock(symbol):
    """Analyze a stock and return predictions"""
    try:
        # Validate symbol
        if not symbol or len(symbol) > 10:
            return jsonify({"error": "Invalid stock symbol"}), 400
        
        print(f"📊 Analyzing {symbol}...")
        
        # Fetch stock data from Yahoo Finance
        stock_data = yf.download(symbol, period='3mo', progress=False)
        
        if stock_data.empty:
            return jsonify({"error": f"No data found for symbol: {symbol}"}), 404
        
        # Prepare data
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        stock_data.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Remove any rows with missing data
        stock_data = stock_data.dropna()
        
        if len(stock_data) < 20:
            return jsonify({"error": "Not enough historical data for analysis"}), 400
        
        # Analyze stock
        analyzer = StockAnalyzer()
        prediction, status = analyzer.train_and_predict(stock_data)
        
        if prediction is None:
            return jsonify({"error": status}), 400
            
        # Get trend analysis
        trend = analyzer.analyze_trend(stock_data)
        
        # Get trading recommendation
        recommendation, confidence = analyzer.get_trading_recommendation(prediction, trend)
        
        # Return complete analysis
        return jsonify({
            "success": True,
            "symbol": symbol,
            "prediction": prediction,
            "trend": trend,
            "recommendation": recommendation,
            "confidence": confidence,
            "data_points": len(stock_data)
        })
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()})

if __name__ == '__main__':
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
