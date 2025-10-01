from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Stock API Working!", "status": "active"})

@app.route('/analyze/<symbol>')
def analyze_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1mo")
        
        if hist.empty:
            return jsonify({"error": "No data found"}), 404
        
        current_price = round(hist['Close'].iloc[-1], 2)
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "current_price": current_price,
            "predicted_price": round(current_price * 1.02, 2),
            "predicted_change_percent": 2.0,
            "recommendation": "🟢 BUY",
            "confidence": 7
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
