from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define numeric columns (the ones you scaled)
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    data = request.json  # receive JSON input
    df = pd.DataFrame([data])  # convert to DataFrame

    # Scale numeric features
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Make prediction
    churn_prob = model.predict_proba(df)[:,1][0]  # probability of churn
    churn_pred = int(churn_prob > 0.5)  # optional: 0/1 prediction

    return jsonify({
        'churn_probability': float(churn_prob),
        'churn_prediction': churn_pred
    })

if __name__ == '__main__':
    app.run(debug=True)