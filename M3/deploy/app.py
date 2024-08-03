from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
RF_model = joblib.load('best_RF_model.joblib')

# Route to render the homepage
@app.route('/')
def home():
    return render_template('home.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = RF_model.predict(features)
    response = {
        'prediction': int(prediction[0])
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
