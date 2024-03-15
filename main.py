from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import joblib
import numpy as np 

app = Flask(__name__)

model = load_model('./content/model')
scale = joblib.load('./scale.pkl')
stand = joblib.load('./stand.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        differences = np.array(data['differences']).reshape(1, -1)
        scale_data1 = scale.transform(differences)
        scale_data2 = stand.transform(scale_data1)
        prediction = model.predict(scale_data2)

        return jsonify({'prediction': str(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
