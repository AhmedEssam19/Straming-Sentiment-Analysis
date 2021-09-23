from flask import Flask, request, jsonify
from model_wrapper import ModelWrapper

app = Flask(__name__)
print('Loading Model...')
model = ModelWrapper()
print('Model Loaded Successfully.')


@app.route('/predict_polarity', methods=['POST'])
def predict_polarity():
    if request.method == 'POST':
        text = request.form.get('text')

        if text is None:
            return jsonify({
                'prediction': None
            })

        return jsonify({
            'prediction': model.predict(text)
        })