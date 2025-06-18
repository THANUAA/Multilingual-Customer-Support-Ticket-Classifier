import os
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model, vectorizer, and label encoder
model = joblib.load('models/ticket_classifier.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'Empty input'}), 400
    
    # Preprocess like in training
    text = text.lower()
    import re, string
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    # Vectorize & predict
    vect_text = vectorizer.transform([text])
    pred_encoded = model.predict(vect_text)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return jsonify({'prediction': pred_label})

if __name__ == '__main__':
    # Run on port 5000 or environment PORT if set
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
