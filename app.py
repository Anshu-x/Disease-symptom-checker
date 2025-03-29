from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load trained model and vectorizer
with open("knn_model.pkl", "rb") as model_file:
    knn_classifier = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

@app.route("/")
def home():
    return "Disease Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", "")

        if not symptoms:
            return jsonify({"error": "The 'symptoms' field is required."}), 400

        # Preprocess symptoms
        preprocessed_symptom = preprocess_text(symptoms)
        symptom_tfidf = tfidf_vectorizer.transform([preprocessed_symptom])

        # Predict the disease
        predicted_disease = knn_classifier.predict(symptom_tfidf)[0]
        return jsonify({"Predicted Disease": predicted_disease})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)