"""Disaster Response API for message classification."""

import os
import re
import logging
import joblib
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NLTK Resources
nltk_resources = ['punkt', 'wordnet', 'omw-1.4']
resource_paths = {
    'punkt': 'tokenizers/punkt',
    'wordnet': 'corpora/wordnet',
    'omw-1.4': 'corpora/omw-1.4'
}
for resource in nltk_resources:
    try:
        nltk.data.find(resource_paths[resource])
    except LookupError:
        logger.info(f"⬇️ Downloading NLTK resource: {resource}")
        nltk.download(resource, quiet=True)

# Custom Transformers
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return pd.DataFrame([len(str(text)) for text in X])

class WordCountExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return pd.DataFrame([len(word_tokenize(str(text))) for text in X])

# Text Processing
url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
def tokenize(text):
    text = re.sub(url_regex, "urlplaceholder", str(text))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(tok).lower() for tok in tokens]

# Model Loading
BASE_PATH = os.getenv("MODEL_BASE_PATH", "saved_models")
MODEL_NAMES = ["catboost", "lgbm", "xgb", "voting_classifier"]
LOADED_MODELS = {}
first_categories = None

logger.info("🔍 Loading models...")
if not os.path.exists(BASE_PATH):
    logger.error(f"❌ Model directory {BASE_PATH} does not exist.")
else:
    for model_name in MODEL_NAMES:
        try:
            model_dir = os.path.join(BASE_PATH, model_name)
            pipeline_path = os.path.join(model_dir, f"{model_name}_pipeline.pkl")
            categories_path = os.path.join(model_dir, f"{model_name}_categories.pkl")

            if os.path.exists(pipeline_path) and os.path.exists(categories_path):
                pipeline = joblib.load(pipeline_path)
                categories = joblib.load(categories_path)

                if first_categories is None:
                    first_categories = categories
                elif set(categories) != set(first_categories):
                    logger.error(f"❌ Categories for {model_name} do not match expected categories.")
                    continue

                LOADED_MODELS[model_name] = {"pipeline": pipeline, "categories": categories}
                logger.info(f"✅ Loaded: {model_name}")
            else:
                logger.warning(f"⚠️ Model files not found for {model_name} in {model_dir}. Skipping.")
        except (joblib.JoblibInvalidPickleError, FileNotFoundError, AttributeError) as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")

if not LOADED_MODELS:
    logger.warning("‼️ No models were loaded. The API might not function as expected.")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if not LOADED_MODELS:
        logger.error("❌ No models available for prediction.")
        return jsonify({"error": "No models available for prediction."}), 500

    json_input = request.get_json()
    if not json_input or "message" not in json_input:
        logger.warning("⚠️ Invalid input received.")
        return jsonify({"error": "Invalid input: 'message' field is required in JSON payload."}), 400

    message = json_input.get("message", "").strip()
    if not message:
        logger.warning("⚠️ Empty message received.")
        return jsonify({"error": "Empty message received."}), 400

    test_input = pd.Series([message])
    THRESHOLD = 0.5
    output_data = {name: [] for name in MODEL_NAMES}

    for model_name, model_assets in LOADED_MODELS.items():
        try:
            probas = model_assets["pipeline"].predict_proba(test_input)
            categories = model_assets["categories"]

            if isinstance(probas, list):
                probas = np.array([p[0][1] if len(p[0]) > 1 else p[0][0] for p in probas])
            else:
                probas = probas[0]

            model_output = []
            for cat, prob in zip(categories, probas):
                if prob >= THRESHOLD:
                    model_output.append({"category": cat, "confidence": float(round(prob, 3))})
            output_data[model_name] = model_output
        except Exception as e:
            logger.error(f"❌ Error predicting probabilities for {model_name}: {e}")

    logger.info("✅ Prediction completed successfully.")
    return jsonify(output_data)

# Application Entry Point
if __name__ == '__main__':
    port = int(os.getenv("API_PORT", 9000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    logger.info(f"🚀 Starting Flask app on port {port} with debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
