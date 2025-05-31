"""Disaster Response Model Training (CatBoost, XGBoost, LightGBM with VotingClassifier)"""
# %pip install -q scikit-learn joblib nltk pandas numpy sqlalchemy xgboost catboost lightgbm

# Imports
import os, re, joblib, logging, subprocess
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'punkt_tab', 'wordnet', 'omw-1.4'], quiet=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Custom Transformers
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return pd.DataFrame([len(text) for text in X])

class WordCountExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return pd.DataFrame([len(text.split()) for text in X])

# Tokenizer
url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
def tokenize(text):
    if not isinstance(text, str): return []
    text = re.sub(url_regex, "urlplaceholder", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

# Data Loading
def load_data(database_path):
    engine = create_engine(f'sqlite:///{database_path}')
    df = pd.read_sql_table('Messages', engine)
    X = df['message'].values
    y = df.drop(columns=['id', 'message', 'original', 'genre'], errors='ignore')
    y['related'] = y['related'].replace(2, 1)
    category_names = [col for col in y.columns if y[col].nunique() > 1]
    return X, y[category_names], category_names

# Model Persistence
def save_model(model, model_name, category_names, vectorizer, output_path):
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(model, os.path.join(output_path, f"{model_name}_pipeline.pkl"))
    joblib.dump(vectorizer, os.path.join(output_path, f"{model_name}_vectorizer.pkl"))
    joblib.dump(category_names, os.path.join(output_path, f"{model_name}_categories.pkl"))
    print(f"✅ Saved {model_name} model, vectorizer, and category names to {output_path}")

# Model Evaluation
def save_metrics(model, X_test, y_test, category_names, model_name, output_path):
    # Create model_metrics directory in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_dir = os.path.join(project_root, 'model_metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    report_lines = []
    for i, col in enumerate(category_names):
        report = classification_report(y_test[col], y_pred[:, i], zero_division=0)
        report_lines.append(f"Category: {col}\n{report}\n")
    
    # Save only to model_metrics directory
    metrics_path = os.path.join(metrics_dir, f"{model_name}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"✅ Metrics saved to {metrics_path}")

# Pipeline Construction
def build_pipeline(classifier, vectorizer):
    return Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([('tfidf', vectorizer)])),
            ('text_len', Pipeline([('text_len', TextLengthExtractor())])),
            ('word_cnt', Pipeline([('word_cnt', WordCountExtractor())]))
        ])),
        ('clf', MultiOutputClassifier(classifier))
    ])

# Vectorizer Fitting
def fit_vectorizer(X_train):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=5,
        max_features=10000
    )
    vectorizer.fit(X_train)
    print("✅ TF-IDF Vectorizer fitted")
    return vectorizer

def check_gpu_available():
    """Check if NVIDIA GPU is available by running nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi'], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
        return result.returncode == 0
    except:
        return False

# Main Training Function
def train_disaster_model(n=200): 
    base_path = os.path.dirname(os.path.abspath(__file__)) 
    db_path = os.path.join(base_path, 'Data/DisasterResponse.db')
    output_base = os.path.join(base_path, 'saved_models')
    
    # Check for GPU availability
    use_gpu = check_gpu_available()
    device = 'GPU' if use_gpu else 'CPU'
    print(f"🔍 Using {device} for training")

    print("📥 Loading data...")
    X, y, category_names = load_data(db_path)

    print("🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Fitting vectorizer...")
    vectorizer = fit_vectorizer(X_train)

    # Define classifiers with GPU if available, otherwise CPU
    if use_gpu:
        classifiers = {
            'xgb': XGBClassifier(
                n_estimators=n,
                tree_method='gpu_hist',
                predictor='gpu_predictor',
                random_state=42,
                gpu_id=0
            ),
            'lgbm': LGBMClassifier(
                n_estimators=n,
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                random_state=42
            ),
            'catboost': CatBoostClassifier(
                iterations=n,
                task_type='GPU',
                devices='0',
                verbose=0,
                random_seed=42
            )
        }
    else:
        classifiers = {
            'xgb': XGBClassifier(
                n_estimators=n,
                tree_method='hist',
                random_state=42,
                n_jobs=-1
            ),
            'lgbm': LGBMClassifier(
                n_estimators=n,
                random_state=42,
                n_jobs=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=n,
                task_type='CPU',
                verbose=0,
                random_seed=42,
                thread_count=-1
            )
        }

    trained_models = []
    for name, clf in classifiers.items():
        print(f"🚀 Training {name}...")
        model_path = os.path.join(output_base, name)
        pipe = build_pipeline(clf, vectorizer)
        pipe.fit(X_train, y_train)
        save_model(pipe, name, category_names, vectorizer, model_path)
        save_metrics(pipe, X_test, y_test, category_names, name, model_path)
        trained_models.append((name, pipe.named_steps['clf'].estimator))

    print("🔗 Training VotingClassifier...")
    voting_clf = VotingClassifier(estimators=trained_models, voting='soft')
    voting_pipeline = build_pipeline(voting_clf, vectorizer)
    voting_pipeline.fit(X_train, y_train)

    ensemble_path = os.path.join(output_base, 'voting_classifier')
    save_model(voting_pipeline, 'voting_classifier', category_names, vectorizer, ensemble_path)
    save_metrics(voting_pipeline, X_test, y_test, category_names, 'voting_classifier', ensemble_path)

    print("✅ All training complete!")

if __name__ == "__main__":
    import argparse
    import sys
    
    # Set default value
    n_estimators = 10
    
    # Check if argument is provided
    if len(sys.argv) > 1:
        try:
            n_estimators = int(sys.argv[1])
            print(f"🚀 Starting training with {n_estimators} estimators...")
        except ValueError:
            print("⚠️ Please provide a valid number for estimators. Using default (10).")
    else:
        print(f"ℹ️  No estimator count provided. Using default ({n_estimators})...")
    
    train_disaster_model(n=n_estimators)
