# Disaster Response Pipeline

A production-grade machine learning pipeline for classifying emergency messages during natural disasters and humanitarian crises. Designed for real-world deployment, the system helps emergency response teams make informed decisions quickly by categorizing messages into actionable categories.

---

## 🎥 Demo

![Disaster Response Pipeline Demo](disaster_response.gif)

---

## 🚀 Key Features

* **Real-World Classification**: Accurately classifies messages into multiple relevant categories (e.g., "medical help", "shelter", "food") based on urgency and context.
* **Powerful Ensemble Modeling**: Combines XGBoost, LightGBM, and CatBoost using a voting strategy for robust predictions.
* **Streamlit Dashboard**: Easy-to-use interface for first responders and analysts.
* **Flask REST API**: Backend-ready for integration with mobile apps or crisis management tools.
* **Modular Codebase**: Separation of concerns for preprocessing, modeling, inference, and UI.

---

## 🌐 Realistic Use Case

**Scenario**: A major earthquake hits a densely populated urban area. Relief coordinators receive thousands of messages via SMS, social media, and emergency forms.

**Objective**: Use this system to quickly classify messages like:

* "Trapped under collapsed building near 5th Avenue, send rescue."
* "Need clean water and medical attention for injured in central park."

**Outcome**: The dashboard highlights categories such as "search and rescue", "water", and "medical help" enabling responders to triage and act immediately.

---

## 🛠️ Installation

### 1. Prerequisites

* Python 3.8+
* pip
* Git

### 2. Clone the Repository

```bash
git clone https://github.com/omkarbhad/Disaster-Response-Pipeline.git
cd Disaster-Response-Pipeline
```

### 3. Set Up a Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download NLTK Assets

```bash
python -m nltk.downloader punkt wordnet stopwords
```

---

## 🏃‍♂️ Quick Start

### Step 1: Process Raw Data

```bash
python Data/process_data.py
```

### Step 2: Train Models

```bash
# Default
python models_training.py

# With custom parameters
python models_training.py 100
```

### Step 3: Launch Backend API

```bash
python backend.py
```

### Step 4: Launch Frontend

```bash
streamlit run frontend.py
```

### Step 5: Access Dashboard

[http://localhost:8501](http://localhost:8501)

---

## 📂 Project Structure

```
Disaster-Response-Pipeline/
├── Data/
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── process_data.py
│   └── DisasterResponse.db
├── model_metrics/
│   ├── xgb_metrics.txt
│   ├── lgbm_metrics.txt
│   ├── catboost_metrics.txt
│   └── voting_classifier_metrics.txt
├── saved_models/
│   ├── xgb/
│   ├── lgbm/
│   ├── catboost/
│   └── voting_classifier/
├── backend.py
├── frontend.py
├── models_training.py
└── requirements.txt
```

---

## 🤖 Modeling Approach

* **XGBoost**: Regularized gradient boosting.
* **LightGBM**: Fast and memory-efficient.
* **CatBoost**: Categorical-feature friendly boosting.
* **Voting Classifier**: Ensemble to improve generalization.

Training is performed on a SQLite-managed dataset for scalable and reproducible experimentation.

---

## 📊 Performance Overview

The Voting Classifier shows strong results across critical categories:

| Category | F1 Score |
| -------- | -------- |
| related  | \~0.88   |
| food     | \~0.80   |
| water    | \~0.65   |
| shelter  | \~0.62   |

Detailed metrics are logged in `model_metrics/`.

---

## 🌐 API Overview

### Endpoint

* `POST /predict` — Accepts a disaster-related message and returns predicted categories.

**Example:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message":"Help! We need medical supplies and water in downtown"}'
```

**Response Format:**

```json
[
  {"Category": "medical_help", "Confidence": 0.91},
  {"Category": "water", "Confidence": 0.87},
  {"Category": "aid_related", "Confidence": 0.85}
]
```

---

## 📋 License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
