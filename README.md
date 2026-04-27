# Text Mining and Clustering Web App

A Streamlit app for unsupervised text mining on CSV feedback data.

## Features

- Upload a CSV file and pick the text column.
- Clean text with NLTK preprocessing.
- Run topic modeling with Gensim LDA.
- Run K-Means clustering with an elbow plot.
- Download results with assigned cluster labels.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

- **Windows**: `.venv\Scripts\activate`
- **Mac/Linux**: `source .venv/bin/activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run

Activate the virtual environment (if not already active), then run:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Notes

- NLTK resources are downloaded automatically on first run.
- The app expects a CSV with at least one text column.
