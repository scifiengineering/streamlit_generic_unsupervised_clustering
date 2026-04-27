# Text Mining and Clustering Web App

A Streamlit app for unsupervised text mining on CSV feedback data.

## Features

- Upload a CSV file and pick the text column.
- Clean text with NLTK preprocessing.
- Run topic modeling with Gensim LDA.
- Run K-Means clustering with an elbow plot.
- Download results with assigned cluster labels.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

- NLTK resources are downloaded automatically on first run.
- The app expects a CSV with at least one text column.
