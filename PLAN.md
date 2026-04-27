# Master Plan: Generic Unsupervised Text Mining & Clustering Web App

## Phase 1: Environment & Repository Setup

### 1. Git Initialization

- Create a local folder and run `git init`.
- Create a `.gitignore` file to exclude `.venv/`, `__pycache__/`, and `.csv` data files.

### 2. Branching Strategy

- Create a new branch for each feature:
  - `feature/env-setup`: Basic structure and environment.
  - `feature/data-upload`: File uploader and column selection.
  - `feature/nlp-pipeline`: Text cleaning and NLTK bootstrapping.
  - `feature/analysis-lda`: Topic modeling implementation.
  - `feature/analysis-kmeans`: Elbow plot and clustering logic.

### 3. Environment

- Run `python -m venv .venv`.
- Activate the environment and install: `streamlit`, `pandas`, `scikit-learn`, `gensim`, `nltk`, `matplotlib`.

## Phase 2: Required Deliverables (File Structure)

- `app.py`: The single entry point for the Streamlit application.
- `requirements.txt`: List of all dependencies.
- `README.md`: Simple instructions on how to run the app and what it does.

## Phase 3: The Build Logic (Feature by Feature)

### 1. Data Loading & Validation (`feature/data-upload`)

- Uploader: `st.file_uploader` for CSVs.
- Column selection: `st.selectbox` to choose the feedback column.
- Validation: Basic check to ensure the file is not empty and the selected column contains text.

### 2. NLTK Preprocessing (`feature/nlp-pipeline`)

- Bootstrapping: Include code to automatically download `stopwords`, `punkt`, and `wordnet` on the first run.
- Cleaning: Lowercase, remove punctuation and stopwords, and lemmatize.
- Performance: Use `@st.cache_data` for the preprocessing function so it does not re-run every time a button is clicked.

### 3. Topic Modeling (`feature/analysis-lda`)

- LDA engine: Use Gensim.
- Parameters: Set a fixed `random_state` for reproducibility.
- UI: Slider for Number of Topics and display the top 10 keywords per topic.

### 4. K-Means Clustering (`feature/analysis-kmeans`)

- Vectorization: `TfidfVectorizer`.
- Elbow plot: Generate a Matplotlib plot for $k=2$ to $k=10$.
- Assignment: Assign cluster IDs and show a table with the results.
- Export: Add `st.download_button` to download the final CSV with cluster labels.

## Items Removed (Over-kills)

- ~~Silhouette scores and coherence~~: Too complex for a basic assignment; the Elbow plot is enough.
- ~~Redaction and privacy policy~~: Unnecessary for a local academic project.
- ~~Unit testing and smoke tests~~: Manual testing (checking if it runs) is sufficient for this scope.
- ~~Dictionary filtering and advanced LDA tuning~~: Standard defaults are fine for discovery.
- ~~Complex error handling~~: Standard Python `try/except` blocks for file loading are enough.

## Acceptance Criteria (The Done Definition)

- [ ] App launches without errors.
- [ ] User can upload a CSV and pick a column.
- [ ] The Elbow plot renders correctly.
- [ ] Clusters are assigned and the final table is downloadable.
- [ ] Git history shows clear branches for each feature.
