# Word2Vec Integration Guide

## Overview
This Streamlit app now supports **two clustering methods**:
1. **TF-IDF** (frequency-based) - Original method
2. **Word2Vec** (semantic embeddings) - NEW

## What Changed

### New Python Functions

```python
# 1. Train Word2Vec model
train_word2vec(cleaned_docs) -> Word2Vec
# Trains model on token-level cleaned documents
# Returns cached Word2Vec model for reuse

# 2. Convert documents to vectors
get_word2vec_vectors(cleaned_docs, w2v_model) -> np.ndarray
# Averages embeddings of words in each document
# Returns (n_docs, 100) shaped array

# 3. Elbow plot for Word2Vec
build_elbow_plot_w2v(vectors, max_k, random_state) -> (ks, inertias)
# Generates elbow plot data using Word2Vec vectors

# 4. Auto-detect optimal clusters
find_optimal_k_w2v(vectors, max_k, random_state) -> int
# Uses second derivative method to find elbow point

# 5. Run clustering
run_kmeans_w2v(vectors, k, random_state) -> list[int]
# Performs K-Means on Word2Vec vectors
```

### UI Changes

#### Before
```
Clustering Tab
└── Elbow Plot (TF-IDF)
└── Cluster Results (TF-IDF only)
└── Download Button
```

#### After
```
Clustering Tab
├── Sub-tab: TF-IDF Results
│   ├── Elbow Plot (blue, TF-IDF)
│   ├── Cluster Results (TF-IDF)
│   └── Download Button (clustered_results_tfidf.csv)
│
└── Sub-tab: Word2Vec Results
    ├── Elbow Plot (green, Word2Vec)
    ├── Cluster Results (Word2Vec)
    └── Download Button (clustered_results_w2v.csv)
```

## How It Works

### Word2Vec Vectorization Process

1. **Training** (automatic, cached)
   ```
   Input: List of cleaned documents
           ↓
   Tokenize: ["word1", "word2", ...] per document
           ↓
   Word2Vec: Train model with 100-dim vectors
           ↓
   Output: Word2Vec model (gensim.models.Word2Vec)
   ```

2. **Vectorization** (automatic, cached)
   ```
   Input: Cleaned documents + trained W2V model
           ↓
   For each document:
     - Split into tokens
     - Get embedding for each token
     - Average all embeddings
           ↓
   Output: np.ndarray of shape (n_docs, 100)
   ```

3. **Clustering**
   ```
   Input: Document vectors (n_docs, 100)
           ↓
   K-Means: Find optimal K using elbow method
           ↓
   Output: Cluster assignments for each document
   ```

## Parameter Tuning

### Word2Vec Model Parameters
In `train_word2vec()` function:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `vector_size` | 100 | Dimensionality of word embeddings |
| `window` | 5 | Context window size for training |
| `min_count` | 1 | Minimum word frequency threshold |

To adjust, modify the function call:
```python
w2v_model = train_word2vec(
    analysis_df["cleaned_text"].tolist(),
    vector_size=50,    # Smaller embeddings
    window=3,          # Smaller context
    min_count=2        # Ignore rare words
)
```

## Comparison Guide

### When to Use TF-IDF
- Quick analysis needed
- Text is homogeneous in domain
- Interpretability important (see which words matter)
- Limited computational resources
- Small dataset

### When to Use Word2Vec
- Semantic similarity matters
- Domain-specific terminology present
- Need to capture word relationships
- Sufficient computational resources
- Larger dataset available

## Example Output

### TF-IDF Results
```
K = 3 clusters

Elbow values:
k=2: inertia=245.3
k=3: inertia=189.2
k=4: inertia=156.8
...

Cluster distribution: [45, 52, 48]
```

### Word2Vec Results
```
K = 4 clusters

Elbow values:
k=2: inertia=8945.2
k=3: inertia=7123.5
k=4: inertia=6234.1
...

Cluster distribution: [38, 55, 52, 45]
```

## Performance Considerations

### Caching Strategy
All compute-intensive functions are cached:
- `@st.cache_data` for deterministic functions
- `@st.cache_resource` for NLTK resources

**Impact**: After first run with same data, subsequent reruns are instant.

### Speed Estimates (on typical dataset)
- Text preprocessing: 2-5 seconds
- Word2Vec training: 5-15 seconds
- Vector generation: 1-3 seconds
- K-Means + elbow plot: 5-10 seconds

**Total**: ~15-35 seconds for first run (includes TF-IDF + Word2Vec)

## Troubleshooting

### "No valid documents to train Word2Vec model"
- **Cause**: All documents are empty after preprocessing
- **Fix**: Check if stopwords are too aggressive; review cleaned_text preview tab

### Word2Vec plot looks different from TF-IDF
- **Expected**: Different vectorization methods give different inertia values
- **Note**: Compare relative shapes, not absolute values
- **Tip**: Use the optimal K values recommended by each method

### Performance is slow
- **Check**: Dataset size (try with subset first)
- **Try**: Reduce `vector_size` in `train_word2vec()` (e.g., 50 instead of 100)
- **Tip**: First run creates cache; subsequent runs are faster

## Technical Stack
- **Vectorization (TF-IDF)**: scikit-learn
- **Vectorization (Word2Vec)**: gensim
- **Clustering**: scikit-learn KMeans
- **Visualization**: matplotlib
- **UI Framework**: Streamlit
- **Data processing**: pandas, numpy

## Files
- `app.py` - Main application (433 lines)
- `requirements.txt` - Dependencies (7 packages)
- `text_data.csv` - Sample data
- `INTEGRATION_SUMMARY.md` - Technical summary
- `WORD2VEC_GUIDE.md` - This file

## Next Steps
To further enhance the app:
1. Add Doc2Vec for alternative document embeddings
2. Implement silhouette score visualization
3. Add 3D PCA projection of Word2Vec embeddings
4. Enable hyperparameter tuning in UI
5. Add model export functionality
