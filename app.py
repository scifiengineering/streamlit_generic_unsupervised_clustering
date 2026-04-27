import re
import string
from io import StringIO

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="Text Mining Web App", page_icon=":brain:", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    .stAlert {border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def bootstrap_nltk() -> None:
    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


@st.cache_data
def read_uploaded_csv(uploaded_bytes: bytes) -> pd.DataFrame:
    text = uploaded_bytes.decode("utf-8", errors="replace")
    return pd.read_csv(
        StringIO(text),
        quoting=1,
        quotechar='"',
        skipinitialspace=True,
        on_bad_lines="skip",
    )


@st.cache_data
def preprocess_texts(texts: list[str]) -> list[str]:
    bootstrap_nltk()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    cleaned_docs: list[str] = []
    punctuation_table = str.maketrans("", "", string.punctuation)

    for text in texts:
        normalized = re.sub(r"\s+", " ", str(text).strip().lower())
        tokens = word_tokenize(normalized)
        kept_tokens: list[str] = []

        for token in tokens:
            token = token.translate(punctuation_table)
            if not token or token in stop_words:
                continue
            lemma = lemmatizer.lemmatize(token)
            if lemma:
                kept_tokens.append(lemma)

        cleaned_docs.append(" ".join(kept_tokens))

    return cleaned_docs


@st.cache_data
def run_lda(
    cleaned_docs: list[str], num_topics: int, random_state: int
) -> list[tuple[int, str]]:
    tokenized_docs = [doc.split() for doc in cleaned_docs if doc.strip()]
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_docs]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        passes=10,
    )
    return lda_model.print_topics(num_topics=num_topics, num_words=10)


@st.cache_data
def build_elbow_plot(
    cleaned_docs: list[str], max_k: int, random_state: int
) -> tuple[list[int], list[float]]:
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(cleaned_docs)

    ks = list(range(2, max_k + 1))
    inertias: list[float] = []

    for k in ks:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(matrix)
        inertias.append(float(model.inertia_))

    return ks, inertias


@st.cache_data
def find_optimal_k(cleaned_docs: list[str], max_k: int, random_state: int) -> int:
    """Find optimal K using the elbow method with rate of change."""
    ks, inertias = build_elbow_plot(cleaned_docs, max_k, random_state)

    if len(ks) < 3:
        return ks[0]

    # Calculate second differences to find elbow
    first_diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    second_diffs = [
        first_diffs[i] - first_diffs[i + 1] for i in range(len(first_diffs) - 1)
    ]

    # Find the point with max second difference (the elbow)
    if second_diffs:
        elbow_idx = second_diffs.index(max(second_diffs)) + 2
        return ks[elbow_idx]
    return ks[0]


@st.cache_data
def suggest_optimal_topics(num_docs: int) -> int:
    """Suggest optimal number of topics based on document count."""
    if num_docs < 50:
        return 3
    elif num_docs < 100:
        return 4
    elif num_docs < 500:
        return 5
    else:
        return min(8, max(5, num_docs // 100))


@st.cache_data
def run_kmeans(cleaned_docs: list[str], k: int, random_state: int) -> list[int]:
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(cleaned_docs)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return model.fit_predict(matrix).tolist()


st.title("Generic Unsupervised Text Mining and Clustering")
st.caption("Upload any CSV, pick a text column, and run LDA plus K-Means analysis.")

bootstrap_nltk()

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    uploaded_bytes = uploaded_file.getvalue()
    source_df = read_uploaded_csv(uploaded_bytes)
except Exception as exc:
    st.error(
        f"Failed to read CSV: {exc}. Try ensuring proper quoting and commas in text are within quotes."
    )
    st.stop()

if source_df.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

candidate_columns = [
    col
    for col in source_df.columns
    if source_df[col].dtype == "object"
    or str(source_df[col].dtype).startswith("string")
]
if not candidate_columns:
    candidate_columns = source_df.columns.tolist()

# Auto-select feedback column if it exists, otherwise use the first/only column
if "feedback" in candidate_columns:
    selected_col = "feedback"
elif len(candidate_columns) == 1:
    selected_col = candidate_columns[0]
else:
    selected_col = st.selectbox(
        "Select the feedback/text column", options=candidate_columns
    )

work_df = source_df[[selected_col]].copy()
work_df.columns = ["text"]
work_df["text"] = work_df["text"].astype(str).str.strip()
work_df = work_df[work_df["text"].ne("")]

if work_df.empty:
    st.error("Selected column has no usable text rows.")
    st.stop()

st.success(f"Loaded {len(work_df)} text rows from '{selected_col}'.")

with st.sidebar:
    st.subheader("Analysis Settings")
    seed = st.number_input(
        "Random State", min_value=0, max_value=9999, value=42, step=1
    )

# Auto-select optimal number of topics
topic_count = suggest_optimal_topics(len(work_df))
st.sidebar.info(f"**Topics Auto-Selected:** {topic_count}")

max_k_allowed = min(10, len(work_df))
if max_k_allowed < 2:
    st.error("At least 2 text rows are required for K-Means clustering.")
    st.stop()

run_clicked = st.button("Run Analysis", type="primary")

if not run_clicked:
    st.info("Click Run Analysis to begin.")
    st.stop()

with st.spinner("Preprocessing text..."):
    cleaned_docs = preprocess_texts(work_df["text"].tolist())

valid_idx = [idx for idx, doc in enumerate(cleaned_docs) if doc.strip()]
if len(valid_idx) < 2:
    st.error("Not enough non-empty cleaned text rows after preprocessing.")
    st.stop()

analysis_df = work_df.iloc[valid_idx].copy()
analysis_df["cleaned_text"] = [cleaned_docs[i] for i in valid_idx]

# Compute optimal K
optimal_k = find_optimal_k(
    analysis_df["cleaned_text"].tolist(),
    min(10, len(analysis_df)),
    int(seed),
)
st.sidebar.info(f"**Optimal K Auto-Detected:** {optimal_k} clusters")

preview_tab, lda_tab, cluster_tab = st.tabs(
    ["Data Preview", "Topic Modeling", "Clustering"]
)

with preview_tab:
    st.subheader("Cleaned Feedback Text Preview")
    preview_df = analysis_df[["cleaned_text"]].copy()
    preview_df.columns = ["cleaned_feedback"]
    st.dataframe(preview_df, use_container_width=True)

with lda_tab:
    st.subheader("LDA Topics")
    with st.spinner("Running LDA..."):
        topics = run_lda(analysis_df["cleaned_text"].tolist(), topic_count, int(seed))

    topic_rows = []
    for topic_id, topic_str in topics:
        topic_rows.append({"topic": topic_id, "top_keywords": topic_str})
    st.dataframe(pd.DataFrame(topic_rows), use_container_width=True)

with cluster_tab:
    st.subheader("Elbow Plot and Cluster Assignment")
    with st.spinner("Running K-Means..."):
        elbow_ks, elbow_inertias = build_elbow_plot(
            analysis_df["cleaned_text"].tolist(),
            min(10, len(analysis_df)),
            int(seed),
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(elbow_ks, elbow_inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Plot")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    st.info(f"**Clustering with K = {optimal_k} clusters**")
    labels = run_kmeans(analysis_df["cleaned_text"].tolist(), optimal_k, int(seed))
    result_df = analysis_df[["cleaned_text"]].copy()
    result_df.columns = ["cleaned_feedback"]
    result_df["cluster_id"] = labels

    st.subheader("Clustered Results")
    st.dataframe(result_df, use_container_width=True)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download clustered CSV",
        data=csv_bytes,
        file_name="clustered_results.csv",
        mime="text/csv",
    )
