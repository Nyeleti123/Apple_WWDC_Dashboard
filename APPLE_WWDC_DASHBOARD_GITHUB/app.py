import streamlit as st
# Force clearing Streamlit's cache
st.cache_data.clear()
st.cache_resource.clear()

from PIL import Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import json
import gdown
import os
import time
from streamlit_lottie import st_lottie

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Apple WWDC Sentiment Analysis Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# RELATIVE PATHS
# -------------------------------
BASE_PATH = Path(__file__).parent  # relative to app.py
ASSETS_PATH = BASE_PATH / "assets"
CLIPART_PATH = BASE_PATH / "clipart"
MODEL_PACKAGE_PATH = BASE_PATH / "Model Package"
MODEL_PATH = MODEL_PACKAGE_PATH / "chunked_svm_balanced_model.pkl"
VECT_PATH = MODEL_PACKAGE_PATH / "chunked_tfidf_vectorizer.pkl"

# -------------------------------
# GOOGLE DRIVE FILES
# -------------------------------
VECTOR_FILE_ID = "1QClOOxVRd7E89wrjWXMjGM06OdwgR7uK"

os.makedirs(MODEL_PACKAGE_PATH, exist_ok=True)

if not VECT_PATH.exists():
    url = f"https://drive.google.com/uc?id={VECTOR_FILE_ID}"
    st.info("Downloading vectorizer from Google Drive... This may take a moment.")
    try:
        gdown.download(url, str(VECT_PATH), quiet=False)
    except Exception as e:
        st.error(f"Failed to download vectorizer: {e}")
        st.stop()

# -------------------------------
# CSS STYLING
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap');
html, body, [class*="css"] { font-family: 'Lato', sans-serif; background-color: #FAF9F6; color: #333333; }
h1, h2, h3 { color: #333333; font-weight: 700; }
.stSidebar { background-color: #DDEBF7 !important; }
.css-1v0mbdj, .css-1dp5vir { background-color: #FAF9F6 !important; }
.highlight-heading { background: linear-gradient(to right, #A1C4FD, #C2E9FB); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
img { border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin-bottom: 15px; }
.footer { text-align: center; color: gray; font-size: 14px; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# SESSION STATE
# -------------------------------
for key in ["dashboard_entered", "model_loaded", "model", "vectorizer", "lottie_assets"]:
    if key not in st.session_state:
        if key in ["dashboard_entered", "model_loaded"]:
            st.session_state[key] = False
        elif key == "lottie_assets":
            st.session_state[key] = {}
        else:
            st.session_state[key] = None
            
# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
import gc

def load_lottie(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Critical Lottie file missing: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

def show_loading_animation(animation_json, height=150):
    st_lottie(animation_json, speed=1, loop=True, quality="low", height=height)

def get_model():
    if "model" not in st.session_state:
        st.session_state.model = joblib.load(MODEL_PATH)
    return st.session_state.model

def get_vectorizer():
    if "vectorizer" not in st.session_state:
        gc.collect()
        with st.spinner("Loading vectorizer..."):
            st.session_state.vectorizer = joblib.load(VECT_PATH, mmap_mode='r')
    return st.session_state.vectorizer

@st.cache_data
def transform_text(vectorizer, texts):
    return vectorizer.transform(texts)

@st.cache_data
def compute_lda(texts, n_topics=3, n_words=10):
    vectorizer_lda = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer_lda.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    topics = []
    topic_distributions = []
    for i, topic in enumerate(lda.components_):
        top_words = [vectorizer_lda.get_feature_names_out()[idx] for idx in topic.argsort()[-n_words:]][::-1]
        topics.append(f"Topic {i+1}: {', '.join(top_words)}")
        topic_distributions.append(topic / topic.sum())
    return topics, topic_distributions
    
# -------------------------------
# PRELOAD LOTTIE ANIMATIONS
# -------------------------------
if not st.session_state.lottie_assets:
    st.session_state.lottie_assets["home"] = load_lottie(ASSETS_PATH / "home_page.json")
    st.session_state.lottie_assets["loading"] = load_lottie(ASSETS_PATH / "loading.json")
    st.session_state.lottie_assets["team"] = load_lottie(ASSETS_PATH / "teamwork.json")
    st.session_state.lottie_assets["under_buttons"] = load_lottie(ASSETS_PATH / "under_buttons.json")

# -------------------------------
# HOME PAGE / LANDING
# -------------------------------
if not st.session_state.dashboard_entered:
    st.markdown('<div class="highlight-heading"><h1>Welcome to the Apple WWDC Sentiment Analysis Dashboard</h1></div>', unsafe_allow_html=True)
    show_loading_animation(st.session_state.lottie_assets["home"], height=300)
    st.markdown("#### A Data Science Journey by Ctrl Alt Elite")
    if st.button("Enter Dashboard"):
        st.session_state.dashboard_entered = True
        st.rerun()
    st.stop()
    
# -------------------------------
# SIDEBAR & PAGES
# -------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=70)
st.sidebar.title("Ctrl Alt Elite Dashboard")
st.sidebar.markdown("#### Apple WWDC Sentiment Analysis Project")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Visualizations",
        "Timeline",
        "Data Collection & Cleaning",
        "The Team",
        "Make Predictions",
        "Visualize Predictions",
        "References"
    ]
)

# -------------------------------
# PAGES (OVERVIEW, VISUALIZATIONS, ETC.)
# -------------------------------
if page == "Project Overview":
    st.markdown('<div class="highlight-heading"><h1>Apple WWDC Sentiment Analysis</h1></div>', unsafe_allow_html=True)
    st.markdown("### A Data Science Journey by Ctrl Alt Elite")
    st.markdown("""
    #### Project Goal
    Analyze public sentiment around Apple's Worldwide Developers Conference (WWDC) and product announcements using NLP and Machine Learning.
    
    #### Approach
    - Data collected from online platforms discussing WWDC.
    - Text cleaned, lemmatized, labeled using TextBlob.
    - SVM model trained using TF-IDF vectorization.
    - Sentiment distribution and trends analyzed and visualized.

    #### Outcome
    The model provides meaningful insights into user perceptions of Apple events and products.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image(ASSETS_PATH / "Overall_Model_Accuracy.png", caption="Overall Model Accuracy", width='stretch')
    with col2:
        st.image(ASSETS_PATH / "Model_Performance_Bar.png", caption="Model Performance (F1 Scores)", width='stretch')

elif page == "Visualizations":
    st.markdown('<div class="highlight-heading"><h1>Visualizations</h1></div>', unsafe_allow_html=True)
    st.markdown("### Key Insights from the Model & Data")
    col1, col2 = st.columns(2)
    with col1:
        st.image(ASSETS_PATH / "Sentiment_Distribution_Bar.png", caption="Sentiment Distribution Across Tweets", width='stretch')
        st.image(ASSETS_PATH / "Senitment_Prop_Pie.png", caption="Sentiment Proportion Pie Chart", width='stretch')
        st.image(ASSETS_PATH / "Confusion_Matrix.png", caption="Confusion Matrix", width='stretch')
    with col2:
        st.image(ASSETS_PATH / "Actual_Vs_Predicted_Bar.png", caption="Actual vs Predicted Sentiments", width='stretch')
    st.markdown("""
    #### Interpretation
    - Bar chart: Positive sentiments dominate discussions.
    - Pie chart: Proportional split between sentiment types.
    - Confusion matrix: Shows prediction accuracy for each sentiment class.
    """)

elif page == "Timeline":
    st.markdown('<div class="highlight-heading"><h1>Project Timeline</h1></div>', unsafe_allow_html=True)
    st.markdown("""
    #### Timeline Overview
    Key stages from initial data collection to dashboard deployment:

    | Phase | Description | Tools Used |
    |:------|:-------------|:------------|
    | Data Collection | Scraped WWDC-related posts & comments | Python, PRAW, AcademicTorrents |
    | Data Cleaning | Removed noise, stopwords, lemmatization | NLTK, Regex |
    | Labeling | Sentiment labeling using TextBlob | Python |
    | Model Training | Built SVM classifier with TF-IDF | Scikit-learn |
    | Evaluation | Analyzed performance using Accuracy & F1 | Pandas, Matplotlib |
    | Dashboard Design | Built interactive dashboard | Streamlit |
    """)

elif page == "Data Collection & Cleaning":
    st.markdown('<div class="highlight-heading"><h1>Data Collection & Cleaning</h1></div>', unsafe_allow_html=True)
    st.markdown("""
    #### Data Collection
    - Reddit posts & comments about WWDC and Apple products.
    - Focused on iPhone, iOS, iPad, Watch, AirPods.
    - ~4.1 million lines collected (2020–2025).

    #### Data Cleaning
    - Lowercasing, remove URLs, mentions, emojis, extra spaces
    - Remove stopwords, lemmatize
    - Sentiment labeling with TextBlob
    """)

elif page == "The Team":
    st.markdown('<div class="highlight-heading"><h1>Meet the Team – Ctrl Alt Elite</h1></div>', unsafe_allow_html=True)
    show_loading_animation(st.session_state.lottie_assets["team"], height=250)
    st.markdown("We built the full sentiment analysis pipeline, model, and dashboard.")
    team = {
        "Rendani": "Lead Data Analyst & Documentation",
        "Nyeleti": "Model Training & Dashboard Designer",
        "Kabelo": "Data Cleaning & Preprocessing",
        "Jeremy": "Research & Data Labeling",
        "Thapelo": "Data Visualization & Documentation",
        "Qayiya": "Project Integration & Testing"
    }
    for name, role in team.items():
        st.markdown(f"**{name}** — {role}")
    st.markdown("---")
    st.markdown("We are Ctrl Alt Elite – Combining creativity, logic, and teamwork to make data beautiful.")

elif page == "Make Predictions":
    st.markdown('<div class="highlight-heading"><h1>Make Predictions</h1></div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV and select a text column to predict sentiments.")
    uploaded_pred = st.file_uploader("Upload CSV for Predictions", type=["csv"], key="predict_uploader")
    if uploaded_pred:
        df_pred = pd.read_csv(uploaded_pred)
        text_columns = df_pred.select_dtypes(include=['object']).columns.tolist()
        selected_col = st.selectbox("Select text column for prediction", text_columns, key="predict_column_select")
        if st.button("Predict Sentiments", key="predict_button"):
            model = get_model()
            vectorizer = get_vectorizer()
            
            with st.spinner("Predicting sentiments..."):
                X = transform_text(vectorizer, df_pred[selected_col].fillna(""))
                df_pred["predicted_sentiment"] = model.predict(X)
            st.success("Predictions added!")
            st.dataframe(df_pred.head())
            csv = df_pred.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV with Predictions",
                data=csv,
                file_name="predicted_sentiments.csv",
                mime="text/csv",
                key="predict_download"
            )

elif page == "Visualize Predictions":
    st.markdown('<div class="highlight-heading"><h1>Predictions Visualizations</h1></div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV with `predicted_sentiment` column to explore insights interactively.")
    uploaded_vis = st.file_uploader("Upload CSV for Visualization", type=["csv"], key="visualize_uploader")
    if uploaded_vis:
        df_vis = pd.read_csv(uploaded_vis)
        if "predicted_sentiment" not in df_vis.columns:
            st.error("CSV must contain a 'predicted_sentiment' column.")
        else:
            st.success("CSV loaded successfully!")
            text_columns = df_vis.select_dtypes(include=['object']).columns.tolist()
            text_columns = [c for c in text_columns if c != "predicted_sentiment"]
            if text_columns:
                text_col = st.selectbox("Select text column for Topic Modeling", text_columns, key="visualize_topic_select")
            else:
                st.warning("No text column available for topic modeling.")
                text_col = None
            st.subheader("Summary Statistics")
            total_texts = len(df_vis)
            avg_len = df_vis[text_col].dropna().apply(lambda x: len(str(x).split())).mean() if text_col else 0
            sentiment_counts = df_vis["predicted_sentiment"].value_counts(normalize=True) * 100
            summary_text = (
                f"Total Rows: {total_texts:,}\n"
                f"Average Text Length: {avg_len:.2f} words\n\n"
                f"Sentiment Breakdown (%):\n"
            )
            for s, p in sentiment_counts.items():
                summary_text += f"   - {s}: {p:.1f}%\n"
            st.text(summary_text)
            st.markdown("Interpretation: Shows sentiment prevalence and text characteristics.")
            with st.spinner("Generating visualizations..."):
                counts = df_vis["predicted_sentiment"].value_counts()
                fig, ax = plt.subplots(figsize=(6,4))
                ax.bar(counts.index, counts.values, color=["#FF7043", "#42A5F5", "#66BB6A"])
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Number of Posts")
                ax.set_title("Sentiment Distribution")
                for i, v in enumerate(counts.values):
                    ax.text(i, v + max(counts.values)*0.02, str(v), ha="center", fontweight="bold")
                st.pyplot(fig)
                fig, ax = plt.subplots(figsize=(5,4))
                ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=["#FF7043", "#42A5F5", "#66BB6A"], startangle=90, textprops={"color": "black"})
                ax.set_title("Sentiment Percentage Share")
                st.pyplot(fig)
                if text_col:
                    words = " ".join(str(t) for t in df_vis[text_col].dropna()).split()
                    most_common = Counter(words).most_common(10)
                    words_list, counts_words = zip(*most_common)
                    fig, ax = plt.subplots(figsize=(7,4))
                    ax.barh(words_list[::-1], counts_words[::-1], color="#9C27B0")
                    ax.set_xlabel("Frequency")
                    ax.set_ylabel("Word")
                    ax.set_title("Top Words")
                    st.pyplot(fig)
                    product_keywords = ["iphone", "macbook", "ios", "ipad", "watch", "airpods"]
                    df_vis["mentioned_products"] = df_vis[text_col].apply(lambda text: [word for word in str(text).lower().split() if word in product_keywords])
                    product_counts = {}
                    for products in df_vis["mentioned_products"]:
                        for p in products:
                            product_counts[p] = product_counts.get(p, 0) + 1
                    if product_counts:
                        fig, ax = plt.subplots(figsize=(6,4))
                        ax.bar(product_counts.keys(), product_counts.values(), color="#FF7043")
                        ax.set_title("Most Mentioned Apple Products")
                        st.pyplot(fig)
                    sample_texts = df_vis[text_col].dropna()
                    if len(sample_texts) > 2000:
                        sample_texts = sample_texts.sample(2000, random_state=42)
                    topics, topic_distributions = compute_lda(sample_texts)
                    st.subheader("Topic Modeling (Main Themes)")
                    for t in topics:
                        st.write("- ", t)
                    st.subheader("Topic Prevalence Across Words")
                    fig, ax = plt.subplots(figsize=(8,4))
                    for i, dist in enumerate(topic_distributions):
                        ax.plot(range(1, len(dist)+1), dist, label=f"Topic {i+1}")
                    ax.set_xlabel("Word Index")
                    ax.set_ylabel("Normalized Importance")
                    ax.set_title("Topic Distribution Line Graph")
                    ax.legend()
                    st.pyplot(fig)

elif page == "References":
    st.markdown('<div class="highlight-heading"><h1>References</h1></div>', unsafe_allow_html=True)
    st.markdown("""
- TextBlob: https://textblob.readthedocs.io/en/dev/
- Scikit-learn: https://scikit-learn.org/stable/
- Reddit API (PRAW): https://praw.readthedocs.io/en/latest/
- Streamlit: https://docs.streamlit.io/
- Matplotlib: https://matplotlib.org/stable/index.html
- Joblib: https://joblib.readthedocs.io/en/latest/
    """)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown('<div class="footer">Ctrl Alt Elite – Apple WWDC Sentiment Analysis Dashboard | 2025</div>', unsafe_allow_html=True)
