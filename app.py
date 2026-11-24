# app.py - reviewsense - Sentiment Analysis Dashboard (Streamlit)
import io
import re

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from spellchecker import SpellChecker

st.set_page_config(page_title="ReviewSense", page_icon="🎬", layout="wide")

spell = SpellChecker()


# ---------- Helper functions ----------
def fix_spelling(text):
    """Fix spelling mistakes, e.g. awsome -> awesome"""

    def fix_word(match):
        word = match.group(0)
        if "'" in word or word.lower() in spell:  # skip known words and don't
            return word
        return spell.correction(word) or word

    return re.sub(r"[A-Za-z']+", fix_word, text)


@st.cache_resource
def train_model():
    """Train once, calculate all metrics, and keep everything in memory."""
    df = pd.read_csv("reviews.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42,
        stratify=df["sentiment"],
    )
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True),
        LogisticRegression(C=30, max_iter=1000),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Words the model finds most positive / most negative

    words = model.named_steps["tfidfvectorizer"].get_feature_names_out()
    weights = model.named_steps["logisticregression"].coef_[0]
    order = weights.argsort()
    top_neg = pd.DataFrame({"Importance": -weights[order[:10]]}, index=words[order[:10]])
    top_pos = pd.DataFrame({"Importance": weights[order[-10:][::-1]]}, index=words[order[-10:][::-1]])

    # Model file for the download button

    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, pos_label="positive"),
        "recall": recall_score(y_test, pred, pos_label="positive"),
        "f1": f1_score(y_test, pred, pos_label="positive"),
        "cm": confusion_matrix(y_test, pred, labels=["negative", "positive"]),
        "top_pos": top_pos,
        "top_neg": top_neg,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "model_bytes": buffer.getvalue(),
        "sample": df.sample(8, random_state=1),
    }


def predict(text, use_spell):
    """Return cleaned text, label, confidence and both probabilities."""
    clean = fix_spelling(text) if use_spell else text
    probs = model.predict_proba([clean])[0]
    return clean, model.classes_[probs.argmax()], probs.max(), probs


def clear_history():
    st.session_state.history = []


def set_text(text):
    st.session_state.review_text = text


# ---------- Load model ----------

with st.spinner("Training model on 30,000 reviews (first load takes ~30 seconds)..."):
    art = train_model()
model = art["model"]

if "history" not in st.session_state:
    st.session_state.history = []
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# ---------- Sidebar ----------

with st.sidebar:
    st.header("⚙️ Settings")
    use_spell = st.toggle("Auto-correct spelling", value=True,
                          help="Fixes typos like 'awsome' before predicting")
    st.divider()
    st.subheader("About")
    st.write("Classifies movie reviews as **positive** or **negative** using "
             "TF-IDF + Logistic Regression.")
    st.caption(f"Trained on {art['train_size']:,} reviews, tested on {art['test_size']:,}.")
    st.caption("Built with Python, scikit-learn, pandas and Streamlit.")

# ---------- Header and KPI cards ----------

st.title("🎬 ReviewSense - Sentiment Analysis Dashboard")
st.caption("Type a review, upload a file, or explore how well the model performs.")

k1, k2, k3, k4 = st.columns(4)
cards = [
    (k1, "🎯 Accuracy", art["accuracy"], "Share of all test reviews predicted correctly"),
    (k2, "🔍 Precision", art["precision"], "Of reviews predicted positive, how many really were positive"),
    (k3, "📡 Recall", art["recall"], "Of all truly positive reviews, how many the model found"),
    (k4, "⚖️ F1 Score", art["f1"], "Balance between precision and recall"),
]
for col, label, value, tip in cards:
    with col.container(border=True):
        st.metric(label, f"{value:.1%}", help=tip)

# ---------- Tabs ----------

tab1, tab2, tab3, tab4 = st.tabs(
    ["✍️ Predict", "📂 Batch Predict", "📊 Model Performance", "⬇️ Download"]
)

# ----- Tab 1: single prediction -----

with tab1:
    st.write("**Try an example:**")
    examples = {
        "😊 Positive": "Great film, wonderful acting and a brilliant story. I really enjoyed it!",
        "😞 Negative": "This movie was boring and badly written. A complete waste of time.",
        "🤔 Mixed": "The acting was great but the story was dull and far too long.",
        "✏️ Typos": "The movie was awsome and the acting was brillant",
    }
    cols = st.columns(len(examples))
    for col, (name, text) in zip(cols, examples.items()):
        col.button(name, on_click=set_text, args=(text,))

    review = st.text_area("Enter your review:", key="review_text", height=120)

    if st.button("Analyze Sentiment", type="primary"):
        if review.strip() == "":
            st.warning("Please enter some text.")
        else:
            clean, label, confidence, probs = predict(review, use_spell)
            left, right = st.columns([1, 1])
            with left:
                with st.container(border=True):
                    if label == "positive":
                        st.success("Positive 😊")
                    else:
                        st.error("Negative 😞")
                    st.write(f"Confidence: **{confidence:.0%}**")
                    st.progress(float(confidence))
                    if clean.lower() != review.lower():
                        st.caption(f"Spelling corrected to: {clean}")
            with right:
                st.bar_chart(pd.DataFrame({"Probability": probs}, index=model.classes_))

            st.session_state.history.insert(0, {
                "Review": review[:80] + ("..." if len(review) > 80 else ""),
                "Sentiment": label.capitalize(),
                "Confidence": f"{confidence:.0%}",
            })

    if st.session_state.history:
        st.subheader("🕘 History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)
        c1, c2 = st.columns(2)
        c1.download_button("Download history (CSV)", history_df.to_csv(index=False),
                           file_name="history.csv", mime="text/csv")
        c2.button("Clear history", on_click=clear_history)

# ----- Tab 2: batch prediction -----
with tab2:
    st.write("Upload a CSV file with a column named **review** to predict many reviews at once.")
    sample_csv = "review\nThe movie was fantastic\nWorst film I have ever seen\n"
    st.download_button("Download sample CSV", sample_csv, file_name="sample_reviews.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        if "review" not in data.columns:
            st.error("The file must have a column named 'review'.")
        else:
            data = data.dropna(subset=["review"]).head(500)  # limit to 500 rows
            texts = data["review"].astype(str)
            if use_spell:
                texts = texts.apply(fix_spelling)
            probs = model.predict_proba(texts)
            data["sentiment"] = model.classes_[probs.argmax(axis=1)]
            data["confidence"] = probs.max(axis=1).round(2)

            counts = data["sentiment"].value_counts()
            m1, m2, m3 = st.columns(3)
            m1.metric("Reviews analyzed", len(data))
            m2.metric("Positive", int(counts.get("positive", 0)))
            m3.metric("Negative", int(counts.get("negative", 0)))
            st.bar_chart(counts)
            st.dataframe(data)
            st.download_button("Download results (CSV)", data.to_csv(index=False),
                               file_name="predictions.csv", mime="text/csv")
            st.caption("Only the first 500 rows are analyzed.")

# ----- Tab 3: model performance -----
with tab3:
    st.subheader("Confusion Matrix")
    st.caption("Rows are the real labels, columns are what the model predicted.")
    cm = pd.DataFrame(
        art["cm"],
        index=["Actual: negative", "Actual: positive"],
        columns=["Predicted: negative", "Predicted: positive"],
    )
    st.dataframe(cm)

    st.subheader("Most Important Words")
    w1, w2 = st.columns(2)
    with w1:
        st.write("**Top positive words**")
        st.bar_chart(art["top_pos"])
    with w2:
        st.write("**Top negative words**")
        st.bar_chart(art["top_neg"])

    with st.expander("Look at some training data"):
        sample = art["sample"].copy()
        sample["review"] = sample["review"].str[:200] + "..."
        st.dataframe(sample)

# ----- Tab 4: download -----
with tab4:
    st.subheader("Download the trained model")
    st.write("Get the trained pipeline (TF-IDF + Logistic Regression) as a `.pkl` file.")
    st.download_button("⬇️ Download model (.pkl)", art["model_bytes"],
                       file_name="sentiment_model.pkl", mime="application/octet-stream")
    st.write("Use it in your own Python code:")
    st.code(
        "import joblib\n"
        "model = joblib.load('sentiment_model.pkl')\n"
        "print(model.predict(['What a great movie!']))",
        language="python",
    )
