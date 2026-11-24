# ReviewSense: Movie Review Sentiment Analyzer

An interactive NLP web app that classifies movie reviews as **positive** or **negative**.

## Features
- KPI cards for Accuracy, Precision, Recall and F1 score
- Single review prediction with confidence and probability chart
- One-click example reviews and prediction history (downloadable)
- Batch prediction: upload a CSV and download the results
- Model performance tab: confusion matrix and most important words
- Automatic spelling correction (can be switched off in the sidebar)
- Download the trained model as a `.pkl` file

## Tech Stack
Python, pandas, scikit-learn (TF-IDF with bigrams + Logistic Regression), pyspellchecker, Streamlit

## Dataset
`reviews.csv` - 30,000 shuffled movie reviews (15,000 positive, 15,000 negative), a subset of the IMDB 50K reviews dataset. Duplicates and HTML tags were removed.

## How it works
1. Spelling mistakes are fixed with pyspellchecker
2. Reviews are converted to numbers using TF-IDF (single words and word pairs)
3. A Logistic Regression model is trained on 80% of the data when the app starts
4. The remaining 20% (6,000 reviews) is used to measure Accuracy (~91.6%), Precision (~91.0%) and Recall (~92.5%)
5. The user types a review and the app predicts the sentiment

## Run locally
```
python -m venv venv
pip install -r requirements.txt
streamlit run app.py
```

## Live Demo
Add your Streamlit link here after deploying.
