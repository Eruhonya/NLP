# NLP Projects

A collection of two end-to-end Natural Language Processing projects, each covering a distinct text-classification task — from exploratory data analysis and feature engineering to model comparison and evaluation.

| Project | Task | Dataset | Best model | Headline result |
|---|---|---|---|---|
| [Yelp Sentiment Analysis](./yelp-sentiment-analysis) | Binary sentiment classification | Yelp reviews (10k) | Logistic Regression / Stacking | F1-macro **0.92–0.93** |
| [Email Spam Detection](./email-spam-detection) | Binary spam vs. ham classification | Enron emails | Linear SVC / Logistic Regression | F1 / Accuracy **~0.99** |

## Overview

Both projects share a common methodology — TF-IDF text vectorization, careful handling of class imbalance (`class_weight='balanced'`, macro-averaged metrics), and a comparison of multiple classifiers — but differ in domain and emphasis:

- **Yelp Sentiment Analysis** focuses on *deep feature engineering*: lexicon-based sentiment scores (AFINN, VADER, Bing, NRC, TextBlob), spaCy lemmatization, Yeo-Johnson transforms, feature selection, hyperparameter tuning, ensembling and error analysis.
- **Email Spam Detection** focuses on *production trade-offs*: data-quality forensics on the Enron corpus, the contribution of metadata features vs. pure text, and a runtime/scalability analysis comparing model speed against accuracy.

## Common tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · NLTK · spaCy · matplotlib · seaborn

## Repository structure

```
NLP/
├── README.md
├── yelp-sentiment-analysis/
│   ├── README.md
│   ├── Yelp_Sentiment_Analysis.ipynb
│   └── data/
└── email-spam-detection/
    ├── README.md
    ├── Email_Spam_Detection.ipynb
    └── data/
```
