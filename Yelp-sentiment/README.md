# Yelp Reviews — Binary Sentiment Classification

Predicting whether a Yelp review is **positive** or **negative** from its text, with an emphasis on lexicon-based feature engineering, rigorous model comparison, and honest analysis of where classical bag-of-words approaches break down.

## Problem

Given the text of a Yelp review, classify its sentiment as positive or negative. The label is derived from the star rating: **1★ → negative (0)**, **5★ → positive (1)**. Intermediate ratings (2–4★) are excluded to obtain cleaner class separation — a deliberate simplification whose trade-offs are discussed in the conclusions.

## Dataset

A well-known educational subset of **10,000 reviews** originating from Kaggle's *Yelp Business Rating Prediction* competition, itself derived from the **Yelp Dataset Challenge**. Each row is one review of one business by one user.

| Column | Description |
|---|---|
| `stars` | Rating of the **business** by the reviewer (1–5) — source of the target |
| `text` | Review text — main feature source |
| `cool` / `useful` / `funny` | Votes the **review itself** received from other users |
| `business_id` / `user_id` / `review_id` / `date` / `type` | Identifiers and metadata |

**Source:** Yelp Open Dataset (released for educational use) / Kaggle.

## Approach

1. **EDA** — target balance, review length, temporal patterns, vote distributions, and per-user / per-business activity.
2. **Feature engineering** — structural features (word/sentence/digit/punctuation counts via spaCy) and lexicon-based sentiment scores from **AFINN, VADER, Bing, NRC and TextBlob**, including normalized and binarized variants.
3. **Feature relevance** — Mutual Information, Pearson and Spearman correlation to identify the most informative features.
4. **Text vectorization** — `CountVectorizer` baseline, then **TF-IDF** with spaCy lemmatization (unigrams + bigrams, `min_df=5`, `sublinear_tf=True`).
5. **Numeric transforms** — Yeo-Johnson to reduce skew, with scaler comparison (MinMax / Standard / Robust).
6. **Feature selection** — `SelectKBest` with chi² (optimum around k ≈ 2590).
7. **Modeling & tuning** — Logistic Regression, Linear SVC, Multinomial NB, Random Forest, XGBoost, ExtraTrees, PassiveAggressive; GridSearchCV tuning; a stacking ensemble.
8. **Error analysis** — inspection of misclassified reviews and what bag-of-words misses.

## Results

| Model | F1-macro (Test) | Balanced Acc. (Test) | Train/Test gap | Notes |
|---|---|---|---|---|
| **Stacking (5 base + meta-LR)** | **0.930** | 0.928 | 0.065 | Best absolute metrics |
| **Logistic Regression** | 0.921 | **0.934** | **0.038** | Best balance; final choice |
| Multinomial NB | ~0.885 | 0.902 | smallest | Strong, robust baseline |
| Linear SVC | ~0.853 | 0.837 | larger | Overfits, needs regularization |
| XGBoost / Random Forest | lower | lower | large | Overfit on sparse TF-IDF |

**Final model: Logistic Regression.** The stacking ensemble scores marginally higher (+0.9% F1-macro), but Logistic Regression is faster, interpretable and generalizes better (smallest train/test gap) — the small accuracy gain does not justify the added complexity and overfitting risk.

## Key takeaways

- Quality text preprocessing and vectorization matter more than feature breadth — most engineered features added noise rather than signal; only `month` and `user_count` gave a small, stable lift.
- Under class imbalance, F1-macro and Balanced Accuracy are far more informative than accuracy or ROC-AUC.
- Linear models are the sweet spot for sparse TF-IDF text: high quality, minimal overfitting, fast training.
- Bag-of-words has real limits — it misses word order, negation, sarcasm and narrative arc (e.g. a grateful review opening with "worst day ever" gets misread as negative).

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · NLTK · spaCy · AFINN · VADER · NRCLex · TextBlob · matplotlib · seaborn

## How to run

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost nltk spacy textblob textstat emoji nrclex afinn vaderSentiment
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
```

Then download the required NLTK data inside the notebook:

```python
import nltk
for pkg in ['stopwords', 'opinion_lexicon', 'wordnet', 'punkt', 'punkt_tab']:
    nltk.download(pkg)
```
