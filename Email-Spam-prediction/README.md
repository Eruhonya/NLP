# Email Spam Detection — Enron Corpus

Classifying emails as **spam** or **ham** (legitimate) using the Enron email corpus, with a focus on data-quality forensics and the engineering trade-off between model accuracy and runtime performance.

## Problem

Given the raw text of an email, predict whether it is **spam (1)** or **ham (0)**. The dataset is imbalanced (~75% ham / ~25% spam), so precision, recall and F1 are tracked rather than accuracy alone.

## Dataset

The **Enron Email Dataset** — one of the few large, real-world, publicly available corporate email corpora. It emerged from the investigation into the 2001 collapse of Enron Corporation: the U.S. Department of Justice seized internal correspondence, the FERC made much of it public in 2003, and researchers later cleaned and structured it for analysis. The official corpus covers communication from 1998 to early 2002.

| Column | Description |
|---|---|
| `text` | Raw text content of the email |
| `spam` | Binary label — spam (1) or ham (0) |

## Approach

1. **EDA** — class balance, missing-value checks, and word-length analysis of ham vs. spam.
2. **Data-quality forensics (Research Note)** — diagnosing and resolving real artifacts in the corpus:
   - *Chronological noise:* fuzzy date parsing misread numbers in spam subject lines (e.g. "80% discount") as years and backfilled missing years with the current system date. Resolved with a strict 1998–2002 chronological mask.
   - *Post-bankruptcy emails:* verified that valid ham into early 2002 is historically accurate (liquidation and investigations ran for years).
   - *Outliers:* ultra-short emails (2–20 words) are legitimate quick-replies; ultra-long emails (8,000+ words) are verified news digests about the Enron crisis.
3. **Feature engineering** — metadata features (email length, word count, uppercase ratio, exclamation count, presence of "spammy" keywords) alongside cleaned text (header stripping, URL/punctuation removal).
4. **Modeling** — two pipeline variants compared: **with metadata features** vs. **pure text only**, across SVM, Random Forest, XGBoost, Logistic Regression and Multinomial Naive Bayes, all using TF-IDF (unigrams + bigrams).
5. **Runtime analysis** — measuring training and prediction time per model to assess production scalability.

## Results

The best results came from **pure text + a well-tuned TF-IDF vectorizer** — adding hand-engineered metadata features actually *degraded* performance and confused the models.

| Model | Quality (F1 / Accuracy) | Prediction speed | Best for |
|---|---|---|---|
| **Linear SVC** | **Maximum (~0.99)** | ~0.63 s (≈3× slower) | High-stakes filtering where precision is paramount |
| **Logistic Regression** | High (just behind SVC) | **~0.20 s (fastest)** | High-volume services where speed and scale dominate |

## Key takeaways

- **Pure NLP beat meta-features:** manually engineered signals (length, exclamation count, keyword flags) hurt rather than helped; clean text + TF-IDF was strongest.
- **No single "best" model — it depends on requirements:** SVC for maximum precision on confidential/high-value mail; Logistic Regression for real-time processing at massive scale, saving substantial compute.
- **Data forensics matter:** a large share of the work was identifying and correcting parsing artifacts before modeling — the difference between an academic notebook and a production-minded approach.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · seaborn

