# Email Spam Detector

A beginner-friendly **Machine Learning / NLP** project that classifies emails as **spam** or **not spam** using Python, text preprocessing, TF-IDF features, and a **Multinomial Naive Bayes** classifier.

---

## Project overview

| Item | Details |
|------|---------|
| **Goal** | Predict whether an email is spam |
| **Labels** | `0` = not spam (ham), `1` = spam |
| **Model** | `MultinomialNB` (scikit-learn) |
| **Features** | TF-IDF (`TfidfVectorizer`, max 5000 features) |
| **Main file** | `Email_Spam_Detector.ipynb` |

---

## Dataset

| Column | Description |
|--------|-------------|
| `text` | Full email content |
| `spam` | `0` = not spam, `1` = spam |

- **File:** `emails.csv` (place in this project folder after downloading from Kaggle)

- **Size:** ~5,574 messages (ham + spam)

---

## Tech stack

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- NLTK

---

## Project structure

```
Email_Spam-Detector/
├── emails.csv                  # Dataset from Kaggle (not included in repo)
├── Email_Spam_Detector.ipynb   # Full pipeline
├── outputs/                    # Saved plots (created when you run the notebook)
│   ├── 1_spam_vs_not_spam.png
│   ├── 2_pie.png
│   ├── 3_word_length.png
│   └── 4_confusion_matrix.png
└── README.md
```

---

## Setup

1. **Clone or download** this project folder.

2. **Add the dataset from Kaggle** — download  then save/prepare `emails.csv` in the project root (same folder as the notebook). A free Kaggle account is required.

3. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   ```

   **Windows:**

   ```powershell
   .\venv\Scripts\activate
   ```

4. **Install dependencies:**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nltk
   ```

5. **Run the notebook** from the project folder:

   ```bash
   jupyter notebook Email_Spam_Detector.ipynb
   ```

   Or open the `.ipynb` file in **VS Code** / **Cursor** and run all cells in order.

---

## Pipeline (steps in the notebook)

| Step | What it does |
|------|----------------|
| **1. EDA** | Load data, check nulls, class counts, word-length plots |
| **2. Preprocessing** | Lowercase, remove punctuation/numbers, stopwords, stemming (NLTK) |
| **3. TF-IDF** | Convert clean text to numbers; **fit on train only** |
| **4. Train/test split** | 80% train / 20% test with `stratify` |
| **5. Train model** | `MultinomialNB` |
| **6. Evaluate** | Accuracy, precision, recall, F1, confusion matrix |
| **7. Predict** | `predict_spam("your message here")` |

**Correct order:** preprocess → split → TF-IDF (fit train) → train → evaluate → predict

---

## Example results

Results vary slightly by run; typical values on this dataset:

| Metric | Approx. value |
|--------|----------------|
| Accuracy | ~0.98 |
| Precision | ~0.99 |
| Recall | ~0.93 |

---

## Try your own text

After running all cells, use the `predict_spam()` function in the last cell:

```python
predict_spam("Congratulations! You won $1000. Click here now.")
# Spam (1)

predict_spam("Meeting is on Tuesday at 3pm. Thanks.")
# Not spam (0)
```

---

## Why Naive Bayes for spam?

- Spam detection depends on **which words appear** (e.g. `free`, `winner`, `click`).
- **Multinomial Naive Bayes** works well with **word counts / TF-IDF**.
- It is **fast** and easy to understand for beginners.
- It assumes words are independent (not fully true in real language), but still performs well on spam vs ham.

Learn more: [scikit-learn — Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

---

## Common beginner mistakes

1. **Train/test split on raw text** before preprocessing — always use `clean_text`.
2. **Fitting TF-IDF on the full dataset** before split — fit on `X_train` only, then `transform` test.
3. **Skipping `stratify=y`** — important when spam is fewer than ham (~24% spam).
4. **Different preprocessing at prediction time** — use the same `preprocess_text()` function.
5. **Saving plots after `plt.show()`** — use `savefig` before `show()` (or skip `show()`).

---


---

## Author

Beginner ML/NLP project — learning portfolio.

---
