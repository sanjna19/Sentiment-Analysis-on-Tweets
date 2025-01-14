# Sentiment Analysis of Tweets: Obama and Romney

## Project Overview
This project focuses on the **sentiment analysis** of tweets related to former U.S. President Barack Obama and politician Mitt Romney. The goal is to classify tweets into **positive**, **negative**, and **neutral** sentiments, leveraging **Natural Language Processing (NLP)** and **machine learning models**. This analysis offers insights into public attitudes during their campaign times.

---

## Problem Statement
Understanding public sentiment is crucial for gauging political campaigns' effectiveness and public opinion. However, raw tweet data is noisy, unstructured, and requires significant preprocessing to extract meaningful insights. This project addresses these challenges by developing machine learning models to classify sentiments accurately.

---

## Objectives
1. Preprocess raw tweets to prepare a clean and structured dataset.
2. Experiment with various machine learning models (Logistic Regression, Naive Bayes, and SVM).
3. Compare model performances to identify the most effective sentiment classifier.

---

## Dataset
The dataset consists of tweets related to Barack Obama and Mitt Romney. Each tweet is labeled with one of the following sentiments:
- **Positive** (`1`)
- **Negative** (`-1`)
- **Neutral** (`0`)
- **Mixed** (`2`)

### Preprocessing Steps
1. **Text Cleaning:** Removed special characters, mentions, hashtags, and hyperlinks.
2. **Tokenization:** Split tweets into individual words.
3. **Stopword Removal:** Eliminated common words like "and" and "the."
4. **Stemming and Lemmatization:** Reduced words to their base forms (e.g., "running" → "run").
5. **TF-IDF Vectorization:** Transformed cleaned text into numerical features while capturing term importance.

---

## Methodology
### **Machine Learning Models**
1. **Logistic Regression:**
   - Chosen for its high performance with feature-rich datasets.
   - Supports probabilistic predictions and handles imbalanced datasets well.
2. **Naive Bayes:**
   - Computationally efficient but assumes feature independence, which may limit its performance.
3. **Support Vector Machines (SVM):**
   - Effective in capturing class boundaries but less robust with noisy or imbalanced data.

### **Feature Engineering**
- Combined **TF-IDF features** with metadata (e.g., tweet length, number of hashtags, mentions).
- Used **Synthetic Minority Over-sampling Technique (SMOTE)** to balance class distribution during training.

---

## Experimental Results
### Barack Obama Dataset
| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 85%      | 87%       | 86%    | 85%      |
| SVM                 | 78%      | 73%       | 74%    | 72%      |
| Naive Bayes         | 62%      | 63%       | 62%    | 62%      |

### Mitt Romney Dataset
| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 71%      | 73%       | 70%    | 72%      |
| SVM                 | 57%      | 56%       | 57%    | 55%      |
| Naive Bayes         | 58%      | 58%       | 58%    | 52%      |

---

## Conclusion
- **Logistic Regression** consistently outperformed SVM and Naive Bayes, achieving the highest accuracy and F1 scores for both datasets.
- SVM performed reasonably well but struggled with class imbalances.
- Naive Bayes was the least effective due to its assumption of feature independence, which is not suitable for textual data.

**Future Work:**
- Explore deep learning models like **BERT** for better sentiment classification.
- Integrate advanced feature engineering techniques to capture sentiment nuances.
- Analyze datasets with stronger sentiment distributions for more granular insights.

---

## Technologies Used
- **Programming Language:** Python
- **Libraries:** scikit-learn, pandas, NumPy, NLTK, matplotlib
- **Algorithms:** Logistic Regression, Naive Bayes, SVM

