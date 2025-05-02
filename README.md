# 🐦 Tweet Virality Prediction using Machine Learning

This project aims to predict whether a tweet will go **viral** based on its content and engagement metrics like **likes, retweets, and hashtags**. It uses natural language processing (NLP) and machine learning to classify tweets as `Viral` or `Non-Viral`.

---

## 📁 Dataset

- `tweet_content-engagement_dataset.csv`: Contains 500 tweets with the following columns:
  - `Tweet Content`: The text of the tweet
  - `Likes`: Number of likes
  - `Retweets`: Number of retweets
  - `Number of Hashtags`: Count of hashtags in the tweet
  - `Virality`: Target label - `Viral` or `Non-Viral`

The dataset is **balanced** with an equal number of viral and non-viral tweets.

---

## 🧠 Objective

To build a machine learning model that predicts a tweet’s virality using:
- Tweet text (via TF-IDF)
- Likes, Retweets, and Hashtag Count

---

## 📌 Steps Followed

1. **Data Preprocessing**
   - Duplicate removal, null checks
2. **Exploratory Data Analysis**
   - Virality counts, distributions of likes/retweets
3. **Feature Engineering**
   - TF-IDF on tweet content
   - Combine with numerical features
4. **Model Training**
   - Logistic Regression for binary classification
5. **Evaluation**
   - Confusion matrix, accuracy, precision, recall, F1-score

---

## 📊 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`

---