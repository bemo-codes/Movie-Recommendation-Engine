# CineMatch AI – Automated Movie Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Similarity%20%26%20Clustering-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

An automated Machine Learning movie recommendation web application built with Streamlit and Scikit-Learn using the TMDB 5000 Kaggle dataset. **CineMatch AI** combines **TF-IDF Text Vectorization**, **Cosine Similarity** content matching, and **Automated Silhouette-based K-Means Clustering**.

---

## 🔬 Machine Learning Pipeline

### 1. Feature Engineering & TF-IDF Vectorization
Genres, plot overviews, and keywords are cleaned and concatenated into a unified text feature space. Text is converted into numerical vectors using `TfidfVectorizer` (Vocabulary limit: 5,000 terms):

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{\vert{}D\vert{}}{\vert{}\{d \in D : t \in d\}\vert{}}\right)$$

### 2. Cosine Similarity Matching
Content proximity between target movies and candidate titles is calculated using pairwise Cosine Similarity:

$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\Vert{}A\Vert{} \Vert{}B\Vert{}}$$

### 3. Automated K-Means Clustering via Silhouette Analysis
The engine automatically evaluates cluster quality across $K \in [3, 8]$ using Silhouette Scores:

$$S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where $a(i)$ represents mean intra-cluster distance and $b(i)$ represents mean nearest-cluster distance. The system automatically selects $K$ corresponding to the maximum mean Silhouette Score.

---

## Features

- **Movie Similarity Matcher:** Search any movie to generate top recommendations with match percentage scores.
- **Genre & Mood Explorer:** Filter titles by single or multiple genres with real-time sorting (Rating, Popularity, Release Date).
- **Automated K-Means Analytics:** Real-time optimization report showing Silhouette Score curve and cluster distribution.
- **Surprise Pick Generator:** 1-Click randomized selection for undecided users filtered by rating thresholds.

---

## Project Structure

```text
Movie-Recommendation-Engine/
│
├── MovieRecommender.py
├── tmdb_5000_movies.csv
├── requirements.txt
├── .gitignore
├── README.md
├── LICENSE
└── screenshots/