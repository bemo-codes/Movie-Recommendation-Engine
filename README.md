# CineMatch AI – Intelligent Movie Discovery

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Similarity%20%26%20Clustering-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

CineMatch AI is a personalized movie recommendation engine built with Python, Streamlit, and Scikit-Learn. Powered by the TMDB 5000 dataset, it combines **TF-IDF Text Vectorization**, **Cosine Similarity** content matching, and **Automated Silhouette-optimized K-Means Clustering** wrapped in a highly immersive, cyber-neon glassmorphic UI.

## ✨ Features

- **🎯 AI Similarity Matcher:** Search for any movie to generate top recommendations accompanied by calculated percentage match scores.
- **🎭 Mood & Genre Explorer:** Filter and explore titles by single or multiple genres with dynamic multi-attribute sorting (Rating, Popularity, Release Date).
- **🎲 Smart Surprise Pick:** A 1-click randomized recommendation generator for undecided users, filtered by rating thresholds.
- **⚡ Engine Analytics:** Real-time optimization report rendering the Silhouette Score curve and movie count distribution across auto-selected K-Means clusters.
- **🌌 Immersive UI:** Custom cyber-neon CSS theme featuring frosted glass cards, gradient badges, and interactive hover animations.

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit (with custom HTML/CSS injection)
- **Machine Learning:** Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`, `KMeans`, `silhouette_score`)
- **Data Manipulation & Viz:** Pandas, NumPy, Matplotlib, Seaborn

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/bemo-codes/Movie-Recommendation-Engine.git](https://github.com/bemo-codes/Movie-Recommendation-Engine.git)
cd Movie-Recommendation-Engine

2. Install Dependencies
Bash
pip install -r requirements.txt
3. Add the Dataset
Download the TMDB 5000 Movie Dataset (tmdb_5000_movies.csv) from Kaggle and place it in the root directory of the project.

4. Run the Application
Bash
streamlit run MovieRecommender.py
The application will launch locally at http://localhost:8501.

👤 Author
Shivam Ramesh Kurlekar

GitHub: bemo-codes

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.