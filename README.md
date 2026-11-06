# 🍿 CineMatch AI - Automated Movie Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?logo=pandas&logoColor=white)
![TF-IDF](https://img.shields.io/badge/TF--IDF-Text_Vectorization-blueviolet)
![K-Means](https://img.shields.io/badge/K--Means-Clustering-green)
![Dataset](https://img.shields.io/badge/Dataset-TMDB_5000-yellow)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?logo=plotly&logoColor=white)

CineMatch AI is a machine-learning-powered movie discovery web application built with **Python** and **Streamlit**. It helps users discover movies based on content similarity, explore films by genre, and get random highly rated movie suggestions.

The project uses the **TMDB 5000 Movie Dataset** and combines TF-IDF text vectorization, cosine similarity, and K-Means clustering to organize and recommend movies. It also includes an interactive analytics dashboard that shows how the clustering model selects its number of clusters.

---

## ✨ Features

### 1. 🎯 AI Similarity Matcher
- Select a movie from the dataset.
- Find other movies with similar content.
- Rank recommendations using cosine similarity.
- Filter recommendations using a minimum movie-rating threshold.
- Choose how many recommendations to display.
- View match scores, genres, taglines, overviews, ratings, runtime, and cluster labels.

### 2. 🎭 Mood & Genre Explorer
- Browse movies by one or more genres.
- Filter results using the minimum rating setting.
- Sort movies by rating, popularity, or newest release year.
- Display results using interactive movie cards.

### 3. 🎲 Smart Surprise Pick
- Generate a random movie suggestion.
- Optionally filter suggestions by genre.
- Only consider movies rated at least 7.0 or the higher minimum rating selected in the sidebar.

### 4. ⚡ Engine Analytics
- Evaluate K-Means clustering for values of K from 3 to 8.
- Use silhouette scores to identify the optimal cluster count.
- Display clustering scores in a table.
- Visualize the silhouette optimization curve.

### 5. 🎨 Modern UI
- Immersive dark cyber/neon theme.
- Glassmorphism-inspired movie cards.
- Interactive hover animations.
- Gradient genre badges.
- Movie similarity match indicators.
- Responsive two-column layout.

---

## 🧠 Machine Learning Approach

CineMatch AI uses **Content-Based Filtering** to recommend movies by comparing their metadata and textual descriptions.

### 1. Data Preprocessing
The application processes the TMDB dataset by:
- Extracting movie genres.
- Extracting movie keywords.
- Handling missing movie overviews and taglines.
- Extracting release years.
- Combining genres, keywords, and overviews into a single feature column.

### 2. TF-IDF Vectorization
TF-IDF (Term Frequency-Inverse Document Frequency) converts movie descriptions into numerical vectors.

Configuration:
- Removes English stop words.
- Uses a maximum vocabulary size of 5,000 features.

### 3. Cosine Similarity
Cosine similarity measures how similar two movie feature vectors are.

When a user selects a movie, the system:
1. Calculates similarity scores against other movies.
2. Sorts movies by similarity.
3. Filters results based on the minimum rating.
4. Returns the top recommended movies.

The displayed match percentage is the cosine similarity score multiplied by 100. It represents content similarity, not the probability that a user will enjoy the movie.

### 4. K-Means Clustering
K-Means clustering groups movies based on their TF-IDF feature vectors.

The application:
1. Tests cluster counts from K = 3 to K = 8.
2. Calculates the silhouette score for each configuration.
3. Selects the K with the highest silhouette score.
4. Trains the final clustering model.
5. Assigns cluster labels to movies.

The clustering results are displayed in the Engine Analytics dashboard.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data processing and manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning algorithms |
| TF-IDF | Text feature extraction |
| Cosine Similarity | Movie similarity calculation |
| K-Means | Movie clustering |
| Matplotlib | Data visualization |
| Seaborn | Visualization library |
| HTML & CSS | Custom UI styling |

---

## 📂 Project Structure

```text
CineMatch-AI/
│
├── MovieRecommender.py       # Main Streamlit application
├── tmdb_5000_movies.csv      # Required movie dataset
└── README.md                 # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.x
- pip (Python package installer)
- TMDB 5000 Movie Dataset

### Step 1: Clone the Repository

```bash
git clone https://github.com/bemo-codes/Movie-Recommendation-Engine
```

Navigate to the project directory:

```bash
cd <YOUR_REPOSITORY_FOLDER>
```

### Step 2: Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

### Step 4: Download the Dataset

Download the TMDB 5000 Movie Dataset from Kaggle:

**Dataset:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Extract the downloaded archive and place `tmdb_5000_movies.csv` in the project's root directory, alongside `MovieRecommender.py`.

**Important:** The application specifically requires the file named `tmdb_5000_movies.csv`.

### Step 5: Run the Application

Execute the following command:

```bash
streamlit run MovieRecommender.py
```

The application will start locally.

Open the following URL in your browser:

```text
http://localhost:8501
```

---

## 🎮 How to Use

1. Launch the application using Streamlit.
2. Adjust the number of recommendations and minimum rating from the sidebar.
3. Navigate to the **AI Similarity Matcher** tab.
4. Select your favorite movie and click **Discover Matches**.
5. Explore movies using the **Mood & Genre Explorer**.
6. Try the **Smart Surprise Pick** feature to discover random movies.
7. Visit **Engine Analytics** to explore clustering performance and silhouette scores.

---

## ⚡ Performance Optimization

- Uses Streamlit caching to avoid repeatedly loading the dataset.
- Caches the machine learning pipeline using `st.cache_resource`.
- Uses TF-IDF to represent movie descriptions efficiently.
- Calculates cosine similarity for movie recommendations.
- Uses silhouette scores to evaluate clustering configurations.

**Note:** The first application run may take longer because TF-IDF vectorization, the full cosine similarity matrix, and K-Means clustering need to be computed.

---

## ⚠️ Current Limitations

- Recommendations are based on movie genres, keywords, and overviews.
- Does not currently support collaborative filtering.
- Does not maintain user accounts or watch history.
- Match percentages indicate content similarity, not predicted user preferences.
- Movie information depends on the supplied dataset.
- Clustering optimization is limited to K values between 3 and 8.
- Does not fetch live movie information or movie posters.

---

## 🔮 Future Enhancements

- Integrate movie posters and external movie information links.
- Include cast, director, and production information in recommendations.
- Add filters for language, release year, and runtime.
- Implement smarter movie search using fuzzy matching.
- Introduce user profiles and personalized watchlists.
- Explore hybrid recommendation systems.
- Add automated testing and a requirements.txt file.
- Deploy the application online for public access.

---

## 👨‍💻 Author

**Shivam Ramesh Kurlekar**

---

## 📄 Dataset Attribution

This project uses the TMDB 5000 Movie Dataset distributed through Kaggle.

All dataset ownership and intellectual property rights belong to their respective owners. Refer to the original dataset page for applicable terms and attribution requirements.

---

<p align="center">
  Built with Python, Machine Learning, and a passion for movies. 🍿
</p>
