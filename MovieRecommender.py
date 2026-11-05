#############################################################################
#
#   Importing required libraries
#
#############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#############################################################################
#
#   Function name:      extract_names
#   Input:              obj_str (string)
#   Description:        Safely parses JSON-like string columns (genres, keywords)
#                       from the TMDB dataset into clean Python lists.
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def extract_names(obj_str):
    try:
        items = ast.literal_eval(obj_str)
        return [item['name'] for item in items if 'name' in item]
    except Exception:
        return []

#############################################################################
#
#   Function name:      load_data
#   Input:              data_path (string)
#   Description:        Loads TMDB dataset, cleans release dates, extracts 
#                       text features, and caches the result.
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
@st.cache_data
def load_data(data_path="tmdb_5000_movies.csv"):
    df = pd.read_csv(data_path)
    
    # Extract list of genres and keywords
    df['genres_list'] = df['genres'].apply(extract_names)
    df['keywords_list'] = df['keywords'].apply(extract_names)
    
    # Text strings for TF-IDF Vectorization
    df['genres_str'] = df['genres_list'].apply(lambda x: " ".join(x))
    df['keywords_str'] = df['keywords_list'].apply(lambda x: " ".join(x))
    df['overview'] = df['overview'].fillna("")
    df['tagline'] = df['tagline'].fillna("")
    
    # Extract Release Year
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    
    # Combine text features for ML model input
    df["combined_features"] = (
        df["genres_str"] + " " + 
        df["keywords_str"] + " " + 
        df["overview"]
    )
    
    return df

#############################################################################
#
#   Function name:      find_optimal_k
#   Input:              tfidf_matrix (sparse matrix), k_min (int), k_max (int)
#   Description:        Automates K-Means cluster optimization by evaluating 
#                       Silhouette Scores across a range of K values.
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def find_optimal_k(tfidf_matrix, k_min=3, k_max=8):
    best_k = k_min
    best_score = -1
    scores = {}

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, labels)
        scores[k] = score

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score, scores

#############################################################################
#
#   Function name:      build_ml_pipeline
#   Input:              _df (DataFrame)
#   Description:        Computes TF-IDF matrix, Cosine Similarity matrix, 
#                       and executes automated Silhouette-based K-Means.
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
@st.cache_resource
def build_ml_pipeline(_df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(_df["combined_features"])
    
    # Cosine Similarity Matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Automated K-Means Search via Silhouette Score
    optimal_k, best_silhouette, k_scores = find_optimal_k(tfidf_matrix, k_min=3, k_max=8)
    
    # Fit model using auto-detected optimal K
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    return tfidf_matrix, cosine_sim, cluster_labels, optimal_k, best_silhouette, k_scores

#############################################################################
#
#   Function name:      render_movie_card
#   Input:              row (Series), match_score (float or None)
#   Description:        Renders a UI card for a movie with badges, 
#                       metrics, and plot summary expander.
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def render_movie_card(row, match_score=None):
    with st.container(border=True):
        col_title, col_meta = st.columns([3, 1])
        
        with col_title:
            year_str = f" ({row['release_year']})" if row['release_year'] > 0 else ""
            st.markdown(f"### 🎬 **{row['title']}**{year_str}")
            
            if isinstance(row['genres_list'], list) and len(row['genres_list']) > 0:
                genre_tags = " • ".join([f"`{g}`" for g in row['genres_list'][:4]])
                st.markdown(f"**Genres:** {genre_tags}")
        
        with col_meta:
            if match_score is not None:
                st.metric("Match Score", f"{match_score:.0f}%")
            st.markdown(f"⭐ **{row['vote_average']:.1f}** / 10")

        if row['tagline']:
            st.markdown(f"_*\"{row['tagline']}\"*_")

        with st.expander("📖 View Synopsis & Details"):
            st.write(row['overview'] if row['overview'] else "No overview available.")
            st.caption(f"**Runtime:** {int(row['runtime']) if pd.notnull(row['runtime']) else 'N/A'} mins | **Auto Cluster:** {row['Cluster']}")

#############################################################################
#
#   Function name:      main
#   Input:              None
#   Description:        Main application controller for Streamlit tabs, ML 
#                       recommendation generation, and automated cluster analytics.
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def main():
    st.set_page_config(page_title="CineMatch AI - Movie Engine", page_icon="🍿", layout="wide")

    # Header
    st.title("🍿 CineMatch AI – Automated Movie Recommendation Engine")
    st.write("Smart content recommendation system powered by **TF-IDF Cosine Similarity** and **Automated Silhouette K-Means Clustering**.")

    # Load dataset
    try:
        df = load_data("tmdb_5000_movies.csv")
    except Exception:
        st.error("⚠️ Dataset `tmdb_5000_movies.csv` not found! Please download it from Kaggle and place it in the project folder.")
        return

    # Sidebar settings
    st.sidebar.header("⚙️ User Controls")
    top_n = st.sidebar.slider("Number of Recommendations", 3, 12, 6)
    min_rating = st.sidebar.slider("Minimum Rating Filter (⭐)", 0.0, 9.0, 5.0, 0.5)

    # Execute ML Pipeline with Automated K-Means
    with st.spinner("🤖 Running Automated TF-IDF Vectorization & Optimal K-Means Clustering..."):
        tfidf_matrix, cosine_sim, cluster_labels, optimal_k, best_silhouette, k_scores = build_ml_pipeline(df)
        df["Cluster"] = cluster_labels

    all_genres = sorted(list(set([g for sublist in df['genres_list'] for g in sublist])))

    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Movie Similarity Matcher", 
        "🎭 Genre & Mood Explorer", 
        "🎲 Smart Surprise Pick",
        "⚡ Automated Clustering Analytics"
    ])

    # ---------------------------------------------------------
    # TAB 1: Movie-to-Movie Similarity Recommender
    # ---------------------------------------------------------
    with tab1:
        st.subheader("Find Movies Similar to Your Favorite")
        
        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            selected_movie = st.selectbox("Search or select a movie you like:", options=df["title"].values, index=0)
        with col_s2:
            st.write(" ")
            st.write(" ")
            recommend_btn = st.button("✨ Recommend Movies", type="primary", use_container_width=True)

        if selected_movie:
            selected_row = df[df["title"] == selected_movie].iloc[0]
            st.info(f"**Selected:** {selected_row['title']} ({selected_row['release_year']}) — ⭐ {selected_row['vote_average']}/10 | Auto-assigned to **Cluster {selected_row['Cluster']}**")

        if recommend_btn and selected_movie:
            idx = df[df["title"].str.lower() == selected_movie.lower()].index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            recommended_indices = []
            for i, score in sim_scores[1:]:
                if df.iloc[i]["vote_average"] >= min_rating:
                    recommended_indices.append((i, score * 100))
                if len(recommended_indices) == top_n:
                    break
            
            st.divider()
            st.markdown(f"### 🎯 Top {len(recommended_indices)} Recommendations for You:")
            
            grid_cols = st.columns(2)
            for rank, (movie_idx, match_score) in enumerate(recommended_indices):
                col = grid_cols[rank % 2]
                with col:
                    render_movie_card(df.iloc[movie_idx], match_score=match_score)

    # ---------------------------------------------------------
    # TAB 2: Genre & Mood Filtering
    # ---------------------------------------------------------
    with tab2:
        st.subheader("Explore Movies by Genre & Rating")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            selected_genres = st.multiselect("Select genres:", options=all_genres, default=["Action", "Science Fiction"])
        with col_g2:
            sort_by = st.selectbox("Sort By:", ["Rating (High to Low)", "Popularity", "Release Year (Newest)"])

        if selected_genres:
            filtered_df = df[df['genres_list'].apply(lambda x: any(g in x for g in selected_genres))]
            filtered_df = filtered_df[filtered_df['vote_average'] >= min_rating]

            if sort_by == "Rating (High to Low)":
                filtered_df = filtered_df.sort_values(by="vote_average", ascending=False)
            elif sort_by == "Popularity":
                filtered_df = filtered_df.sort_values(by="popularity", ascending=False)
            else:
                filtered_df = filtered_df.sort_values(by="release_year", ascending=False)

            st.write(f"Found **{len(filtered_df)}** movies matching your criteria.")
            st.divider()

            grid_cols = st.columns(2)
            for idx, (_, row) in enumerate(filtered_df.head(top_n).iterrows()):
                col = grid_cols[idx % 2]
                with col:
                    render_movie_card(row)
        else:
            st.warning("Please select at least one genre.")

    # ---------------------------------------------------------
    # TAB 3: Smart Surprise Pick
    # ---------------------------------------------------------
    with tab3:
        st.subheader("Undecided? Get a Recommended Surprise Pick")
        
        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            pick_genre = st.selectbox("Preferred Genre (Optional):", ["Any Genre"] + all_genres)
        with col_r2:
            st.write(" ")
            st.write(" ")
            surprise_btn = st.button("🎲 Pick Random Movie", type="primary", use_container_width=True)

        if surprise_btn:
            candidates = df[df['vote_average'] >= max(6.5, min_rating)]
            if pick_genre != "Any Genre":
                candidates = candidates[candidates['genres_list'].apply(lambda x: pick_genre in x)]
            
            if not candidates.empty:
                random_movie = candidates.sample(1).iloc[0]
                st.success("🎉 Recommended Surprise Movie:")
                render_movie_card(random_movie)
            else:
                st.warning("No candidate movies found matching criteria.")

    # ---------------------------------------------------------
    # TAB 4: Automated Clustering Analytics
    # ---------------------------------------------------------
    with tab4:
        st.subheader("⚡ Automated K-Means Optimization Report")
        st.success(f"🤖 Machine Learning Pipeline auto-selected **Optimal K = {optimal_k}** with a peak Silhouette Score of **{best_silhouette:.4f}**")

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.write("**Silhouette Score Evaluation across K Range:**")
            scores_df = pd.DataFrame(list(k_scores.items()), columns=["Clusters (K)", "Silhouette Score"])
            st.dataframe(scores_df, use_container_width=True)

        with col_a2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(scores_df["Clusters (K)"], scores_df["Silhouette Score"], marker="o", color="#1f77b4", linewidth=2)
            ax.axvline(x=optimal_k, color="red", linestyle="--", label=f"Optimal K ({optimal_k})")
            ax.set_title("Automated K Selection via Silhouette Analysis")
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Silhouette Score")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        st.subheader("Cluster Distribution Summary")
        fig_dist, ax_dist = plt.subplots(figsize=(8, 3.5))
        cluster_counts = df["Cluster"].value_counts().sort_index()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis", ax=ax_dist)
        ax_dist.set_title(f"Movie Distribution across {optimal_k} Automated Clusters")
        ax_dist.set_xlabel("Cluster ID")
        ax_dist.set_ylabel("Movie Count")
        ax_dist.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig_dist)

if __name__ == "__main__":
    main()