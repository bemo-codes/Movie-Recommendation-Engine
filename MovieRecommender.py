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
#   UI Enhancement: Inject Custom CSS (Highly Vibrant Immersive Cyber-Theme)
#
#############################################################################
def inject_custom_css():
    css = """
    <style>
        /* Main Cinematic Gradient Background */
        .stApp {
            background: linear-gradient(135deg, #070913 0%, #0f1123 50%, #170d2b 100%);
            color: #f1f5f9;
        }
        
        /* Vibrant custom sidebar */
        [data-testid="stSidebar"] {
            background-color: #0b0c16 !important;
            border-right: 1px solid rgba(139, 92, 246, 0.2);
        }
        
        /* Glassmorphic Cards with Cyber-Neon Glow */
        .movie-card {
            background: rgba(15, 23, 42, 0.45);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 24px;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            position: relative;
            overflow: hidden;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        
        /* Interactive Glowing Hover & Scale Animations */
        .movie-card:hover {
            transform: translateY(-8px) scale(1.02);
            border-color: #f43f5e;
            box-shadow: 0 15px 35px rgba(244, 63, 94, 0.3), 
                        0 0 15px rgba(99, 102, 241, 0.2), 
                        inset 0 0 15px rgba(244, 63, 94, 0.05);
            background: rgba(15, 23, 42, 0.6);
        }
        
        /* Movie Title Gradient typography */
        .movie-title {
            color: #ffffff;
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 1.55rem;
            font-weight: 850;
            letter-spacing: -0.5px;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.15);
            padding-right: 80px; /* Space for Match badge */
        }
        
        .movie-year { 
            color: #94a3b8; 
            font-weight: 500; 
            font-size: 1.1rem; 
            margin-left: 6px;
        }
        
        /* Highlighted Tagline */
        .tagline { 
            color: #22d3ee; 
            font-style: italic; 
            font-size: 0.98rem; 
            margin-bottom: 14px;
            font-weight: 500;
            border-left: 2px solid #ec4899;
            padding-left: 10px;
        }
        
        /* Plot Overview description */
        .movie-overview { 
            color: #cbd5e1; 
            font-size: 0.92rem; 
            line-height: 1.65; 
            margin-bottom: 24px;
            flex-grow: 1;
        }
        
        /* Vivid Multi-colored Genre Badges */
        .genre-container { 
            margin-bottom: 18px; 
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        
        .genre-tag {
            background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
            color: #ffffff;
            padding: 4px 14px;
            border-radius: 50px;
            font-size: 0.76rem;
            font-weight: 700;
            box-shadow: 0 3px 8px rgba(236, 72, 153, 0.25);
            letter-spacing: 0.3px;
        }
        
        /* Vibrant Glowing Ribbon for Match Score */
        .match-badge {
            position: absolute;
            top: 22px;
            right: -35px;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 6px 42px;
            font-size: 0.8rem;
            font-weight: 800;
            transform: rotate(45deg);
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
            letter-spacing: 1px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
        }
        
        /* Card Footer elements */
        .card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding-top: 18px;
            font-size: 0.88rem;
            color: #94a3b8;
            margin-top: auto;
        }
        
        .rating { 
            color: #fbbf24; 
            font-weight: 850; 
            font-size: 1.05rem; 
            text-shadow: 0 0 10px rgba(251, 191, 36, 0.25);
        }
        
        .runtime {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            color: #a7f3d0;
            font-weight: 500;
        }
        
        .cluster-badge { 
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
            padding: 4px 12px; 
            border-radius: 8px; 
            font-size: 0.78rem; 
            font-weight: 700; 
            color: #38bdf8; 
            border: 1px solid rgba(56, 189, 248, 0.2);
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        
        /* Slider & Controls High Contrast Styles */
        .stSlider [data-baseweb="slider"] { margin-bottom: 20px; }
        .stButton button {
            background: linear-gradient(90deg, #ec4899 0%, #8b5cf6 100%) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(236, 72, 153, 0.6) !important;
        }
        
        /* Custom Header effects */
        .sparkle-header {
            background: linear-gradient(90deg, #38bdf8 0%, #ec4899 50%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 0px;
            letter-spacing: -1.5px;
            filter: drop-shadow(0px 2px 10px rgba(139, 92, 246, 0.3));
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

#############################################################################
#
#   Function name:      render_immersive_card
#   Input:              row (Series), match_score (float or None)
#   Description:        Generates zero-indent HTML for the custom-styled 
#                       movie cards to prevent code block parsing.
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def render_immersive_card(row, match_score=None):
    # Format Genres
    genres = row.get('genres_list', [])
    genre_html = "".join([f'<span class="genre-tag">{g}</span>' for g in genres[:4]])
    
    # Format Match Badge Ribbon
    match_html = f'<div class="match-badge">{match_score:.0f}% MATCH</div>' if match_score else ""
    
    # Format Overview (truncate to maintain symmetric alignment)
    overview = row['overview'] if pd.notnull(row['overview']) and row['overview'] else "No synopsis available."
    if len(overview) > 220:
        overview = overview[:217] + "..."
        
    # Format Tagline
    tagline = f'<p class="tagline">"{row["tagline"]}"</p>' if pd.notnull(row['tagline']) and row['tagline'] else ""
    
    # Details & parameters
    runtime = int(row['runtime']) if pd.notnull(row['runtime']) and row['runtime'] > 0 else 'N/A'
    year_str = f"({row['release_year']})" if row['release_year'] > 0 else ""

    # Generate raw card HTML block without code indentation
    card_html = f"""<div class="movie-card">
{match_html}
<h3 class="movie-title">{row['title']} <span class="movie-year">{year_str}</span></h3>
<div class="genre-container">{genre_html}</div>
{tagline}
<p class="movie-overview">{overview}</p>
<div class="card-footer">
<span class="rating">⭐ {row['vote_average']:.1f} <span style="font-size:0.75rem; color:#64748b;">/ 10</span></span>
<span class="runtime">⏱️ {runtime} min</span>
<span class="cluster-badge">Cluster {row['Cluster']}</span>
</div>
</div>"""
    st.html(card_html)

#############################################################################
#
#   Dataset Parse & Machine Learning helpers
#
#############################################################################
def extract_names(obj_str):
    try:
        items = ast.literal_eval(obj_str)
        return [item['name'] for item in items if 'name' in item]
    except Exception:
        return []

@st.cache_data
def load_data(data_path="tmdb_5000_movies.csv"):
    df = pd.read_csv(data_path)
    df['genres_list'] = df['genres'].apply(extract_names)
    df['keywords_list'] = df['keywords'].apply(extract_names)
    df['genres_str'] = df['genres_list'].apply(lambda x: " ".join(x))
    df['keywords_str'] = df['keywords_list'].apply(lambda x: " ".join(x))
    df['overview'] = df['overview'].fillna("")
    df['tagline'] = df['tagline'].fillna("")
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    
    df["combined_features"] = df["genres_str"] + " " + df["keywords_str"] + " " + df["overview"]
    return df

def find_optimal_k(tfidf_matrix, k_min=3, k_max=8):
    best_k, best_score = k_min, -1
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

@st.cache_resource
def build_ml_pipeline(_df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(_df["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    optimal_k, best_silhouette, k_scores = find_optimal_k(tfidf_matrix, k_min=3, k_max=8)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    return tfidf_matrix, cosine_sim, cluster_labels, optimal_k, best_silhouette, k_scores

#############################################################################
#
#   Visual App Dashboard
#
#############################################################################
def main():
    st.set_page_config(page_title="CineMatch AI", page_icon="🍿", layout="wide")
    
    # Style injection 
    inject_custom_css()

    # Title Hero
    st.markdown("<h1 class='sparkle-header'>🍿 CineMatch AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #38bdf8; font-size: 1.2rem; font-weight: 500; margin-bottom: 40px;'>Intelligent Movie Discovery powered by Machine Learning</p>", unsafe_allow_html=True)

    try:
        df = load_data("tmdb_5000_movies.csv")
    except Exception:
        st.error("⚠️ Dataset `tmdb_5000_movies.csv` not found in root workspace directory.")
        return

    # Sidebar Engine Parameters
    st.sidebar.markdown("### ⚙️ Engine Parameters")
    top_n = st.sidebar.slider("Number of Recommendations", 2, 12, 6)
    min_rating = st.sidebar.slider("Minimum Rating Filter (⭐)", 0.0, 9.0, 5.0, 0.5)

    # ML Pipeline
    with st.spinner("🤖 Booting AI Engine & processing model components..."):
        tfidf_matrix, cosine_sim, cluster_labels, optimal_k, best_silhouette, k_scores = build_ml_pipeline(df)
        df["Cluster"] = cluster_labels

    all_genres = sorted(list(set([g for sublist in df['genres_list'] for g in sublist])))

    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 AI Similarity Matcher", 
        "🎭 Mood & Genre Explorer", 
        "🎲 Smart Surprise Pick",
        "⚡ Engine Analytics"
    ])

    # ---------------------------------------------------------
    # TAB 1: Movie-to-Movie Similarity Matcher
    # ---------------------------------------------------------
    with tab1:
        st.write("### Find Movies Similar to Your Favorites")
        
        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            selected_movie = st.selectbox("Search or select a target movie:", options=df["title"].values, index=0, label_visibility="collapsed")
        with col_s2:
            recommend_btn = st.button("✨ Discover Matches", type="primary", use_container_width=True)

        if recommend_btn and selected_movie:
            idx = df[df["title"].str.lower() == selected_movie.lower()].index[0]
            sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
            
            recommended_indices = []
            for i, score in sim_scores[1:]:
                if df.iloc[i]["vote_average"] >= min_rating:
                    recommended_indices.append((i, score * 100))
                if len(recommended_indices) == top_n: 
                    break
            
            st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)
            
            # Grid display configuration for glassmorphic cards
            grid_cols = st.columns(2)
            for rank, (movie_idx, match_score) in enumerate(recommended_indices):
                with grid_cols[rank % 2]:
                    render_immersive_card(df.iloc[movie_idx], match_score=match_score)

    # ---------------------------------------------------------
    # TAB 2: Mood & Genre Explorer
    # ---------------------------------------------------------
    with tab2:
        st.write("### Curate by Cinematic Genre")
        
        col_g1, col_g2 = st.columns([2, 1])
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

            st.caption(f"Showing top {top_n} results from {len(filtered_df)} matches.")
            st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)

            grid_cols = st.columns(2)
            for idx, (_, row) in enumerate(filtered_df.head(top_n).iterrows()):
                with grid_cols[idx % 2]:
                    render_immersive_card(row)

    # ---------------------------------------------------------
    # TAB 3: Smart Surprise Pick
    # ---------------------------------------------------------
    with tab3:
        st.write("### Let the Engine Decide")
        
        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
        with col_r2:
            pick_genre = st.selectbox("Filter Surprise by Genre (Optional):", ["Any Genre"] + all_genres)
            surprise_btn = st.button("🎲 Generate Random Masterpiece", type="primary", use_container_width=True)

            if surprise_btn:
                candidates = df[df['vote_average'] >= max(7.0, min_rating)]
                if pick_genre != "Any Genre":
                    candidates = candidates[candidates['genres_list'].apply(lambda x: pick_genre in x)]
                
                if not candidates.empty:
                    st.markdown("<br>", unsafe_allow_html=True)
                    render_immersive_card(candidates.sample(1).iloc[0])
                else:
                    st.warning("No highly-rated candidate movies found for this genre.")

    # ---------------------------------------------------------
    # TAB 4: Engine Analytics
    # ---------------------------------------------------------
    with tab4:
        st.write("### AI Clustering Optimization Report")
        st.info(f"**Engine Output:** Pipeline auto-selected **Optimal K = {optimal_k}** achieving a peak Silhouette Score of **{best_silhouette:.4f}**")

        col_a1, col_a2 = st.columns([1, 2])
        with col_a1:
            scores_df = pd.DataFrame(list(k_scores.items()), columns=["Clusters (K)", "Silhouette Score"])
            st.dataframe(scores_df, use_container_width=True)

        with col_a2:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('#070913')
            ax.set_facecolor('#070913')
            
            ax.plot(scores_df["Clusters (K)"], scores_df["Silhouette Score"], marker="o", color="#e11d48", linewidth=3)
            ax.axvline(x=optimal_k, color="#38bdf8", linestyle="--", label=f"Optimal K ({optimal_k})")
            ax.set_title("Silhouette Optimization Curve", color="#f8fafc", weight="bold")
            ax.set_xlabel("Number of Clusters (K)", color="#94a3b8")
            ax.set_ylabel("Silhouette Score", color="#94a3b8")
            ax.grid(True, color="#1e293b", linestyle="--", alpha=0.5)
            ax.legend()
            
            st.pyplot(fig)

if __name__ == "__main__":
    main()