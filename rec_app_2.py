"""
Movie Recommendation System — Streamlit App
Item-Based Collaborative Filtering using MovieLens Dataset

To run locally:
    pip install streamlit pandas numpy scikit-learn
    streamlit run movie_recommender_app.py

To deploy on Streamlit Cloud:
    1. Push this file + ratings.csv + movies.csv to a GitHub repo
    2. Go to https://streamlit.io/cloud and connect your repo
    3. Set the main file to movie_recommender_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Movie Recommender',
    page_icon='🎬',
    layout='centered',
)


# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
    <style>
        .title { font-size: 2.4rem; font-weight: 800; margin-bottom: 0; }
        .subtitle { color: gray; margin-top: 0; margin-bottom: 2rem; }
        .rec-card {
            background: #1e1e2e;
            border-radius: 10px;
            padding: 0.7rem 1.2rem;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .rec-rank { color: #888; font-size: 0.85rem; min-width: 2rem; }
        .rec-title { font-weight: 600; font-size: 1rem; flex: 1; padding: 0 1rem; }
        .rec-title a { color: inherit; text-decoration: none; }
        .rec-title a:hover { color: #7dd3fc; text-decoration: underline; }
        .rec-score { color: #7dd3fc; font-size: 0.9rem; font-weight: 600; }
        .imdb-badge {
            font-size: 0.7rem; font-weight: 700; color: #000;
            background: #f5c518; border-radius: 3px;
            padding: 1px 5px; margin-left: 8px;
            text-decoration: none; vertical-align: middle;
        }
        .imdb-badge:hover { background: #e6b800; }
    </style>
""", unsafe_allow_html=True)


# ── Data Loading (cached so it only runs once) ────────────────────────────────

@st.cache_data(show_spinner='Loading dataset...')
def load_data(data_path, min_ratings):
    """Load, merge, filter, and pivot the MovieLens dataset."""
    ratings = pd.read_csv(f'{data_path}/ratings.csv')
    movies  = pd.read_csv(f'{data_path}/movies.csv')

    df = ratings.merge(movies[['movieId', 'title']], on='movieId')

    movie_counts   = df.groupby('title')['rating'].count()
    popular_movies = movie_counts[movie_counts >= min_ratings].index
    df             = df[df['title'].isin(popular_movies)]

    matrix = df.pivot_table(index='userId', columns='title', values='rating')
    return matrix


@st.cache_data(show_spinner='Computing similarity matrices...')
def compute_all_similarities(_matrix):
    """Compute all three similarity matrices. Underscore prefix avoids hashing the df."""
    # Cosine similarity
    filled = _matrix.fillna(0).T
    cosine_sim = pd.DataFrame(
        cosine_similarity(filled),
        index=filled.index,
        columns=filled.index
    )

    # Normalized cosine similarity
    user_mean  = _matrix.mean(axis=1)
    normalized = _matrix.sub(user_mean, axis=0).fillna(0)
    norm_sim   = pd.DataFrame(
        cosine_similarity(normalized.T),
        index=_matrix.columns,
        columns=_matrix.columns
    )

    # Pearson correlation
    pearson_sim = _matrix.corr(method='pearson')

    return cosine_sim, norm_sim, pearson_sim


# ── IMDB Link Lookup ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_imdb_lookup(data_path):
    """
    Build a title -> IMDB URL lookup dict from links.csv and movies.csv.
    links.csv has: movieId, imdbId, tmdbId
    imdbId is stored as a number — full URL is https://www.imdb.com/title/tt{imdbId:07d}/
    """
    movies = pd.read_csv(f'{data_path}/movies.csv')
    links  = pd.read_csv(f'{data_path}/links.csv')

    merged = movies.merge(links[['movieId', 'imdbId']], on='movieId', how='left')

    # Zero-pad imdbId to 7 digits and construct the IMDB URL
    merged['imdb_url'] = merged['imdbId'].apply(
        lambda x: f'https://www.imdb.com/title/tt{int(x):07d}/' if pd.notna(x) else None
    )

    return dict(zip(merged['title'], merged['imdb_url']))


# ── Recommender Logic ─────────────────────────────────────────────────────────

def get_recommendations(movie_title, n, similarity_df):
    """Return top-N recommendations as a DataFrame, or None if no match."""
    matches = [t for t in similarity_df.index if movie_title.lower() in t.lower()]

    if not matches:
        return None, None, []

    matched_title = matches[0]
    scores = similarity_df[matched_title].drop(index=matched_title).sort_values(ascending=False)

    result_df = scores.head(n).reset_index()
    result_df.columns = ['Movie Title', 'Similarity Score']
    result_df.index  += 1
    result_df['Similarity Score'] = result_df['Similarity Score'].round(4)

    return result_df, matched_title, matches


# ── Sidebar: Settings ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header('⚙️ Settings')

    data_path = st.text_input(
        'Dataset path',
        value='ml-latest',
        help='Path to the folder containing ratings.csv and movies.csv'
    )

    min_ratings = st.slider(
        'Minimum ratings per movie',
        min_value=100, max_value=2000, value=500, step=100,
        help='Movies with fewer ratings than this will be excluded'
    )

    method = st.radio(
        'Similarity method',
        options=['Cosine Similarity', 'Normalized Cosine', 'Pearson Correlation'],
        help=(
            '**Cosine**: fast, standard.\n\n'
            '**Normalized Cosine**: corrects for user rating bias.\n\n'
            '**Pearson**: best quality, slower to compute.'
        )
    )

    n_recs = st.slider('Number of recommendations', 5, 25, 10)

    st.markdown('---')
    st.markdown('**About**\n\nItem-based collaborative filtering on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset.')


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown('<p class="title">🎬 Movie Recommender</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Item-based collaborative filtering · MovieLens dataset</p>', unsafe_allow_html=True)

# Load data
try:
    with st.spinner('Loading and preparing data...'):
        matrix = load_data(data_path, min_ratings)
        cosine_sim, norm_sim, pearson_sim = compute_all_similarities(matrix)

    imdb_lookup = load_imdb_lookup(data_path)

    similarity_map = {
        'Cosine Similarity':    cosine_sim,
        'Normalized Cosine':    norm_sim,
        'Pearson Correlation':  pearson_sim,
    }
    chosen_sim = similarity_map[method]

except FileNotFoundError:
    st.error(
        f'Could not find one or more required files (`ratings.csv`, `movies.csv`, `links.csv`) '
        f'in `{data_path}/`. Please update the dataset path in the sidebar.'
    )
    st.stop()

# ── Search Box ────────────────────────────────────────────────────────────────

st.markdown('### 🔍 Find a movie')
movie_input = st.text_input(
    label='Enter a movie title',
    placeholder='e.g. Inception, Toy Story, Dark Knight...',
    label_visibility='collapsed'
)

# Show autocomplete suggestions as the user types
if movie_input and len(movie_input) >= 2:
    suggestions = [t for t in cosine_sim.index if movie_input.lower() in t.lower()][:8]
    if suggestions:
        selected = st.selectbox(
            'Did you mean...',
            options=suggestions,
            label_visibility='collapsed'
        )
        movie_input = selected

# ── Recommend Button ──────────────────────────────────────────────────────────

if st.button('Get Recommendations', type='primary', use_container_width=True):
    if not movie_input:
        st.warning('Please enter a movie title first.')
    else:
        result_df, matched_title, all_matches = get_recommendations(movie_input, n_recs, chosen_sim)

        if result_df is None:
            st.error(
                f'No movie found matching **"{movie_input}"**. '
                'Try a shorter or partial title.'
            )
        else:
            st.success(f'Showing recommendations for **"{matched_title}"** · Method: {method}')

            if len(all_matches) > 1:
                with st.expander(f'ℹ️ {len(all_matches)} matches found — click to see others'):
                    for m in all_matches:
                        st.write(f'- {m}')

            st.markdown('### 🎯 Top Recommendations')

            # Render each result as a styled card with IMDB link
            for idx, row in result_df.iterrows():
                score_pct  = f"{row['Similarity Score'] * 100:.1f}%"
                title      = row['Movie Title']
                imdb_url   = imdb_lookup.get(title)

                # Build the title portion — clickable if IMDB URL exists
                if imdb_url:
                    imdb_badge = f'<a class="imdb-badge" href="{imdb_url}" target="_blank">IMDb</a>'
                    title_html = f'<a href="{imdb_url}" target="_blank">{title}</a>{imdb_badge}'
                else:
                    title_html = title

                st.markdown(f"""
                    <div class="rec-card">
                        <span class="rec-rank">#{idx}</span>
                        <span class="rec-title">{title_html}</span>
                        <span class="rec-score">{score_pct}</span>
                    </div>
                """, unsafe_allow_html=True)

            # Also show as a downloadable table
            with st.expander('📋 View as table / download'):
                st.dataframe(result_df, use_container_width=True)
                csv = result_df.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label='Download CSV',
                    data=csv,
                    file_name=f'recommendations_{matched_title[:20]}.csv',
                    mime='text/csv'
                )

# ── Dataset Stats ─────────────────────────────────────────────────────────────

with st.expander('📊 Dataset info'):
    col1, col2, col3 = st.columns(3)
    col1.metric('Movies in model', f'{matrix.shape[1]:,}')
    col2.metric('Users',           f'{matrix.shape[0]:,}')
    sparsity = 1 - (matrix.notna().sum().sum() / (matrix.shape[0] * matrix.shape[1]))
    col3.metric('Matrix sparsity', f'{sparsity:.1%}')
