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
import streamlit.components.v1 as components
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
        .title { font-size: 4.5rem; font-weight: 800; margin-bottom: 0; }
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
        .rec-score { color: #7dd3fc; font-size: 0.9rem; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)


# ── Data Loading (cached so it only runs once) ────────────────────────────────

@st.cache_resource(show_spinner='Loading similarity matrices...')
def load_similarities():
    cosine_sim  = pd.read_parquet('similarity_cosine.parquet')
    norm_sim    = pd.read_parquet('similarity_normalized.parquet')
    pearson_sim = pd.read_parquet('similarity_pearson.parquet')
    imdb_lookup = pd.read_parquet('imdb_lookup.parquet')
    imdb_lookup = dict(zip(imdb_lookup['title'], imdb_lookup['imdb_url']))
    return cosine_sim, norm_sim, pearson_sim, imdb_lookup

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
        cosine_sim, norm_sim, pearson_sim, imdb_lookup = load_similarities()

    similarity_map = {
        'Cosine Similarity':    cosine_sim,
        'Normalized Cosine':    norm_sim,
        'Pearson Correlation':  pearson_sim,
    }
    chosen_sim = similarity_map[method]

except FileNotFoundError:
    st.error(
        'Could not find similarity parquet files. '
        'Run the precompute script first to generate them.'
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

            # Render each result as a styled card
            cards_html = """
                <style>
                    body { margin: 0; padding: 0; background: transparent; font-family: sans-serif; }
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
                    .rec-title { font-weight: 600; font-size: 0.88rem; flex: 1; padding: 0 1rem; color: #ffffff; }
                    .rec-title a { color: inherit; text-decoration: none; }
                    .rec-title a:hover { text-decoration: underline; }
                    .rec-score { color: #7dd3fc; font-size: 0.9rem; font-weight: 600; }
                </style>
            """
            
            for idx, row in result_df.iterrows():
                score_pct = f"{row['Similarity Score'] * 100:.1f}%"
                title     = row['Movie Title']
                imdb_url  = imdb_lookup.get(title)
            
                title_content = f'<a href="{imdb_url}" target="_blank">{title}</a>' if imdb_url else title
            
                cards_html += f"""
                    <div class="rec-card">
                        <span class="rec-rank">#{idx}</span>
                        <span class="rec-title">{title_content}</span>
                        <span class="rec-score">{score_pct}</span>
                    </div>
                """
            
            components.html(cards_html, height=n_recs * 46)

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
    col1, col2 = st.columns(2)
    col1.metric('Movies in model', f'{cosine_sim.shape[0]:,}')
    col2.metric('Similarity method', method)
