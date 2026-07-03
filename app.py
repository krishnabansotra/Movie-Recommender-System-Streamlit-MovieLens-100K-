import streamlit as st
import pandas as pd
from recommender import get_movie_list, recommend_content_based

# --- Page config ---
st.set_page_config(
    page_title="Movie Recommender System",
    layout="wide",
)

# --- Custom CSS ---
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# --- Banner ---
st.markdown(
    """
    <section class='top-banner'>
      <div class='banner-copy'>
        <span class='hero-badge'>MovieLens 100K</span>
        <h1 class='main-title'>Movie recommendations</h1>
        <p class='sub-title'>Discover similar titles instantly — powered by MovieLens data.</p>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# --- Main layout ---
with st.container():
    left, right = st.columns([3, 1], gap="large")

    # --- Left panel ---
    with left:
        st.markdown(
            """
            <div class='panel-card'>
              <h2>Search by movie title</h2>
              <p>Choose a title from the MovieLens corpus, adjust the recommendation depth, and discover titles that share the same tone and genre.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        movie = st.selectbox("Selected movie", get_movie_list())
        num_recommendations = st.slider("Recommendations", 5, 15, 10)
        min_ratings = st.slider("Minimum rating count", 0, 200, 20)

        if st.button("Show similar movies"):
            recommendations = recommend_content_based(movie, num_recommendations, min_ratings)
            st.markdown(
                f"<div class='selected-movie'>Selected movie: <strong>{movie}</strong></div>",
                unsafe_allow_html=True,
            )

            if recommendations:
                df = pd.DataFrame(recommendations)
                df.index = df.index + 1
                df.index.name = "Rank"
                st.table(df)
            else:
                st.warning("No recommendations available for this movie with the chosen filters.")

    # --- Right panel (now empty, since 'How it works' and 'Pro tips' removed) ---
    with right:
        pass

# --- Sidebar ---
st.sidebar.markdown("# Movie Recommender")
st.sidebar.markdown("Use the controls below to refine how recommendations are generated.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Source:** MovieLens 100K dataset")
st.sidebar.markdown("**Engine:** Content-based similarity")
st.sidebar.markdown("**Goal:** One movie → curated recommendations")

# --- Footer ---
st.markdown(
    "<footer class='footer'>Designed for fast, polished movie discovery.</footer>",
    unsafe_allow_html=True,
)
