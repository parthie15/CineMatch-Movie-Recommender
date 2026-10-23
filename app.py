
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

@st.cache_data
def load_posters():
    posters = pd.read_csv(
        "data/movie_posters.csv",
        header=None,
        names=["movieId", "poster_url"]
    )
    return posters.drop_duplicates("movieId")

# ----------------------------
# 1. Page configuration
# ----------------------------
st.set_page_config(
    page_title="CineMatch | Movie Discovery",
    page_icon="🎬",
    layout="wide"
)

DATA_DIR = Path(__file__).parent / "data"

# ----------------------------
# 2. Load and prepare data
# ----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")
    posters = load_posters()

    # Average user rating and rating count per movie
    stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    movies = movies.merge(stats, on="movieId", how="left")
    movies["avg_rating"] = movies["avg_rating"].fillna(0)
    movies["rating_count"] = movies["rating_count"].fillna(0)
    movies = movies.merge(
        posters,
        on="movieId",
        how="left"
    )



    # Clean movie genres and titles
    movies["genres"] = movies["genres"].fillna("").str.replace(
    "|", " ", regex=False
    )
    movies["title"] = movies["title"].fillna("Unknown")

    # Combine all user-provided tags for each movie
    tags["tag"] = tags["tag"].fillna("").str.lower()

    movie_tags = (
        tags.groupby("movieId")["tag"]
        .apply(lambda values: " ".join(values))
        .reset_index()
    )

    # Add tags to the movie dataset
    movies = movies.merge(movie_tags, on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("")

    # Build richer movie descriptions
    movies["content"] = (
        movies["genres"] + " " + movies["tag"]
    )

    return movies



@st.cache_resource
def build_model(content):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(content)

    similarities = cosine_similarity(matrix)

    return similarities


try:
    movies = load_data()
    similarity = build_model(movies["content"])
except Exception as e:
    st.error(
        "Could not load the dataset. Check that data/movies.csv "
        "and data/ratings.csv exist."
    )
    st.stop()

# ----------------------------
# 3. Recommendation function
# ----------------------------
def recommend_movies(movie_title, number=10):
    matches = movies.index[movies["title"] == movie_title].tolist()

    if not matches:
        return movies.head(0)

    movie_index = matches[0]
    scores = list(enumerate(similarity[movie_index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Exclude the selected movie itself
    scores = [
        (idx, score) for idx, score in scores
        if idx != movie_index
    ][:number]

    indices = [idx for idx, score in scores]
    result = movies.iloc[indices].copy()
    result["similarity_score"] = [
        score for idx, score in scores
    ]

    return result


# ----------------------------
# 5. Interface
# ----------------------------
st.title("🎬 CineMatch")
st.subheader("Find your next favorite movie")
st.write(
    "Choose a film you love. Our recommendation engine "
    "will discover movies with similar genres."
)

st.divider()

col1, col2 = st.columns([3, 1])

with col1:
    selected_movie = st.selectbox(
        "Search for a movie",
        options=sorted(movies["title"].unique()),
        index=None,
        placeholder="e.g. Toy Story (1995)"
    )

with col2:
    number = st.selectbox(
        "Recommendations",
        [5, 10, 15, 20],
        index=1
    )

if st.button("✨ Find Similar Movies", type="primary"):
    if selected_movie:
        results = recommend_movies(selected_movie, number)

        st.subheader(f"Because you liked {selected_movie}")

        if results.empty:
            st.warning("No recommendations found.")
        else:
            cols = st.columns(5)

            for i, (_, movie) in enumerate(results.iterrows()):
                with cols[i % 5]:
                    poster = movie.get("poster_url")

                    if pd.notna(poster) and str(poster).startswith("https://"):
                        st.image(poster, width="stretch")
                    else:
                        st.markdown("🎬")
                        st.caption("Poster unavailable")

                    st.markdown(f"**{movie['title']}**")
                    st.caption(movie["genres"].replace("|", " · "))

                    st.write(
                        f"⭐ {movie['avg_rating']:.1f}/5"
                    )
                    st.progress(
                        min(
                            max(
                                float(movie["similarity_score"]),
                                0.0
                            ),
                            1.0
                        )
                    )
                    st.caption(
                        f"Similarity: "
                        f"{movie['similarity_score']:.0%}"
                    )
    else:
        st.warning("Please select a movie first.")

st.divider()
st.caption(
    "CineMatch uses TF-IDF and cosine similarity. "
    "Recommendations are based on movie genres, not "
    "personal viewing history."
)
