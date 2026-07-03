import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Genre columns from MovieLens 100K u.item
genre_columns = [
    "Unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
]

columns = [
    "MovieID",
    "Title",
    "ReleaseDate",
    "VideoReleaseDate",
    "IMDbURL"
] + genre_columns

movies = pd.read_csv(
    "data/u.item",
    sep="|",
    names=columns,
    encoding="latin-1",
    header=None
)

ratings = pd.read_csv(
    "data/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
    header=None,
    encoding="latin-1"
)

movie_stats = (
    ratings
    .groupby("movie_id")["rating"]
    .agg(["count", "mean"])
    .rename(columns={"count": "rating_count", "mean": "avg_rating"})
    .reset_index()
)

movies = movies.merge(
    movie_stats,
    how="left",
    left_on="MovieID",
    right_on="movie_id"
)

movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
movies["avg_rating"] = movies["avg_rating"].fillna(0.0)

genre_features = movies[genre_columns].fillna(0)
similarity = cosine_similarity(genre_features)


def get_dataset_stats():
    num_users = int(ratings["user_id"].nunique())
    num_movies = int(movies["MovieID"].nunique())
    num_ratings = int(len(ratings))
    avg_rating = float(ratings["rating"].mean())
    sparsity = 1.0 - num_ratings / (num_users * num_movies)

    return {
        "num_users": num_users,
        "num_movies": num_movies,
        "num_ratings": num_ratings,
        "avg_rating": avg_rating,
        "sparsity": sparsity,
    }


def get_rating_distribution():
    return ratings["rating"].value_counts().sort_index()


def get_top_users(n=10):
    users = (
        ratings
        .groupby("user_id")["rating"]
        .agg(["count", "mean"])
        .rename(columns={"count": "ratings_count", "mean": "avg_rating"})
        .reset_index()
        .sort_values(["ratings_count", "avg_rating"], ascending=[False, False])
        .head(n)
    )
    return users


def get_top_movies_by_rating_count(n=20, min_count=50):
    return (
        movies[movies["rating_count"] >= min_count]
        .sort_values(["rating_count", "avg_rating"], ascending=[False, False])
        .head(n)
    )


def get_top_movies_by_avg_rating(n=20, min_count=50):
    return (
        movies[movies["rating_count"] >= min_count]
        .sort_values(["avg_rating", "rating_count"], ascending=[False, False])
        .head(n)
    )


def get_movie_list():
    return sorted(movies["Title"].tolist())


def get_user_list():
    return sorted(ratings["user_id"].unique().tolist())


def recommend_by_popularity(top_n=10, min_ratings=50):
    top = get_top_movies_by_rating_count(top_n, min_ratings)
    return top[["Title", "rating_count", "avg_rating"]].rename(
        columns={"Title": "Movie Name", "rating_count": "Rating Count", "avg_rating": "Avg Rating"}
    )


def recommend_content_based(movie_name, top_n=10, min_ratings=0):
    if movie_name not in movies["Title"].values:
        return []

    movie_index = int(movies[movies["Title"] == movie_name].index[0])
    scores = list(enumerate(similarity[movie_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = []

    for index, score in scores[1: top_n + 1]:
        row = movies.iloc[index]
        if row["rating_count"] < min_ratings:
            continue
        recommendations.append(
            {
                "Movie Name": row["Title"],
                "Similarity (%)": round(float(score) * 100, 2),
                "Rating Count": int(row["rating_count"]),
                "Avg Rating": round(float(row["avg_rating"]), 2),
            }
        )

    return recommendations


def load_svd_model(path="svd_model.pkl"):
    if not os.path.exists(path):
        return None

    with open(path, "rb") as file:
        return pickle.load(file)


svd_model = load_svd_model()


def is_svd_available():
    return svd_model is not None


def predict_svd_rating(user_id, movie_id):
    if svd_model is None:
        return None

    if user_id not in svd_model["user_map"] or movie_id not in svd_model["item_map"]:
        return None

    u = svd_model["user_map"][user_id]
    i = svd_model["item_map"][movie_id]
    score = (
        svd_model["global_mean"]
        + svd_model["user_bias"][u]
        + svd_model["item_bias"][i]
        + np.dot(svd_model["P"][u], svd_model["Q"][i])
    )
    return float(score)


def recommend_svd(user_id, top_n=10):
    if svd_model is None:
        return []

    rated_movie_ids = set(ratings.loc[ratings["user_id"] == user_id, "movie_id"].tolist())
    candidate_movies = movies[~movies["MovieID"].isin(rated_movie_ids)]
    predictions = []

    for _, row in candidate_movies.iterrows():
        est = predict_svd_rating(user_id, int(row.MovieID))
        if est is None:
            continue
        predictions.append(
            {
                "Movie Name": row["Title"],
                "Estimated Rating": round(float(est), 3),
                "Rating Count": int(row["rating_count"]),
                "Avg Rating": round(float(row["avg_rating"]), 2),
            }
        )

    predictions = sorted(predictions, key=lambda x: x["Estimated Rating"], reverse=True)
    return predictions[:top_n]


def get_user_history(user_id):
    history = (
        ratings[ratings["user_id"] == user_id]
        .merge(movies[["MovieID", "Title"]], left_on="movie_id", right_on="MovieID", how="left")
        .sort_values("rating", ascending=False)
    )
    return history[["movie_id", "Title", "rating"]]
