import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_mappings(ratings):
    user_ids = sorted(ratings["user_id"].unique())
    movie_ids = sorted(ratings["movie_id"].unique())
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {mid: idx for idx, mid in enumerate(movie_ids)}
    return user_map, item_map


def train_matrix_factorization(
    ratings,
    n_factors=50,
    n_epochs=25,
    learning_rate=0.005,
    reg=0.02,
    random_state=42,
):
    user_map, item_map = build_mappings(ratings)
    n_users = len(user_map)
    n_items = len(item_map)

    P = np.random.normal(0, 0.1, (n_users, n_factors))
    Q = np.random.normal(0, 0.1, (n_items, n_factors))
    user_bias = np.zeros(n_users, dtype=np.float64)
    item_bias = np.zeros(n_items, dtype=np.float64)
    global_mean = ratings["rating"].mean()

    rating_idx = ratings.apply(lambda row: (user_map[row["user_id"]], item_map[row["movie_id"]], row["rating"]), axis=1).tolist()

    for epoch in range(n_epochs):
        np.random.shuffle(rating_idx)
        total_error = 0.0
        for u, i, r in rating_idx:
            pred = global_mean + user_bias[u] + item_bias[i] + np.dot(P[u], Q[i])
            error = r - pred
            total_error += error ** 2

            user_bias[u] += learning_rate * (error - reg * user_bias[u])
            item_bias[i] += learning_rate * (error - reg * item_bias[i])
            P[u] += learning_rate * (error * Q[i] - reg * P[u])
            Q[i] += learning_rate * (error * P[u] - reg * Q[i])

        rmse = np.sqrt(total_error / len(rating_idx))
        print(f"Epoch {epoch + 1}/{n_epochs} RMSE: {rmse:.4f}")

    return {
        "P": P,
        "Q": Q,
        "user_bias": user_bias,
        "item_bias": item_bias,
        "global_mean": float(global_mean),
        "user_map": user_map,
        "item_map": item_map,
    }


def evaluate_model(model, test_ratings):
    errors = []
    inv_user_map = {v: k for k, v in model["user_map"].items()}
    inv_item_map = {v: k for k, v in model["item_map"].items()}

    for _, row in test_ratings.iterrows():
        user_id = row["user_id"]
        movie_id = row["movie_id"]
        rating = row["rating"]

        if user_id not in model["user_map"] or movie_id not in model["item_map"]:
            continue

        u = model["user_map"][user_id]
        i = model["item_map"][movie_id]
        pred = model["global_mean"] + model["user_bias"][u] + model["item_bias"][i] + np.dot(model["P"][u], model["Q"][i])
        errors.append((rating - pred) ** 2)

    if len(errors) == 0:
        return None

    return float(np.sqrt(np.mean(errors)))


def train_svd_model(data_path="data/u.data", model_path="svd_model.pkl"):
    ratings = pd.read_csv(
        data_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        header=None,
        encoding="latin-1",
    )
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    model = train_matrix_factorization(
        train_ratings,
        n_factors=50,
        n_epochs=25,
        learning_rate=0.005,
        reg=0.02,
    )

    rmse = evaluate_model(model, test_ratings)
    if rmse is not None:
        print(f"Test RMSE: {rmse:.4f}")

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print("Model trained successfully.")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train_svd_model()
