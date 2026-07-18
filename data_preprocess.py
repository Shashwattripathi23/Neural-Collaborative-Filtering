import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
random.seed(42)


def negative_sampling(user_df, unique_movie_ids, user_df_unique_movie_ids, neg_samples, user_id):
    # For a given user, sample negatives (movies the user has NOT interacted with)
    # and label them as interaction=0 to train an implicit-feedback model.
    n = len(user_df)
    non_interacted_movie_set = list(
        set(unique_movie_ids) - set(user_df_unique_movie_ids))

    if len(non_interacted_movie_set) == 0:
        # Edge-case: user has interacted with every movie in the dataset.
        raise ValueError(
            f"No negative candidates available for user {user_id}")

    # Sample with replacement so we can generate n * neg_samples rows even when
    # the candidate set is small.
    new_movie_ids = random.choices(non_interacted_movie_set, k=n * neg_samples)

    neg_df = pd.DataFrame({
        "userId": [user_id] * len(new_movie_ids),
        "movieId": new_movie_ids,
        "interaction": [0] * len(new_movie_ids)
    })

    return neg_df


# We cant have new users in the test or the validation split
# So we do the split per user 70-15-15 randomly
def get_dataset(file_path="ml-1m/ratings.dat"):
    # Load MovieLens ratings (original format uses '::' separator).
    df = pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )

    # Map original ids to contiguous integer codes [0..n-1]
    # so they can be used as embedding indices.
    df["userId"] = df["userId"].astype("category").cat.codes
    df["movieId"] = df["movieId"].astype("category").cat.codes

    n_users = df["userId"].nunique()
    n_movies = df["movieId"].nunique()
    print(f"Users: {n_users}  Movies: {n_movies}  Interactions: {len(df)}")

    # Convert explicit ratings to implicit feedback:
    # treat ratings >= 4 as positive interactions (interaction=1).
    df = df[df.rating >= 4].copy()
    df['rating'] = 1
    df.rename(columns={"rating": "interaction"}, inplace=True)

    # Timestamp is not needed for random splitting.
    df = df.drop(['timestamp'], axis=1)

    # How many negative samples to generate per positive interaction.
    neg_samples = 4

    train_parts = []
    val_parts = []
    test_parts = []

    # Global movie id set (used to sample negatives for each user).
    unique_movie_ids = df.movieId.unique().tolist()

    for user_id, user_df in df.groupby("userId"):
        # Movies this user has positive interactions with.
        user_df_unique_movie_ids = user_df.movieId.unique().tolist()
        n = len(user_df)

        if n < 4:
            # For very small histories, keep everything in train to avoid
            # extremely tiny val/test splits.
            neg_df = negative_sampling(user_df, unique_movie_ids=unique_movie_ids,
                                       user_df_unique_movie_ids=user_df_unique_movie_ids, neg_samples=neg_samples, user_id=user_id)
            user_df = pd.concat([user_df, neg_df], ignore_index=True)
            train_parts.append(user_df)
            continue

        # Per-user split prevents cold-start users appearing only in val/test.
        # Requested random split ratio: 70% train, 15% val, 15% test.
        train_u, temp_u = train_test_split(
            user_df,
            test_size=0.30,
            random_state=42,
            shuffle=True
        )

        # Add negatives only to train and validation (test stays positives-only for ranking eval).
        neg_df = negative_sampling(train_u, unique_movie_ids=unique_movie_ids,
                                   user_df_unique_movie_ids=user_df_unique_movie_ids, neg_samples=neg_samples, user_id=user_id)
        train_u = pd.concat([train_u, neg_df], ignore_index=True)

        val_u, test_u = train_test_split(
            temp_u,
            test_size=0.50,
            random_state=42,
            shuffle=True
        )

        val_neg_df = negative_sampling(val_u, unique_movie_ids=unique_movie_ids,
                                       user_df_unique_movie_ids=user_df_unique_movie_ids, neg_samples=neg_samples, user_id=user_id)
        val_u = pd.concat([val_u, val_neg_df], ignore_index=True)

        train_parts.append(train_u)
        val_parts.append(val_u)
        test_parts.append(test_u)

    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    # Persist splits for reuse by training/evaluation scripts.
    train_df.to_csv("ml-1m/train_neg_4.csv", index=False)
    val_df.to_csv("ml-1m/val_neg_4.csv", index=False)
    test_df.to_csv("ml-1m/test.csv", index=False)

    return train_df, val_df, test_df, n_users, n_movies


if __name__ == "__main__":
    get_dataset()
