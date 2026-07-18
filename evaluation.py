import math
import copy
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NCF


# Paths to the preprocessed splits written by data_preprocess.py
TRAIN_PATH = "ml-1m/train_neg_4.csv"
TEST_PATH = "ml-1m/test.csv"
VAL_PATH = "ml-1m/val_neg_4.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 10
EPOCHS = 20
PATIENCE = 3
N_TRIALS = 15

# Hyperparameter tuning search space.
# Kept small to limit runtime while still exploring model capacity.
GRID = {
    "embed_dim": [16, 32, 64],
    "hidden": [
        [32, 16],
        [64, 32, 16],
        [128, 64, 32, 16],
    ],
    "dropout": [0.1, 0.3],
    "lr": [1e-3],
    "batch_size": [1024],
}


class InteractionDataset(Dataset):
    def __init__(self, df):
        # Convert dataframe columns to tensors.
        self.users = torch.tensor(df["userId"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movieId"].values, dtype=torch.long)
        self.labels = torch.tensor(
            df["interaction"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.labels[idx]


def run_epoch(model, loader, optimizer, criterion, train=True):
    # One epoch over a loader; returns average loss.
    model.train() if train else model.eval()
    total_loss = 0.0
    total_n = 0

    # Only compute gradients during training.
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for users, movies, labels in loader:
            users = users.to(DEVICE)
            movies = movies.to(DEVICE)
            labels = labels.to(DEVICE)

            # Model outputs logits; criterion applies sigmoid internally.
            logits = model(users, movies)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            total_n += len(labels)

    return total_loss / total_n

# Normalized Discounted Cumulative Gain (NDCG@k): rank-aware relevance metric.


def ndcg_at_k(topk_items, true_items, k=10):
    dcg = 0.0
    for rank, item in enumerate(topk_items[:k], start=1):
        if item in true_items:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(true_items), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def recall_at_k(topk_items, true_items):
    # Recall@k = (# of relevant items in top-k) / (# of relevant items).
    if len(true_items) == 0:
        return 0.0
    hits = len(set(topk_items) & true_items)
    return hits / len(true_items)


def build_history(df):
    # Build user -> set(items) history from positive interactions.
    hist = defaultdict(set)
    for row in df[df["interaction"] == 1].itertuples(index=False):
        hist[row.userId].add(row.movieId)
    return hist


def evaluate(model, test_df, train_history, n_movies, k=10):
    # Full-ranking evaluation:
    # For each user, score all unseen items and rank them; compute Recall@k and NDCG@k.
    model.eval()
    recalls, ndcgs = [], []

    # user -> set of positive items in the evaluation split.
    user_positive_items = test_df.groupby(
        "userId")["movieId"].apply(set).to_dict()

    with torch.no_grad():
        for user, true_items in user_positive_items.items():
            # Filter out items seen in training so we recommend "new" items.
            seen = train_history.get(user, set())
            candidates = np.array(
                [m for m in range(n_movies) if m not in seen])

            if len(candidates) == 0:
                continue

            # Create a (user, movie) batch for scoring all candidate movies.
            users_t = torch.full((len(candidates),), user,
                                 dtype=torch.long, device=DEVICE)
            movies_t = torch.tensor(
                candidates, dtype=torch.long, device=DEVICE)

            # Convert logits -> probabilities and rank descending.
            scores = torch.sigmoid(model(users_t, movies_t)).cpu().numpy()
            ranked = candidates[np.argsort(-scores)]
            topk_items = ranked[:k].tolist()

            recalls.append(recall_at_k(topk_items, true_items))
            ndcgs.append(ndcg_at_k(topk_items, true_items, k=k))

    return np.mean(recalls), np.mean(ndcgs)


# Load preprocessed CSVs.
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
val_df = pd.read_csv(VAL_PATH)

# Some splits might be positives-only; ensure the column exists.
if "interaction" not in test_df.columns:
    test_df["interaction"] = 1

if "interaction" not in val_df.columns:
    val_df["interaction"] = 1

# Ensure userId/movieId are mapped consistently across train/val/test.
# This is important if any split was written/loaded with different categorical encodings.
all_users = pd.Index(
    pd.concat([train_df["userId"], val_df["userId"], test_df["userId"]]).unique())
all_movies = pd.Index(pd.concat(
    [train_df["movieId"], val_df["movieId"], test_df["movieId"]]).unique())
user_map = {u: i for i, u in enumerate(all_users)}
movie_map = {m: i for i, m in enumerate(all_movies)}

train_df["userId"] = train_df["userId"].map(user_map)
train_df["movieId"] = train_df["movieId"].map(movie_map)
val_df["userId"] = val_df["userId"].map(user_map)
val_df["movieId"] = val_df["movieId"].map(movie_map)
test_df["userId"] = test_df["userId"].map(user_map)
test_df["movieId"] = test_df["movieId"].map(movie_map)

# Drop any rows that map to NaN (unknown ids) and restore integer dtype.
train_df = train_df.dropna().astype({"userId": int, "movieId": int})
val_df = val_df.dropna().astype({"userId": int, "movieId": int})
test_df = test_df.dropna().astype({"userId": int, "movieId": int})

n_users = len(user_map)
n_movies = len(movie_map)

train_history = build_history(train_df)
full_history = train_history

val_positive = val_df[val_df["interaction"] == 1].copy()
test_positive = test_df[test_df["interaction"] == 1].copy()

results = []


def objective(trial):
    # Optuna objective: train a model for this trial and return validation NDCG@TOP_K.
    embed_dim = trial.suggest_categorical("embed_dim", GRID["embed_dim"])
    hidden = trial.suggest_categorical("hidden", GRID["hidden"])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
    lr = trial.suggest_categorical("lr", [5e-4, 1e-3, 2e-3])
    batch_size = trial.suggest_categorical("batch_size", GRID["batch_size"])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5])

    print(
        f"\nTesting: embed_dim={embed_dim}, hidden={hidden}, dropout={dropout}, lr={lr}, batch={batch_size}, weight_decay={weight_decay}")

    train_loader = DataLoader(InteractionDataset(
        train_df), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(InteractionDataset(val_df), batch_size=batch_size)

    model = NCF(n_users, n_movies, embed_dim, hidden, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0

    train_losses = []
    val_losses = []
    best_train_losses = None
    best_val_losses = None

    for epoch in range(EPOCHS):
        # Optimize BCE loss on (positive + sampled negative) interactions.
        train_loss = run_epoch(model, train_loader,
                               optimizer, criterion, train=True)
        val_loss = run_epoch(model, val_loader, optimizer,
                             criterion, train=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1:02d}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_train_losses = train_losses.copy()
            best_val_losses = val_losses.copy()
            bad_epochs = 0
        else:
            bad_epochs += 1

        # Allow Optuna to prune bad trials early.
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping based on validation loss.
        if bad_epochs >= PATIENCE:
            break

    # Evaluate the best checkpoint on positive-only validation items using full ranking.
    model.load_state_dict(best_state)
    recall, ndcg = evaluate(model, val_positive,
                            train_history, n_movies, TOP_K)

    print(f"val Recall@10={recall:.4f} | val NDCG@10={ndcg:.4f}")

    trial.set_user_attr("state_dict", best_state)
    trial.set_user_attr("train_losses", best_train_losses)
    trial.set_user_attr("val_losses", best_val_losses)
    trial.set_user_attr("val_recall@10", float(recall))
    trial.set_user_attr("val_ndcg@10", float(ndcg))

    results.append({
        "trial": trial.number,
        "embed_dim": embed_dim,
        "hidden": hidden,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "val_recall@10": float(recall),
        "val_ndcg@10": float(ndcg),
    })

    return float(ndcg)


study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
)

# Run hyperparameter optimization.
study.optimize(objective, n_trials=N_TRIALS)

best_trial = study.best_trial
best = {
    "embed_dim": best_trial.params["embed_dim"],
    "hidden": best_trial.params["hidden"],
    "dropout": best_trial.params["dropout"],
    "lr": best_trial.params["lr"],
    "batch_size": best_trial.params["batch_size"],
    "weight_decay": best_trial.params["weight_decay"],
    "val_recall@10": best_trial.user_attrs["val_recall@10"],
    "val_ndcg@10": best_trial.user_attrs["val_ndcg@10"],
    "state_dict": best_trial.user_attrs["state_dict"],
    "train_losses": best_trial.user_attrs["train_losses"],
    "val_losses": best_trial.user_attrs["val_losses"],
}

print("\nBest config:")
print({k: v for k, v in best.items() if k not in {
      "state_dict", "train_losses", "val_losses"}})

best_model = NCF(n_users, n_movies,
                 best["embed_dim"], best["hidden"], best["dropout"]).to(DEVICE)
best_model.load_state_dict(best["state_dict"])

# Final evaluation on the test split.
test_recall, test_ndcg = evaluate(
    best_model, test_positive, full_history, n_movies, TOP_K)
print(f"\nTest Recall@10={test_recall:.4f}")
print(f"Test NDCG@10={test_ndcg:.4f}")

# Plot loss curves (train vs val) for the best trial.
plt.figure(figsize=(8, 5))
plt.plot(best["train_losses"], label="Train loss")
plt.plot(best["val_losses"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Convergence of best hyperparameter setting")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ml-1m/best_model_convergence.png", dpi=300)
plt.show()

# Save trial summary for reporting.
results_df = pd.DataFrame(results).sort_values(
    by="val_ndcg@10", ascending=False)
results_df.to_csv("ml-1m/optuna_results.csv", index=False)
print("\nSaved results to ml-1m/optuna_results.csv")

