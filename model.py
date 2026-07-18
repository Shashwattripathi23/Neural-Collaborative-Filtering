import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from data_preprocess import get_dataset


# Basic configs (paths + training hyperparameters)
dataPath = "ml-1m/ratings.dat"
savingPath = "output/"

os.makedirs(savingPath, exist_ok=True)
bestModelPath = os.path.join(savingPath, "ncf_best.pt")

EMBED_DIM = 32
HIDDEN = [64, 32, 16]
BATCH_SIZE = 1024
EPOCHS = 20
LR = 1e-3
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# Build train/val/test splits (and also writes the CSVs under ml-1m/)
train_df, val_df, test_df, n_users, n_movies = get_dataset(dataPath)


class InteractionDataset(Dataset):
    def __init__(self, df):
        # Convert dataframe columns to tensors for model input.
        self.users = torch.tensor(df["userId"].values,     dtype=torch.long)
        self.movies = torch.tensor(df["movieId"].values,    dtype=torch.long)
        self.labels = torch.tensor(
            df["interaction"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.labels[idx]


train_loader = DataLoader(InteractionDataset(
    train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(InteractionDataset(val_df),   batch_size=BATCH_SIZE)


# Neural Collaborative Filtering (NCF): GMF + MLP branches concatenated.
class NCF(nn.Module):
    def __init__(self, n_users, n_movies, embed_dim, hidden, dropout):
        super().__init__()
        # GMF branch: element-wise product of user/movie embeddings.
        self.gmf_user = nn.Embedding(n_users,  embed_dim)
        self.gmf_movie = nn.Embedding(n_movies, embed_dim)

        # MLP branch: concatenate user/movie embeddings then apply feed-forward layers.
        self.mlp_user = nn.Embedding(n_users,  embed_dim)
        self.mlp_movie = nn.Embedding(n_movies, embed_dim)

        mlp_layers = []
        in_dim = embed_dim * 2
        for out_dim in hidden:
            mlp_layers += [nn.Linear(in_dim, out_dim),
                           nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Final layer: concatenate [gmf, mlp_out] then predict a single logit.
        self.output = nn.Linear(embed_dim + hidden[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user, self.gmf_movie, self.mlp_user, self.mlp_movie]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user, movie):
        # GMF path
        gmf = self.gmf_user(user) * self.gmf_movie(movie)

        # MLP path
        mlp_in = torch.cat(
            [self.mlp_user(user), self.mlp_movie(movie)], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Combine both representations
        x = torch.cat([gmf, mlp_out], dim=-1)
        return self.output(x).squeeze(-1)


model = NCF(n_users, n_movies, EMBED_DIM, HIDDEN, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# BCEWithLogitsLoss expects raw logits (sigmoid is applied inside the loss).
criterion = nn.BCEWithLogitsLoss()

print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# Train for one epoch and return (avg_loss, accuracy)
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    # Only track gradients during training.
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for users, movies, labels in loader:
            users, movies, labels = users.to(
                DEVICE), movies.to(DEVICE), labels.to(DEVICE)
            preds = model(users, movies)
            loss = criterion(preds, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            # Convert logits -> probabilities and threshold at 0.5 for accuracy.
            correct += ((torch.sigmoid(preds) >= 0.5)
                        == labels.bool()).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


start_epoch = 0
best_val_loss = float("inf")
patience = 3
epochs_without_improvement = 0

for epoch in range(start_epoch, EPOCHS):
    # One full pass over train + validation loaders.
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader, train=False)

    print(f"Epoch {epoch+1:02d}/{EPOCHS}  "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
          f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

    # Early-stopping criterion based on validation loss.
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        epochs_without_improvement = 0
        # Save the best model weights seen so far.
        torch.save(model.state_dict(), bestModelPath)
        print(f"Best model saved (val_loss={best_val_loss:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s)")

    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break


print("Training complete.")
print(f"Best model : {bestModelPath}")
#
#
# # Optional: load best saved model before evaluation
# model.load_state_dict(torch.load(bestModelPath, map_location=DEVICE))
# model.to(DEVICE)
#
# recall_10, ndcg_10 = full_ranking_evaluation(
#     model=model,
#     test_path="ml-1m/test.csv",
#     n_movies=n_movies,
#     device=DEVICE,
#     k=10
# )
#
# print(f"Recall@10: {recall_10:.4f}")
# print(f"NDCG@10:   {ndcg_10:.4f}")
