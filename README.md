# Neural Collaborative Filtering (NCF) — Assignment 1

This project trains and evaluates a Neural Collaborative Filtering (NCF) recommender on the MovieLens 1M ratings dataset (implicit feedback).

## Repository contents

- `data_preprocess.py`
  - Loads `ml-1m/ratings.dat`
  - Converts explicit ratings to implicit interactions (`rating >= 4` → `interaction = 1`)
  - Performs a per-user random split (70% train / 15% val / 15% test)
  - Adds negative samples (4 negatives per positive) to train and validation
  - Writes: `ml-1m/train_neg_4.csv`, `ml-1m/val_neg_4.csv`, `ml-1m/test.csv`

- `model.py`
  - Defines the `NCF` model (GMF + MLP branches)
  - Trains using `BCEWithLogitsLoss` and reports loss/accuracy
  - Saves the best model checkpoint to `output/ncf_best.pt` (early stopping on val loss)

- `evaluation.py`
  - Runs Optuna hyperparameter tuning
  - Evaluates with full-ranking metrics: `Recall@10` and `NDCG@10`
  - Saves: `ml-1m/optuna_results.csv` and `ml-1m/best_model_convergence.png`

## Data setup

Place the MovieLens 1M `ratings.dat` file at:

```
ml-1m/ratings.dat
```

If `ml-1m/ratings.dat` is missing, preprocessing and training will fail.

## Environment setup (Windows / PowerShell)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## How to run

### 1) Preprocess data

Generates train/val/test CSVs under `ml-1m/`:

```powershell
python data_preprocess.py
```

Expected outputs:

- `ml-1m/train_neg_4.csv`
- `ml-1m/val_neg_4.csv`
- `ml-1m/test.csv`

### 2) Train the NCF model

Trains the model and saves the best checkpoint:

```powershell
python model.py
```

Expected outputs:

- Console logs for epochs (train/val loss + accuracy)
- `output/ncf_best.pt`

### 3) Hyperparameter search + evaluation

Runs Optuna tuning and evaluates the best model on the test set:

```powershell
python evaluation.py
```

Note: `evaluation.py` imports `NCF` from `model.py`. Since `model.py` contains top-level training code, running `evaluation.py` may also trigger the training script before the Optuna run.

Expected outputs:

- Printed `Test Recall@10` and `Test NDCG@10`
- `ml-1m/optuna_results.csv`
- `ml-1m/best_model_convergence.png`

## Configuration

Key constants you can adjust in the scripts:

- `data_preprocess.py`: `neg_samples` (currently 4)
- `model.py`: `EMBED_DIM`, `HIDDEN`, `BATCH_SIZE`, `EPOCHS`, `LR`, `DROPOUT`, `patience`
- `evaluation.py`: `N_TRIALS`, `EPOCHS`, `PATIENCE`, `TOP_K`, and the `GRID`

## GPU / CUDA

All scripts select the device automatically:

- If CUDA is available, they use `cuda`
- Otherwise they fall back to `cpu`
