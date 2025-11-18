from datasets import load_dataset
import numpy as np

DATASET_IN = "xw1234gan/SPO_DATASET_WITH_EMBEDDDING_PCA"
DATASET_OUT = "xw1234gan/EMBEDDING_PCA_COMPLETE"


def main():
    print(f"Loading dataset from Hugging Face: {DATASET_IN}")
    ds = load_dataset(DATASET_IN, split="train")
    print(ds)

    # -------------------------
    # 1. Compute difficulty = mean(vhat) per prompt
    # -------------------------
    print("\nComputing difficulty (mean of vhat per row)...")
    vhat_col = ds["vhat"]

    def to_mean(v):
        # handle scalar, list, or numpy-y things
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.array(v, dtype="float32")
            return float(arr.mean())
        else:
            return float(v)

    difficulty = np.array([to_mean(v) for v in vhat_col], dtype="float32")
    print("difficulty shape:", difficulty.shape)

    ds = ds.add_column("difficulty", difficulty.tolist())

    # -------------------------
    # 2. Build whole_vector = [semantic_embedding_PCA..., difficulty]
    # -------------------------
    print("\nStacking semantic_embedding_PCA...")
    X_pca = np.array(ds["semantic_embedding_PCA"], dtype="float32")  # [N, D]
    print("semantic_embedding_PCA shape:", X_pca.shape)

    whole_vector = np.concatenate(
        [X_pca, difficulty[:, None]], axis=1
    )  # [N, D+1]
    print("whole_vector shape:", whole_vector.shape)

    ds = ds.add_column("whole_vector", whole_vector.tolist())

    # -------------------------
    # 3. Standardize like SPaRFT: each dim to mean 0, std 1
    #    (PCA dims and difficulty each standardized, then concatenated)
    # -------------------------
    print("\nStandardizing PCA dims and difficulty (mean 0, std 1)...")

    # PCA dims
    mu_pca = X_pca.mean(axis=0)                   # [D]
    sigma_pca = X_pca.std(axis=0)                 # [D]
    sigma_pca[sigma_pca == 0] = 1.0               # avoid divide-by-zero
    X_pca_std = (X_pca - mu_pca) / sigma_pca      # [N, D]

    # difficulty dim
    mu_diff = difficulty.mean()
    sigma_diff = difficulty.std()
    if sigma_diff == 0:
        sigma_diff = 1.0
    diff_std = (difficulty - mu_diff) / sigma_diff  # [N]

    whole_vector_std = np.concatenate(
        [X_pca_std, diff_std[:, None]], axis=1
    )  # [N, D+1]

    print("whole_vector_standardised shape:", whole_vector_std.shape)

    ds = ds.add_column(
        "whole_vector_standardised",
        whole_vector_std.tolist()
    )

    # -------------------------
    # 4. Push to Hub
    # -------------------------
    print(f"\nPushing dataset to Hugging Face Hub as: {DATASET_OUT}")
    ds.push_to_hub(DATASET_OUT)

    print("\nDone.")
    print("Columns:", ds.column_names)
    first = ds[0]
    print("First row keys:", first.keys())
    print("difficulty:", first["difficulty"])
    print("len(whole_vector):", len(first["whole_vector"]))
    print("len(whole_vector_standardised):", len(first["whole_vector_standardised"]))


if __name__ == "__main__":
    main()
