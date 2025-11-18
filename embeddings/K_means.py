from datasets import load_dataset
from sklearn.cluster import KMeans   # <-- K-means import
import numpy as np

DATASET_IN = "xw1234gan/EMBEDDING_PCA_COMPLETE"
DATASET_OUT = "xw1234gan/EMBEDDING_PCA_COMPLETE_GROUPED"
N_CLUSTERS = 10


def main():
    print(f"Loading dataset from Hugging Face: {DATASET_IN}")
    ds = load_dataset(DATASET_IN, split="train")
    print(ds)

    # -------------------------
    # 1. Extract whole_vector_standardised as feature matrix
    # -------------------------
    print("\nStacking whole_vector_standardised into a matrix...")
    X = np.array(ds["whole_vector_standardised"], dtype="float32")  # [N, D+1]
    print("Feature matrix shape:", X.shape)

    # -------------------------
    # 2. Run K-means with 10 clusters (Euclidean)
    # -------------------------
    print(f"\nRunning KMeans with {N_CLUSTERS} clusters...")
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=0,
        n_init=10,      # explicit for stability
    )
    labels = kmeans.fit_predict(X)   # [N], values 0..9
    print("Cluster labels shape:", labels.shape)

    # convert to 1..10 as requested
    groups = (labels + 1).astype(int)
    print("Unique groups:", sorted(set(groups.tolist())))

    ds_grouped = ds.add_column("K_means_group", groups.tolist())

    # -------------------------
    # 3. Push grouped dataset to Hub
    # -------------------------
    print(f"\nPushing dataset to Hugging Face Hub as: {DATASET_OUT}")
    ds_grouped.push_to_hub(DATASET_OUT)

    print("\nDone.")
    print("Columns:", ds_grouped.column_names)
    first = ds_grouped[0]
    print("First row keys:", first.keys())
    print("First row K_means_group:", first["K_means_group"])


if __name__ == "__main__":
    main()
