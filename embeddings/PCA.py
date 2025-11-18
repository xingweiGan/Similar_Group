from datasets import load_dataset
from sklearn.decomposition import PCA
import numpy as np

# CHANGED: use dataset names instead of Arrow paths / local output dir
DATASET_NAME_IN = "xw1234gan/SPO_DATASET_WITH_EMBEDDDING"          # input dataset on HF
DATASET_NAME_OUT = "xw1234gan/SPO_DATASET_WITH_EMBEDDDING_PCA"     # output dataset on HF
N_COMPONENTS = 10


def main():
    # CHANGED: load from HF dataset directly
    print(f"Loading dataset from Hugging Face: {DATASET_NAME_IN}")
    ds = load_dataset(
        DATASET_NAME_IN,
        split="train",
    )
    print(ds)

    # Extract embeddings into a numpy array [N, D]
    print("\nStacking semantic_embedding into a matrix...")
    X = np.array(ds["semantic_embedding"], dtype="float32")
    print("Original embedding shape:", X.shape)

    print(f"\nRunning PCA to {N_COMPONENTS} dimensions...")
    pca = PCA(n_components=N_COMPONENTS, random_state=0)
    X_pca = pca.fit_transform(X).astype("float32")
    print("Compressed embedding shape:", X_pca.shape)

    # CHANGED: column name you requested
    col_name = "semantic_embedding_PCA"
    print(f"\nAdding new column '{col_name}' to dataset...")
    ds_pca = ds.add_column(col_name, X_pca.tolist())

    # CHANGED: instead of save_to_disk, push to Hugging Face Hub
    print(f"\nPushing PCA-compressed dataset to Hugging Face Hub as: {DATASET_NAME_OUT}")
    ds_pca.push_to_hub(DATASET_NAME_OUT)

    print("\nDone.")

    

if __name__ == "__main__":
    main()
