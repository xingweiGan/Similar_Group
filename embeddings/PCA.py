from datasets import load_dataset
from sklearn.decomposition import PCA
import numpy as np

DATASET_ARROW_PATH = "hf://datasets/xw1234gan/Qwen3Emb_DAPO16k/data-00000-of-00001.arrow"
N_COMPONENTS = 10
OUTPUT_DIR = "Dataset_with_embeddings_pca10"


def main():
    print(f"Loading Arrow file from Hugging Face: {DATASET_ARROW_PATH}")
    ds = load_dataset(
        "arrow",
        data_files={"train": DATASET_ARROW_PATH},
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

    col_name = "semantic_embedding_pca10"
    print(f"\nAdding new column '{col_name}' to dataset...")
    ds_pca = ds.add_column(col_name, X_pca.tolist())

    print(f"\nSaving compressed dataset to: {OUTPUT_DIR}")
    ds_pca.save_to_disk(OUTPUT_DIR)

    print("\nDone.")
    print(ds_pca)
    print("Columns:", ds_pca.column_names)
    first = ds_pca[0]
    print(f"Length of {col_name} for first row:", len(first[col_name]))


if __name__ == "__main__":
    main()
