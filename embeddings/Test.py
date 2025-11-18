from datasets import load_dataset
import numpy as np

DATASET_NAME = "xw1234gan/SPO_DATASET_WITH_EMBEDDDING_PCA"

ds = load_dataset(DATASET_NAME, split="train")

# Stack PCA embeddings into [N, 10]
X = np.array(ds["semantic_embedding_PCA"], dtype="float32")

def closest_prompts(idx, k=10):
    """
    Return indices of k nearest neighbors of prompt idx
    (excluding itself) based on semantic_embedding_PCA.
    """
    query = X[idx]                             # shape [10]
    dists = np.linalg.norm(X - query, axis=1)  # [N]
    nn_indices = np.argsort(dists)             # ascending distance
    nn_indices = nn_indices[nn_indices != idx] # drop itself
    return nn_indices[:k]

target_idx = 0

print(f"Target prompt idx={target_idx}:")
print(ds[target_idx]["prompt"])
print("=" * 60)

neighbors = closest_prompts(target_idx, k=10)
for j in neighbors:
    print(f"Neighbor idx={j}, dist={np.linalg.norm(X[j] - X[target_idx]):.4f}")
    print("prompt:", ds[j]["prompt"])
    print("-" * 40)
