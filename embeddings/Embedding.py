from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ------------------------
# Config
# ------------------------

MATH_TASK = (
    "Represent math problems so that problems testing similar mathematical "
    "concepts and multi-step reasoning patterns are embedded close together."
)

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_LENGTH = 512
OUTPUT_DIR = "Dataset_with_embeddings"


# ------------------------
# Helpers
# ------------------------

def format_instruction(query: str) -> str:
    """Wrap the raw problem text with a Qwen-style math instruction."""
    return f"Instruct: {MATH_TASK}\nQuery: {query}"


def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Qwen3-Embedding uses last-token pooling with left padding.
    """
    # left padding if all sequences have attention_mask[:, -1] == 1
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seq_lengths = attention_mask.sum(dim=1) - 1  # index of last token
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            seq_lengths
        ]


def main():
    print("Loading dataset...")
    ds = load_dataset(
        "dingzihan737/SPO_Qwen3-8B_DAPO_16k_ReTool_Binary",
        split="train"
    )
    print(ds)

    print(f"\nLoading model {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True  # Qwen models usually need this
    ).to(DEVICE)
    model.eval()

    # We close over tokenizer/model/DEVICE in this function:
    def embed_batch(batch):
        texts = [format_instruction(p) for p in batch["prompt"]]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state          # [B, T, H]
            pooled = last_token_pool(
                hidden, inputs["attention_mask"]
            )                                          # [B, H]
            embeddings = F.normalize(pooled, p=2, dim=1)  # [B, H]

        batch["semantic_embedding"] = embeddings.cpu().tolist()
        return batch

    print("\nComputing embeddings for all rows...")
    ds_with_emb = ds.map(
        embed_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        desc="Embedding prompts"
    )

    print(f"\nSaving to disk at: {OUTPUT_DIR}")
    ds_with_emb.save_to_disk(OUTPUT_DIR)

    print("\nDone. Dataset info with new column:")
    print(ds_with_emb)
    print("Columns:", ds_with_emb.column_names)
    print("First row keys:", ds_with_emb[0].keys())
    print("Embedding length for row 0:",
          len(ds_with_emb[0]["semantic_embedding"]))


if __name__ == "__main__":
    main()
