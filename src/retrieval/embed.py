from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
OUTPUT_FILE = Path("data/processed/embeddings.npy")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# Load chunks
# ============================================================

def load_chunks():
    """Load processed Victorian Road Rules chunks."""

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks


# ============================================================
# Main embedding pipeline
# ============================================================

def main():

    print("Loading chunks...")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    if not chunks:
        print("ERROR: No chunks found.")
        return

    # --------------------------------------------------------
    # Prepare retrieval text
    # --------------------------------------------------------

    # Include the section heading together with the content.
    # This gives the embedding model useful semantic context.

    texts = [
        f"{chunk['section']}\n{chunk['text']}"
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print()
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_FILE,
        embeddings
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("Embedding generation complete.")
    print(f"Model: {MODEL_NAME}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
