from pathlib import Path

import faiss
import numpy as np


# ============================================================
# Configuration
# ============================================================

EMBEDDINGS_FILE = Path("data/processed/embeddings.npy")
INDEX_FILE = Path("indexes/road_rules.faiss")


# ============================================================
# Main FAISS indexing pipeline
# ============================================================

def main():

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_FILE)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2-dimensional matrix."
        )

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    print()
    print(f"Creating FAISS index with dimension {dimension}...")

    # Embeddings were normalized in embed.py.
    # Inner product therefore corresponds to cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    # --------------------------------------------------------
    # Add vectors
    # --------------------------------------------------------

    index.add(embeddings)

    print(f"Vectors added to index: {index.ntotal}")

    # --------------------------------------------------------
    # Save index
    # --------------------------------------------------------

    INDEX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("FAISS index creation complete.")
    print(f"Index type: {type(index).__name__}")
    print(f"Dimension: {index.d}")
    print(f"Vectors: {index.ntotal}")
    print(f"Saved to: {INDEX_FILE}")


if __name__ == "__main__":
    main()
