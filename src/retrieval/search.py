from pathlib import Path
import json
import sys

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
INDEX_FILE = Path("indexes/road_rules.faiss")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 3


# ============================================================
# Load chunks
# ============================================================

def load_chunks():
    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks


# ============================================================
# Search
# ============================================================

def search(query, top_k=DEFAULT_TOP_K):

    chunks = load_chunks()

    index = faiss.read_index(str(INDEX_FILE))

    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index contains {index.ntotal} vectors "
            f"but {len(chunks)} chunks were loaded."
        )

    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # Encode query using the SAME model used for documents
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # FAISS search
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (index_id, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1
    ):

        if index_id < 0:
            continue

        chunk = chunks[index_id]

        result = {
            "rank": rank,
            "score": float(score),
            **chunk,
        }

        results.append(result)

    return results


# ============================================================
# Display results
# ============================================================

def display_results(query, results):

    print()
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    for result in results:

        print()
        print(f"RANK: {result['rank']}")
        print(f"SCORE: {result['score']:.4f}")
        print(f"CHUNK ID: {result['chunk_id']}")
        print(f"SOURCE: {result['source_title']}")
        print(f"SECTION: {result['section']}")
        print(f"URL: {result['url']}")

        print()
        print("TEXT:")
        print(result["text"])

        print("-" * 70)


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input(
            "Enter a Victorian road rules question: "
        ).strip()

    if not query:
        print("ERROR: Question cannot be empty.")
        return

    results = search(
        query,
        top_k=DEFAULT_TOP_K
    )

    display_results(
        query,
        results
    )


if __name__ == "__main__":
    main()
