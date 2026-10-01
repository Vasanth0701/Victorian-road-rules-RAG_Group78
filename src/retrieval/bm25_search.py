from pathlib import Path
import json
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi


CHUNKS_FILE = Path("data/processed/chunks.jsonl")
DEFAULT_TOP_K = 3


def load_chunks():
    """Load the same chunks used by the dense FAISS retriever."""

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks


def tokenize(text):
    """
    Simple tokenizer for BM25.

    Converts text to lowercase and extracts
    alphanumeric terms.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def search(query, top_k=DEFAULT_TOP_K):

    chunks = load_chunks()

    # Use the same section + text representation
    # conceptually used by the dense retriever.
    documents = [
        f"{chunk['section']} {chunk['text']}"
        for chunk in chunks
    ]

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    # Sort document indices from highest to lowest score
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, index_id in enumerate(
        ranked_indices,
        start=1
    ):
        chunk = chunks[index_id]

        results.append({
            "rank": rank,
            "score": float(scores[index_id]),
            **chunk,
        })

    return results


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
