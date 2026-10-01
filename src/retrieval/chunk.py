from pathlib import Path
import json
import re


# --------------------------------------------------
# Paths
# --------------------------------------------------

RAW_DIR = Path("data/raw/road-rule-sources")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def extract_url(value):
    """
    Extract a clean URL from either a Markdown link
    or a plain URL.
    """

    # Markdown format:
    # [text](https://example.com)
    match = re.search(r"\]\((https?://[^)]+)\)", value)

    if match:
        return match.group(1)

    # Plain URL
    match = re.search(r"https?://[^\s\]]+", value)

    if match:
        return match.group(0)

    return value.strip()


def parse_document(file_path):
    """
    Parse one Victorian Road Rules Markdown document.

    Each ## section becomes one retrieval chunk.
    Document metadata is preserved for every chunk.
    """

    text = file_path.read_text(encoding="utf-8")

    # --------------------------------------------------
    # Extract document-level metadata
    # --------------------------------------------------

    metadata = {
        "source_id": "",
        "organisation": "",
        "source_title": "",
        "url": "",
        "category": file_path.parent.name,
        "filename": file_path.name,
    }

    for line in text.splitlines():

        line = line.strip()

        if line.startswith("Source ID:"):
            metadata["source_id"] = (
                line.split(":", 1)[1].strip()
            )

        elif line.startswith("Organisation:"):
            metadata["organisation"] = (
                line.split(":", 1)[1].strip()
            )

        elif line.startswith("Source title:"):
            metadata["source_title"] = (
                line.split(":", 1)[1].strip()
            )

        elif line.startswith("URL:"):

            raw_url = line.split(":", 1)[1].strip()

            metadata["url"] = extract_url(raw_url)

    # --------------------------------------------------
    # Split document by level-2 Markdown headings
    # --------------------------------------------------

    sections = re.split(
        r"(?m)^##\s+",
        text
    )

    chunks = []

    # sections[0] contains:
    # document title + metadata
    #
    # Therefore start from sections[1].
    for index, section in enumerate(
        sections[1:],
        start=1
    ):

        section = section.strip()

        if not section:
            continue

        lines = section.splitlines()

        if not lines:
            continue

        # First line after ## is the section title
        section_title = lines[0].strip()

        # Remaining lines are section content
        content = "\n".join(
            lines[1:]
        ).strip()

        # Ignore empty sections
        if not content:
            continue

        # --------------------------------------------------
        # Create structured chunk
        # --------------------------------------------------

        chunk = {
            "chunk_id": (
                f"{metadata['source_id']}_{index:03d}"
            ),
            "source_id": metadata["source_id"],
            "organisation": metadata["organisation"],
            "source_title": metadata["source_title"],
            "category": metadata["category"],
            "section": section_title,
            "url": metadata["url"],
            "filename": metadata["filename"],
            "text": content,
        }

        chunks.append(chunk)

    return chunks


# --------------------------------------------------
# Main chunking pipeline
# --------------------------------------------------

def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Only process actual source documents:
    # S01..., S02..., etc.
    #
    # README.md is intentionally excluded.
    source_files = sorted(
        RAW_DIR.rglob("S*.md")
    )

    print(
        f"Found {len(source_files)} source documents."
    )

    all_chunks = []

    # --------------------------------------------------
    # Process each source document
    # --------------------------------------------------

    for file_path in source_files:

        chunks = parse_document(file_path)

        print(
            f"{file_path.name}: "
            f"{len(chunks)} chunks"
        )

        all_chunks.extend(chunks)

    # --------------------------------------------------
    # Save chunks as JSONL
    # --------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output:

        for chunk in all_chunks:

            output.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print(
        f"Total chunks: {len(all_chunks)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
