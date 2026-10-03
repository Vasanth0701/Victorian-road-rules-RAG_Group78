import json
import urllib.request

from retrieval.search import search


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
TOP_K = 3


def build_context(results):
    context_parts = []

    for result in results:
        context_parts.append(
            f"""Source: {result['source_title']}
Section: {result['section']}
URL: {result['url']}
Text: {result['text']}"""
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(question, context):
    prompt = f"""
You are a Victorian road rules question-answering assistant.

Use ONLY the retrieved context below to answer the user's question.

Instructions:
1. Read all retrieved context carefully before answering.
2. Look for statements that directly answer the question.
3. If the context contains the answer, state it clearly and concisely.
4. Do not ignore an answer just because it appears in a list or bullet point.
5. Do not use outside knowledge or invent information.
6. Only if the answer is genuinely absent from ALL retrieved context, say:
"I could not find enough information in the retrieved road rules."

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    data = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"]

def ask_road_rules(question):
    if not question or not question.strip():
        return {
            "answer": "Question cannot be empty.",
            "sources": []
        }

    results = search(question.strip(), top_k=TOP_K)
    context = build_context(results)
    answer = generate_answer(question.strip(), context)

    sources = []

    for result in results:
        sources.append({
            "title": result["source_title"],
            "section": result["section"],
            "url": result["url"]
        })

    return {
        "answer": answer,
        "sources": sources
    }
def main():
    question = input("Enter a Victorian road rules question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    print("\nRetrieving relevant road rules...")

    results = search(question, top_k=TOP_K)

    context = build_context(results)

    print("\nGenerating answer with Llama 3.2...\n")

    answer = generate_answer(question, context)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for result in results:
        print(
            f"[{result['rank']}] "
            f"{result['source_title']} - "
            f"{result['section']}"
        )
        print(result["url"])
        print()


if __name__ == "__main__":
    main()