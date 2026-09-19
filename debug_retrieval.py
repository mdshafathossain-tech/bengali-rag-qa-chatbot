"""
debug_retrieval.py
===================

A diagnostic tool that shows BOTH:
  1. Raw FAISS-only similarity search (what Phase 3 used to use) — with
     distance scores, so you can see how weak/strong the semantic match is.
  2. The final hybrid (BM25 + FAISS) retriever's result — what the chatbot
     actually uses now.

Comparing the two tells you directly whether a question is a case BM25
rescues (exact keyword match the embedding model missed) or whether both
still fail (meaning the passage may not be well-represented in any chunk,
or needs a bigger k).

HOW TO RUN
----------
    python debug_retrieval.py

Then type any question. Type 'exit' to quit.
"""

from rag_chain import load_vectorstore, load_retriever, RETRIEVER_TOP_K


def main():
    print("Loading FAISS vector store ...")
    vectorstore = load_vectorstore()
    print("Building hybrid (BM25 + FAISS) retriever ...")
    hybrid_retriever = load_retriever(vectorstore)

    print(f"\nRetriever debug tool (k={RETRIEVER_TOP_K}). Type 'exit' to quit.\n")
    while True:
        query = input("প্রশ্ন > ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break

        print("\n--- [A] Raw FAISS-only similarity search (old behavior) ---")
        faiss_results = vectorstore.similarity_search_with_score(query, k=RETRIEVER_TOP_K)
        for i, (doc, score) in enumerate(faiss_results, start=1):
            chapter = doc.metadata.get("chapter_name", "unknown")
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"[{i}] score={score:.4f} | {chapter}\n    {preview}...")

        print("\n--- [B] Hybrid (BM25 + FAISS) retriever (current behavior) ---")
        hybrid_results = hybrid_retriever.invoke(query)
        for i, doc in enumerate(hybrid_results, start=1):
            chapter = doc.metadata.get("chapter_name", "unknown")
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"[{i}] {chapter}\n    {preview}...")

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
