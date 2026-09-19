"""
rag_chain.py
============

Phase 3: RAG Chain & LLM Integration for the Bengali Book Knowledge-Base Chatbot.

Pipeline:
    retriever (FAISS, k=3) -> format_docs -> Bengali prompt -> LLM -> StrOutputParser()

Assumes Phase 1 (data ingestion) and Phase 2 (FAISS chosen as the vector DB)
are already done, and that a FAISS index already exists at
`./vector_bench/faiss_index` (built with the same embedding model used here).

HOW TO RUN
----------
1. Install dependencies:

    pip install langchain langchain-core langchain-community langchain-huggingface \
                sentence-transformers faiss-cpu rank-bm25

   Then install ONE of the LLM backends depending on which option you use
   (see "LLM BACKEND SELECTION" below):

    # Option A (local, default) — Ollama
    pip install langchain-ollama
    # and separately install & run the Ollama app: https://ollama.com
    # then pull a model, e.g.:  ollama pull gemma2:2b

    # Option B — Groq API
    pip install langchain-groq

    # Option B — OpenAI API
    pip install langchain-openai

2. Run:

    python rag_chain.py

   Then type Bengali questions at the prompt. Type `exit` or `quit` to stop.

LLM BACKEND SELECTION
----------------------
Controlled by the `LLM_PROVIDER` environment variable. Defaults to "ollama".

    # Option A (default): local Ollama model
    export LLM_PROVIDER=ollama
    export OLLAMA_MODEL=gemma2:2b        # or llama3, etc.

    # Option B: Groq API
    export LLM_PROVIDER=groq
    export GROQ_API_KEY=your_key_here
    export GROQ_MODEL=llama-3.1-8b-instant

    # Option B: OpenAI API
    export LLM_PROVIDER=openai
    export OPENAI_API_KEY=your_key_here
    export OPENAI_MODEL=gpt-4o-mini

No code changes are needed to switch — just set the environment variable(s)
before running the script.
"""

import os
import sys

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FAISS_INDEX_DIR = "./vector_bench/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Raised from 3 to 5: hybrid retrieval (below) pulls in more candidates from
# two different ranking signals, so a slightly wider net avoids losing a
# document that one signal ranks lower than the other.
RETRIEVER_TOP_K = 5

BENGALI_RAG_PROMPT = ChatPromptTemplate.from_template(
    """তুমি একজন সহায়ক বাংলা ভাষার সহকারী, যে একটি নির্দিষ্ট বইয়ের বিষয়বস্তুর উপর ভিত্তি করে প্রশ্নের উত্তর দেয়।

নিচের নিয়মগুলো কঠোরভাবে অনুসরণ করো:
১. শুধুমাত্র নিচের "প্রসঙ্গ" অংশে দেওয়া তথ্যের ভিত্তিতে উত্তর দাও। বাইরের কোনো জ্ঞান ব্যবহার করবে না।
২. যদি প্রসঙ্গে প্রশ্নের উত্তর খুঁজে না পাও, তাহলে নিজে থেকে কোনো তথ্য তৈরি (hallucinate) না করে বিনয়ের সাথে বাংলায় বলো যে এই তথ্যটি বইয়ে পাওয়া যায়নি।
৩. উত্তর স্পষ্ট, সংক্ষিপ্ত এবং সহজবোধ্য বাংলায় দাও।
৪. সম্ভব হলে উত্তরটি প্রসঙ্গের ভাষা ও শৈলীর সাথে সামঞ্জস্যপূর্ণ রাখো।

প্রসঙ্গ:
{context}

প্রশ্ন: {question}

উত্তর:"""
)


# ---------------------------------------------------------------------------
# Step 1: Load the persisted FAISS vector store
# ---------------------------------------------------------------------------
def load_vectorstore(index_dir: str = FAISS_INDEX_DIR) -> FAISS:
    """Loads the FAISS index built in Phase 2, using the same embedding model
    so query vectors are compatible with the stored vectors.

    `allow_dangerous_deserialization=True` is required because FAISS.load_local
    unpickles a docstore file; this is safe here since the index was created
    by our own Phase 2 script and never comes from an untrusted source.
    """
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(
            f"FAISS index not found at '{index_dir}'. "
            "Run the Phase 2 benchmarking/indexing script first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


# ---------------------------------------------------------------------------
# Step 1b: Hybrid (BM25 + FAISS) retriever
# ---------------------------------------------------------------------------
def load_retriever(vectorstore: FAISS, k: int = RETRIEVER_TOP_K):
    """Builds a hybrid retriever combining:
      - FAISS (dense/semantic search): good at matching PARAPHRASES and
        meaning, weaker on exact rare terms/names in a multilingual model
        that has seen little of this book's colloquial Bengali dialect.
      - BM25 (sparse/keyword search): guarantees that a chunk containing the
        exact words in the question (e.g. "পরীর দীঘি", "সৃষ্টি") ranks highly,
        regardless of how well the embedding model captures their meaning.

    For a small, single-book corpus like this one, this combination recovers
    "which paragraph literally mentions X" questions that pure dense search
    can miss. `weights=[0.5, 0.5]` gives both signals equal say; raise the
    BM25 weight if factual/keyword questions still underperform, or raise
    the FAISS weight if paraphrased/conceptual questions start underperforming.

    NOTE: this reads documents straight out of the FAISS index you already
    built and saved (via its in-memory docstore), so no re-indexing or
    re-chunking of scraped_book.json is required to use this.
    """
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    all_docs = list(vectorstore.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
    )


# ---------------------------------------------------------------------------
# Step 2: Helpers to format retrieved documents and extract source metadata
# ---------------------------------------------------------------------------
def format_docs(docs: list[Document]) -> str:
    """Joins retrieved chunk texts into a single context block for the prompt."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def extract_sources(docs: list[Document]) -> list[dict]:
    """Extracts de-duplicated chapter/source metadata from retrieved chunks,
    so the chatbot can cite where each answer came from.
    """
    seen = set()
    sources = []
    for doc in docs:
        chapter = doc.metadata.get("chapter_name", "unknown")
        url = doc.metadata.get("source_url", "")
        key = (chapter, url)
        if key not in seen:
            seen.add(key)
            sources.append({"chapter_name": chapter, "source_url": url})
    return sources


# ---------------------------------------------------------------------------
# Step 3: LLM factory — modular switch between Ollama (local) and API options
# ---------------------------------------------------------------------------
def get_llm():
    """Returns a chat model instance based on the LLM_PROVIDER environment
    variable. Defaults to Ollama (local, free, private).

    Supported values for LLM_PROVIDER: "ollama" (default), "groq", "openai".
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        # --- OPTION A (default): local model via Ollama ---------------
        # Requires the Ollama app running locally (https://ollama.com) and
        # a model pulled, e.g.:  ollama pull gemma2:2b
        from langchain_ollama import ChatOllama

        model_name = os.getenv("OLLAMA_MODEL", "gemma2:2b")
        return ChatOllama(model=model_name, temperature=0.2)

    elif provider == "groq":
        # --- OPTION B: Groq API (fast hosted inference) ---------------
        # export LLM_PROVIDER=groq
        # export GROQ_API_KEY=your_key_here
        from langchain_groq import ChatGroq

        # .strip() guards against stray leading/trailing whitespace sneaking
        # into the key when set via shells like PowerShell (e.g. quoting
        # `" key "` instead of `"key"`), which otherwise produces a confusing
        # generic "Connection error." instead of a clear auth error.
        api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        # NOTE: Groq deprecated llama-3.1-8b-instant and llama-3.3-70b-versatile
        # on 2026-08-16. openai/gpt-oss-20b is the recommended lightweight
        # replacement; see https://console.groq.com/docs/deprecations for the
        # current model list if this default stops working again in the future.
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        return ChatGroq(model=model_name, api_key=api_key, temperature=0.2)

    elif provider == "openai":
        # --- OPTION B: OpenAI API --------------------------------------
        # export LLM_PROVIDER=openai
        # export OPENAI_API_KEY=your_key_here
        from langchain_openai import ChatOpenAI

        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.2)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use 'ollama', 'groq', or 'openai'."
        )


# ---------------------------------------------------------------------------
# Step 4: Assemble the LCEL RAG chain
# ---------------------------------------------------------------------------
def build_answer_chain(llm):
    """Builds the generation half of the pipeline (prompt -> llm -> parser).
    Retrieval is kept as a separate explicit step (see `answer_question`
    below) so we can both stream the answer AND report the exact source
    chunks used — a single monolithic `retriever | format_docs | prompt |
    llm | StrOutputParser()` chain would retrieve internally and hide the
    documents from the caller, making source citation impossible.
    """
    return BENGALI_RAG_PROMPT | llm | StrOutputParser()


def answer_question(question: str, retriever, answer_chain, stream: bool = True):
    """Runs one full RAG turn:
        1. retriever -> relevant chunks
        2. format_docs -> context string
        3. answer_chain (prompt -> llm -> StrOutputParser()) -> answer text
    Returns (answer_text, sources_list). If stream=True, also prints tokens
    to stdout as they arrive.
    """
    docs = retriever.invoke(question)
    context = format_docs(docs)
    sources = extract_sources(docs)

    chain_input = {"context": context, "question": question}

    if stream:
        chunks = []
        for token in answer_chain.stream(chain_input):
            print(token, end="", flush=True)
            chunks.append(token)
        print()  # newline after streaming finishes
        answer_text = "".join(chunks)
    else:
        answer_text = answer_chain.invoke(chain_input)
        print(answer_text)

    return answer_text, sources


# ---------------------------------------------------------------------------
# Step 5: Interactive CLI loop
# ---------------------------------------------------------------------------
def main():
    print("Loading FAISS vector store ...")
    vectorstore = load_vectorstore()
    print("Building hybrid (BM25 + FAISS) retriever ...")
    retriever = load_retriever(vectorstore)

    print(f"Initializing LLM (provider = {os.getenv('LLM_PROVIDER', 'ollama')}) ...")
    llm = get_llm()
    answer_chain = build_answer_chain(llm)

    print("\nবাংলা RAG চ্যাটবট প্রস্তুত। প্রশ্ন লিখুন (প্রস্থান করতে 'exit' লিখুন)।\n")

    while True:
        try:
            question = input("প্রশ্ন > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nবিদায়!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("বিদায়!")
            break

        print("উত্তর > ", end="", flush=True)
        try:
            _, sources = answer_question(question, retriever, answer_chain, stream=True)
        except Exception as exc:  # keep the CLI alive on backend errors
            print(f"\n[ত্রুটি] উত্তর তৈরি করতে সমস্যা হয়েছে: {exc}", file=sys.stderr)
            continue

        if sources:
            print("\nতথ্যসূত্র (Sources):")
            for src in sources:
                if src["source_url"]:
                    print(f"  - {src['chapter_name']} ({src['source_url']})")
                else:
                    print(f"  - {src['chapter_name']}")
        print()  # blank line between turns


if __name__ == "__main__":
    main()
