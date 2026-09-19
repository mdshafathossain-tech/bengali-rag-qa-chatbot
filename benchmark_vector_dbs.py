"""
benchmark_vector_dbs.py
========================

Benchmarks FAISS vs ChromaDB as local vector stores for a Bengali-language
RAG knowledge-base chatbot built on a scraped novel (~22,000 words / 9 chapters).

WHAT THIS SCRIPT DOES
----------------------
1. Loads chapters + metadata from `scraped_book.json`.
2. Splits chapter text into chunks with RecursiveCharacterTextSplitter
   (chunk_size=600, chunk_overlap=100).
3. Embeds chunks with the multilingual sentence-transformer
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   (via `langchain-huggingface`), which handles Bengali well.
4. Builds a FAISS index and a ChromaDB collection from the SAME chunks,
   measuring for each:
     - Index build time (s)
     - Peak process RAM during indexing (MB)
     - On-disk footprint of the persisted index (MB)
     - Average query latency across 5 Bengali test queries (ms)
5. Prints a formatted comparison table and a short recommendation.

HOW TO RUN
----------
1. Install dependencies (see requirements block below or requirements.txt):

    pip install langchain langchain-community langchain-huggingface \
                langchain-text-splitters sentence-transformers \
                faiss-cpu chromadb psutil

2. Place `scraped_book.json` in the same folder as this script (or pass
   its path with --data).

3. Run:

    python benchmark_vector_dbs.py --data scraped_book.json

   Optional flags:
     --chunk-size 600         (default 600, matches spec)
     --chunk-overlap 100      (default 100, matches spec)
     --top-k 3                (default 3, retrieval depth for latency test)
     --workdir ./vector_bench (default ./vector_bench, where indexes are saved)

NOTES ON THE METRICS
---------------------
- "Peak RAM" is measured with `psutil`, sampling the CURRENT PROCESS's
  resident set size (RSS) from a background thread every 50ms while each
  index is being built. This captures native-library memory (FAISS's C++
  core, Chroma's DuckDB/SQLite backend, etc.) that a pure-Python profiler
  like `tracemalloc` would miss, since tracemalloc only tracks Python-level
  allocations.
- "Disk Size" is the total size of the persisted index directory on disk
  (FAISS: a folder with `.faiss` + `.pkl` files; ChromaDB: its persistent
  client directory), computed by walking the directory tree.
- "Avg Latency" is the average wall-clock time for `similarity_search`
  (top-k) across the 5 sample queries, repeated a few times per query and
  averaged, to smooth out first-call warm-up effects.
- Embedding model loading time is NOT counted in the indexing time for
  either DB (it's a one-time shared cost), but it IS included once before
  both benchmarks run, so both DBs are compared on equal footing.
"""

import argparse
import json
import os
import shutil
import statistics
import threading
import time
from pathlib import Path

import psutil

# ---------------------------------------------------------------------------
# LangChain building blocks
# ---------------------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma


# ---------------------------------------------------------------------------
# 5 sample Bengali queries relevant to a novel/story context
# (Tuned to the sample book "হাজার বছর ধরে" but generic enough for any
#  Bengali fiction knowledge base — feel free to edit these.)
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = [
    "মন্তু আর টুনির সম্পর্ক কেমন ছিল?",
    "বুড়ো মকবুলের পরিবারে কতজন সদস্য ছিল?",
    "পরীর দীঘির উৎপত্তির কাহিনী কী?",
    "গ্রামে বন্যার সময় কী ঘটেছিল?",
    "হীরনের বিবাহিত জীবন সম্পর্কে কী জানা যায়?",
]


# ---------------------------------------------------------------------------
# Peak-RAM sampler
# ---------------------------------------------------------------------------
class PeakRAMSampler:
    """Samples this process's RSS memory in a background thread and tracks
    the peak value observed while it is running. Used to bracket a block of
    code (e.g. index-building) and report the highest memory point reached.
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread = None
        self.peak_rss_bytes = 0
        self.start_rss_bytes = 0

    def _sample_loop(self):
        while not self._stop_event.is_set():
            rss = self._process.memory_info().rss
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
            time.sleep(self.interval_s)

    def __enter__(self):
        self.start_rss_bytes = self._process.memory_info().rss
        self.peak_rss_bytes = self.start_rss_bytes
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_increase_mb(self) -> float:
        """Peak RSS observed minus RSS right before the block started, in MB.
        This isolates the memory the indexing step itself added, rather than
        the whole process's baseline footprint (Python interpreter, the
        already-loaded embedding model, etc.).
        """
        return max(0.0, (self.peak_rss_bytes - self.start_rss_bytes) / (1024 ** 2))


# ---------------------------------------------------------------------------
# Data loading & chunking
# ---------------------------------------------------------------------------
def load_documents(json_path: str) -> list[Document]:
    """Loads `scraped_book.json` and converts each chapter record into a
    LangChain Document, preserving useful metadata for later filtering /
    citation in the chatbot.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    docs = []
    for rec in records:
        docs.append(
            Document(
                page_content=rec["content"],
                metadata={
                    "book_name": rec.get("book_name", ""),
                    "chapter_name": rec.get("chapter_name", ""),
                    "source_url": rec.get("source_url", ""),
                    "word_count": rec.get("word_count", 0),
                },
            )
        )
    return docs


def split_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Splits chapter-level documents into smaller overlapping chunks.
    RecursiveCharacterTextSplitter is Unicode-safe, so it works correctly
    on Bengali (Unicode) text out of the box.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "।", "?", "!", " ", ""],
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Disk-size helper
# ---------------------------------------------------------------------------
def get_dir_size_mb(path: str) -> float:
    """Recursively sums file sizes under `path` and returns the total in MB."""
    total_bytes = 0
    for root, _, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                total_bytes += os.path.getsize(fpath)
            except OSError:
                pass
    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Index builders (each returns: vectorstore, index_time_s, peak_ram_mb, disk_mb)
# ---------------------------------------------------------------------------
def build_faiss_index(chunks, embeddings, save_dir: str):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    with PeakRAMSampler() as sampler:
        start = time.perf_counter()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(save_dir)
        elapsed = time.perf_counter() - start

    disk_mb = get_dir_size_mb(save_dir)
    return vectorstore, elapsed, sampler.peak_increase_mb, disk_mb


def build_chroma_index(chunks, embeddings, save_dir: str, collection_name: str):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    with PeakRAMSampler() as sampler:
        start = time.perf_counter()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=save_dir,
        )
        elapsed = time.perf_counter() - start

    disk_mb = get_dir_size_mb(save_dir)
    return vectorstore, elapsed, sampler.peak_increase_mb, disk_mb


# ---------------------------------------------------------------------------
# Query-latency benchmark
# ---------------------------------------------------------------------------
def benchmark_query_latency(
    vectorstore, queries: list[str], top_k: int, repeats: int = 3
) -> float:
    """Runs each query `repeats` times (discarding nothing — all runs count,
    since a RAG chatbot in production sees a mix of cold and warm queries)
    and returns the average latency across all runs, in milliseconds.
    """
    latencies_ms = []
    for query in queries:
        for _ in range(repeats):
            start = time.perf_counter()
            vectorstore.similarity_search(query, k=top_k)
            latencies_ms.append((time.perf_counter() - start) * 1000)
    return statistics.mean(latencies_ms)


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------
def print_comparison_table(results: dict):
    headers = ["Vector DB", "Indexing Time (s)", "Peak RAM (MB)", "Disk Size (MB)", "Avg Latency (ms)"]
    rows = []
    for name, m in results.items():
        rows.append([
            name,
            f"{m['index_time_s']:.3f}",
            f"{m['peak_ram_mb']:.2f}",
            f"{m['disk_mb']:.2f}",
            f"{m['avg_latency_ms']:.2f}",
        ])

    col_widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) + 2
        for i in range(len(headers))
    ]

    def fmt_row(cells):
        return "".join(c.ljust(w) for c, w in zip(cells, col_widths))

    sep = "-" * sum(col_widths)
    print("\n" + "=" * sum(col_widths))
    print("VECTOR DATABASE BENCHMARK — FAISS vs ChromaDB")
    print("=" * sum(col_widths))
    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print(sep)


def print_recommendation(results: dict):
    faiss_m = results["FAISS"]
    chroma_m = results["ChromaDB"]

    print("\nRECOMMENDATION")
    print("-" * 60)

    lighter = "FAISS" if faiss_m["peak_ram_mb"] <= chroma_m["peak_ram_mb"] else "ChromaDB"
    smaller = "FAISS" if faiss_m["disk_mb"] <= chroma_m["disk_mb"] else "ChromaDB"
    faster_index = "FAISS" if faiss_m["index_time_s"] <= chroma_m["index_time_s"] else "ChromaDB"
    faster_query = "FAISS" if faiss_m["avg_latency_ms"] <= chroma_m["avg_latency_ms"] else "ChromaDB"

    print(f"- Lower peak RAM during indexing : {lighter}")
    print(f"- Smaller on-disk footprint      : {smaller}")
    print(f"- Faster index build             : {faster_index}")
    print(f"- Faster query retrieval         : {faster_query}")

    print(
        "\nFor a small (~22K word) Bengali knowledge base running on a "
        "resource-constrained local machine, FAISS is typically the better "
        "default: it is a thin, dependency-light library with no background "
        "server/DB engine, which tends to give it a smaller memory and disk "
        "footprint and faster raw similarity search at this data scale.\n"
        "ChromaDB becomes more attractive if you need built-in persistence "
        "with metadata filtering, multi-collection management, or a "
        "client-server deployment model as the knowledge base grows beyond "
        "a single local script — it trades some extra overhead for that "
        "convenience and flexibility.\n"
        "The numbers above are specific to this machine and this dataset — "
        "re-run the script on your target hardware before finalizing the "
        "choice for production."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark FAISS vs ChromaDB on a Bengali book RAG corpus.")
    parser.add_argument("--data", default="scraped_book.json", help="Path to scraped_book.json")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--workdir", default="./vector_bench", help="Directory to store built indexes")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Load + split data -----------------------------------------------
    print(f"Loading documents from {args.data} ...")
    raw_docs = load_documents(args.data)
    print(f"Loaded {len(raw_docs)} chapter document(s).")

    print(f"Splitting into chunks (chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}) ...")
    chunks = split_documents(raw_docs, args.chunk_size, args.chunk_overlap)
    print(f"Produced {len(chunks)} chunks.")

    # 2. Load embedding model once, shared by both DBs -------------------
    print("Loading embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print("Embedding model ready.\n")

    results = {}

    # 3. Benchmark FAISS ---------------------------------------------------
    print("Building FAISS index ...")
    faiss_dir = str(workdir / "faiss_index")
    faiss_store, faiss_time, faiss_ram, faiss_disk = build_faiss_index(chunks, embeddings, faiss_dir)
    print(f"FAISS built in {faiss_time:.3f}s | peak RAM Δ {faiss_ram:.2f} MB | disk {faiss_disk:.2f} MB")

    print("Benchmarking FAISS query latency ...")
    faiss_latency = benchmark_query_latency(faiss_store, SAMPLE_QUERIES, args.top_k)
    print(f"FAISS avg latency: {faiss_latency:.2f} ms\n")

    results["FAISS"] = {
        "index_time_s": faiss_time,
        "peak_ram_mb": faiss_ram,
        "disk_mb": faiss_disk,
        "avg_latency_ms": faiss_latency,
    }

    # 4. Benchmark ChromaDB -------------------------------------------------
    print("Building ChromaDB index ...")
    chroma_dir = str(workdir / "chroma_index")
    chroma_store, chroma_time, chroma_ram, chroma_disk = build_chroma_index(
        chunks, embeddings, chroma_dir, collection_name="bengali_book"
    )
    print(f"ChromaDB built in {chroma_time:.3f}s | peak RAM Δ {chroma_ram:.2f} MB | disk {chroma_disk:.2f} MB")

    print("Benchmarking ChromaDB query latency ...")
    chroma_latency = benchmark_query_latency(chroma_store, SAMPLE_QUERIES, args.top_k)
    print(f"ChromaDB avg latency: {chroma_latency:.2f} ms\n")

    results["ChromaDB"] = {
        "index_time_s": chroma_time,
        "peak_ram_mb": chroma_ram,
        "disk_mb": chroma_disk,
        "avg_latency_ms": chroma_latency,
    }

    # 5. Report --------------------------------------------------------------
    print_comparison_table(results)
    print_recommendation(results)


if __name__ == "__main__":
    main()
