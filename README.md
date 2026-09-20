# বাংলা RAG নলেজ-বেস চ্যাটবট — "হাজার বছর ধরে"

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a single Bengali
novel, citing the chapter each answer came from, and clearly saying so when an answer isn't
in the book.

> ⚠️ **Source disclosure:** This book was ingested from **ebanglalibrary.com**, not
> bn.wikisource.org. See [Book Information](#book-information) below. If your assignment
> requires a strict Wikisource source, replace `scraped_book.json` with a re-crawl of a
> Wikisource title and rerun the pipeline — the rest of this repo (chunking, embeddings,
> vector store, retriever, LLM, UI) needs no changes.

---

## Book Information

| | |
|---|---|
| **Title** | হাজার বছর ধরে (Hajar Bochhor Dhore) |
| **Author** | জহির রায়হান (Zahir Raihan) |
| **First published** | 1964 |
| **Source used** | [ebanglalibrary.com](https://www.ebanglalibrary.com/) — **not** bn.wikisource.org |
| **Chapters ingested** | 9 (all chapters of the book) |
| **Total size** | ~22,000 words |

**Description:** A short novel depicting the unchanging, cyclical life of a poor joint family
in a rural Bengal village over generations — poverty, floods, marriage, and survival — centered
on the household of the old patriarch Makbul.

---

## Setup & Running Instructions

### Required Python version
Python 3.10–3.12 recommended. (Python 3.13 also works, but requires the `numpy`/`langchain-core`
pins already set in `requirements.txt` — see the comments in that file if you hit an
`ImportError` after installing anything outside it.)

### Installation
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Running the pipeline (in order)
```bash
# Phase 1–2: (already done) scraped_book.json exists, and vector DB benchmark was run
python benchmark_vector_dbs.py --data scraped_book.json   # optional: reproduce the FAISS vs Chroma comparison

# Phase 3: build the FAISS index + test the RAG chain in the terminal
python rag_chain.py

# Debugging retrieval only (no LLM call) — useful if an answer seems wrong
python debug_retrieval.py

# Phase 4: full chatbot web UI
python app.py
# open http://127.0.0.1:7860
```

### Choosing an LLM backend
Set before running `rag_chain.py` / `app.py` (see comments in `rag_chain.py` for full details):
```bash
# Local (default, free, private)
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=gemma2:2b

# or Groq (fast hosted API)
export LLM_PROVIDER=groq
export GROQ_API_KEY=your_key

# or OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key
```

---

## Technical Details

### Embedding model
**`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**, loaded via
`langchain-huggingface`.

**Why this model:**
- It's explicitly multilingual — trained on parallel sentence pairs across 50+ languages,
  including Bengali — unlike a default English-only sentence embedding model, which would
  perform poorly on Bengali queries and text.
- It's compact (~470 MB, 12-layer MiniLM backbone), so it runs fast on CPU with low RAM, which
  matters for a locally-hosted chatbot with no GPU (validated empirically in the FAISS vs
  ChromaDB benchmark — see `benchmark_vector_dbs.py`).
- It's a well-established, widely-used sentence-embedding model for cross-lingual semantic
  search tasks like this one (as opposed to a general-purpose multilingual LM not fine-tuned
  for sentence similarity).

**How it supports Bengali:** The model was fine-tuned via multilingual knowledge distillation
from a strong English "teacher" sentence-embedding model, using parallel corpora that include
Bengali among the covered languages, so Bengali sentences are mapped into (roughly) the same
embedding space as their English/other-language paraphrases — enabling semantic (not just
keyword) retrieval over Bengali text.

### Chunking & preprocessing
- **Chunk size:** 600 characters
- **Chunk overlap:** 100 characters
- **Splitter:** LangChain's `RecursiveCharacterTextSplitter`, with a Bengali-aware separator
  list (`["\n\n", "\n", "।", "?", "!", " ", ""]`) so splits prefer paragraph breaks, then the
  Bengali sentence-ending danda (।), before falling back to word/character boundaries — this
  avoids cutting mid-sentence wherever possible.
- **Why 600/100:** The book is prose-heavy with long dialogue passages; 600 characters keeps
  each chunk to roughly one to a few sentences of context (enough for a self-contained fact),
  while a 100-character overlap (~17%) reduces the chance that a fact spanning a chunk boundary
  is lost entirely from both neighboring chunks.
- **Preprocessing:** Chapter text was scraped as clean article body text (ads/navigation/HTML
  stripped at the scraping stage), stored as one JSON record per chapter with the fields below,
  then split into overlapping chunks before embedding.
- **Metadata preserved per chunk:**
  - `book_name` — always "হাজার বছর ধরে"
  - `chapter_name` — e.g. "০১. মস্ত বড় অজগরের মত সড়কটা"
  - `source_url` — the exact page the chapter was scraped from
  - *(Section name: this book has no sub-chapter section headings — each chapter is a single
    continuous narrative — so `chapter_name` is the finest citation granularity available.)*

### Vector database
**FAISS**, chosen after benchmarking against ChromaDB on this exact dataset (see
`benchmark_vector_dbs.py`), comparing:
- Indexing time (s)
- Peak RAM during indexing (MB)
- On-disk index size (MB)
- Average query latency (ms) across 5 sample Bengali queries

| Vector DB | Indexing Time (s) | Peak RAM (MB) | Disk Size (MB) | Avg Latency (ms) |
|---|---|---|---|---|
| FAISS | 8.862 | 115.56 | 0.74 | 23.67 |
| ChromaDB | 13.426 | 30.79 | 4.03 | 32.05 |

FAISS was selected for being a lighter-weight, dependency-minimal library with no background
database engine, which suits a small (~22K word), single-book, locally-hosted deployment.

### Retriever configuration
A **hybrid retriever** combining two ranking signals via LangChain's `EnsembleRetriever`:
- **BM25** (keyword/sparse search, `k=5`) — guarantees exact-term matches (names, rare nouns)
  rank highly, which the embedding model alone can under-rank for a book written partly in
  colloquial rural Bengali dialect.
- **FAISS** (dense/semantic search, `k=5`) — catches paraphrased or conceptual questions that
  don't share exact wording with the source text.
- **Weights:** 0.5 / 0.5, fused via reciprocal rank fusion. Results are not truncated after
  fusion, so up to 10 chunks may reach the LLM as context for a single question.

### LLM
Switchable via the `LLM_PROVIDER` environment variable (see Setup above):
- **Ollama** (local, default) — `gemma2:2b` or any locally pulled model
- **Groq** — `openai/gpt-oss-20b` (fast hosted inference)
- **OpenAI** — `gpt-4o-mini`

All three go through the same LangChain LCEL chain: `prompt | llm | StrOutputParser()`.

### Prompt design
The system prompt (in Bengali) instructs the model to:
1. Answer **only** from the given context — no outside knowledge.
2. If the answer isn't in the context, say plainly (in Bengali) that the information isn't in
   the book, rather than guessing.
3. Keep answers clear, concise, and in plain Bengali.

---

## RAG Pipeline

```
ebanglalibrary.com (chapter pages)
        │  scraping
        ▼
scraped_book.json  (book_name, chapter_name, source_url, content)
        │  RecursiveCharacterTextSplitter (chunk_size=600, overlap=100)
        ▼
Text chunks + metadata
        │  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        ▼
Embeddings
        │  stored in
        ▼
FAISS index (./vector_bench/faiss_index)
        │  at query time:
        │  user question → embedded → hybrid retrieval (BM25 + FAISS, k=5 each)
        ▼
Top retrieved chunks + their chapter/source metadata
        │  formatted into the Bengali RAG prompt
        ▼
LLM (Ollama / Groq / OpenAI)
        ▼
Final answer (Bengali) + chapter citation(s), shown in the Gradio UI
```

---

## Project Structure
```
├── scraped_book.json         # Phase 1: ingested book (9 chapters)
├── benchmark_vector_dbs.py   # Phase 2: FAISS vs ChromaDB comparison
├── rag_chain.py              # Phase 3: RAG chain, LLM factory, hybrid retriever, CLI
├── debug_retrieval.py        # Retrieval-only diagnostic tool (no LLM call)
├── app.py                    # Phase 4: Gradio web UI
├── test_questions.md         # 10 test questions with expected answers + chapter
├── requirements.txt
└── README.md
```

---

## Limitations / Known Trade-offs
- The small local LLM option (`gemma2:2b` via Ollama) can occasionally struggle to extract the
  right fact when the retrieved context includes several noisy/irrelevant chunks; a hosted
  model (Groq/OpenAI) generally gives more reliable answers over the same retrieved context.
- BM25's contribution helps most on questions containing rare/named terms (character names,
  place names); for questions built from very common words, it adds less signal and dense
  (FAISS) ranking dominates.