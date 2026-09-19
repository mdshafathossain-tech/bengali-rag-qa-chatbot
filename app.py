"""
app.py
======

Phase 4: Web UI for the "হাজার বছর ধরে" (Zahir Raihan) Bengali RAG Knowledge-Base
Chatbot, built with Gradio on top of the Phase 3 RAG pipeline (`rag_chain.py`).

FEATURES
--------
- Custom Soft theme + CSS (Bengali-friendly font, gradient header, badge, styled
  citation cards).
- Token-by-token streaming of answers straight from the LangChain LCEL chain.
- Clear visual separation between the user's question, the streamed answer, and
  a collapsible "তথ্যসূত্র (Sources)" panel with chapter name + clickable URL.
- Clickable example-question buttons, a Clear Chat button, and a Regenerate
  button (re-runs the last question through the chain).
- Friendly, Bengali-language error banners instead of raw stack traces when the
  FAISS index is missing, the LLM backend is unreachable, an API key is
  missing/invalid, or a rate limit is hit.

HOW TO RUN
----------
1. Make sure Phase 2 (FAISS index at ./vector_bench/faiss_index) and Phase 3
   (`rag_chain.py`, in the same folder as this file) are already working.

2. Install the UI dependency (pinned — Gradio 6.0+ moved `theme`/`css` out of
   `gr.Blocks()` and dropped `Chatbot(type=...)`, which this script relies on).
   This is already covered if you're installing from the project's
   requirements.txt: pip install -r requirements.txt

3. Configure your LLM backend exactly as in Phase 3 (env vars), e.g.:

    # PowerShell, Groq example
    $env:LLM_PROVIDER="groq"
    $env:GROQ_API_KEY="your_actual_key"
    $env:GROQ_MODEL="openai/gpt-oss-20b"

    # or for local Ollama (default), just make sure the Ollama app is running
    # and the model is pulled: ollama pull gemma2:2b

4. Launch the app:

    python app.py

   Gradio will print a local URL (usually http://127.0.0.1:7860) — open it in
   your browser.

TROUBLESHOOTING: "ValueError: When localhost is not accessible ..."
---------------------------------------------------------------------
This means Gradio's internal self-check (an HTTP request to 127.0.0.1 to
confirm the machine can reach itself) failed. On Windows this is almost
always a system/VPN/antivirus proxy intercepting even loopback traffic.
Before relying on this script's automatic share=True fallback (which makes
the app briefly PUBLIC), try the local fix first:

    # PowerShell — check if a proxy is configured:
    Get-ChildItem Env: | Where-Object { $_.Name -match 'proxy' }

    # If HTTP_PROXY / HTTPS_PROXY are set, exclude localhost from it:
    $env:NO_PROXY = "127.0.0.1,localhost"
    $env:no_proxy = "127.0.0.1,localhost"
    python app.py

If that doesn't help, check Windows Defender Firewall / antivirus settings
allow python.exe on private networks. Only if none of that works should you
rely on the share=True fallback baked into this script's entry point below.

TROUBLESHOOTING: "TypeError: argument of type 'bool' is not iterable"
-------------------------------------------------------------------------
A known, harmless version mismatch between `pydantic` (2.11+) and
`gradio_client`'s API-schema parser — unrelated to this script's logic and
does not stop the app from working. To silence it:

    pip install "pydantic==2.10.6"
"""

import os
import traceback

import gradio as gr

# ---------------------------------------------------------------------------
# Import the RAG building blocks from Phase 3 (rag_chain.py must be in the
# same directory, or importable on PYTHONPATH).
# ---------------------------------------------------------------------------
from rag_chain import (
    load_vectorstore,
    load_retriever,
    get_llm,
    build_answer_chain,
    format_docs,
    extract_sources,
    RETRIEVER_TOP_K,
)

# ---------------------------------------------------------------------------
# Sample questions shown as clickable buttons
# ---------------------------------------------------------------------------
SAMPLE_QUESTIONS = [
    "পরীর দীঘি কীভাবে সৃষ্টি হয়েছিল?",
    "মকবুলের বড় বউয়ের নাম কী?",
    "টুনির বয়স কত?",
    "হীরন কার মেয়ে?",
    "মন্তুর মা-বাবা কবে মারা যান?",
    "নন্তু শেখের কাজ কী ছিল?",
    "গনু মোল্লা সারাদিন কী করে?",
    "মকবুলের বাড়িতে মোট কত ঘর মানুষ থাকে?",
    "আম্বিয়ার স্বামী ও ছেলে কীভাবে মারা যান?",
    "টুনি কোন ঘরে থাকে?",
    "ফাতেমা দেখতে কেমন?",
    "মিয়া বাড়ির ইতিহাস কী?",
    "মন্তু আর টুনির সম্পর্ক কেমন ছিল?",
    "তেরশ সনের বন্যার সময় গ্রামে কী ঘটেছিল?",
    "কাসেম শিকদার কীভাবে এই গ্রামে এসে বসতি স্থাপন করেছিলেন?",
    "ছমিরণ বিবির পরিণতি কী হয়েছিল?",
    "বুড়ো মকবুলের পরিবারে কতজন বউ ছিল এবং তাদের নাম কী কী?",
    "মন্তু আর টুনি একরাতে জমীর মুন্সির পুকুরে মাছ ধরতে গিয়ে কী অভিজ্ঞতা হয়েছিল?",
    "হীরনের বিয়ের প্রস্তাব কোথা থেকে এবং কার সাথে এসেছিল?",
    "মকবুল কেন মন্তুর বিয়ে আম্বিয়ার সঙ্গে ঠিক করতে চেয়েছিলেন?",
    "আম্বিয়া কে?",
]

APP_TITLE = "হাজার বছর ধরে"
APP_SUBTITLE = "জহির রায়হানের উপন্যাস নিয়ে গড়া বাংলা RAG নলেজ-বেস চ্যাটবট"
APP_BADGE = "📖 Powered by FAISS + LangChain + Bengali LLM"


# ---------------------------------------------------------------------------
# Custom CSS: Bengali-friendly typography, header styling, citation cards
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;500;600;700&display=swap');

/* Apply the Bengali-friendly font everywhere text is rendered */
.gradio-container, .gradio-container * {
    font-family: 'Hind Siliguri', 'Noto Sans Bengali', 'Segoe UI', sans-serif !important;
}

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* Header block */
#app-header {
    text-align: center;
    padding: 18px 12px 8px 12px;
}
#app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 2px;
    background: linear-gradient(90deg, #1f6f5c, #3aa17e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#app-header p.subtitle {
    font-size: 1.05rem;
    color: #555;
    margin-top: 0;
}
#app-header .badge {
    display: inline-block;
    margin-top: 6px;
    padding: 4px 14px;
    border-radius: 999px;
    background: #eaf6f1;
    color: #1f6f5c;
    font-size: 0.85rem;
    font-weight: 600;
    border: 1px solid #bfe4d6;
}

/* Chat bubbles: larger, more readable Bengali text */
.message.user, .message.bot {
    font-size: 1.02rem !important;
    line-height: 1.7 !important;
}

/* Citation card styling (rendered as <details> inside assistant messages) */
details.sources-card {
    margin-top: 10px;
    padding: 8px 12px;
    border-radius: 10px;
    background: #f6f9f8;
    border: 1px solid #dce8e3;
}
details.sources-card summary {
    cursor: pointer;
    font-weight: 600;
    color: #1f6f5c;
}
details.sources-card ul {
    margin: 8px 0 2px 0;
    padding-left: 20px;
}

/* Example question buttons */
.example-btn-wrap {
    flex-wrap: wrap !important;
    gap: 6px !important;
}
.example-btn button {
    border-radius: 999px !important;
    font-size: 0.9rem !important;
    width: auto !important;
    white-space: normal !important;
}

/* Error banner */
#init-error {
    background: #fdecea;
    border: 1px solid #f5c2c0;
    color: #7a1f1a;
    padding: 10px 14px;
    border-radius: 8px;
}
"""


# ---------------------------------------------------------------------------
# One-time startup: load the vector store, retriever, LLM and answer chain.
# Any failure here (missing index, missing API key, unreachable Ollama, etc.)
# is captured instead of crashing the whole app, so the UI can still launch
# and show a friendly explanation.
# ---------------------------------------------------------------------------
INIT_ERROR = None
retriever = None
answer_chain = None

try:
    _vectorstore = load_vectorstore()
    retriever = load_retriever(_vectorstore)  # hybrid BM25 + FAISS
    _llm = get_llm()
    answer_chain = build_answer_chain(_llm)
except Exception as exc:  # noqa: BLE001 - deliberately broad: any startup issue
    INIT_ERROR = str(exc)
    print("Startup error while initializing the RAG pipeline:")
    print(traceback.format_exc())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_sources_html(sources: list[dict]) -> str:
    """Renders retrieved-chunk metadata as a collapsible citation card."""
    if not sources:
        return ""

    items = []
    for src in sources:
        chapter = src.get("chapter_name", "unknown")
        url = src.get("source_url", "")
        if url:
            items.append(f'<li><b>{chapter}</b> — <a href="{url}" target="_blank">সূত্র দেখুন</a></li>')
        else:
            items.append(f"<li><b>{chapter}</b></li>")

    return (
        '\n\n<details class="sources-card">'
        "<summary>📚 তথ্যসূত্র (Sources) দেখুন</summary>"
        f"<ul>{''.join(items)}</ul>"
        "</details>"
    )


def friendly_error_message(exc: Exception) -> str:
    """Maps common backend failures to a polite, actionable Bengali message
    instead of surfacing raw stack traces in the chat UI.
    """
    text = str(exc).lower()

    if "api_key" in text or "environmentvariable" in text or "environment variable" in text:
        return (
            "⚠️ **কনফিগারেশন সমস্যা:** LLM API কী সঠিকভাবে সেট করা নেই। "
            "টার্মিনালে `GROQ_API_KEY` / `OPENAI_API_KEY` এনভায়রনমেন্ট ভ্যারিয়েবল "
            "সঠিকভাবে সেট করা আছে কিনা যাচাই করুন (অতিরিক্ত স্পেস নেই তো?)।"
        )
    if "rate limit" in text or "429" in text:
        return (
            "⚠️ **রেট লিমিট অতিক্রম হয়েছে:** এই মুহূর্তে অনেক বেশি অনুরোধ পাঠানো হয়েছে। "
            "কিছুক্ষণ অপেক্ষা করে আবার চেষ্টা করুন।"
        )
    if "model_not_found" in text or "404" in text or "does not exist" in text:
        return (
            "⚠️ **মডেল পাওয়া যায়নি:** সেট করা LLM মডেলের নাম ভুল অথবা এটি ডিপ্রিকেটেড হতে পারে। "
            "`GROQ_MODEL` / `OLLAMA_MODEL` এনভায়রনমেন্ট ভ্যারিয়েবল যাচাই করুন।"
        )
    if "connection" in text or "timeout" in text or "refused" in text:
        return (
            "⚠️ **সংযোগ সমস্যা:** LLM সার্ভারের সাথে সংযোগ স্থাপন করা যায়নি। "
            "যদি Ollama ব্যবহার করেন, তাহলে Ollama অ্যাপ চালু আছে কিনা যাচাই করুন। "
            "API ব্যবহার করলে ইন্টারনেট সংযোগ এবং API কী যাচাই করুন।"
        )

    # Fallback: generic but still friendly, with the raw error for debugging.
    return f"⚠️ **দুঃখিত, একটি সমস্যা হয়েছে:** {exc}"


# ---------------------------------------------------------------------------
# Core chat logic (streaming generator)
# ---------------------------------------------------------------------------
def user_submit(message: str, history: list):
    """Appends the user's message to the chat history and clears the textbox."""
    if not message or not message.strip():
        return "", history
    history = history + [{"role": "user", "content": message.strip()}]
    return "", history


def bot_respond(history: list):
    """Streams the assistant's answer for the most recent user message, then
    appends a formatted source-citation panel once streaming completes.
    Yields the growing `history` list so Gradio's Chatbot updates live.
    """
    if not history or history[-1]["role"] != "user":
        return

    question = history[-1]["content"]
    history = history + [{"role": "assistant", "content": ""}]
    yield history

    if INIT_ERROR:
        history[-1]["content"] = (
            "⚠️ **অ্যাপ চালু করা যায়নি:** RAG পাইপলাইন ইনিশিয়ালাইজ করার সময় সমস্যা হয়েছে।\n\n"
            f"বিস্তারিত: `{INIT_ERROR}`\n\n"
            "FAISS ইনডেক্স, এনভায়রনমেন্ট ভ্যারিয়েবল, এবং LLM ব্যাকএন্ড (Ollama/Groq) সেটআপ যাচাই করুন।"
        )
        yield history
        return

    try:
        docs = retriever.invoke(question)
        context = format_docs(docs)
        sources = extract_sources(docs)

        partial = ""
        for token in answer_chain.stream({"context": context, "question": question}):
            partial += token
            history[-1]["content"] = partial
            yield history

        history[-1]["content"] = partial + build_sources_html(sources)
        yield history

    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a friendly banner
        print("Error while answering a question:")
        print(traceback.format_exc())
        history[-1]["content"] = friendly_error_message(exc)
        yield history


def regenerate(history: list):
    """Drops the last assistant turn (keeping the user's question) so
    `bot_respond` can be re-run on it.
    """
    if history and history[-1]["role"] == "assistant":
        history = history[:-1]
    return history


def clear_chat():
    return [], ""


def use_example(question: str):
    return question


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="teal",
        font=[gr.themes.GoogleFont("Hind Siliguri"), "sans-serif"],
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title=f"{APP_TITLE} — বাংলা RAG চ্যাটবট") as demo:

        # ---- Header -------------------------------------------------
        gr.HTML(
            f"""
            <div id="app-header">
                <h1>📖 {APP_TITLE}</h1>
                <p class="subtitle">{APP_SUBTITLE}</p>
                <span class="badge">{APP_BADGE}</span>
            </div>
            """
        )

        if INIT_ERROR:
            gr.Markdown(
                f"**⚠️ অ্যাপ চালু করতে সমস্যা হয়েছে।** বিস্তারিত: `{INIT_ERROR}`\n\n"
                "নিচে প্রশ্ন করলেও একই সতর্কবার্তা দেখানো হবে যতক্ষণ না সমস্যাটি সমাধান হয়।",
                elem_id="init-error",
            )

        # ---- Chat window ---------------------------------------------
        chatbot = gr.Chatbot(
            type="messages",
            height=480,
            label=None,
            show_label=False,
            avatar_images=(None, None),
            render_markdown=True,
        )

        # ---- Input row --------------------------------------------------
        with gr.Row():
            msg = gr.Textbox(
                placeholder="আপনার প্রশ্ন এখানে বাংলায় লিখুন... (যেমনঃ মন্তু আর টুনির সম্পর্ক কেমন ছিল?)",
                show_label=False,
                scale=8,
                container=False,
            )
            send_btn = gr.Button("পাঠান 📤", variant="primary", scale=1)

        # ---- Example question buttons -----------------------------------
        with gr.Accordion(f"নমুনা প্রশ্ন ({len(SAMPLE_QUESTIONS)}টি) — ক্লিক করে জিজ্ঞেস করুন", open=False):
            with gr.Row(elem_classes="example-btn-wrap"):
                example_buttons = [
                    gr.Button(q, elem_classes="example-btn") for q in SAMPLE_QUESTIONS
                ]

        # ---- Control row: Clear / Regenerate -----------------------------
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat")
            regen_btn = gr.Button("🔄 Regenerate")

        # ---- Wiring --------------------------------------------------
        # Submit via Enter key in the textbox
        msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
            bot_respond, chatbot, chatbot
        )
        # Submit via the Send button
        send_btn.click(user_submit, [msg, chatbot], [msg, chatbot]).then(
            bot_respond, chatbot, chatbot
        )

        # Example buttons: fill the textbox, then run the same submit flow
        for btn, question in zip(example_buttons, SAMPLE_QUESTIONS):
            btn.click(use_example, gr.State(question), msg).then(
                user_submit, [msg, chatbot], [msg, chatbot]
            ).then(bot_respond, chatbot, chatbot)

        # Clear chat
        clear_btn.click(clear_chat, None, [chatbot, msg])

        # Regenerate last answer
        regen_btn.click(regenerate, chatbot, chatbot).then(
            bot_respond, chatbot, chatbot
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    # .queue() is required so the streaming generator (`bot_respond`) can
    # push incremental updates to the UI as tokens arrive.
    queued_app = app.queue()

    try:
        # Preferred: fully local, private, no data leaves the machine.
        # If this fails with "localhost is not accessible", it's almost
        # always a system/VPN/antivirus proxy intercepting the loopback
        # request Gradio uses to verify it can reach itself — see the
        # troubleshooting notes in this file's docstring before assuming
        # the share=True fallback below is required.
        queued_app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
        )
    except ValueError as exc:
        if "localhost is not accessible" not in str(exc):
            raise
        print(
            "\n[Warning] Could not reach 127.0.0.1 locally (likely a proxy, VPN, "
            "or firewall issue — see the troubleshooting notes at the top of "
            "this file for a fix that keeps the app private).\n"
            "Falling back to a TEMPORARY PUBLIC LINK. Anyone with that link "
            "can use this chatbot and consume your LLM API quota until you "
            "stop this script.\n"
        )
        queued_app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=True,
        )