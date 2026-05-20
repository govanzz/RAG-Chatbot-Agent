"""Streamlit UI components and styling for the Mandai-themed chatbot."""

from __future__ import annotations

from html import escape

import streamlit as st

from config import (
    EMBEDDING_MODEL_NAME,
    PROJECT_NAME,
    REQUESTED_DEFAULT_MODEL,
    TOP_K,
)
from llm import OllamaConnectionError, get_installed_models, select_ollama_model
from rag import get_indexed_chunk_count, unique_sources, vectorstore_exists
from utils import clean_multiline_text


def configure_page() -> None:
    st.set_page_config(page_title=PROJECT_NAME, page_icon="M", layout="centered")


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --mandai-forest: #143a2a;
            --mandai-deep: #0d2a20;
            --mandai-green: #2f6b3f;
            --mandai-leaf: #7d9f45;
            --mandai-moss: #dfead2;
            --mandai-sand: #f6f1e7;
            --mandai-cream: #fffaf0;
            --mandai-amber: #d78a2d;
            --mandai-rust: #b86134;
            --mandai-river: #5f8f99;
            --mandai-bark: #6a4b2f;
            --mandai-ink: #1f2f24;
            --mandai-muted: #5d6d60;
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(20, 58, 42, 0.08), rgba(246, 241, 231, 0.98) 260px),
                repeating-linear-gradient(135deg, rgba(47, 107, 63, 0.055) 0 1px, transparent 1px 22px),
                var(--mandai-sand);
            color: var(--mandai-ink);
        }

        .block-container {
            max-width: 1060px;
            padding-top: 1.3rem;
            padding-bottom: 6rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #103326 0%, #1c563b 64%, #f2e8d6 64%, #f2e8d6 100%);
            border-right: 1px solid rgba(20, 58, 42, 0.18);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label {
            color: var(--mandai-cream);
        }

        [data-testid="stSidebar"] label {
            font-weight: 800;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: rgba(255, 250, 240, 0.9);
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] input {
            background: var(--mandai-cream);
            color: var(--mandai-ink);
            border: 1px solid rgba(20, 58, 42, 0.2);
            border-radius: 8px;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] input::placeholder {
            color: var(--mandai-ink);
            -webkit-text-fill-color: var(--mandai-ink);
        }

        [data-testid="stSidebar"] [data-baseweb="select"] svg {
            color: var(--mandai-forest);
            fill: var(--mandai-forest);
        }

        [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {
            background: rgba(255, 250, 240, 0.35);
        }

        [data-testid="stSidebar"] [data-testid="stAlert"] {
            background: #e5f1dc;
            border: 1px solid rgba(47, 107, 63, 0.24);
            border-radius: 8px;
        }

        [data-testid="stSidebar"] [data-testid="stAlert"],
        [data-testid="stSidebar"] [data-testid="stAlert"] * {
            color: var(--mandai-ink);
            -webkit-text-fill-color: var(--mandai-ink);
        }

        [data-testid="stSidebar"] .stButton button {
            background: var(--mandai-cream);
            color: var(--mandai-forest);
            -webkit-text-fill-color: var(--mandai-forest);
            border: 1px solid rgba(20, 58, 42, 0.18);
            border-radius: 8px;
            font-weight: 800;
        }

        [data-testid="stSidebar"] .stButton button p,
        [data-testid="stSidebar"] .stButton button span {
            color: var(--mandai-forest);
            -webkit-text-fill-color: var(--mandai-forest);
            font-weight: 800;
        }

        [data-testid="stSidebar"] code {
            background: var(--mandai-cream);
            color: var(--mandai-forest);
        }

        .mandai-sidebar-brand {
            padding: 1rem 0 1.15rem;
            border-bottom: 1px solid rgba(255, 250, 240, 0.22);
            margin-bottom: 1rem;
        }

        .mandai-mark {
            width: 46px;
            height: 46px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--mandai-cream);
            color: var(--mandai-forest);
            font-weight: 900;
            letter-spacing: 0;
            margin-bottom: 0.75rem;
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.16);
        }

        .mandai-sidebar-brand h2 {
            font-size: 1.26rem;
            line-height: 1.16;
            margin: 0;
            color: var(--mandai-cream);
        }

        .mandai-sidebar-brand p {
            margin: 0.45rem 0 0;
            font-size: 0.86rem;
            color: rgba(255, 250, 240, 0.86);
        }

        .mandai-side-note {
            margin: 1.2rem 0 0.35rem;
            padding: 0.85rem;
            border-radius: 8px;
            background: rgba(255, 250, 240, 0.14);
            border: 1px solid rgba(255, 250, 240, 0.18);
            color: rgba(255, 250, 240, 0.92);
            font-size: 0.86rem;
            line-height: 1.45;
        }

        .mandai-hero {
            position: relative;
            overflow: hidden;
            border-radius: 8px;
            padding: 2rem;
            margin-bottom: 1rem;
            color: var(--mandai-cream);
            background:
                linear-gradient(120deg, rgba(13, 42, 32, 0.97), rgba(47, 107, 63, 0.9)),
                repeating-linear-gradient(30deg, rgba(255, 250, 240, 0.08) 0 1px, transparent 1px 34px);
            border: 1px solid rgba(255, 250, 240, 0.22);
            box-shadow: 0 24px 70px rgba(20, 58, 42, 0.18);
        }

        .mandai-hero::after {
            content: "";
            position: absolute;
            inset: auto -8% -44% 48%;
            height: 178px;
            background:
                linear-gradient(135deg, transparent 0 40%, rgba(255, 250, 240, 0.14) 40% 46%, transparent 46% 100%),
                linear-gradient(45deg, transparent 0 42%, rgba(255, 250, 240, 0.12) 42% 48%, transparent 48% 100%);
            opacity: 0.75;
            transform: rotate(-4deg);
        }

        .mandai-kicker {
            display: inline-flex;
            align-items: center;
            padding: 0.36rem 0.62rem;
            border-radius: 999px;
            background: rgba(255, 250, 240, 0.13);
            border: 1px solid rgba(255, 250, 240, 0.22);
            color: #f7e3bb;
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .mandai-hero h1 {
            margin: 0.9rem 0 0.45rem;
            max-width: 720px;
            font-size: 2.28rem;
            line-height: 1.05;
            color: var(--mandai-cream);
            letter-spacing: 0;
        }

        .mandai-hero p {
            max-width: 760px;
            margin: 0;
            color: rgba(255, 250, 240, 0.88);
            font-size: 1rem;
        }

        .mandai-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.15rem;
        }

        .mandai-chip {
            border-radius: 999px;
            padding: 0.42rem 0.7rem;
            background: rgba(255, 250, 240, 0.14);
            border: 1px solid rgba(255, 250, 240, 0.22);
            color: var(--mandai-cream);
            font-size: 0.84rem;
            font-weight: 700;
        }

        [data-testid="stChatMessage"] {
            border-radius: 8px;
            border: 1px solid rgba(20, 58, 42, 0.13);
            background: rgba(255, 250, 240, 0.9);
            box-shadow: 0 12px 35px rgba(47, 107, 63, 0.08);
            margin-bottom: 0.85rem;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #edf4df;
            border-color: rgba(125, 159, 69, 0.32);
        }

        [data-testid="stChatMessage"] p {
            color: var(--mandai-ink);
            line-height: 1.58;
        }

        .source-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.65rem;
            margin: 0.7rem 0 0.3rem;
        }

        .source-card {
            display: block;
            border-radius: 8px;
            padding: 0.78rem 0.85rem;
            background: #f9efd9;
            border: 1px solid rgba(106, 75, 47, 0.18);
            text-decoration: none;
        }

        .source-card strong {
            color: var(--mandai-forest);
            display: block;
            font-size: 0.94rem;
            line-height: 1.25;
            margin-bottom: 0.25rem;
        }

        .source-card span {
            color: var(--mandai-bark);
            font-size: 0.78rem;
            overflow-wrap: anywhere;
        }

        .chunk-box {
            border: 1px solid rgba(47, 107, 63, 0.16);
            border-radius: 8px;
            padding: 0.85rem;
            margin: 0.55rem 0;
            background: rgba(255, 250, 240, 0.94);
        }

        .chunk-box strong {
            color: var(--mandai-forest);
        }

        .chunk-box span {
            color: var(--mandai-bark);
            font-size: 0.82rem;
        }

        .chunk-box a {
            color: var(--mandai-green);
            overflow-wrap: anywhere;
            font-size: 0.82rem;
        }

        .chunk-box p {
            color: var(--mandai-ink);
            margin-bottom: 0;
        }

        .stChatInput {
            background: rgba(246, 241, 231, 0.78);
        }

        [data-testid="stChatInput"] {
            border-radius: 8px;
            border: 1px solid rgba(47, 107, 63, 0.18);
            box-shadow: 0 12px 35px rgba(20, 58, 42, 0.12);
        }

        a {
            color: var(--mandai-green);
        }

        div[data-testid="stExpander"] {
            border-color: rgba(47, 107, 63, 0.18);
            background: rgba(255, 250, 240, 0.72);
            border-radius: 8px;
        }

        @media (max-width: 640px) {
            .mandai-hero {
                padding: 1.35rem;
            }
            .mandai-hero h1 {
                font-size: 1.65rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sources(chunks: list[dict]) -> None:
    sources = unique_sources(chunks)
    if sources:
        cards = []
        for source in sources:
            title = escape(str(source["title"]))
            url = escape(str(source["url"]), quote=True)
            cards.append(
                (
                    f'<a class="source-card" href="{url}" '
                    'target="_blank" rel="noopener">'
                    f"<strong>{title}</strong>"
                    f"<span>{url}</span>"
                    "</a>"
                )
            )
        st.markdown(
            (
                '<div style="font-weight: 800; color: #143a2a; '
                'margin-top: 0.85rem;">Sources</div>'
                f'<div class="source-grid">{"".join(cards)}</div>'
            ),
            unsafe_allow_html=True,
        )

    if chunks:
        with st.expander("Source evidence used for this answer"):
            for chunk in chunks:
                title = escape(str(chunk.get("page_title", "Untitled")))
                url = escape(str(chunk.get("source_url", "")), quote=True)
                section = escape(str(chunk.get("section_heading") or "General"))
                raw_text = clean_multiline_text(str(chunk.get("text", "")))
                raw_text = raw_text.replace("### ", "").replace("## ", "").replace("# ", "")
                if len(raw_text) > 520:
                    raw_text = f"{raw_text[:520].rstrip()}..."
                text = escape(raw_text)
                score = float(chunk.get("score", 0.0))
                st.markdown(
                    (
                        '<div class="chunk-box">'
                        f"<strong>{title}</strong><br>"
                        f"<span>{section} | score {score:.3f}</span><br>"
                        f'<a href="{url}">{url}</a>'
                        f"<p>{text}</p>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )


def render_sidebar() -> tuple[str, int]:
    st.sidebar.markdown(
        f"""
        <div class="mandai-sidebar-brand">
            <div class="mandai-mark">M</div>
            <h2>{PROJECT_NAME}</h2>
            <p>Grounded answers from locally indexed Mandai website pages.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    installed_models: list[str] = []
    ollama_warning = ""
    try:
        installed_models = get_installed_models()
    except OllamaConnectionError as exc:
        ollama_warning = str(exc)

    auto_model = select_ollama_model(installed_models=installed_models)
    if installed_models:
        selected_index = (
            installed_models.index(auto_model) if auto_model in installed_models else 0
        )
        selected_model = st.sidebar.selectbox(
            "Current Ollama model",
            installed_models,
            index=selected_index,
        )
        if REQUESTED_DEFAULT_MODEL not in installed_models:
            st.sidebar.warning(
                "llama3.1:8b is not installed. Run: ollama pull llama3.1:8b"
            )
    else:
        selected_model = st.sidebar.text_input("Current Ollama model", value=auto_model)
        st.sidebar.error("Ollama is not running. Start it with: ollama serve")
        if ollama_warning:
            st.sidebar.caption(ollama_warning)

    top_k = st.sidebar.slider("Retrieved chunks (top_k)", 1, 8, TOP_K)
    st.sidebar.caption(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    st.sidebar.caption(f"Indexed chunks: {get_indexed_chunk_count()}")

    if vectorstore_exists():
        st.sidebar.success("Vectorstore found.")
    else:
        st.sidebar.warning("Run ingestion first:")
        st.sidebar.code("python ingest_web.py")

    st.sidebar.markdown(
        """
        <div class="mandai-side-note">
            Built for local RAG demos: retrieve first, answer second, cite always.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("Refresh index status"):
        st.cache_resource.clear()
        st.rerun()

    return selected_model, top_k


def render_hero(selected_model: str, top_k: int) -> None:
    st.markdown(
        f"""
        <section class="mandai-hero">
            <div class="mandai-kicker">Local-first wildlife guide</div>
            <h1>{PROJECT_NAME}</h1>
            <p>Ask about Mandai parks, visits, wildlife experiences, and conservation. Answers stay grounded in retrieved official website content.</p>
            <div class="mandai-chips">
                <span class="mandai-chip">Mandai website RAG</span>
                <span class="mandai-chip">Ollama: {escape(selected_model)}</span>
                <span class="mandai-chip">Top {top_k} chunks</span>
                <span class="mandai-chip">{get_indexed_chunk_count()} indexed chunks</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
