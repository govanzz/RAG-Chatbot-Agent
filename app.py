"""Streamlit chatbot entry point for the Mandai Website RAG Assistant."""

from __future__ import annotations

import streamlit as st

from config import CHAT_LOG_PATH
from llm import (
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    generate_answer,
)
from rag import MandaiRetriever, VectorStoreMissingError, unique_sources
from ui import (
    configure_page,
    inject_theme,
    render_hero,
    render_sidebar,
    render_sources,
)
from utils import append_jsonl, utc_now_iso


configure_page()
inject_theme()


@st.cache_resource(show_spinner=False)
def get_retriever() -> MandaiRetriever:
    return MandaiRetriever()


def log_query(question: str, chunks: list[dict], model: str) -> None:
    append_jsonl(
        CHAT_LOG_PATH,
        {
            "timestamp": utc_now_iso(),
            "user_question": question,
            "retrieved_source_urls": [
                source["url"] for source in unique_sources(chunks)
            ],
            "model_used": model,
        },
    )


selected_model, top_k = render_sidebar()
render_hero(selected_model, top_k)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me about Mandai Wildlife Reserve, its parks, visitor planning, and conservation work. I will answer only from retrieved Mandai website content.",
            "chunks": [],
            "model": selected_model,
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("chunks", []))

question = st.chat_input("Ask a question about Mandai Wildlife Reserve")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chunks: list[dict] = []
        model_used = selected_model
        try:
            retriever = get_retriever()
            chunks = retriever.retrieve(question, top_k=top_k)
            with st.spinner("Reading retrieved Mandai content..."):
                result = generate_answer(question, chunks, model=selected_model)
            answer = result.answer
            model_used = result.model
            st.markdown(answer)
            render_sources(chunks)
        except VectorStoreMissingError:
            answer = "Vectorstore is missing. Run: `python ingest_web.py`"
            st.error(answer)
        except OllamaConnectionError:
            answer = "Ollama is not running. Start it with: `ollama serve`"
            st.error(answer)
        except OllamaModelNotFoundError as exc:
            answer = str(exc)
            st.error(answer)
        except OllamaError as exc:
            answer = f"Ollama error: {exc}"
            st.error(answer)
        except RuntimeError as exc:
            answer = str(exc)
            st.error(answer)
        finally:
            log_query(question, chunks, model_used)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "chunks": chunks,
            "model": model_used,
        }
    )
