# Mandai Website RAG Assistant

A local-first Streamlit chatbot that retrieves official Mandai Wildlife Group / Mandai Wildlife Reserve website content and answers questions using only the retrieved website context.

The app uses local embeddings, a local FAISS vectorstore, and Ollama for local generation. It does not use OpenAI or paid APIs.

## Features

- Scrapes only a fixed configured list of official Mandai URLs. The current list contains 28 pages.
- Stores cleaned page text in `data/raw_pages/`.
- Chunks pages with overlap and stores chunk metadata.
- Embeds chunks with `all-MiniLM-L6-v2`.
- Stores vectors in FAISS under `vectorstore/`.
- Uses Ollama for grounded local answer generation.
- Shows citations and optional retrieved evidence in a Mandai-inspired UI.
- Logs every query to `logs/chat_log.jsonl`.

## Project Structure

- `app.py` - Streamlit entry point and chat/RAG flow.
- `ui.py` - Mandai-inspired layout, styling, sidebar, sources, and retrieved chunk cards.
- `ingest_web.py` - Scrapes configured Mandai pages and builds the index.
- `rag.py` - Chunking, embeddings, FAISS storage, and retrieval.
- `llm.py` - Ollama model selection and grounded answer generation.
- `config.py` - URLs, paths, model names, and runtime settings.
- `utils.py` - Shared helpers.

## Tech Stack

- Python
- Streamlit
- Ollama
- SentenceTransformers
- FAISS
- BeautifulSoup
- requests
- python-dotenv

## Model Setup

The default model is `llama3.1:8b`, with `phi3:latest` as fallback:

```bash
ollama pull llama3.1:8b
ollama serve
```

On this workstation, `llama3.1:8b` is already installed, so the app will use it unless you set a different model in `.env` or `config.py`.

To use the installed model explicitly:

```bash
OLLAMA_MODEL=llama3.1:8b
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Copy the example environment file if you want to override defaults:

```bash
copy .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

## Ingest Website Content

Run:

```bash
python ingest_web.py
```

This scrapes the configured Mandai URLs, writes `.txt` files to `data/raw_pages/`, creates embeddings, and saves the FAISS index plus metadata in `vectorstore/`. The scraper does not crawl the website; add more URLs manually in `config.py` if you want broader coverage.

## Run the App

```bash
python -m streamlit run app.py
```

If the vectorstore is missing, the app will tell you to run:

```bash
python ingest_web.py
```

If Ollama is not running, start it with:

```bash
ollama serve
```

## Example Questions

- What parks are part of Mandai Wildlife Reserve?
- What can visitors do at Singapore Zoo?
- What is Night Safari known for?
- What conservation work does Mandai support?
- How should I plan a family visit?
- What is Bird Paradise?
- What is River Wonders?
- What accessibility or visitor information is available?
- How do I get to Mandai Wildlife Reserve?
- What ticket or pass options are mentioned?
- What should visitors know before going to Night Safari?

## Repository Link

Public GitHub repository link: to be added after publishing.

## Limitations

- Answers are limited to the configured Mandai pages and the quality of extracted text.
- The first embedding run may download the SentenceTransformers model if it is not cached locally.
- Website content can change, so rerun ingestion to refresh the local index.
- The chatbot will refuse or give an insufficient-context answer when retrieval does not contain enough relevant content.
- FAISS and SentenceTransformers wheels may vary by Python version; Python 3.10-3.12 is usually the smoothest setup for local ML packages.
