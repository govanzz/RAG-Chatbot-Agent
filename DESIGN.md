# Design

## Overall Architecture

The application has three main layers:

1. `ingest_web.py` scrapes the configured Mandai URLs and stores cleaned page text locally.
2. `rag.py` chunks the stored pages, embeds the chunks with SentenceTransformers, writes a FAISS index, and retrieves top matching chunks at query time.
3. `app.py` runs the Streamlit chat flow, calls retrieval, and asks `llm.py` to generate grounded answers through Ollama.
4. `ui.py` owns the Mandai-inspired presentation layer: theme CSS, sidebar, hero, source cards, and retrieved chunk display.

Configuration lives in `config.py`, shared helpers live in `utils.py`, and runtime logs are written under `logs/`.

## Why Local-First LLM Was Chosen

The assistant is designed to run without paid APIs and without sending user questions or retrieved Mandai content to hosted model providers. Ollama keeps generation local, which supports privacy, repeatability, and offline-friendly development after models are installed.

## Website Ingestion Strategy

The scraper only visits the URLs listed in `config.py`. The current configuration contains 28 official Mandai pages, including top-level park pages, visit pages, ticket pages, transport pages, know-before-you-go pages, and selected activity or zone pages. It does not crawl links or discover new pages. Requests use a clear user-agent, timeout, and delay between pages. Each scraped page is saved as a `.txt` file in `data/raw_pages/` with title, source URL, scrape timestamp, headings, and cleaned body text.

The parser removes scripts, styles, navigation, headers, footers, forms, iframes, empty lines, and repeated whitespace where possible. If one page fails, ingestion logs the failure to `logs/ingest_failures.jsonl` and continues with the remaining configured URLs.

## Chunking Strategy

Documents are split into sections using extracted Markdown-style headings, then chunked into about 700 characters with about 120 characters of overlap. This keeps chunks small enough for precise retrieval while preserving nearby context across chunk boundaries.

Each chunk stores:

- source URL
- page title
- section heading when available
- raw page path
- chunk ID
- chunk text

## Embedding Strategy

The embedding model is `all-MiniLM-L6-v2`, loaded locally through SentenceTransformers. It is compact, fast, and strong enough for a small website RAG project. Embeddings are normalized so FAISS inner-product search behaves like cosine similarity.

## FAISS Vector Storage

FAISS stores the vector index in `vectorstore/index.faiss`. Chunk metadata is stored separately in `vectorstore/metadata.json`. This keeps retrieval fast while preserving human-readable citation data.

## Retrieval Strategy

For each user question, the app embeds the question and retrieves the top `k` chunks, with `top_k` defaulting to 4. Retrieved chunks are passed to the LLM as the only allowed context. The UI shows citations under each answer and keeps the raw retrieved evidence in an optional expander.

## Prompt Engineering Choices

The system prompt instructs the model to:

- act as a Mandai website knowledge assistant
- answer only from retrieved context
- avoid invented facts
- say when retrieved content is insufficient
- keep answers concise
- include citations with page titles and URLs

The app also returns a fixed insufficient-context answer when retrieval is empty or weak.

## Citation Strategy

Every retrieved chunk carries its page title and source URL. The prompt asks the model to cite those details in the answer, and the Streamlit UI lists retrieved sources below each response regardless of how the model phrases the answer.

## UI Strategy

The UI is intentionally secondary to the RAG pipeline. It uses a Mandai-inspired nature palette without copying the official site exactly: deep reserve greens, warm sand backgrounds, moss highlights, and amber/rust accents. The sidebar controls use explicit dark text inside light controls to avoid disappearing labels. The main page stays focused on the chat experience, citations, and optional retrieved source evidence.

## Error Handling

- Missing vectorstore: the UI tells the user to run `python ingest_web.py`.
- Ollama not running: the UI tells the user to run `ollama serve`.
- Missing `llama3.1:8b`: the UI tells the user to run `ollama pull llama3.1:8b`.
- Page scrape failures: ingestion logs the error and continues.
- Missing Python packages: runtime errors point back to `pip install -r requirements.txt`.

## Privacy Considerations

User questions, retrieved chunks, embeddings, vector indexes, and chat logs stay on the local machine. The only network calls are to the configured Mandai website pages during ingestion and to the local Ollama server during chat.

## Limitations

- The assistant only knows the configured URLs.
- It may miss content rendered only by client-side JavaScript.
- It depends on local model quality and installed Ollama models.
- Website updates require rerunning ingestion.
- The first embedding run may need internet access to download `all-MiniLM-L6-v2` unless it is already cached.

## Future Improvements

- Add more official Mandai URLs to `MANDAI_URLS`.
- Add scheduled re-ingestion.
- Add HTML extraction rules tailored to Mandai page templates.
- Add reranking for better retrieval precision.
- Add tests for chunking, metadata parsing, and retrieval behavior.
- Add a small evaluation set for answer grounding.
