# Development Conversation Summary

This document summarizes the development process of the Mandai Website RAG Assistant in a professional conversation-style format. It captures the main implementation decisions, architecture choices, debugging steps, retrieval improvements, interface refinements, and validation activities completed during development.

---

# Initial Project Request

User: Build a local-first Website RAG AI Agent chatbot in Python for Mandai Wildlife Group / Mandai Wildlife Reserve website content. The application should use Streamlit, Ollama, SentenceTransformers, FAISS, BeautifulSoup, requests, and dotenv. It should include website ingestion, local vector storage, grounded answer generation, citations, logging, and a professional project structure.

Assistant: Created a GitHub-ready Python project structure including:

- `app.py`
- `ui.py`
- `ingest_web.py`
- `rag.py`
- `llm.py`
- `config.py`
- `utils.py`
- `requirements.txt`
- `.env.example`
- `data/raw_pages/`
- `vectorstore/`
- `logs/`

Assistant: Implemented the application as a local-first RAG chatbot. The ingestion pipeline scrapes configured Mandai website pages, cleans the HTML content, stores raw text locally, chunks the content, embeds the chunks with SentenceTransformers, and saves the vectors in FAISS. The Streamlit application retrieves relevant chunks for each question and uses Ollama to generate grounded answers with citations.

---

# Local Model Selection

User: Use the best local model already available instead of requiring a model that is not installed.

Assistant: Checked the available Ollama models and found `llama3.1:8b` and `phi3:latest`. Selected `llama3.1:8b` as the default model because it provides stronger answer quality for this assistant. Kept `phi3:latest` as a fallback option.

Assistant: Updated the model-selection logic so the app uses the preferred local model when available and provides a clear message if Ollama is not running or the selected model is missing.

---

# Dependency Verification

User: Confirm whether installing the Python dependencies worked correctly.

Assistant: Reviewed the installation result and verified that the required packages were available, including Streamlit, BeautifulSoup, requests, python-dotenv, SentenceTransformers, FAISS, and NumPy.

Assistant: Recommended launching Streamlit with:

```bash
python -m streamlit run app.py
```

Assistant: This command avoids common PATH issues on Windows when command-line scripts are installed outside the global terminal path.

---

# Website Ingestion

User: Run the website ingestion process and confirm whether it works.

Assistant: Verified that `python ingest_web.py` successfully scraped the configured official Mandai pages and created the local RAG assets.

Assistant: Confirmed that ingestion produced:

- Cleaned page text files in `data/raw_pages/`.
- A FAISS index in `vectorstore/index.faiss`.
- Metadata in `vectorstore/metadata.json`.
- Indexed chunks ready for semantic retrieval.

Assistant: The scraper was kept intentionally controlled. It only visits official Mandai URLs configured in the project and does not recursively crawl the website. This makes the data source predictable, easier to validate, and safer for a focused RAG assistant.

---

# Expanded Mandai Website Coverage

User: Check whether the scraper only uses configured pages, and add more useful Mandai pages if needed.

Assistant: Confirmed that the scraper only visits the fixed URL list. Expanded the configured source list to 28 official Mandai pages, covering park pages, visitor information, ticketing, transport, selected activities, conservation, and animal or zone pages.

Assistant: The ingestion strategy remains bounded and deliberate. New coverage can be added by adding more official Mandai URLs to the configured list and rerunning ingestion.

---

# Chunking and Retrieval Strategy

User: Ensure the assistant retrieves relevant information accurately from the website content.

Assistant: Implemented a chunking strategy that splits cleaned page text into approximately 700-character chunks with about 120 characters of overlap. This keeps chunks small enough for precise retrieval while preserving nearby context across chunk boundaries.

Assistant: Each chunk stores useful metadata, including:

- Source URL
- Page title
- Section heading when available
- Raw page path
- Chunk ID
- Chunk text

Assistant: Embedded chunks using `all-MiniLM-L6-v2`, normalized the vectors, and stored them in FAISS. At question time, the user query is embedded and matched against the FAISS index to retrieve the most relevant chunks.

---

# Retrieval Validation

User: Test whether retrieval and answer generation are working.

Assistant: Performed a retrieval smoke test using a Mandai-related question. The retriever returned relevant chunks from the local FAISS index, and the Ollama model generated a grounded response from the retrieved evidence.

Assistant: Confirmed that the assistant could answer questions about Mandai attractions using local website content rather than relying on general model knowledge.

---

# Grounded Answer Behavior

User: Make sure the chatbot does not invent answers when it does not have enough information.

Assistant: Added answer-generation instructions requiring the model to use only retrieved Mandai website context. The assistant is expected to answer concisely, cite relevant sources, and clearly state when the retrieved content is insufficient.

Assistant: Added an insufficient-context behavior so the app can refuse unsupported questions instead of guessing. For example, a question about nearby hotels should not be answered unless hotel information exists in the retrieved Mandai website content.

---

# Terminal Weight-Loading Message

User: Why does the terminal say that it is loading weights?

Assistant: Explained that the message comes from the SentenceTransformers embedding model loading locally. It is related to embedding the user question for retrieval and is not caused by Ollama, the Streamlit website, or live website scraping.

Assistant: Reduced unnecessary terminal noise during normal app usage so the Streamlit terminal output stays cleaner.

---

# Streamlit Runtime Issue

User: The terminal shows noisy Streamlit or Transformers messages, including optional module warnings.

Assistant: Identified the issue as Streamlit file-watcher behavior interacting with optional Transformers modules. Added Streamlit configuration to reduce unnecessary file-watching noise during local development.

Assistant: The app remains functionally unchanged, but the development terminal is cleaner and easier to interpret.

---

# User Interface Improvements

User: Improve the chatbot interface so it looks more professional and aligned with the Mandai theme.

Assistant: Refined the Streamlit interface with a Mandai-inspired visual direction using deep greens, warm sand tones, readable source cards, and a cleaner chat-focused layout.

Assistant: Improved usability by adjusting contrast, simplifying visual elements, and keeping the primary focus on the chatbot response and citations.

---

# UI Module Separation

User: Move the UI code into a separate file instead of keeping everything inside `app.py`.

Assistant: Refactored the Streamlit presentation layer into `ui.py`. This separated layout, styling, sidebar rendering, source cards, and retrieved-evidence display from the main application flow.

Assistant: Kept `app.py` focused on orchestration: loading the retriever, handling chat state, calling retrieval, generating answers, logging queries, and passing display data to the UI helpers. This made the project structure cleaner, easier to maintain, and more professional for review.

---

# Separate UI Section for Retrieved Evidence

User: Add a separate UI section so the retrieved source evidence is clearly separated from the chatbot's final answer.

Assistant: Added a dedicated retrieved-evidence area below the generated response. This section separates the assistant's final answer from the raw chunks retrieved by FAISS, making the interface easier to understand for both users and evaluators.

Assistant: Kept the evidence section optional so the main chat response remains clean, while still allowing the retrieval process to be inspected when needed.

---

# Source Evidence Display

User: What is the extra UI section showing chunks of text under the answer?

Assistant: Explained that this section shows the retrieved evidence chunks used by the RAG pipeline before answer generation. It is useful for transparency and debugging because it shows what the assistant found in FAISS before producing the response.

Assistant: Renamed the section to make its purpose clearer and placed it inside an optional expander so it does not distract from the main answer. Long chunks were trimmed, raw heading clutter was reduced, and scraped text artifacts were cleaned where possible.

---

# Source UI Rendering Fix

User: The sources section is showing HTML code instead of a clean UI.

Assistant: Identified that the HTML for the source cards was indented in a way that caused Streamlit Markdown to treat it as a code block. Updated the source-card rendering so the HTML is emitted as flat inline markup.

Assistant: Confirmed that source cards and retrieved evidence now render as intended instead of appearing as raw code.

---

# Sample Question Request

User: Provide sample questions that can be used to test the Mandai chatbot.

Assistant: Suggested sample questions covering the main expected use cases: Mandai parks, Singapore Zoo, Night Safari, Bird Paradise, River Wonders, Rainforest Wild Asia, transport, tickets, visitor information, conservation, and an out-of-scope hotel question.

Assistant: The sample questions were selected to validate both successful grounded answers and safe insufficient-context behavior. This helped confirm that the assistant can answer when relevant Mandai content is retrieved and avoid guessing when the local website context does not contain enough information.

---

# Validation Commands

User: Verify that the project files still compile after the fixes.

Assistant: Ran a Python compile check across the main application files:

```bash
python -m py_compile app.py ui.py ingest_web.py rag.py llm.py config.py utils.py
```

Assistant: Confirmed that the compile check completed successfully.

---

# Final Outcome

Assistant: Completed a functional local-first Website RAG Assistant for Mandai website content. The final application includes controlled website ingestion, cleaned local text storage, chunking with metadata, SentenceTransformer embeddings, FAISS retrieval, Ollama-based grounded generation, citations, optional retrieved-evidence inspection, query logging, and a polished Streamlit chat interface.

Assistant: The project demonstrates a complete retrieval-augmented generation workflow that prioritizes privacy, source grounding, transparent retrieval, and practical local execution.
