# Running on Another System

This guide explains how to run the Mandai Website RAG Assistant on a different computer after cloning the GitHub repository.

## 1. Prerequisites

Install these first:

- Git
- Python 3.10, 3.11, or 3.12
- Ollama

Recommended Python version: Python 3.11.

The first setup requires internet access to install Python packages, download the Ollama model, download the SentenceTransformers embedding model if it is not already cached, and scrape the configured Mandai website pages.

## 2. Clone the Repository

```bash
git clone https://github.com/govanzz/RAG-Chatbot-Agent.git
cd RAG-Chatbot-Agent
```

## 3. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install Python Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Install and Start Ollama

Install Ollama from:

```text
https://ollama.com
```

Then pull the default local model:

```bash
ollama pull llama3.1:8b
```

Start Ollama:

```bash
ollama serve
```

If Ollama is already running in the background, this command may say the port is already in use. That is usually fine.

## 6. Optional Environment File

The app works with the default settings, but you can copy the example environment file if you want to override model or runtime settings.

On Windows:

```powershell
copy .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

To explicitly use the default model, set:

```text
OLLAMA_MODEL=llama3.1:8b
```

## 7. Build the Local RAG Index

Run ingestion before starting the chatbot:

```bash
python ingest_web.py
```

This command:

- Scrapes the configured official Mandai URLs.
- Saves cleaned page text under `data/raw_pages/`.
- Creates text chunks with metadata.
- Generates embeddings with SentenceTransformers.
- Builds the FAISS index under `vectorstore/`.

The generated raw pages, vectorstore files, and logs are intentionally not committed to GitHub. Each system should generate them locally.

## 8. Run the Streamlit App

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in a browser and ask Mandai-related questions.

## 9. Quick Test Questions

Try:

- What parks are part of Mandai Wildlife Reserve?
- What is Bird Paradise?
- How do I get to Mandai Wildlife Reserve?
- What ticket or pass options are mentioned?
- What conservation work does Mandai support?

The assistant should answer from retrieved Mandai website content and show source citations.

## 10. Common Issues

### `streamlit` command not found

Use:

```bash
python -m streamlit run app.py
```

### Vectorstore missing

Run:

```bash
python ingest_web.py
```

### Ollama is not running

Start Ollama:

```bash
ollama serve
```

### Model missing

Pull the default model:

```bash
ollama pull llama3.1:8b
```

### FAISS installation problem

Use Python 3.10, 3.11, or 3.12, recreate the virtual environment, and reinstall dependencies:

```bash
pip install -r requirements.txt
```

### First answer is slow

The first run may load the embedding model and local LLM into memory. Later requests are usually faster.

## 11. Refreshing Website Content

If the Mandai website changes, rerun:

```bash
python ingest_web.py
```

This rebuilds the local raw page files, metadata, embeddings, and FAISS index.
