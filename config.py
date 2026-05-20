"""Central configuration for the Mandai Website RAG Assistant."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "Mandai Website RAG Assistant"
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_PAGES_DIR = DATA_DIR / "raw_pages"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
LOGS_DIR = BASE_DIR / "logs"

FAISS_INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"
CHAT_LOG_PATH = LOGS_DIR / "chat_log.jsonl"
INGEST_LOG_PATH = LOGS_DIR / "ingest_failures.jsonl"

MANDAI_URLS = [
    "https://www.mandai.com/en.html",
    "https://www.mandai.com/en/singapore-zoo.html",
    "https://www.mandai.com/en/singapore-zoo/visit.html",
    "https://www.mandai.com/en/night-safari.html",
    "https://www.mandai.com/en/night-safari/visit.html",
    "https://www.mandai.com/en/river-wonders.html",
    "https://www.mandai.com/en/bird-paradise.html",
    "https://www.mandai.com/en/bird-paradise/animals-and-zones.html",
    "https://www.mandai.com/en/rainforest-wild-asia.html",
    "https://www.mandai.com/en/rainforest-wild-asia/animals-and-zones.html",
    "https://www.mandai.com/en/plan-your-visit.html",
    "https://www.mandai.com/en/plan-your-visit/getting-to-and-around.html",
    "https://www.mandai.com/en/plan-your-visit/getting-to-and-around/getting-to-our-mandai-wildlife-parks.html",
    "https://www.mandai.com/en/plan-your-visit/know-before-you-go/singapore-zoo.html",
    "https://www.mandai.com/en/plan-your-visit/know-before-you-go/night-safari.html",
    "https://www.mandai.com/en/plan-your-visit/itinerary/singapore-zoo-itinerary.html",
    "https://www.mandai.com/en/conservation.html",
    "https://www.mandai.com/en/tickets-and-passes.html",
    "https://www.mandai.com/en/tickets-and-passes/single-attractions.html",
    "https://www.mandai.com/en/tickets-and-passes/single-attractions/bird-paradise.html",
    "https://www.mandai.com/en/tickets-and-passes/single-attractions/river-wonders.html",
    "https://www.mandai.com/en/memberships/wildpass.html",
    "https://www.mandai.com/en/bird-paradise/things-to-do/dine.html",
    "https://www.mandai.com/en/night-safari/things-to-do/shows/creatures-of-the-night-show.html",
    "https://www.mandai.com/en/river-wonders/things-to-do/activities/once-upon-a-river.html",
    "https://www.mandai.com/en/see-and-do/rides/river-wonders/amazon-river-quest.html",
    "https://www.mandai.com/en/rainforest-wild-asia/animals-and-zones/the-canopy.html",
    "https://www.mandai.com/en/rainforest-wild-asia/animals-and-zones/the-karsts.html",
]

USER_AGENT = os.getenv(
    "USER_AGENT",
    "MandaiWebsiteRAGAssistant/1.0 (+local-first educational RAG project)",
)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "1.0"))

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.18"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
REQUESTED_DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", REQUESTED_DEFAULT_MODEL)
FALLBACK_OLLAMA_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "phi3:latest")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "90"))
AUTO_SELECT_INSTALLED_MODEL = os.getenv("AUTO_SELECT_INSTALLED_MODEL", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_model_preference = os.getenv("OLLAMA_MODEL_PREFERENCE", "")
if _model_preference:
    OLLAMA_MODEL_PREFERENCE = [
        model.strip() for model in _model_preference.split(",") if model.strip()
    ]
else:
    OLLAMA_MODEL_PREFERENCE = [
        DEFAULT_OLLAMA_MODEL,
        FALLBACK_OLLAMA_MODEL,
        "llama3.2:1b",
        "gemma3:1b",
    ]
