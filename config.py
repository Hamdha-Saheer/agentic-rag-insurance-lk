# config.py — Updated with domain indexes + web search toggle
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ── API Keys ─────────────────────────────────────────
GROQ_API_KEY   = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# ── Model ────────────────────────────────────────────
LLM_MODEL   = 'llama-3.1-8b-instant'
TEMPERATURE = 0
MAX_TOKENS  = 512

# ── RAG Settings ─────────────────────────────────────
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K_DOCS      = 5  

# ── Web Search Toggle ─────────────────────────────────
# Set False to test system WITHOUT Tavily (for dissertation comparison)
USE_WEB_SEARCH = True

# ── Paths ────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_PDF_DIR   = os.path.join(BASE_DIR, 'data', 'raw_pdfs')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Single index (kept as backup)
VECTORSTORE_PATH = os.path.join(BASE_DIR, 'vectorstore', 'insurance_index')

# Separate domain indexes (supervisor feedback)
VECTORSTORE_PATHS = {
    'motor':   os.path.join(BASE_DIR, 'vectorstore', 'motor_index'),
    'health':  os.path.join(BASE_DIR, 'vectorstore', 'health_index'),
    'life':    os.path.join(BASE_DIR, 'vectorstore', 'life_index'),
    'general': os.path.join(BASE_DIR, 'vectorstore', 'general_index'),
}

# ── Domains ──────────────────────────────────────────
DOMAINS = ['motor', 'life', 'health', 'general']
