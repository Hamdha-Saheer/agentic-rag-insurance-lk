# Agentic RAG for Sri Lanka Insurance Adjudication

A multi-agent Retrieval-Augmented Generation framework for insurance guidance in Sri Lanka. Built with LangGraph, FAISS, and Llama-3.1-8B-Instant via Groq. Covers Motor, Health (Agrahara), Life, and General (Suraksha) domains.


---

## Architecture

- Supervisor Agent routes queries to 1-2 domain agents via parallel fork-join (LangGraph)
- Four isolated FAISS vector indexes prevent cross-domain contamination
- Persona-aware routing tailors results by age, occupation, and vehicle type
- Configurable web search toggle (Tavily API) for real-time regulatory data
- Zero-temperature Groq inference for deterministic, grounded responses

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Llama-3.1-8B-Instant via Groq |
| Embeddings | all-MiniLM-L6-v2 (384-dim) |
| Vector DB | FAISS (4 domain indexes) |
| Agents | LangGraph 0.2+ |
| Web Search | Tavily API |
| OCR | Tesseract 5.5.0 |
| Frontend | Streamlit |
| Evaluation | RAGAS + DeepEval |

Total project cost: LKR 0.00

## Knowledge Base

63 official IRCSL and NITF PDF documents split across four domains: Motor (19 docs, 450 chunks), Health (22 docs, 580 chunks), Life (8 docs, 220 chunks), General (14 docs, 300 chunks). Chunking: size=500, overlap=50. 16 scanned PDFs processed with Tesseract OCR at 200 DPI.

Note: Source PDFs and FAISS indexes are not included in this repository. Place PDFs in data/raw_pdfs/ and run the ingestion pipeline to generate indexes.

## Setup

pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
streamlit run app.py

Get free API keys from Groq (https://console.groq.com) and Tavily (https://tavily.com).

## Evaluation Results

| Metric | RAGAS | DeepEval |
|--------|:-----:|:--------:|
| Faithfulness | 0.8083 | 0.8063 |
| Answer Relevancy | 0.8337 | 0.8286 |
| Context Precision | 0.8023 | N/A |

Dual-mode testing showed Health performs better with local retrieval (0.674 vs 0.620) while Regulatory benefits from web augmentation (0.150 vs 0.951).

## Project Structure

config.py                  # Configuration
app.py                     # Streamlit UI
agents/
  multi_agent.py            # LangGraph supervisor + domain agents
  rag_pipeline.py           # FAISS retrieval + LLM generation
  domain_router.py          # Query routing + persona boost
evaluation/
  ground_truth.json         # 35 Q&A pairs
  evaluator.py              # RAGAS evaluation script
  results/                  # CSV result files
