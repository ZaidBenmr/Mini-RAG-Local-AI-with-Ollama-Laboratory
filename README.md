# FSTT Local AI Assistant

A fully local assistant: Flask web UI → Ollama → local LLM, with a keyword (TF-IDF) retriever over `documents/`,
conversation history, structured (JSON) answers and response-time display.

## Setup
```bash
ollama pull qwen3.5:4b
ollama pull gemma3:4b
pip install -r source/requirements.txt
```

## Run
```bash
cd source
python app.py            # http://127.0.0.1:5001
python chatbot.py        # console chatbot
python benchmark.py qwen3.5:4b gemma3:4b
```
Environment variables: `MODEL` (default `qwen3.5:4b`), `OLLAMA_URL`, `PORT`.

## Contents
- `source/app.py` — Flask web app (RAG + history + structured output + timing)
- `source/chatbot.py` — console chatbot with memory
- `source/rag.py` — local TF-IDF retriever and prompt builder
- `source/benchmark.py` — speed / memory benchmark of local models
- `documents/` — the course knowledge base
- `screenshots/` — screenshots of each step
- `report.pdf` — lab report
