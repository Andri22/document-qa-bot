# Document Q&A Bot

An AI-powered chatbot that answers questions from your PDF documents.
Built with LangChain, FastAPI, and Streamlit. Features RAG retrieval,
session-based memory, and source citations.

## Features

- **Upload PDFs** — ingest documents into a vector database
- **Ask questions** — get answers grounded in your documents
- **Session memory** — follow-up questions remember past conversation
- **Source citations** — every answer shows which document it came from
- **Filter by document** — ask questions scoped to a specific file
- **Persistent history** — chat history survives server restarts (SQLite)

## Tech Stack

| Layer        | Technology                        |
|--------------|-----------------------------------|
| LLM          | OpenAI GPT-4o-mini                |
| Embeddings   | OpenAI text-embedding-3-small     |
| Vector Store | ChromaDB                          |
| RAG          | LangChain                         |
| Backend      | FastAPI                           |
| Frontend     | Streamlit                         |
| Memory       | SQLChatMessageHistory (SQLite)    |

## Prerequisites

- Python 3.11+
- OpenAI API key

## How to Run

### 1. Clone and setup

```bash
git clone https://github.com/Andri22/document-qa-bot.git
cd document-qa-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in your `.env`:

| Variable           | Description               | Default               |
|--------------------|---------------------------|-----------------------|
| `OPENAI_API_KEY`   | Your OpenAI API key       | —                     |
| `CHROMA_PATH`      | Vector store directory    | `./vectorstore`       |
| `COLLECTION_NAME`  | ChromaDB collection name  | `example_collection`  |
| `EMBEDDING_MODEL`  | Embedding model           | `text-embedding-3-small` |
| `LLM_MODEL`        | Chat model                | `gpt-4o-mini`         |

### 3. Run locally

Open two terminals:

```bash
# Terminal 1 — FastAPI backend
PYTHONPATH=stage4_api uvicorn stage4_api.main:app --reload --port 8000

# Terminal 2 — Streamlit frontend
streamlit run stage3_ui/app.py
```

Open `http://localhost:8501` in your browser.

### 4. Run with Docker

```bash
docker compose up --build
```

- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

## API Endpoints

| Method | Path           | Description                        |
|--------|----------------|------------------------------------|
| GET    | `/health`      | Health check                       |
| POST   | `/upload`      | Upload PDF(s) for ingestion        |
| POST   | `/chat`        | Ask a question (with session_id)   |
| GET    | `/documents`   | List all ingested documents        |

## Project Structure

```
├── stage3_ui/          # Streamlit frontend
│   └── app.py
├── stage4_api/         # FastAPI backend
│   ├── main.py
│   └── app/
│       ├── routes/
│       │   ├── chat.py       # Chat endpoint + memory
│       │   ├── upload.py     # Upload endpoint + file validation
│       │   └── documents.py  # List ingested documents
│       └── core/
│           ├── rag.py        # RAG engine
│           ├── ingest.py     # PDF ingestion pipeline
│           └── helper.py     # Config helpers
├── vectorstore/        # ChromaDB persistence (gitignored)
├── data/               # Sample PDFs
├── scripts/            # Utility scripts
├── docker-compose.yml
└── requirements.txt
```

## Project Stages

- [x] Stage 1 — LangChain Foundations
- [x] Stage 2 — RAG Core
- [x] Stage 3 — Streamlit UI
- [x] Stage 4 — FastAPI Backend
- [x] Stage 5 — Docker + Deployment
- [x] Stage 6 — CI/CD
