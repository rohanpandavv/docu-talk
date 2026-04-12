# DocuTalk

DocuTalk lets you have a conversation with your documents. Upload a PDF or text file, index it, and ask grounded questions against the selected document.

It works using **Retrieval-Augmented Generation (RAG)**: instead of asking an LLM to rely on its training data alone (which can lead to hallucinations), we feed it the exact excerpts from your document that are relevant to your question. The LLM's job is reduced from "know everything" to "read and summarize what's in front of it," which is generally more reliable.

## Architecture

```mermaid
flowchart LR
    subgraph Frontend
        A[Streamlit UI]
    end

    subgraph Backend
        B[FastAPI]
        C[LangChain Services]
        G[(Document Registry)]
    end

    subgraph External APIs
        D[OpenAI Embeddings API]
        E[Anthropic Claude API]
    end

    F[(ChromaDB)]

    A -- "upload file" --> B
    B -- "chunk text" --> C
    C -- "embed chunks" --> D
    D -- "vectors" --> F
    C -- "document metadata" --> G

    A -- "ask question" --> B
    B -- "resolve active/selected document" --> G
    B -- "embed query" --> D
    D -- "query vector" --> F
    F -- "top-k chunks" --> C
    C -- "context + question" --> E
    E -- "answer" --> A
```

### How the pipeline works

**Indexing (upload):**
1. User uploads a PDF/TXT via the Streamlit frontend
2. FastAPI extracts text (using `pypdf` for PDFs)
3. Text is split into 1000-character chunks with 200-character overlap using LangChain's `RecursiveCharacterTextSplitter`
4. Each chunk is embedded via OpenAI's `text-embedding-3-small` and stored in ChromaDB with document-scoped metadata
5. The uploaded document is registered as the active document for future chat requests

**Querying (chat):**
1. User asks a question against either an explicit `document_id` or the active document
2. The question is embedded using the same OpenAI model
3. ChromaDB returns the top 3 most similar chunks for that specific document
4. The chunks are injected as context into a prompt template
5. Claude Haiku 4.5 generates an answer based on the retrieved context
6. The API returns the answer along with lightweight source snippets

## Tech Stack

| Component | Choice | Role |
|-----------|--------|------|
| Frontend | Streamlit | Chat UI + file upload |
| Backend | FastAPI | REST API server |
| LLM | Claude Haiku 4.5 | Answer generation |
| Embeddings | OpenAI `text-embedding-3-small` | Text-to-vector conversion |
| Vector Store | ChromaDB | Similarity search over document chunks |
| Orchestration | LangChain | Chaining retrieval + LLM calls |

## Design Decisions and Tradeoffs

### Why separate embedding and LLM providers?

We use **OpenAI for embeddings** and **Anthropic (Claude) for generation**. Below are the reasons why:

- OpenAI's `text-embedding-3-small` is cheap ($0.02/1M tokens), fast, and well-supported in the LangChain ecosystem
- Claude Haiku 4.5 is used for generation because it offers a strong quality-to-cost ratio for RAG tasks, follows instructions well, and works effectively with provided context.

**Tradeoff:** Two API keys are required, and there is a small latency overhead because the app depends on two providers.

### Why document-scoped retrieval?

Each upload is now assigned a `document_id`, and retrieval is filtered to that document instead of searching one shared global pool.

- Prevents chunk leakage between unrelated uploads
- Makes multi-document iteration safer while keeping the API simple
- Allows the frontend to keep chatting against the last indexed document without re-uploading it

**Tradeoff:** Adds document state management and a small registry layer on top of the vector store.

### Why cloud-based embeddings over local models?

Originally, the project used `sentence-transformers` to run the `all-MiniLM-L6-v2` embedding model locally. I later switched to OpenAI as I did not want to run an embedding model locally.

**Tradeoff:** Adds an external API dependency and per-request cost. For a personal/research tool, the cost is usually very low for small documents.

### Why ChromaDB?

- Zero-config, embedded vector database, no separate server to run
- Persists to disk out of the box (`chroma_db/` directory)
- Good enough for single-user, small-to-medium-sized document collections

**Tradeoff:** Not suitable for production-scale workloads. For larger deployments, Pinecone, Weaviate, or pgvector would be better choices.

### Why `RecursiveCharacterTextSplitter` with 1000/200?

- 1,000-character chunks are small enough to stay specific but large enough to preserve context
- 200-character overlap ensures sentences at chunk boundaries aren't lost
- `RecursiveCharacterTextSplitter` tries to split on paragraph or sentence boundaries before falling back to characters, which helps preserve semantic coherence

**Tradeoff:** Fixed chunk sizes don't adapt to document structure. Semantic chunking or document-aware splitting, such as splitting by section headers, could improve retrieval quality but would add complexity.

### Why FastAPI + Streamlit instead of a single app?

- **Wanted a UI instead of API docs** : I wanted to see how the project would feel in a chat interface instead of only accessing it through the API documentation. That made the project more alive to me. Streamlit was the easiest way to spin up a chat user interface and I was already familiar with the library.
I also did not want to spend a lot of time on other JavaScript libraries for the frontend because my main focus was understanding how RAG works, not building a chat UI.

- **Separation of concerns** : This backend can be reused with any frontend application.

**Tradeoff:** Two processes to run. For a production app, a React/Next.js frontend would give more control over UX.

## Setup

### Prerequisites

- Python 3.10+
- [Anthropic API key](https://console.anthropic.com/)
- [OpenAI API key](https://platform.openai.com/)

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`

```bash
cp .env.example .env
```

Open the .env file 
```bash
nano .env
```
Add your API keys and keep the timeout/logging defaults unless you want to tune them:

```env
LOG_LEVEL=INFO
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
ANTHROPIC_CHAT_MODEL=claude-haiku-4-5-20251001
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVE_K=3
MAX_UPLOAD_SIZE_BYTES=10485760
PROVIDER_MAX_RETRIES=2
OPENAI_TIMEOUT_SECONDS=30
ANTHROPIC_TIMEOUT_SECONDS=30
CHROMA_ANONYMIZED_TELEMETRY=false
```

Start the server:

```bash
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the contents of `.env.example` to `.env`

```bash
cp .env.example .env
```

Open the .env file 
```bash
nano .env
```

Replace the `BACKEND_URL` with your FastAPI's address. The server by default runs on port 8000.

```env
BACKEND_URL=http://localhost:8000
REQUEST_TIMEOUT_SECONDS=60
```

Start the app:

```bash
streamlit run app.py
```

## API Endpoints

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| POST | `/upload` | `multipart/form-data` (file) | Upload a PDF or TXT file for indexing |
| GET | `/documents` | - | List indexed documents and the current active document |
| POST | `/documents/{document_id}/activate` | - | Mark a document as the default target for chat requests |
| DELETE | `/documents/{document_id}` | - | Remove a document and its embeddings |
| POST | `/chat` | `{"question": "...", "document_id": "optional"}` | Ask a question about the active document or an explicit document id |

### Example responses

`POST /upload`

```json
{
  "message": "Document indexed successfully!",
  "document_id": "2f2f1e74-3d8d-4c2e-b7eb-4b986c9f4201",
  "filename": "paper.pdf",
  "chunk_count": 12
}
```

`POST /chat`

```json
{
  "answer": "The paper argues that ...",
  "document_id": "2f2f1e74-3d8d-4c2e-b7eb-4b986c9f4201",
  "sources": [
    {
      "source": "paper.pdf",
      "page": 3,
      "chunk_index": 5,
      "excerpt": "..."
    }
  ]
}
```

## Recent Backend Improvements

- Refactored the backend from a single-file prototype into smaller config, schema, and service modules
- Added a persistent document registry so uploads can be listed, activated, and deleted cleanly
- Scoped retrieval to a single document to avoid mixing chunks from different uploads
- Added stronger upload validation, clearer service errors, and source snippets in chat responses
- Added request/provider timeouts and better logging so indexing failures surface instead of hanging silently

## Project Structure

```
DocuTalk/
├── backend/
│   ├── main.py              # FastAPI entrypoint and route wiring
│   ├── config.py            # Environment-backed backend settings
│   ├── logging_config.py    # Logging setup
│   ├── schemas.py           # Request/response models
│   ├── services/
│   │   ├── document_registry.py  # Persistent metadata for indexed docs
│   │   ├── errors.py             # Service-layer exceptions
│   │   └── rag.py                # Ingestion, retrieval, and answer generation
│   ├── tests/
│   │   ├── test_api.py           # API-level tests with service overrides
│   │   └── test_registry.py      # Registry behavior tests
│   ├── requirements.txt
│   └── .env                 # API keys, chunking, logging, and timeout settings
├── frontend/
│   ├── app.py               # Streamlit chat UI with upload/chat timeout handling
│   ├── requirements.txt
│   └── .env                 # BACKEND_URL, REQUEST_TIMEOUT_SECONDS
├── LICENSE
└── README.md
```

## Planned Improvements
- Prepare a strategy to evaluate the performance of the RAG system
- Add score thresholds / better "answer not found in the document" handling
- Support OCR or another fallback path for scanned PDFs
- Improve multi-document workflows beyond the current active-document model


## License

MIT
