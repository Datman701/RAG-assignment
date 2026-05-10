# Document Q&A: RAG-Powered Web Application

A production-ready document question-answering web app powered by LangChain, Gemini, Hugging Face embeddings, and ChromaDB. Upload a PDF, TXT, or CSV file, ask natural language questions, and receive answers grounded exclusively in your document.

---

## 🎯 Features

- **Document Upload**: Upload PDF, TXT, or CSV files (up to 10MB)
- **RAG-Powered Q&A**: Retrieval-Augmented Generation using LangChain
- **Source Attribution**: Answers include retrieved source chunks with metadata
- **Syntax Highlighting**: Code blocks in answers are highlighted
- **Session Persistence**: Chat history persisted in browser localStorage
- **Document Management**: Upload, view, and close documents with ease
- **Ephemeral Storage**: Vector store resets on server restart (no persistence needed)
- **Zero Local Setup**: Deployable to Render free tier with render.yaml

---

## 📁 Project Structure

```
RAG-implementation/
├── main.py                 # FastAPI server - handles /upload, /ask, /sessions endpoints
├── rag_pipeline.py         # RAG pipeline - document processing, embedding, retrieval, QA
├── document_loader.py      # Document loading utilities - PDF, TXT, CSV parsing
├── vectorstore.py          # Vector store creation and management with ChromaDB
├── requirements.txt        # Python dependencies
├── render.yaml             # Render.com deployment configuration
├── .env.example            # Environment variables template (copy to .env)
├── static/
│   └── index.html          # Single-file frontend (HTML/CSS/JS SPA)
├── tests/
│   ├── __init__.py
│   └── test_rag_pipeline.py # Unit tests for RAG pipeline
├── README.md               # This file
└── DEPLOYMENT.md           # Detailed deployment guide
```

### Key Features of the UI

- **Upload Zone**: Drag-and-drop or click to upload documents (PDF, TXT, CSV)
- **Active Document Panel**: Shows document name, type, chunk count, and load time
- **Close Document Button**: Resets the session and allows uploading a new document
- **Chat Interface**: Interactive Q&A with real-time message streaming
- **Source Attribution**: Expandable source references with metadata (page, chunk index)
- **Session Persistence**: Automatically restores previous sessions via browser localStorage

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (HTML/CSS/JS)              │
│  • Single-file SPA (no framework)                       │
│  • localStorage for session persistence                 │
│  • SSE for streaming responses                          │
│  • Client-side validation (file size, type)             │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/JSON
┌────────────────────▼────────────────────────────────────┐
│                FastAPI Backend                          │
│  • POST /upload: Ingest document → generate session_id  │
│  • POST /ask: Generate answer via retrieval chain       │
│  • GET /sessions/<id>: Validate session persistence     │
│  • GET /health: Health check endpoint                   │
│  • In-memory session storage (dict)                     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  RAG Pipeline                           │
│                                                         │
│  1. Load Document (PDF/TXT)                            │
│     └─> PyPDFLoader + TextLoader                       │
│                                                         │
│  2. Chunk Document                                      │
│     └─> RecursiveCharacterTextSplitter                 │
│        • chunk_size: 1000 tokens                        │
│        • chunk_overlap: 200 tokens                      │
│        • separators: ["\n\n", "\n", ". ", " ", ""]     │
│                                                         │
│  3. Embed Chunks                                        │
│     └─> sentence-transformers/all-MiniLM-L6-v2         │
│                                                         │
│  4. Create Vector Store                                │
│     └─> ChromaDB (in-memory, no persistence)           │
│                                                         │
│  5. Retrieve Relevant Chunks                           │
│     └─> Similarity search (k=4 chunks)                 │
│                                                         │
│  6. Generate Answer                                     │
│     └─> Gemini 2.5 Flash + fallback model chain        │
│        • System prompt ensures document-grounded ans    │
│        • Temperature: 0 (deterministic)                │
└──────────────────────────────────────────────────────────┘
```

---

## � Supported File Formats

### PDF
- **Format**: Standard PDF files (text-based, not scanned images)
- **Processing**: Text extracted page-by-page
- **Metadata**: Page numbers included for source attribution
- **Requirement**: Must be text-based PDF (OCR-scanned PDFs are not supported)

### TXT
- **Format**: Plain UTF-8 text files
- **Processing**: Entire file treated as a single document, then chunked semantically
- **Encoding**: Must be UTF-8 encoded
- **Best for**: Articles, transcripts, long-form content

### CSV
- **Format**: Comma/Tab/Semicolon-separated values with or without headers
- **Processing**: Each row becomes a separate document (key-value pairs)
- **Requirements**:
  - **Must contain at least one data row** (header-only files not supported)
  - **Cannot be empty**
  - Cannot contain only blank rows
- **Dialect Detection**: Automatically detects delimiter (comma, tab, semicolon)
- **Metadata**: Row numbers included for traceability
- **Example**:
  ```csv
  Name,Age,Department
  Alice,30,Engineering
  Bob,25,Marketing
  ```

---

## �📊 Chunking Strategy

The chunking strategy balances context preservation with semantic relevance:

### Why chunk at all?
- **Context Window Limits**: GPT-4 has finite token limits. Chunking allows us to fit relevant portions of large documents into prompts.
- **Retrieval Precision**: Smaller chunks are easier to match semantically to user queries via embedding similarity.
- **Efficiency**: We retrieve only the most relevant pieces, not the entire document.

### Our Configuration

```python
chunk_size = 1000        # Tokens per chunk (≈ 250-300 words)
chunk_overlap = 200      # Tokens of overlap between chunks
separators = [
    "\n\n",             # Paragraph breaks (highest priority)
    "\n",               # Line breaks
    ". ",               # Sentence boundaries
    " ",                # Word boundaries
    ""                  # Character boundaries (fallback)
]
```

### Why These Values?

- **1000 tokens**: Large enough to maintain context (e.g., a paragraph) but small enough to prevent token overflow. With k=4 retrieved chunks, we're ~3000-4000 tokens of context, leaving room for the user's question and system prompt in GPT-4's ~4096 context window.

- **200-token overlap**: Prevents information loss at chunk boundaries. Important concepts may span chunk boundaries; overlap ensures they're captured.

- **Hierarchical separators**: Respects document structure. We split on semantic boundaries (paragraphs) first, then fall back to lower-level boundaries if chunks are too large.

### Metadata Enrichment

Each chunk includes metadata for source attribution:

**PDF chunks:**
```python
{
    "source": "filename.pdf",      # Original file name
    "page": 3,                     # Page number (1-based)
    "chunk_index": 0,              # Sequential chunk index
    "type": "pdf"
}
```

**TXT chunks:**
```python
{
    "source": "filename.txt",      # Original file name
    "chunk_index": 0,              # Sequential chunk index
    "type": "txt"
}
```

**CSV chunks:**
```python
{
    "source": "filename.csv",      # Original file name
    "row": 2,                      # Row number (1-based, excluding header)
    "chunk_index": 0,              # Sequential chunk index
    "type": "csv"
}
```

This metadata is returned in sources so users see exactly where answers came from.

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.10+
- Gemini API key (free tier works, cost is minimal)

### Setup

1. **Clone the repo** (or create locally):
```bash
cd RAG-implementation
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure API key**:
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
nano .env
```

5. **Run the server**:
```bash
python main.py
```

6. **Open in browser**:
```
http://localhost:8000
```

---

## 📡 API Endpoints

### `POST /upload`

Upload a document and create a RAG session.

**Request:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "session_id": "a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6",
  "file_name": "document.pdf",
  "chunk_count": 42,
  "message": "Document processed successfully. Created 42 chunks."
}
```

### `POST /ask`

Ask a question about the uploaded document.

**Request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6",
    "question": "What is the main topic of this document?"
  }'
```

**Response:**
```json
{
  "answer": "The main topic of this document is...",
  "sources": [
    {
      "text": "...",
      "metadata": {
        "source": "document.pdf",
        "page": 1,
        "chunk_index": 5
      }
    }
  ]
}
```

### `GET /sessions/{session_id}`

Validate and retrieve session metadata (used for localStorage persistence).

**Request:**
```bash
curl http://localhost:8000/sessions/a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6
```

**Response:**
```json
{
  "session_id": "a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6",
  "file_name": "document.pdf",
  "chunk_count": 42,
  "created_at": "2025-05-07T14:30:00",
  "file_type": "pdf"
}
```

### `GET /health`

Health check endpoint (useful for monitoring).

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-05-07T14:35:00"
}
```

---

## 🌐 Deployment (Render.com)

### Free Tier Deployment

1. **Fork/Push to GitHub** (required for Render)
2. **Connect to Render**:
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
3. **Configure Environment**:
   - In Render dashboard, set environment variable:
    - Key: `GEMINI_API_KEY`
    - Value: Your Gemini API key (from Google AI Studio)
4. **Deploy**:
   - Render auto-deploys on git push
   - Uses `render.yaml` for build/start commands
5. **Access**:
   - Your app is live at `https://your-app-name.onrender.com`

### Expected Startup Time
- ~2-3 minutes on free tier (includes Python install + dependency installation)
- Subsequent deployments are faster

### Cost Estimate
- **Server**: Free tier (512MB RAM, auto-spins down after 15 min inactivity)
- **Gemini API**: usage varies by model and request size

---

## ⚙️ Configuration

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your-gemini-api-key-here

# Optional
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional
LOG_LEVEL=INFO                    # Logging verbosity
MAX_UPLOAD_SIZE=10485760          # Max file size in bytes (10MB default)
```

### Customization

Edit in the source files:

**Chunking strategy** (`rag_pipeline.py`):
```python
chunk_size = 1000          # Increase for larger docs, decrease for fragmented content
chunk_overlap = 200        # Increase to preserve more cross-chunk context
separators = [...]         # Adjust for different document types
```

**Retrieval parameters** (`rag_pipeline.py`):
```python
search_kwargs={"k": 4}     # Number of chunks retrieved (increase for more context, slower)
```

**LLM parameters** (`rag_pipeline.py`):
```python
ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
# temperature=0: Deterministic answers (good for factual Q&A)
# temperature=0.7: More creative answers
```

---

## 🎓 How It Works: Example

### Scenario
User uploads a 50-page PDF about climate change and asks: *"What are the main causes?"*

### Pipeline Execution

1. **Load** (main.py `/upload`)
   - Read PDF with PyPDF
   - Extract text from all 50 pages
   - Create Document objects with page metadata

2. **Chunk** (rag_pipeline.py `create_chunks()`)
   - PDF has ~100,000 tokens total
   - Split into ~100 chunks of 1000 tokens with 200-token overlap
   - Add metadata: source="climate_change.pdf", page=3, chunk_index=15, etc.

3. **Embed** (rag_pipeline.py `create_vectorstore()`)
  - Convert each chunk to vector via sentence-transformers/all-MiniLM-L6-v2
   - Store vectors in ChromaDB (in-memory)

4. **Retrieve** (rag_pipeline.py `answer_question()`)
   - Convert user question to vector
   - Search for 4 most similar chunks
   - Found: chunks on CO2 emissions, methane, deforestation, industrial activity

5. **Generate** (rag_pipeline.py + LangChain)
   - Build prompt:
     ```
     System: You are a document assistant. Use ONLY the provided context...
     Context: [4 retrieved chunks about causes]
     Question: What are the main causes?
     ```
  - Send to Gemini 2.5 Flash
   - Model generates: *"Based on the document, the main causes are: CO2 emissions from manufacturing, methane from agriculture, deforestation reducing CO2 absorption, and..."*

6. **Return** (frontend)
   - Display answer in chat
   - Show expandable sources (each with page number and chunk text)

---

## 🐛 Troubleshooting

### "Gemini API Error"
- **Cause**: Invalid API key or API key not set
- **Solution**: Check `.env` file and ensure `GEMINI_API_KEY` is correct

### "Could not extract text from PDF"
- **Cause**: Scanned PDF (images, not text)
- **Solution**: Use OCR to convert PDF to text first, or upload a text-based PDF

### "Session not found" (404)
- **Cause**: Session expired (server restarted) or invalid session ID
- **Solution**: Re-upload document to create new session

### "File too large"
- **Cause**: Uploaded file > 10MB
- **Solution**: Split file or reduce size

### Slow response time
- **Cause**: Large document (many chunks) or network latency
- **Solution**: On Render free tier, first request after inactivity takes ~30s to start server

---

## 📋 Limitations

1. **No Persistence**: Vector store is in-memory; disappears when server restarts
2. **Single File per Session**: One document per session (could be extended)
3. **10MB File Limit**: Configurable but keeps cold starts fast on free tier
4. **Token Limit**: GPT-4's context window means very long documents may not be fully indexed (users get best-effort retrieval)
5. **No Authentication**: Anyone can access the app (deploy internally or add auth if needed)
6. **Free Tier Spin-Down**: Render free tier auto-stops after 15 min inactivity (costs $5/month to keep running)

---

## 🚀 Future Enhancements

- [ ] Multi-file support (upload multiple documents per session)
- [ ] Streaming responses (chunk answer generation in real-time)
- [ ] PostgreSQL persistence (save vectorstore to database)
- [ ] User authentication & rate limiting
- [ ] Web scraping source (upload URLs, not just files)
- [ ] Advanced retrieval (hybrid search, reranking)
- [ ] Chat memory (context from previous messages)
- [ ] Parallel question answering (ask multiple Qs at once)
- [ ] Analytics dashboard (track usage, queries)

---

## 📚 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | FastAPI | REST API, file serving |
| RAG Framework | LangChain | Orchestrate retrieval + generation |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Semantic search |
| Vector Store | ChromaDB | In-memory vector database |
| LLM | Gemini 2.5 Flash | Answer generation |
| Frontend | Vanilla HTML/CSS/JS | Single-page app |
| Deployment | Render.com | Free tier hosting |

---

## 📄 File Structure

```
notebooklm-rag/
├── main.py                           # FastAPI app + endpoints
├── rag_pipeline.py                   # LangChain RAG pipeline
├── static/
│   └── index.html                    # Single-file frontend
├── requirements.txt                  # Dependencies
├── render.yaml                       # Deployment config
├── .env.example                      # API key template
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

---


## 🎉 Next Steps

1. **Run locally**: Follow [Quick Start](#quick-start-local-development)
2. **Test upload**: Upload a sample PDF and ask a question
3. **Deploy**: Push to GitHub and deploy on Render
4. **Customize**: Adjust chunking, LLM parameters, styling as needed

