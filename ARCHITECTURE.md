# SmartDoc AI - Simplified Architecture Diagram

## 📊 System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                        (Streamlit App)                          │
└────────────────┬────────────────────────────────┬───────────────┘
                 │                                │
                 ▼                                ▼
    ┌────────────────────────┐      ┌────────────────────────┐
    │   📄 Document Screen    │      │    💬 Chat Screen      │
    │  - File uploader        │      │  - Message history     │
    │  - Process button       │      │  - Chat input          │
    │  - Status display       │      │  - Source citations    │
    └────────┬───────────────┘      └─────────┬──────────────┘
             │                                  │
             ▼                                  ▼
    ┌────────────────────────┐      ┌────────────────────────┐
    │ 📋 Document Controller  │      │   💭 Chat Controller   │
    │  - Validate files       │      │  - Process queries     │
    │  - Save uploads         │      │  - Build prompts       │
    │  - Orchestrate flow     │      │  - Handle responses    │
    └────────┬───────────────┘      └─────────┬──────────────┘
             │                                  │
             │                                  │
             ▼                                  ▼
    ┌────────────────────────┐      ┌────────────────────────┐
    │  📑 Document Service    │      │    🤖 LLM Service      │
    │  - Load documents       │      │  (Ollama - qwen2.5:7b) │
    │  - Chunk text           │      │  - Generate answers    │
    │  - Extract metadata     │      │  - Local inference     │
    └────────┬───────────────┘      └────────────────────────┘
             │                                  ▲
             │                                  │
             ▼                                  │
    ┌──────────────────────────────────────────┴──────────────┐
    │           🗄️  Vector Store Service (FAISS)              │
    │  - Store document embeddings                            │
    │  - Similarity search                                    │
    │  - Retrieve relevant context                            │
    └─────────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────────┐
    │    🧠 Embedding Service (HuggingFace Multilingual)      │
    │  - Convert text to 768-dim vectors                      │
    │  - Multilingual support (50+ languages)                 │
    └─────────────────────────────────────────────────────────┘
```

## 🔄 User Journey

### 1️⃣ Document Upload Flow

```
User uploads PDF
       │
       ▼
DocumentScreen validates file
       │
       ▼
DocumentController orchestrates
       │
       ▼
DocumentService loads & chunks
       │
       ▼
Embeddings generated (768-dim vectors)
       │
       ▼
VectorStoreService stores in FAISS
       │
       ▼
Success notification to user
```

### 2️⃣ Question & Answer Flow

```
User asks question
       │
       ▼
ChatScreen captures input
       │
       ▼
ChatController processes query
       │
       ▼
VectorStoreService searches similar chunks (k=3)
       │
       ▼
Relevant context retrieved
       │
       ▼
Prompt built with context + question
       │
       ▼
OllamaLLMService generates answer
       │
       ▼
Answer + sources displayed to user
```

## 🏗️ Layer Architecture

```
┌──────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER (Views)                              │
│  ├─ chat_screen.py        - Chat UI                     │
│  ├─ document_screen.py    - Upload UI                   │
│  ├─ settings_screen.py    - Config UI                   │
│  └─ components.py         - Reusable widgets            │
├──────────────────────────────────────────────────────────┤
│  APPLICATION LAYER (Controllers)                         │
│  ├─ chat_controller.py    - Query orchestration         │
│  └─ document_controller.py - Document management        │
├──────────────────────────────────────────────────────────┤
│  BUSINESS LAYER (Services)                               │
│  ├─ llm_service.py        - AI generation               │
│  ├─ vector_store_service.py - Vector operations         │
│  ├─ document_service.py   - Document processing         │
│  └─ embedding_service.py  - (Integrated in vector)      │
├──────────────────────────────────────────────────────────┤
│  DOMAIN LAYER (Models)                                   │
│  ├─ document_model.py     - Document entities           │
│  ├─ chat_model.py         - Chat entities               │
│  └─ config_model.py       - Configuration               │
├──────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE (Utils)                                  │
│  ├─ logger.py             - Logging                     │
│  ├─ exceptions.py         - Error types                 │
│  └─ constants.py          - Configuration               │
└──────────────────────────────────────────────────────────┘
```

## 🎯 Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **MVC** | Overall | Separation of concerns |
| **Factory** | DocumentLoaderFactory | Create loaders by file type |
| **Singleton** | SessionStateManager | Single source of truth |
| **Dependency Injection** | Controllers | Loose coupling |
| **Repository** | VectorStoreService | Abstract data access |
| **Observer** | Streamlit session_state | State management |

## 📦 Key Components

### Models (Data)
- **Document**: Content + metadata + ID
- **ChatMessage**: Role + content + timestamp
- **ChatHistory**: Message collection with max limit

### Services (Logic)
- **OllamaLLMService**: Local LLM inference
- **FAISSVectorStoreService**: Similarity search
- **DocumentService**: Load, chunk, process

### Controllers (Orchestration)
- **ChatController**: RAG pipeline execution
- **DocumentController**: Upload workflow

### Views (UI)
- **ChatScreen**: Conversation interface
- **DocumentScreen**: File management
- **SettingsScreen**: Configuration

## 🔐 Security & Privacy

```
┌─────────────────────────────────┐
│   USER'S LOCAL MACHINE ONLY     │
│                                 │
│  ┌─────────────────────────┐   │
│  │  Ollama LLM             │   │
│  │  (localhost:11434)      │   │
│  └─────────────────────────┘   │
│                                 │
│  ┌─────────────────────────┐   │
│  │  FAISS Vector DB        │   │
│  │  (In-memory)            │   │
│  └─────────────────────────┘   │
│                                 │
│  ┌─────────────────────────┐   │
│  │  Document Storage       │   │
│  │  (data/uploads/)        │   │
│  └─────────────────────────┘   │
│                                 │
│   ❌ NO EXTERNAL API CALLS      │
│   ❌ NO CLOUD SERVICES          │
│   ✅ 100% LOCAL PROCESSING      │
└─────────────────────────────────┘
```

## 📊 Data Flow Example

```
1. User uploads "research.pdf" (5 pages)
   └─> Saved to: data/uploads/research.pdf

2. DocumentService processes:
   └─> Page 1 → 3 chunks (1000 chars each)
   └─> Page 2 → 3 chunks
   └─> ... 
   └─> Total: 15 chunks created

3. Embedding generation:
   └─> Chunk 1 → [0.23, 0.45, ..., 0.12] (768 dims)
   └─> Chunk 2 → [0.34, 0.21, ..., 0.56] (768 dims)
   └─> ... all 15 chunks embedded

4. FAISS indexing:
   └─> Store all 15 vectors with metadata
   └─> Build similarity search index
   └─> Ready for queries!

5. User asks: "What is the main conclusion?"
   └─> Query embedded → [0.12, 0.67, ..., 0.34]
   └─> FAISS finds 3 most similar chunks
   └─> Chunks sent to Ollama with question
   └─> Ollama generates contextual answer
   └─> User sees answer + citations
```

---

**This simplified architecture ensures:**
✅ Easy to understand
✅ Easy to maintain
✅ Easy to extend
✅ Production-ready
