# Title: Agentic RAG with Groq

A command-line Retrieval-Augmented Generation (RAG) system built using Python, Groq's LLaMA 3.1 model, TF-IDF retrieval, and an agentic decision layer. The system loads documents, chunks and indexes them, and answers user queries by retrieving relevant context and generating accurate responses.

---

## Abstract

This project implements a document-based question answering system using the Retrieval-Augmented Generation (RAG) architecture. It is built with Python and powered by Groq's LLaMA 3.1 8B model.

The system loads PDF and TXT documents from a local folder, splits them into overlapping chunks, and indexes them using TF-IDF vectorization. When a user submits a query, the system computes cosine similarity between the query vector and all chunk vectors to retrieve the most relevant context. An agentic decision layer routes the query — either to the document retrieval pipeline or to a calculator tool based on the nature of the query. A pronoun resolution module resolves contextual references using chat history before retrieval. The retrieved context and query are then passed to Groq's LLaMA 3.1 model, which generates a structured response with an answer and source citation.

The system preserves conversational continuity through a chat history buffer, enabling multi-turn interactions. API credentials are managed securely using environment variables.

This project demonstrates core RAG concepts including document chunking, TF-IDF retrieval, cosine similarity ranking, agentic routing, pronoun resolution, and LLM API integration in a modular Python pipeline.

---

## Key Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Document Chunking with Overlap
- TF-IDF Vectorization
- Cosine Similarity Ranking
- Agentic Decision Making
- Pronoun Resolution and Query Rewriting
- LLM API Integration
- Conversational Context Preservation
- Multi-document Source Attribution
- Environment Variable Management

---

## System Workflow

```text
┌──────────────────────┐
│     User Query       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Pronoun Resolver   │
│  (resolve_query)     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Agent Decision     │
│  (decide_action)     │
└────┬─────────────────┘
     │
     ├──── calculation ──→ ┌──────────────────┐
     │                     │  Calculator Tool  │
     │                     └──────────────────┘
     │
     └── document_search → ┌──────────────────────┐
                            │   TF-IDF Retrieval    │
                            │  (retrieve_chunks)    │
                            └──────────┬───────────┘
                                       ↓
                            ┌──────────────────────┐
                            │  Groq API (LLaMA 3.1) │
                            └──────────┬───────────┘
                                       ↓
                            ┌──────────────────────┐
                            │   Answer + Source     │
                            └──────────────────────┘
```

---

## Project Structure

```text
agentic-rag-groq/
│
├── main.py
├── requirements.txt
├── complete_architecture.txt
│
├── documents/              ← Add your PDF/TXT files here (not tracked by git)
│
├── LICENSES/
│   └── markitdown-MIT.txt
│
├── .env
├── .gitignore
├── LICENSE
└── README.md
```

### File Description

- **main.py** — Core pipeline: document loading, chunking, TF-IDF indexing, retrieval, agentic routing, pronoun resolution, and Groq LLM integration.
- **requirements.txt** — Python dependencies.
- **complete_architecture.txt** — Detailed system architecture notes.
- **documents/** — Folder for user-provided PDF and TXT files. Not tracked by git.
- **LICENSES/markitdown-MIT.txt** — Third-party license for Microsoft MarkItDown.
- **.env** — Stores the Groq API key securely.
- **LICENSE** — MIT License for this project.

---

## Technologies Used

### Backend
- Python
- Groq API
- Python Dotenv
- pypdf
- scikit-learn (TF-IDF, Cosine Similarity)

### AI Model
- LLaMA 3.1 8B Instant (via Groq)

---

## Features

- Multi-document loading (PDF and TXT)
- Overlapping document chunking for context preservation
- TF-IDF vectorization and cosine similarity retrieval
- Agentic routing — document search or calculator based on query type
- Pronoun resolution using chat history
- Best chunk selection per document to avoid redundancy
- Relevance threshold filtering
- Source attribution in every response
- Multi-turn conversational context via chat history buffer
- Secure API key management using environment variables

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Blackblitz777/agentic-rag-groq.git
cd agentic-rag-groq
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / WSL:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY="your_api_key_here"
```

### Add Documents

Create a `documents/` folder and add your PDF or TXT files:

```bash
mkdir documents
cp /path/to/yourfile.pdf documents/
```

### Run the Application

```bash
python main.py
```

---

## Usage

1. Place your PDF or TXT files in the `documents/` folder.
2. Run `python main.py`.
3. The system loads, chunks, and indexes all documents.
4. Type your question at the prompt.
5. The agent decides whether to search documents or calculate.
6. Relevant chunks are retrieved and passed to LLaMA 3.1.
7. The answer and source filename are displayed.
8. Type `quit` or `exit` to stop.

---

## Learning Outcomes

This project helped demonstrate:

- RAG Pipeline Architecture
- Document Chunking and Overlap Strategy
- TF-IDF Vectorization and Cosine Similarity
- Agentic Decision Making
- Pronoun Resolution and Query Rewriting
- LLM API Integration with Groq
- Multi-document Retrieval and Source Attribution
- Conversational Memory Handling
- Python Modular Pipeline Design

---

## Future Improvements

- Replace TF-IDF with semantic embeddings (FAISS + sentence-transformers)
- Flask web interface for browser-based interaction
- MarkItDown integration for multi-format document support
- Persistent chat history using a database
- Streaming responses
- Multi-model selection
- Reranking retrieved chunks before LLM call
- Deployment using Docker and cloud platforms

---

## Third-Party Licenses

- [MarkItDown](https://github.com/microsoft/markitdown) by Microsoft — MIT License. See `LICENSES/markitdown-MIT.txt`.

---

## Author

Developed as part of Generative AI and Large Language Model learning and experimentation.
