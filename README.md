# Monopoly Rule Assistant

A GenAI-powered assistant designed to navigate the complexities of Monopoly rules across various editions. Leveraging **Retrieval-Augmented Generation (RAG)** and **Agentic Workflows**, this assistant provides precise answers by analyzing official rulebooks and video tutorials.

---

##  Key Features

- ** Agentic RAG (LangGraph)**: An advanced reasoning workflow that includes query rewriting, document routing, and iterative answer verification.
- ** Hybrid Search**: Combines Dense Vector Search (ChromaDB) with Sparse Keyword Search (BM25) for superior retrieval accuracy.
- ** Multi-modal Ingestion**: Seamlessly processes PDF rulebooks and transcribes video tutorials (using Faster-Whisper).
- ** Advanced Chunking**: Supports Multiple strategies—Fixed-size, Recursive, and Semantic chunking—to optimize context relevance.
- ** Fully Local**: Powered by **Ollama**, ensuring data privacy and offline capability with models like `llama3.1` and `nomic-embed-text`.
- ** Performance Benchmarking**: Built-in evaluation scripts to compare chunking strategies and RAG accuracy.

---

##  Architecture

### Ingestion Pipeline
1. **Load**: PDFs are converted to Markdown; Videos are transcribed with timestamps.
2. **Chunk**: Documents are split using the selected strategy (Standard, Recursive, or Semantic).
3. **Embed & Index**: Text chunks are embedded using `nomic-embed-text` and stored in a hybrid ChromaDB/BM25 index.

### RAG Workflows
- **Simple RAG**: A streamlined pipeline: Retrieve -> Re-rank (Flashrank) -> Generate.
- **Agentic RAG**: A sophisticated LangGraph-based flow: Router -> Rewriter -> Retriever -> Generator -> Verifier (with automated retries).

---

## Setup & Installation

### Prerequisites
- **Python 3.10+**
- **Ollama**: [Download here](https://ollama.com/)
- **FFmpeg**: Required for Whisper video transcription (optional, only if using video tutorials).

### 1. Clone & Install Dependencies
```bash
pip install pymupdf4llm langchain-ollama langchain-chroma langchain-community langchain-experimental faster-whisper flashrank langgraph python-dotenv
```

### 2. Configure Ollama
Ensure the required models are pulled:
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
# Add any API keys if switching to cloud-based models (optional), previously used gemini
# GOOGLE_API_KEY=your_key_here
```

---

## Usage

### 1. Ingest Data
Place your Monopoly rules (PDFs/Videos) in the `./data/` folder, then run:
```bash
python ingest.py
```
*You can specify a strategy by passing '1' (Standard), '2' (Recursive), or '3' (Semantic) as an argument.*

### 2. Start the Assistant
```bash
python main.py
```
Select between **Simple RAG** or **Agentic RAG** and start asking questions!

### 3. Run Evaluations
Compare the effectiveness of different chunking strategies:
```bash
python run_eval.py
```

---

##  Project Structure

- `main.py`: Interactive CLI entry point.
- `ingest.py`: Multi-modal data processing and vector store creation.
- `graph_agent.py`: LangGraph implementation of the Agentic RAG workflow.
- `rag.py`: Simple hybrid-search RAG implementation.
- `evaluation.py` & `run_eval.py`: RAG evaluation and benchmarking tools.