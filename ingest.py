import time
import pymupdf4llm
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from faster_whisper import WhisperModel 
import os
import sys
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# --- SETUP ---
DATA_PATH = "./data"
DEBUG_PATH = "converted_markdown"
BATCH_SIZE = 100 

if not os.path.exists(DEBUG_PATH):
    os.makedirs(DEBUG_PATH)

# Initialize Models
# Force CPU for whisper on Windows to avoid DLL issues
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

def load_vid_with_timestamps(file_path, filename):
    segments, info = whisper_model.transcribe(file_path)
    docs = []
    for seg in segments:
        doc = Document(
            page_content=seg.text,
            metadata={
                "source": filename,
                "start_time": seg.start,
                "end_time": seg.end
            }
        )
        docs.append(doc)
    return docs

def fixed_size_split(documents, size=256, overlap=50):
    splitter = CharacterTextSplitter(
        separator="", 
        chunk_size=size, 
        chunk_overlap=overlap,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)

def recursive_split(documents, size=256, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(documents)

def semantic_split(documents):
    # Safety: pre-split into 5000-char chunks so SemanticChunker doesn't 
    # accidentally create a chunk larger than our 8k token context window.
    pre_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    docs = pre_splitter.split_documents(documents)
    
    splitter = SemanticChunker(embedding_model)
    return splitter.split_documents(docs)

def main():
    # --- 1. LOADING PHASE (The expensive part) ---
    # We do this once and keep the full documents in memory
    print("\n--- PHASE 1: LOADING & TRANSCRIBING ---")
    base_documents = []
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        
        if filename.lower().endswith(".pdf"):
            print(f"Loading PDF: {filename}...")
            md_text = pymupdf4llm.to_markdown(file_path)
            if md_text.strip():
                base_documents.append(Document(page_content=md_text, metadata={"source": filename}))

        elif filename.lower().endswith((".webm", ".mp4")):
            print(f"Transcribing Video: {filename} (this might take a while)...")
            transcript_docs = load_vid_with_timestamps(file_path, filename)
            base_documents.extend(transcript_docs)

    if not base_documents:
        print("No documents found to process.")
        return

    # --- 2. VECTORIZING PHASE (The fast part) ---
    strategies = {
        "1": ("standard", fixed_size_split),
        "2": ("recursive", recursive_split),
        "3": ("semantic", semantic_split)
    }

    # If the user provided a specific strategy in CLI, only do that one
    # Otherwise, do all three by default for the benchmark
    to_run = [sys.argv[1]] if len(sys.argv) > 1 else ["1", "2", "3"]

    for choice in to_run:
        if choice not in strategies:
            print(f"Skipping invalid strategy choice: {choice}")
            continue

        name, split_func = strategies[choice]
        db_path = f"vector_store_{name}"
        
        print(f"\n--- PHASE 2: PROCESSING STRATEGY [{name.upper()}] ---")
        print(f"Chunking...")
        chunks = split_func(base_documents)
        print(f"Generated {len(chunks)} chunks.")

        print(f"Creating vector store at: {db_path}...")
        db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
        
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            for doc in batch:
                if not doc.page_content.startswith("search_document: "):
                    doc.page_content = f"search_document: {doc.page_content}"
            
            print(f"   Adding batch {i // BATCH_SIZE + 1}/{ (len(chunks) // BATCH_SIZE) + 1}...")
            db.add_documents(documents=batch)
        print(f"Done with {name}!")

    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
