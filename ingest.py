import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
# import whisper
from faster_whisper import WhisperModel 
import os
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

all_docs = []

path = "./data"

model = WhisperModel("small")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

db = Chroma(
    persist_directory="vector_store",
    embedding_function=embedding_model
)

# it was sending too much request at the same time
BATCH_SIZE = 50
DELAY_SECONDS = 60

def load_vid(path):
    segments,info=model.transcribe(path)
    text = " ".join([seg.text for seg in segments])
    return text

def file_already_indexed(db, filename: str):
    results = db.get(where={"source": filename})
    return len(results["ids"]) > 0

for filename in os.listdir(path):

    file_path = os.path.join(path, filename)

    # Skip if vectorized already
    if file_already_indexed(db, filename):
        print(f"Skipping {filename}, already embedded.")
        continue

    if filename.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = filename
        all_docs.extend(docs)

    elif filename.lower().endswith((".webm", ".mp4")):
        transcript = load_vid(file_path)
        all_docs.append(Document(page_content=transcript, metadata={"source": filename}))


print("Choose chunking strategy:")
print("1. Fixed Size")
print("2. Recursive Character Splitter")
print("3. Semantic Chunking")

choice = input("")

def fixed_size(documents, size=1000, overlap=100):
    splitter = CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(documents)

def recursive_split(documents, size = 1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(documents)

def semantic_split(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    splitter = SemanticChunker(embeddings)
    return splitter.split_documents(documents)

if choice == "1":
    print("fixed")
    chunks = fixed_size(all_docs)

elif choice == "2":
    print("recursive")
    chunks = recursive_split(all_docs)

elif choice == "3":
    print("semantic")
    chunks = semantic_split(all_docs)

else:
    raise ValueError("Invalid selection!")

print(f"Generated {len(chunks)} chunks.")


if len(chunks)!=0:
    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        
        print(f"Processing batch {i // BATCH_SIZE + 1}: adding {len(batch)} chunks...")
        
        db.add_documents(documents=batch)
        
        print(f"Waiting for {DELAY_SECONDS} seconds...")
        time.sleep(DELAY_SECONDS)

else:
    print("No new files detected")