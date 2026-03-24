from langchain_chroma import Chroma
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings # Free alternative: local models
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel

load_dotenv()

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

def format_docs(docs):
    # Joins the page content of retrieved documents into a single string.
    # Also strips the Nomic prefix so the LLM doesn't get confused.
    return "\n\n".join(doc.page_content.replace("search_document: ", "").replace("search_query: ", "") for doc in docs)

def inspect(state):
    # Print the state passed between Runnables in a langchain and pass it on
    print(state)
    return state

def get_sources(docs):
    # Extracts the 'source' metadata from a list of documents and returns a unique list.
    return list(set(doc.metadata.get("source", "Unknown Source") for doc in docs))

def create_rag_chain(db_path="vector_store"):
    # Load Environment Variables
    load_dotenv()

    # --- SETUP MODELS ---
    # embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3.1", temperature=0)

    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

    # retriever = db.as_retriever(search_kwargs={'k': 3},
    #                         # search_type="similarity_score_threshold",
    #                         # different type of search type
    #                         search_type="similarity"
    #                         # search_type="mmr",
    #                         # search_kwargs={"k": 4, "fetch_k": 20} 
    # )
    # --- HYBRID SEARCH SETUP ---
    # 1. Sparse Retriever (BM25)
    # We need to pull documents from Chroma to build the BM25 index
    results = db.get()
    all_docs = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(results['documents'], results['metadatas'])
    ]
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 10

    # 2. Dense Retriever (Chroma Vector Search)
    vector_retriever = db.as_retriever(search_kwargs={'k': 10})

    # 3. Ensemble Retriever (Combining both)
    # Reciprocal Rank Fusion (RRF) will be used to merge the results
    # might have to change the weights in the future
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.6, 0.4]
    )

    # --- RE-RANKING SETUP ---
    from langchain_classic.retrievers.document_compressors import FlashrankRerank
    from langchain_classic.retrievers import ContextualCompressionRetriever

    compressor = FlashrankRerank(top_n=3)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    prompt_template = """
    You are a precise Monopoly rules expert, specializing in the many different editions of the game. Your task is to answer questions using ONLY the official rulebook context provided.

    **Source Documents:**
    The context below was retrieved from the following rulebook(s):
    {sources}

    **Your Instructions:**
    1.  First, carefully analyze the user's **Question** and the provided **Context**.
    2.  The **Context** may contain rules from different versions of Monopoly (e.g., Classic, Monopoly Deal, Cheaters Edition). Pay close attention to the **Source Documents** listed above to understand which version(s) are relevant.
    3.  **Ambiguity:** If the question could apply to multiple Monopoly versions with different rules, provide the answer for **Classic Monopoly** first (as a baseline), then concisely note that rules for other versions (like Monopoly Deal) may differ and ask if they need those specifics.
    4.  If the answer is not in the provided **Context**, state that you cannot find the answer in the rulebooks.
    5.  Base your entire answer on the provided text. Do not use any prior knowledge about Monopoly.

    **Context:**
    {context}

    **Question:**
    {question}

    **Answer:**
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    setup_and_retrieval = RunnableParallel(
        {"docs": (lambda x: f"search_query: {x}") | retriever, "question": RunnablePassthrough()}
    )

    prompt_inputs=RunnableParallel(
        question=lambda x: x["question"],
        context=lambda x: format_docs(x["docs"]),
        sources=lambda x: ", ".join(get_sources(x["docs"]))
    )

    rag_chain = (
        setup_and_retrieval
        | RunnableParallel(
            #returning answer and sources only for now
              answer=(
                  prompt_inputs
                #   | RunnableLambda(inspect)
                  | PROMPT
                  | llm
                  | StrOutputParser()
              ),
              docs=lambda x: x["docs"],
              sources=lambda x: get_sources(x["docs"]),
          )
    )
    
    return rag_chain