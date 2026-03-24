from typing import List, TypedDict, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings # Free local alternatives
from langgraph.graph import END, StateGraph

load_dotenv()

MAX_RETRIES = 3

def format_docs(docs: List[Document]) -> str:
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        meta = doc.metadata
        
        if "start_time" in meta:
            location = f"Timestamp {format_timestamp(meta['start_time'])}"
        elif "page" in meta:
            location = f"Page {meta['page'] + 1}"
        else:
            location = "Text segment"

        # Clean up the page content for the LLM (strip Nomic prefixes)
        clean_content = doc.page_content.replace("search_document: ", "").replace("search_query: ", "")
        chunk_text = f"--- Source: {source} ({location}) ---\n{clean_content}"
        formatted_chunks.append(chunk_text)
        
    return "\n\n".join(formatted_chunks)

def format_timestamp(seconds):
    if seconds is None: return ""
    m, s = divmod(int(seconds),60)
    return f"{m:02d}:{s:02d}"

def get_sources(docs: List[Document]) -> List[str]:
    unique_sources = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        meta = doc.metadata
        
        # Check if it's a video chunk (has start_time)
        if "start_time" in meta:
            timestamp = format_timestamp(meta["start_time"])
            entry = f"{source} (Timestamp: {timestamp})"
            
        # Check if it has a page number (Standard PDF Loaders)
        elif "page" in meta:
            # +1 because computer counts from 0, humans count from 1
            entry = f"{source} (Page: {meta['page'] + 1})"
            
        # Fallback for Markdown/Text files without specific location data
        else:
            entry = source
            
        unique_sources.add(entry)
    
    return sorted(list(unique_sources))

class GraphState(TypedDict):
    question: str
    rewritten_question: str           # standalone query after rewriting
    chat_history: List[str]           # list of past Q&A strings
    documents: List[Document]
    generation: str
    verification_decision: str
    retry_count: int

# ─── nodes ────────────────────────────────────────────────────────────────────

def route_question(state: GraphState) -> GraphState:
    """Detect whether the input is a Monopoly rule question or a greeting/off-topic."""
    print("Routing question...")
    question = state["question"]

    router_chain = router_prompt | llm | StrOutputParser()
    decision = router_chain.invoke({"question": question}).strip().upper()

    if "GREETING" in decision:
        print("Router: Greeting detected.")
        return {"verification_decision": "GREETING", "generation": "Hey there! Ask me anything about Monopoly rules!"}
    else:
        print("Router: Rule question detected.")
        return {"verification_decision": ""}

def rewrite_query(state: GraphState) -> GraphState:
    """Rewrite a contextual follow-up into a standalone search query."""
    print("Rewriting query...")
    question = state["question"]
    history = state.get("chat_history", [])

    if not history:
        # No history, use the question as-is
        return {"rewritten_question": question}

    history_text = "\n".join(history[-4:])  # last 2 turns
    rewriter_chain = rewriter_prompt | llm | StrOutputParser()
    rewritten = rewriter_chain.invoke({
        "chat_history": history_text,
        "question": question
    }).strip()

    print(f"Rewritten query: {rewritten}")
    return {"rewritten_question": rewritten}

def retrieve_documents(state: GraphState) -> GraphState:
    """Retrieve documents based on the (potentially rewritten) question."""
    print("Retrieving Relevant Documents...")
    search_query = state.get("rewritten_question") or state["question"]
    # Add Nomic search_query prefix for better performance
    documents = retriever.invoke(f"search_query: {search_query}")
    print(f"Retrieved {len(documents)} documents.")
    return {"documents": documents}

def generate_answer(state: GraphState) -> GraphState:
    """Generate the initial answer based on retrieved documents."""
    print("Generating Initial Answer...")
    question = state["question"]
    documents = state["documents"]
    current_retries = state.get("retry_count", 0)

    context = format_docs(documents)
    sources = ", ".join(get_sources(documents))
    
    prompt_with_inputs = rag_prompt.invoke({
        "context": context, 
        "question": question, 
        "sources": sources
    })
    
    generation = llm.invoke(prompt_with_inputs)
    return {"generation": generation.content, "retry_count": current_retries + 1}

def verify_answer(state: GraphState) -> GraphState:
    """Verify the generated answer against the retrieved context."""
    print("Verifying Answer...")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    sources = ", ".join(get_sources(documents))
    context = format_docs(documents)

    verifier_chain = verifier_prompt | llm | StrOutputParser()
    
    decision = verifier_chain.invoke({
        "question": question,
        "sources": sources,
        "context": context,
        "generation": generation
    }).strip().upper()

    if "CORRECT" in decision:
        print("Verification Agent thinks it's correct")
        return {"verification_decision": "CORRECT"}
    elif "RECLARIFICATION" in decision:
        # If the answer correctly asks the user for clarification, that's valid
        print("Verification Agent detected a clarification request.")
        return {"verification_decision": "RECLARIFICATION"}
    else:
        print("Verification Agent thinks it's ambiguous")
        return {"verification_decision": "AMBIGUOUS"}

def decide_after_routing(state: GraphState) -> str:
    """Route after the router node runs."""
    if state.get("verification_decision") == "GREETING":
        return "greeting"
    return "rewrite"

def decide_next_step(state: GraphState) -> str:
    print("Checking verification...")
    decision = state["verification_decision"]
    print(decision)

    if decision == "CORRECT":
        return "present_final_answer"
    elif decision == "RECLARIFICATION":
        return "present_reclarification_question"
    else:  # AMBIGUOUS
        if state["retry_count"] < MAX_RETRIES:
            print(f"Retrying... (Attempt {state['retry_count']}/{MAX_RETRIES})")
            return "rethink"
        else:
            print("Max retries reached. Returning what we have.")
            return "present_final_answer"

# --- GLOBAL COMPONENTS (Refreshed by create_agentic_rag_workflow) ---
llm = ChatOllama(model="llama3.1", temperature=0)
retriever = None 

def create_agentic_rag_workflow(db_path="vector_store"):
    global retriever
    global llm
    
    # Initialize LLM
    llm = ChatOllama(model="llama3.1", temperature=0)

    # Initialize Embeddings and Chroma DB
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    
    # 1. Sparse Retriever (BM25)
    from langchain_community.retrievers import BM25Retriever
    results = db.get()
    all_docs = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(results['documents'], results['metadatas'])
    ]
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 10

    # 2. Dense Retriever (Vector)
    vector_retriever = db.as_retriever(search_kwargs={'k': 10})

    # 3. Ensemble (Hybrid)
    from langchain_classic.retrievers import EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.7, 0.3]
    )

    # 4. Re-Ranking
    from langchain_classic.retrievers.document_compressors import FlashrankRerank
    from langchain_classic.retrievers import ContextualCompressionRetriever
    compressor = FlashrankRerank(top_n=5)
    
    # Update global retriever for nodes to use
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    workflow = StateGraph(GraphState)

    workflow.add_node("router", route_question)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("verify_answer", verify_answer)

    # Entry: always route first
    workflow.set_entry_point("router")

    # After routing: greeting -> END, rule question -> rewrite
    workflow.add_conditional_edges(
        "router",
        decide_after_routing,
        {
            "greeting": END,
            "rewrite": "rewrite_query",
        }
    )

    workflow.add_edge("rewrite_query", "retriever")
    workflow.add_edge("retriever", "generate_answer")
    workflow.add_edge("generate_answer", "verify_answer")
    
    workflow.add_conditional_edges(
        "verify_answer",
        decide_next_step,
        {
            "present_final_answer": END, 
            "present_reclarification_question": END,
            "rethink": "generate_answer",
        }
    )

    app = workflow.compile()
    return app

# ─── prompts ──────────────────────────────────────────────────────────────────

router_prompt = PromptTemplate(
    template="""You are a question classifier for a Monopoly rules assistant.

Classify the following user input into one of two categories:
- GREETING: The input is a greeting, small talk, off-topic, or not a Monopoly rules question.
- RULES_QUESTION: The input is asking about Monopoly rules, gameplay, or related topics.

User Input: {question}

Output one word only (GREETING or RULES_QUESTION):""",
    input_variables=["question"]
)

rewriter_prompt = PromptTemplate(
    template="""You are a question rewriter. Your job is to take a contextual follow-up question and rewrite it into a clear, standalone search query using the chat history for context.

Chat History:
{chat_history}

Follow-up Question: {question}

Rewrite the question so it can be understood without the chat history. Output only the rewritten question, nothing else:""",
    input_variables=["chat_history", "question"]
)

rag_prompt_template = """
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
rag_prompt = PromptTemplate(
    template=rag_prompt_template, 
    input_variables=["context", "question", "sources"]
)
#using context here instead of sources because I think it should verify based on the given context?
verifier_prompt_template = """
You are a logic gate. Your job is to classify the **Generated Answer** with exactly one word.

**Instructions (Follow in Order):**

1. **CHECK FOR CLARIFICATION:** Does the 'Generated Answer' ask the user a question, ask for specifics, or request which version of the game they are playing?
   * If YES -> Output RECLARIFICATION and STOP.
   * If NO -> Proceed to step 2.

2. **CHECK FOR AMBIGUITY/ERRORS:**
   Compare the 'Generated Answer' against the 'Source Context'.
   * If the answer contradicts the sources -> Output AMBIGUOUS.
   * If the answer is unsupported by the text -> Output AMBIGUOUS.
   * If the answer fails to address conflicting rules from multiple editions -> Output AMBIGUOUS.

3. **CHECK FOR CORRECTNESS:**
   * If the answer is clearly supported by the sources -> Output CORRECT.

**Data:**
Generated Answer: {generation}
Sources: {sources}
Source Context: {context}
User Question: {question}

**Output (one word only — RECLARIFICATION, AMBIGUOUS, or CORRECT):**
"""
verifier_prompt = PromptTemplate(
    template=verifier_prompt_template, 
    input_variables=["question", "sources", "context", "generation"]
)