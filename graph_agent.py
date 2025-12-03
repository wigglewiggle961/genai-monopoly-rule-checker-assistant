from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph

load_dotenv()

MAX_RETRIES = 3

def format_docs(docs: List[Document]) -> str:
    # Joins the page content of retrieved documents into a single string.
    return "\n\n".join(doc.page_content for doc in docs)

def get_sources(docs: List[Document]) -> List[str]:
    """Extracts the 'source' metadata from a list of documents and returns a unique list."""
    return sorted(list(set(doc.metadata.get("source", "Unknown Source") for doc in docs)))

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str  # answer from first agent
    verification_decision: str  
    retry_count: int

# nodes
def retrieve_documents(state: GraphState) -> GraphState:
    # retrieve documents based on question
    print("Retrieving Relevant Documents...")
    question = state["question"]
    documents = retriever.invoke(question)
    print(f"Retrieved {len(documents)} documents.")
    return {"documents": documents}

def generate_answer(state: GraphState) -> GraphState:
    # generates initial answer, will be checked if valid or not 
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
    return {"generation": generation.content, "retry_count":current_retries+1}

def verify_answer(state: GraphState) -> GraphState:
    # verifies the initial answer, because i got different edition of monopoly and there might
    # be ambiguity about the different versions
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
    })
    
    if "CORRECT" in decision.upper():
        print("Verification Agent thinks it's correct")
        return {"verification_decision": "CORRECT"}
    else:
        print("Verification Agent thinks it's ambigious")
        return {"verification_decision": "AMBIGUOUS"}

def decide_next_step(state: GraphState) -> str:
    print("Checking verification...")
    if state["verification_decision"] == "CORRECT":
        return "present_final_answer"
    elif state["verification_decision"] == "RECLARIFICATION":
        return "present_reclarification_question"
    else:
        if state["retry_count"] < MAX_RETRIES:
            print(f"Retrying... (Attempt {state['retry_count']}/{MAX_RETRIES})")
            return "rethink" # Loop back
        else:
            print("Max retries reached. Returning what we have.")
            return "present_final_answer" # Give up and return the ambiguous answer

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db_path = "vector_store"
db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
retriever = db.as_retriever(search_kwargs={'k': 3},
                                # search_type="similarity_score_threshold",
                                # different type of search type
                                search_type="similarity"
                                # search_type="mmr",
                                # search_kwargs={"k": 4, "fetch_k": 20} 

                                )

rag_prompt_template = """
You are a precise Monopoly rules expert, specializing in the many different editions of the game. Your task is to answer questions using ONLY the official rulebook context provided.

**Source Documents:**
The context below was retrieved from the following rulebook(s):
{sources}

**Your Instructions:**
1.  First, carefully analyze the user's **Question** and the provided **Context**.
2.  The **Context** may contain rules from different versions of Monopoly (e.g., Classic, Monopoly Deal, Cheaters Edition). Pay close attention to the **Source Documents** listed above to understand which version(s) are relevant.
3.  **Crucially:** If the user's question is ambiguous and could apply to multiple versions of Monopoly with different rules, you MUST ask for clarification. Do not guess or default to the classic rules.
    *   *Example:* If the user asks "How much money do you start with?" and the context provides rules for both Classic Monopoly and Monopoly Deal, you should ask: "To give you the correct answer, could you please specify which version of Monopoly you're asking about? The starting amount is different for Classic and Monopoly Deal."
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
You are a strict fact-checker. Review the **User Question**, **Sources**, *Source Context*, and the **Generated Answer**.

Your Rules:
1. If the Generated Answer is a clarifying question → respond **RECLARIFICATION**.
2. If the answer is unsupported, unclear, or ambiguous with respect to the sources → respond **AMBIGUOUS**.
3. If the answer is fully supported and unambiguous → respond **CORRECT**.

Return ONLY one word.

User Question:
{question}

Sources:
{sources}

Source Context:
{context}

Generated Answer:
{generation}

Your Decision:

"""
verifier_prompt = PromptTemplate(
    template=verifier_prompt_template, 
    input_variables=["question", "sources", "context", "generation"]
)


def create_agentic_rag_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("verify_answer", verify_answer)

    # define flow
    workflow.set_entry_point("retriever")
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


