from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel

load_dotenv()

def format_docs(docs):
    # Joins the page content of retrieved documents into a single string.
    return "\n\n".join(doc.page_content for doc in docs)

def inspect(state):
    # Print the state passed between Runnables in a langchain and pass it on
    print(state)
    return state

def get_sources(docs):
    # Extracts the 'source' metadata from a list of documents and returns a unique list.
    return list(set(doc.metadata.get("source", "Unknown Source") for doc in docs))

def create_rag_chain():
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

    prompt_template = """
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

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    setup_and_retrieval = RunnableParallel(
        {"docs": retriever, "question": RunnablePassthrough()}
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
              sources=lambda x: get_sources(x["docs"]),
          )
    )
    
    return rag_chain