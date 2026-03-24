from graph_agent import create_agentic_rag_workflow
from rag import create_rag_chain

def format_source(doc):
    """
    Extracts metadata and formats it into a readable string.
    e.g., "video.mp4 (Timestamp: 02:30)" or "rules.pdf (Page: 5)"
    """
    meta = doc.metadata
    source = meta.get("source", "Unknown Source")
    
    # Check for Video Timestamp
    if "start_time" in meta:
        minutes, seconds = divmod(int(meta["start_time"]), 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"
        return f"{source} (Timestamp: {timestamp})"
    
    # Check for PDF Page
    elif "page" in meta:
        # +1 because valid pages start at 1, but computers start at 0
        return f"{source} (Page: {meta['page'] + 1})"
    
    # Default (Just filename)
    return source

monopoly_chain = create_rag_chain()

choice = input("Select mode (1 = simple RAG, 2 = agentic RAG graph): ")

if choice == "1":
    chain = create_rag_chain()
elif choice == "2":
    chain = create_agentic_rag_workflow()
else:
    print("Invalid choice. Defaulting to simple RAG.")
    chain = create_rag_chain()

print("Monopoly Rule Assistant is ready!")
print("Ask a question or type 'exit' to quit.")

chat_history = []  # rolling history for the Query Rewriter

while True:
    user_query = input("Your question: ")
    if user_query.lower() == 'exit':
        break

    sources_list = [] # Reset sources

    if choice == "2":
        initial_state = {
            "question": user_query,
            "rewritten_question": "",
            "chat_history": chat_history,
            "documents": [],
            "generation": "",
            "verification_decision": "",
            "retry_count": 0
        }
        response = chain.invoke(initial_state)
        answer = response.get("generation", "")
        raw_docs = response.get("documents", [])
        sources_list = [format_source(doc) for doc in raw_docs]
        # Update history with this turn
        if answer:
            chat_history.append(f"User: {user_query}")
            chat_history.append(f"Assistant: {answer}")
    else:
        response = chain.invoke(user_query)
        answer = response['answer']
        sources = response['sources']

    print("\nAnswer:")
    print(answer)
    unique_sources = list(dict.fromkeys(sources_list))
    print("\nSources:")
    for source in unique_sources:
        print(f"- {source}")