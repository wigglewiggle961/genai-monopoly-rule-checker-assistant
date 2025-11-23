from graph_agent import create_agentic_rag_workflow
from rag import create_rag_chain

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

while True:
    user_query = input("Your question: ")
    if user_query.lower() == 'exit':
        break

    if choice == "2":
        initial_state = {
            "question": user_query,
            "documents": [],
            "generation": "",
            "verification_decision": "",
        }
        response = chain.invoke(initial_state)
        answer = response.get("generation", "")
        sources = [doc.metadata.get("source", "Unknown Source") for doc in response.get("documents", [])]
    else:
        response = chain.invoke(user_query)
        answer = response['answer']
        sources = response['sources']

    print("\nAnswer:")
    print(answer)
    print("\nSources:")
    for source in sources:
        print(source)