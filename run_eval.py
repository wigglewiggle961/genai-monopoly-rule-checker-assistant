import os
import pandas as pd
import datetime
from dotenv import load_dotenv

from rag import create_rag_chain
from graph_agent import create_agentic_rag_workflow
from evaluation import correctness_evaluator, run_ragas_evaluation

load_dotenv()

eval_dataset = [
    {
        "question": "How do you win in classic Monopoly?",
        "reference_answer": "The last player left in the game after everyone else has gone bankrupt wins the game."
    },
    {
        "question": "How much money does a player start with in classic Monopoly?",
        "reference_answer": "Each player starts with $1500, divided into specific bills."
    },
    {
        "question": "How many property sets do I need to win in Monopoly Deal?",
        "reference_answer": "You need to collect three complete property sets of different colors to win."
    },
    {
        "question": "What is the starting amount of money in Monopoly Empire?",
        "reference_answer": "Each player starts with 1000K."
    },
    {
        "question": "What does the 'sneaky swapper' do in Monopoly Empire?",
        "reference_answer": "It allows you to switch the topmost billboard in one tower with the topmost billboard in any other tower."
    }
]

def run_evaluation():
    db_strategies = [
        "vector_store_standard",
        "vector_store_recursive",
        "vector_store_semantic"
    ]
    
    summary_data = []

    for db_path in db_strategies:
        if not os.path.exists(db_path):
            print(f"Skipping {db_path}, not found.")
            continue

        print(f"\n" + "="*50)
        print(f"EVALUATING: {db_path.upper()}")
        print("="*50)

        simple_chain = create_rag_chain(db_path)
        agentic_chain = create_agentic_rag_workflow(db_path)
        
        print(f"Running evaluation on {len(eval_dataset)} questions...")
        
        simple_eval_samples = []
        agentic_eval_samples = []

        for item in eval_dataset:
            question = item["question"]
            ground_truth = item["reference_answer"]
            print(f"  -> '{question}'")

            # --- SIMPLE RAG ---
            simple_res = simple_chain.invoke(question)
            simple_eval_samples.append({
                "user_input": question,
                "response": simple_res['answer'],
                "retrieved_contexts": [
                    doc.page_content.replace("search_document: ", "") 
                    for doc in simple_res.get("docs", [])
                ],
                "reference": ground_truth
            })

            # --- AGENTIC RAG ---
            agentic_res_state = agentic_chain.invoke({"question": question})
            agentic_eval_samples.append({
                "user_input": question,
                "response": agentic_res_state.get("generation", ""),
                "retrieved_contexts": [
                    doc.page_content.replace("search_document: ", "") 
                    for doc in agentic_res_state.get("documents", [])
                ],
                "reference": ground_truth
            })

        # --- RUN RAGAS EVALUATION ---
        print(f"Calculating Ragas Metrics for {db_path}...")
        simple_ragas = run_ragas_evaluation(simple_eval_samples)
        agentic_ragas = run_ragas_evaluation(agentic_eval_samples)

        # Aggregate means for summary
        s_mean = simple_ragas.to_pandas().mean(numeric_only=True)
        a_mean = agentic_ragas.to_pandas().mean(numeric_only=True)

        summary_data.append({
            "Strategy": db_path.replace("vector_store_", ""),
            "Pipeline": "Simple",
            "Faithfulness": s_mean.get("faithfulness", 0),
            "Relevancy": s_mean.get("answer_relevancy", 0),
            "Precision": s_mean.get("context_precision", 0)
        })
        summary_data.append({
            "Strategy": db_path.replace("vector_store_", ""),
            "Pipeline": "Agentic",
            "Faithfulness": a_mean.get("faithfulness", 0),
            "Relevancy": a_mean.get("answer_relevancy", 0),
            "Precision": a_mean.get("context_precision", 0)
        })

    # --- FINAL DISPLAY ---
    comparison_df = pd.DataFrame(summary_data)
    print("\n\n" + "="*60)
    print("FINAL CHUNKING STRATEGY COMPARISON")
    print("="*60)
    print(comparison_df.to_string(index=False))

    # Save to CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_df.to_csv(f"chunking_benchmark_{timestamp}.csv", index=False)
    print(f"\nSummary saved to chunking_benchmark_{timestamp}.csv")
    print("Verification complete.")

run_evaluation()
