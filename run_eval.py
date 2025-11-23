import pandas as pd
from dotenv import load_dotenv

from rag import create_rag_chain
from graph_agent import create_agentic_rag_workflow
from evaluation import correctness_evaluator

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
        "question": "What happens if you roll doubles three times in a row?",
        "reference_answer": "On your third roll of doubles, you do not move. Instead, you must Go To Jail immediately."
    },
    {
        "question": "Does monopoly have a time limit?",
        "reference_answer": "The rulebook does not specify a time limit for the game. However, a shorter game variant is suggested where the game ends after the second player goes bankrupt."
    }
]

def run_evaluation():
    print("Initializing RAG systems...")
    simple_chain = create_rag_chain()
    agentic_chain = create_agentic_rag_workflow()
    
    print(f"Running evaluation on {len(eval_dataset)} questions...")
    results = []

    for item in eval_dataset:
        question = item["question"]
        reference_answer = item["reference_answer"]
        print(f"\n--- Question: {question} ---")

        # run and eval simple rag
        simple_response = simple_chain.invoke(question)
        simple_answer = simple_response['answer']
        simple_grade = correctness_evaluator(
            question=question,
            reference_answer=reference_answer,
            prediction=simple_answer
        )

        # run and eval agentic rag
        agentic_initial_state = {"question": question}
        agentic_response_state = agentic_chain.invoke(agentic_initial_state)
        agentic_answer = agentic_response_state.get("generation", "Agent failed to generate an answer.")
        agentic_grade = correctness_evaluator(
            question=question,
            reference_answer=reference_answer,
            prediction=agentic_answer
        )

        # comparison
        results.append({
            "question": question,
            "reference_answer": reference_answer,
            "simple_rag_answer": simple_answer,
            "simple_rag_correct": simple_grade["correct"],
            "simple_rag_explanation": simple_grade["explanation"],
            "agentic_rag_answer": agentic_answer,
            "agentic_rag_correct": agentic_grade["correct"],
            "agentic_rag_explanation": agentic_grade["explanation"],
        })

    print("\n\n" + "="*50)
    print("DETAILED EVALUATION RESULTS")
    print("="*50)

    for i, result in enumerate(results):
        print(f"\n----- Question {i+1}/{len(results)} -----")
        print(f"QUESTION: {result['question']}")
        print(f"REFERENCE ANSWER: {result['reference_answer']}")
        print("-" * 20)
        
        print("SIMPLE RAG:")
        print(f"  Correct: {result['simple_rag_correct']}")
        print(f"  Answer: {result['simple_rag_answer']}")
        print(f"  Explanation: {result['simple_rag_explanation']}")
        print("-" * 20)
        
        print("AGENTIC RAG:")
        print(f"  Correct: {result['agentic_rag_correct']}")
        print(f"  Answer: {result['agentic_rag_answer']}")
        print(f"  Explanation: {result['agentic_rag_explanation']}")
        
        print("="*50)

run_evaluation()