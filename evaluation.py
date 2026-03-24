from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from datasets import Dataset

# --- SETUP EVALUATOR ---
# Use existing Llama 3.1 and Nomic Embeddings via LangChain wrappers
grader_llm = ChatOllama(model="llama3.1", temperature=0)
grader_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Wrap them for Ragas
llm_wrapper = LangchainLLMWrapper(grader_llm)
embeddings_wrapper = LangchainEmbeddingsWrapper(grader_embeddings)

# Initialize metrics with these models
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
]

from ragas.run_config import RunConfig

def run_ragas_evaluation(eval_samples: list) -> Dataset:
    """
    Runs Ragas evaluation on a list of samples.
    Each sample must have: 'user_input', 'response', 'retrieved_contexts', and 'reference'.
    """
    # Convert list of dicts to a HuggingFace Dataset (what Ragas expects)
    dataset = Dataset.from_list(eval_samples)
    
    # Run config for local Ollama stability
    run_config = RunConfig(timeout=180, max_workers=1)
    
    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm_wrapper,
        embeddings=embeddings_wrapper,
        run_config=run_config
    )
    
    return result

# --- LEGACY CORRECTNESS GRADER (kept for backward compatibility) ---
from typing_extensions import Annotated, TypedDict

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, "Explain your reasoning for the score step-by-step."]
    correct: Annotated[bool, "True if the answer is correct, False otherwise."]

correctness_grader = grader_llm.with_structured_output(CorrectnessGrade)
correctness_instructions = """You are a teacher grading a quiz. Determine if the STUDENT ANSWER is correct based on the GROUND TRUTH."""

def correctness_evaluator(question: str, reference_answer: str, prediction: str) -> dict:
    grading_prompt_input = f"QUESTION: {question}\nGROUND TRUTH: {reference_answer}\nSTUDENT ANSWER: {prediction}"
    grade = correctness_grader.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": grading_prompt_input}
    ])
    return {"correct": grade["correct"], "explanation": grade["explanation"]}
