from typing_extensions import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

# broke so no advance model for reasoning
grader_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class CorrectnessGrade(TypedDict):
    """The schema for the correctness grade."""
    explanation: Annotated[str, "Explain your reasoning for the score step-by-step."]
    correct: Annotated[bool, "True if the answer is correct, False otherwise."]

correctness_grader = grader_llm.with_structured_output(CorrectnessGrade)

correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.
Your task is to determine if the STUDENT ANSWER is correct based on the GROUND TRUTH.

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.
(2) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate.
(3) The student answer must not contain any statements that contradict the ground truth.

A correctness value of True means that the student's answer is factually accurate based on the ground truth.
A correctness value of False means that the student's answer contains factually inaccurate information.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
"""

def correctness_evaluator(question: str,reference_answer: str,prediction: str,) -> dict:
    grading_prompt_input = f"""
    QUESTION:
    {question}

    GROUND TRUTH ANSWER:
    {reference_answer}

    STUDENT ANSWER:
    {prediction}
    """
    
    grade = correctness_grader.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": grading_prompt_input}
    ])
    
    return {"correct": grade["correct"], "explanation": grade["explanation"]}