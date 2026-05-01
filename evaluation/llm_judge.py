"""LLM-as-a-Judge evaluation module for GraphRAG Hackathon.

Provides PASS/FAIL judgment using HuggingFace Inference API with fallback to Groq.
"""

import logging
import os
from typing import Dict
import requests
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace Inference API settings
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.getenv("HF_API_TOKEN", "")  # Optional, free tier often works without token


def _build_judge_prompt(question: str, answer: str, reference: str) -> str:
    """Build the judge prompt."""
    return f"""You are an expert judge evaluating answer quality.
Question: {question}
Reference Answer: {reference}
Generated Answer: {answer}

Does the generated answer correctly answer the question based on the reference?
Reply with exactly one word: PASS or FAIL"""


def llm_judge(question: str, answer: str, reference: str) -> Dict[str, any]:
    """Judge answer quality using LLM.
    
    Uses HuggingFace Inference API (free tier) with Mistral-7B-Instruct-v0.3.
    Falls back to Groq if HuggingFace fails.
    
    Args:
        question: The original question
        answer: The generated answer to evaluate
        reference: The reference/gold standard answer
        
    Returns:
        Dict with passed (bool) and raw_response (str)
    """
    prompt = _build_judge_prompt(question, answer, reference)
    
    # Try HuggingFace Inference API first
    try:
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 10,
                "temperature": 0.1,
                "return_full_text": False
            }
        }
        
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                raw_response = result[0].get("generated_text", "").strip()
            else:
                raw_response = str(result).strip()
            
            passed = "PASS" in raw_response.upper()
            logger.debug(f"HuggingFace judge result: {raw_response}")
            return {
                "passed": passed,
                "raw_response": raw_response
            }
        else:
            logger.warning(f"HuggingFace API failed: {response.status_code}")
            # Fall through to Groq fallback
    except Exception as e:
        logger.warning(f"HuggingFace API error: {e}")
        # Fall through to Groq fallback
    
    # Fallback: Use Groq with same prompt
    logger.info("Falling back to Groq for LLM judge...")
    try:
        client = Groq(api_key=GROQ_API_KEY)
        # Using a model that is currently available (llama-3.3-70b-versatile)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert judge. Reply with exactly one word: PASS or FAIL"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        raw_response = response.choices[0].message.content.strip()
        passed = "PASS" in raw_response.upper()
        
        return {
            "passed": passed,
            "raw_response": raw_response
        }
    except Exception as e:
        logger.error(f"Groq fallback also failed: {e}")
        return {
            "passed": False,
            "raw_response": f"ERROR: {str(e)}"
        }


if __name__ == "__main__":
    logger.info("Testing LLM Judge...")
    
    # Test cases
    test_cases = [
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Diabetes symptoms include increased thirst, frequent urination, and fatigue.",
            "reference": "Diabetes symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",
            "expected": True
        },
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "The capital of France is Paris.",
            "reference": "Diabetes symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",
            "expected": False
        }
    ]
    
    for i, test in enumerate(test_cases):
        logger.info(f"\nTest case {i+1}:")
        logger.info(f"  Question: {test['question']}")
        logger.info(f"  Answer: {test['answer'][:60]}...")
        logger.info(f"  Expected: {'PASS' if test['expected'] else 'FAIL'}")
        
        result = llm_judge(test["question"], test["answer"], test["reference"])
        logger.info(f"  Judge result: {'PASS' if result['passed'] else 'FAIL'}")
        logger.info(f"  Raw response: {result['raw_response']}")
