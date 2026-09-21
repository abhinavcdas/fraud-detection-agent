"""LLM Operator Factory.

Instantiates:
- 'groq' -> GroqLLMOperator (Production fast LLaMA 3.3 70B)
- 'mock' -> MockLLMOperator (Deterministic offline testing)
"""

import os
from core.interfaces import BaseLLMOperator
from operators.llm.groq_operator import GroqLLMOperator
from operators.llm.mock_operator import MockLLMOperator
from core.logger import get_logger

logger = get_logger("llm_factory")

def get_llm_operator(provider: str = None) -> BaseLLMOperator:
    """Factory method to get the configured LLM operator."""
    selected = (provider or os.getenv("LLM_PROVIDER", "groq")).lower().strip()
    
    # Auto-fallback to mock if groq is selected but GROQ_API_KEY is unset
    if selected == "groq" and not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY is not set. Automatically falling back to MockLLMOperator.")
        return MockLLMOperator()

    if selected == "groq":
        return GroqLLMOperator()
    elif selected == "mock":
        return MockLLMOperator()
    else:
        logger.warning("Unknown LLM provider '{provider}'. Defaulting to mock.", provider=selected)
        return MockLLMOperator()
