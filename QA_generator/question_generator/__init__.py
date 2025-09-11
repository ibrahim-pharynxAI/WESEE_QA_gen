"""
Question generator using vLLM for creating questions from text content.
"""

from .generate import QuestionGenerator
from .managers.vllm_manager import check_vLLM, start_vllm_server

__version__ = "1.0.0"
__all__ = ["QuestionGenerator", "check_vLLM", "start_vllm_server"]
