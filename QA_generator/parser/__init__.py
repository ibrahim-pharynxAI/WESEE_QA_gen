"""
PDF to Markdown converter with LaTeX cleaning capabilities.
"""

from .managers.docling_manager import PDFConverter
from .managers.regex_manager import clean_latex, clean_latex_formulas_in_md

__version__ = "1.0.0"
__all__ = ["PDFConverter", "clean_latex", "clean_latex_formulas_in_md"]
