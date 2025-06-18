"""
Dextra - Data extraction framework for financial documents.

This package provides specialized extractors for different document types
and extraction methods, including regex-based, ML-based, and rule-based extractors.
"""

from .base import (
    Extractor, FieldExtractor, DocumentExtractor, 
    ExtractionResult, DocumentType, ExtractorFactory
)
from .regex_extractor import (
    RegexFieldExtractor, AmountExtractor, DateExtractor,
    TextExtractor, InvoiceNumberExtractor, TaxIDExtractor,
    RegexInvoiceExtractor, RegexReceiptExtractor, RegexExtractorFactory
)
from .ml_extractor import (
    MLFieldExtractor, MLInvoiceExtractor, 
    MLReceiptExtractor, MLExtractorFactory
)
from .extractor_factory import (
    ExtractionMethod, UnifiedExtractorFactory
)

__version__ = "0.1.0"

__all__ = [
    # Base classes
    "Extractor", "FieldExtractor", "DocumentExtractor",
    "ExtractionResult", "DocumentType", "ExtractorFactory",
    
    # Regex extractors
    "RegexFieldExtractor", "AmountExtractor", "DateExtractor",
    "TextExtractor", "InvoiceNumberExtractor", "TaxIDExtractor",
    "RegexInvoiceExtractor", "RegexReceiptExtractor", "RegexExtractorFactory",
    
    # ML extractors
    "MLFieldExtractor", "MLInvoiceExtractor", 
    "MLReceiptExtractor", "MLExtractorFactory",
    
    # Factory
    "ExtractionMethod", "UnifiedExtractorFactory",
]
