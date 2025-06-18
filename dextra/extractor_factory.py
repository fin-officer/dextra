"""
Unified extractor factory for creating extractors based on document type and method.

This module provides a factory for creating appropriate extractors based on
document type and extraction method preferences.
"""

from enum import Enum
from typing import Dict, List, Optional, Union

from .base import DocumentType, Extractor, ExtractorFactory
from .regex_extractor import RegexExtractorFactory
from .ml_extractor import MLExtractorFactory


class ExtractionMethod(str, Enum):
    """Enumeration of supported extraction methods."""
    
    REGEX = "regex"
    ML = "ml"
    HYBRID = "hybrid"


class UnifiedExtractorFactory:
    """Factory for creating extractors based on document type and method."""
    
    def __init__(
        self,
        preferred_method: ExtractionMethod = ExtractionMethod.HYBRID,
        ml_model_name: str = "distilbert-base-cased-distilled-squad"
    ):
        """
        Initialize unified extractor factory.
        
        Args:
            preferred_method: Preferred extraction method
            ml_model_name: Name of the ML model to use
        """
        self.preferred_method = preferred_method
        self.ml_model_name = ml_model_name
        
        # Initialize specialized factories
        self.regex_factory = RegexExtractorFactory()
        self.ml_factory = MLExtractorFactory(ml_model_name)
    
    def create_extractor(
        self, 
        document_type: Union[DocumentType, str],
        method: Optional[Union[ExtractionMethod, str]] = None,
        **kwargs
    ) -> Extractor:
        """
        Create an appropriate extractor for the given document type and method.
        
        Args:
            document_type: Type of document to extract
            method: Extraction method to use (defaults to preferred_method)
            **kwargs: Additional parameters for the extractor
            
        Returns:
            An extractor instance
            
        Raises:
            ValueError: If no extractor is available for the document type
        """
        # Use preferred method if not specified
        if method is None:
            method = self.preferred_method
        
        # Convert string to enum if needed
        if isinstance(method, str):
            try:
                method = ExtractionMethod(method.lower())
            except ValueError:
                method = self.preferred_method
        
        # Create extractor based on method
        if method == ExtractionMethod.REGEX:
            return self.regex_factory.create_extractor(document_type, **kwargs)
        elif method == ExtractionMethod.ML:
            return self.ml_factory.create_extractor(
                document_type, 
                model_name=kwargs.get("model_name", self.ml_model_name),
                **kwargs
            )
        elif method == ExtractionMethod.HYBRID:
            # For hybrid method, we'll create both extractors and combine them later
            # This is a placeholder for now - in a real implementation, we would
            # create a hybrid extractor that uses both methods
            return self.regex_factory.create_extractor(document_type, **kwargs)
        else:
            raise ValueError(f"Unsupported extraction method: {method}")
    
    def get_supported_document_types(self) -> List[DocumentType]:
        """
        Get the list of supported document types.
        
        Returns:
            List of supported document types
        """
        return [
            DocumentType.INVOICE,
            DocumentType.RECEIPT,
        ]
    
    def get_supported_methods(self) -> List[ExtractionMethod]:
        """
        Get the list of supported extraction methods.
        
        Returns:
            List of supported extraction methods
        """
        return list(ExtractionMethod)
