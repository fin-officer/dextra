"""
Base extraction interfaces and abstract classes.

This module provides the core extraction interfaces and abstract base classes
that all extractors should implement.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class ExtractionResult(BaseModel):
    """Result of an extraction operation."""
    
    data: Dict[str, Any] = Field(default_factory=dict)
    """Extracted data as a dictionary of field names to values."""
    
    confidence: Dict[str, float] = Field(default_factory=dict)
    """Confidence scores for each extracted field (0.0-1.0)."""
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Additional metadata about the extraction process."""


class DocumentType(str, Enum):
    """Enumeration of supported document types."""
    
    INVOICE = "invoice"
    RECEIPT = "receipt"
    BANK_STATEMENT = "bank_statement"
    UNKNOWN = "unknown"


class Extractor(ABC):
    """Base extractor interface."""
    
    @abstractmethod
    def extract(self, text: str, **kwargs) -> ExtractionResult:
        """
        Extract data from the provided text.
        
        Args:
            text: Text to extract data from
            **kwargs: Additional extraction parameters
            
        Returns:
            ExtractionResult containing extracted data and metadata
        """
        pass
    
    @property
    def supported_fields(self) -> List[str]:
        """
        Get the list of fields this extractor can extract.
        
        Returns:
            List of field names
        """
        return []


class FieldExtractor(Extractor):
    """Extractor for specific fields."""
    
    def __init__(self, field_name: str):
        """
        Initialize field extractor.
        
        Args:
            field_name: Name of the field this extractor handles
        """
        self.field_name = field_name
    
    @abstractmethod
    def extract_field(self, text: str, **kwargs) -> Tuple[Any, float]:
        """
        Extract a specific field from the text.
        
        Args:
            text: Text to extract from
            **kwargs: Additional extraction parameters
            
        Returns:
            Tuple of (extracted value, confidence score)
        """
        pass
    
    def extract(self, text: str, **kwargs) -> ExtractionResult:
        """
        Extract the field from the text.
        
        Args:
            text: Text to extract from
            **kwargs: Additional extraction parameters
            
        Returns:
            ExtractionResult containing the extracted field
        """
        value, confidence = self.extract_field(text, **kwargs)
        
        return ExtractionResult(
            data={self.field_name: value},
            confidence={self.field_name: confidence},
            metadata={"extractor_type": self.__class__.__name__}
        )
    
    @property
    def supported_fields(self) -> List[str]:
        """
        Get the list of fields this extractor can extract.
        
        Returns:
            List containing the field name
        """
        return [self.field_name]


class DocumentExtractor(Extractor):
    """Extractor for entire documents."""
    
    def __init__(
        self, 
        document_type: DocumentType,
        field_extractors: Optional[Dict[str, FieldExtractor]] = None
    ):
        """
        Initialize document extractor.
        
        Args:
            document_type: Type of document this extractor handles
            field_extractors: Dictionary mapping field names to field extractors
        """
        self.document_type = document_type
        self.field_extractors = field_extractors or {}
    
    def extract(self, text: str, **kwargs) -> ExtractionResult:
        """
        Extract all fields from the document.
        
        Args:
            text: Document text to extract from
            **kwargs: Additional extraction parameters
            
        Returns:
            ExtractionResult containing all extracted fields
        """
        all_data = {}
        all_confidence = {}
        
        # Extract each field using its specialized extractor
        for field_name, extractor in self.field_extractors.items():
            result = extractor.extract(text, **kwargs)
            
            if field_name in result.data:
                all_data[field_name] = result.data[field_name]
                all_confidence[field_name] = result.confidence.get(field_name, 0.0)
        
        return ExtractionResult(
            data=all_data,
            confidence=all_confidence,
            metadata={
                "document_type": self.document_type,
                "extractor_type": self.__class__.__name__
            }
        )
    
    @property
    def supported_fields(self) -> List[str]:
        """
        Get the list of fields this extractor can extract.
        
        Returns:
            List of field names
        """
        return list(self.field_extractors.keys())


class ExtractorFactory(ABC):
    """Factory for creating extractors based on document type."""
    
    @abstractmethod
    def create_extractor(
        self, 
        document_type: Union[DocumentType, str],
        **kwargs
    ) -> Extractor:
        """
        Create an appropriate extractor for the given document type.
        
        Args:
            document_type: Type of document to extract
            **kwargs: Additional parameters for the extractor
            
        Returns:
            An extractor instance
            
        Raises:
            ValueError: If no extractor is available for the document type
        """
        pass
