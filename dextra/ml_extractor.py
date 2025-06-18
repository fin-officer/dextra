"""
ML-based extractors for financial document fields.

This module provides extractors that use machine learning models to extract
fields from financial documents.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import DocumentType, ExtractionResult, FieldExtractor, DocumentExtractor


class MLFieldExtractor(FieldExtractor):
    """Field extractor using transformer-based question answering."""
    
    def __init__(
        self,
        field_name: str,
        questions: List[str],
        model_name: str = "distilbert-base-cased-distilled-squad",
        postprocess_func: Optional[callable] = None
    ):
        """
        Initialize ML field extractor.
        
        Args:
            field_name: Name of the field to extract
            questions: List of questions to ask the model
            model_name: Name of the transformer model to use
            postprocess_func: Optional function to postprocess extracted value
        """
        super().__init__(field_name)
        
        self.questions = questions
        self.model_name = model_name
        self.postprocess_func = postprocess_func
        self.pipeline = None
    
    def _load_pipeline(self):
        """Load the question-answering pipeline if not already loaded."""
        if self.pipeline is None:
            try:
                from transformers import pipeline
                self.pipeline = pipeline("question-answering", model=self.model_name)
            except ImportError:
                raise ImportError(
                    "Transformers library is required for ML extraction. "
                    "Install it with: pip install transformers"
                )
    
    def extract_field(self, text: str, **kwargs) -> Tuple[Any, float]:
        """
        Extract field using question-answering.
        
        Args:
            text: Text to extract from
            **kwargs: Additional extraction parameters
            
        Returns:
            Tuple of (extracted value, confidence score)
        """
        if not text:
            return None, 0.0
        
        # Load pipeline if needed
        self._load_pipeline()
        
        # Limit context length to avoid OOM errors
        max_length = kwargs.get("max_length", 1024)
        if len(text) > max_length:
            text = text[:max_length]
        
        best_answer = None
        best_score = 0.0
        
        # Try each question
        for question in self.questions:
            try:
                result = self.pipeline(question=question, context=text)
                
                if result["score"] > best_score:
                    best_answer = result["answer"]
                    best_score = result["score"]
            except Exception as e:
                # Log error but continue with other questions
                print(f"Error in ML extraction: {str(e)}")
        
        if best_answer and self.postprocess_func:
            best_answer = self.postprocess_func(best_answer)
        
        return best_answer, best_score


class MLInvoiceExtractor(DocumentExtractor):
    """Document extractor for invoices using ML-based field extractors."""
    
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """
        Initialize ML-based invoice extractor.
        
        Args:
            model_name: Name of the transformer model to use
        """
        # Create field extractors
        field_extractors = {
            "invoice_number": MLFieldExtractor(
                "invoice_number",
                [
                    "What is the invoice number?",
                    "What is the invoice ID?",
                    "What is the reference number of this invoice?"
                ],
                model_name
            ),
            "issue_date": MLFieldExtractor(
                "issue_date",
                [
                    "What is the invoice date?",
                    "When was this invoice issued?",
                    "What is the date of this invoice?"
                ],
                model_name
            ),
            "due_date": MLFieldExtractor(
                "due_date",
                [
                    "What is the payment due date?",
                    "When is the payment due?",
                    "What is the due date for this invoice?"
                ],
                model_name
            ),
            "total_amount": MLFieldExtractor(
                "total_amount",
                [
                    "What is the total amount?",
                    "How much is the total?",
                    "What is the total invoice amount?"
                ],
                model_name
            ),
            "seller_name": MLFieldExtractor(
                "seller_name",
                [
                    "Who is the seller?",
                    "What is the name of the company issuing this invoice?",
                    "What is the vendor name?"
                ],
                model_name
            ),
            "buyer_name": MLFieldExtractor(
                "buyer_name",
                [
                    "Who is the buyer?",
                    "What is the name of the customer?",
                    "Who is this invoice addressed to?"
                ],
                model_name
            ),
        }
        
        super().__init__(
            document_type=DocumentType.INVOICE,
            field_extractors=field_extractors
        )


class MLReceiptExtractor(DocumentExtractor):
    """Document extractor for receipts using ML-based field extractors."""
    
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """
        Initialize ML-based receipt extractor.
        
        Args:
            model_name: Name of the transformer model to use
        """
        # Create field extractors
        field_extractors = {
            "date": MLFieldExtractor(
                "date",
                [
                    "What is the receipt date?",
                    "When was this receipt issued?",
                    "What is the date of this receipt?"
                ],
                model_name
            ),
            "total_amount": MLFieldExtractor(
                "total_amount",
                [
                    "What is the total amount?",
                    "How much is the total?",
                    "What is the total receipt amount?"
                ],
                model_name
            ),
            "merchant_name": MLFieldExtractor(
                "merchant_name",
                [
                    "What is the merchant name?",
                    "What is the name of the store?",
                    "Where was this purchase made?"
                ],
                model_name
            ),
        }
        
        super().__init__(
            document_type=DocumentType.RECEIPT,
            field_extractors=field_extractors
        )


class MLExtractorFactory:
    """Factory for creating ML-based extractors."""
    
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """
        Initialize ML extractor factory.
        
        Args:
            model_name: Name of the transformer model to use
        """
        self.model_name = model_name
    
    def create_extractor(
        self, 
        document_type: Union[DocumentType, str],
        **kwargs
    ) -> DocumentExtractor:
        """
        Create an appropriate ML-based extractor for the given document type.
        
        Args:
            document_type: Type of document to extract
            **kwargs: Additional parameters for the extractor
            
        Returns:
            A document extractor instance
            
        Raises:
            ValueError: If no extractor is available for the document type
        """
        model_name = kwargs.get("model_name", self.model_name)
        
        if isinstance(document_type, str):
            try:
                document_type = DocumentType(document_type.lower())
            except ValueError:
                document_type = DocumentType.UNKNOWN
        
        if document_type == DocumentType.INVOICE:
            return MLInvoiceExtractor(model_name)
        elif document_type == DocumentType.RECEIPT:
            return MLReceiptExtractor(model_name)
        else:
            raise ValueError(f"No ML extractor available for document type: {document_type}")
