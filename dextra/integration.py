"""
Integration with document detection and workflow components.

This module provides integration between document detection (dotect)
and extraction (dextra) components to create a complete workflow.
"""

from typing import Dict, Optional, Tuple, Union

from dotect import DetectorFactory, DetectionMethod, DocumentType
from invutil.logger import get_logger

from .base import ExtractionResult
from .extractor_factory import ExtractionMethod, UnifiedExtractorFactory


logger = get_logger(__name__)


class ExtractionWorkflow:
    """
    Workflow for document detection and extraction.
    
    This class integrates document detection and extraction components
    to provide a complete workflow for processing financial documents.
    """
    
    def __init__(
        self,
        detection_method: Union[DetectionMethod, str] = DetectionMethod.HYBRID,
        extraction_method: Union[ExtractionMethod, str] = ExtractionMethod.HYBRID,
        ml_model_path: Optional[str] = None
    ):
        """
        Initialize extraction workflow.
        
        Args:
            detection_method: Method to use for document detection
            extraction_method: Method to use for data extraction
            ml_model_path: Path to pre-trained ML model for detection
        """
        self.detector_factory = DetectorFactory(
            preferred_method=detection_method,
            ml_model_path=ml_model_path
        )
        
        self.extractor_factory = UnifiedExtractorFactory(
            preferred_method=extraction_method
        )
        
        logger.info(
            f"Initialized extraction workflow with "
            f"detection_method={detection_method}, "
            f"extraction_method={extraction_method}"
        )
    
    def process_document(
        self,
        text: str,
        document_type: Optional[Union[DocumentType, str]] = None,
        **kwargs
    ) -> Tuple[ExtractionResult, float]:
        """
        Process a document by detecting its type and extracting data.
        
        Args:
            text: Document text to process
            document_type: Optional document type (if already known)
            **kwargs: Additional processing parameters
            
        Returns:
            Tuple of (extraction_result, confidence)
        """
        # Detect document type if not provided
        detection_confidence = 1.0
        if document_type is None:
            detector = self.detector_factory.create_detector()
            detection_result = detector.detect(text, **kwargs)
            document_type = detection_result.document_type
            detection_confidence = detection_result.confidence
            
            logger.info(
                f"Detected document type: {document_type} "
                f"(confidence: {detection_confidence:.2f})"
            )
        
        # Create appropriate extractor
        extractor = self.extractor_factory.create_extractor(
            document_type=document_type,
            **kwargs
        )
        
        # Extract data
        extraction_result = extractor.extract(text, **kwargs)
        
        # Adjust overall confidence based on detection and extraction
        overall_confidence = detection_confidence * extraction_result.confidence
        
        logger.info(
            f"Extracted {len(extraction_result.fields)} fields "
            f"(confidence: {extraction_result.confidence:.2f})"
        )
        
        return extraction_result, overall_confidence
    
    def batch_process(
        self,
        documents: Dict[str, str],
        **kwargs
    ) -> Dict[str, Tuple[ExtractionResult, float]]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: Dictionary mapping document IDs to document text
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary mapping document IDs to (extraction_result, confidence) tuples
        """
        results = {}
        
        for doc_id, text in documents.items():
            try:
                results[doc_id] = self.process_document(text, **kwargs)
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {str(e)}")
                # Return empty result with zero confidence
                results[doc_id] = (
                    ExtractionResult(document_type=DocumentType.UNKNOWN),
                    0.0
                )
        
        return results
