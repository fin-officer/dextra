"""
Unit tests for the ML extractor module.
"""

import unittest
from unittest.mock import patch, MagicMock

from dextra.base import DocumentType
from dextra.ml_extractor import (
    MLFieldExtractor,
    MLInvoiceExtractor,
    MLReceiptExtractor,
    MLExtractorFactory
)


class TestMLFieldExtractor(unittest.TestCase):
    """Test cases for the MLFieldExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock pipeline for testing
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.return_value = {
            "answer": "test answer",
            "score": 0.85
        }
    
    @patch("transformers.pipeline")
    def test_basic_extraction(self, mock_pipeline_func):
        """Test basic field extraction with QA model."""
        # Set up mock
        mock_pipeline_func.return_value = self.mock_pipeline
        
        # Create extractor
        extractor = MLFieldExtractor(
            field_name="test_field",
            questions=["What is the test value?"]
        )
        
        # Test extraction
        value, confidence = extractor.extract_field("This is a test document.")
        
        # Verify pipeline was called correctly
        mock_pipeline_func.assert_called_once()
        self.mock_pipeline.assert_called_once()
        
        # Verify results
        self.assertEqual(value, "test answer")
        self.assertEqual(confidence, 0.85)
    
    @patch("transformers.pipeline")
    def test_multiple_questions(self, mock_pipeline_func):
        """Test extraction with multiple questions."""
        # Set up mock with different responses for different questions
        mock_pipeline = MagicMock(side_effect=[
            {"answer": "answer1", "score": 0.7},
            {"answer": "answer2", "score": 0.9}
        ])
        mock_pipeline_func.return_value = mock_pipeline
        
        # Create extractor
        extractor = MLFieldExtractor(
            field_name="test_field",
            questions=["Question 1?", "Question 2?"]
        )
        
        # Test extraction
        value, confidence = extractor.extract_field("This is a test document.")
        
        # Verify pipeline was called twice (once for each question)
        self.assertEqual(mock_pipeline.call_count, 2)
        
        # Verify results (should use answer2 since it has higher score)
        self.assertEqual(value, "answer2")
        self.assertEqual(confidence, 0.9)
    
    @patch("transformers.pipeline")
    def test_postprocessing(self, mock_pipeline_func):
        """Test extraction with postprocessing function."""
        # Set up mock
        mock_pipeline_func.return_value = self.mock_pipeline
        
        # Create postprocessing function
        def postprocess(value):
            return value.upper()
        
        # Create extractor
        extractor = MLFieldExtractor(
            field_name="test_field",
            questions=["What is the test value?"],
            postprocess_func=postprocess
        )
        
        # Test extraction
        value, confidence = extractor.extract_field("This is a test document.")
        
        # Verify results
        self.assertEqual(value, "TEST ANSWER")
        self.assertEqual(confidence, 0.85)
    
    @patch("transformers.pipeline")
    def test_error_handling(self, mock_pipeline_func):
        """Test error handling during extraction."""
        # Set up mock to raise an exception
        mock_pipeline = MagicMock(side_effect=Exception("Test error"))
        mock_pipeline_func.return_value = mock_pipeline
        
        # Create extractor
        extractor = MLFieldExtractor(
            field_name="test_field",
            questions=["What is the test value?"]
        )
        
        # Test extraction
        value, confidence = extractor.extract_field("This is a test document.")
        
        # Verify results (should return None with zero confidence)
        self.assertIsNone(value)
        self.assertEqual(confidence, 0.0)


class TestMLInvoiceExtractor(unittest.TestCase):
    """Test cases for the MLInvoiceExtractor class."""
    
    @patch.object(MLFieldExtractor, "extract_field")
    def test_extract_fields(self, mock_extract_field):
        """Test extraction of invoice fields."""
        # Set up mock to return different values for different fields
        mock_extract_field.side_effect = [
            ("INV-2023-001", 0.9),  # invoice_number
            ("2023-01-15", 0.8),    # issue_date
            ("2023-02-15", 0.7),    # due_date
            (1000.0, 0.85),         # total_amount
            ("ABC Company", 0.95),  # seller_name
            ("XYZ Corp", 0.8)       # buyer_name
        ]
        
        # Create extractor
        extractor = MLInvoiceExtractor()
        
        # Test extraction
        result = extractor.extract("This is a test invoice.")
        
        # Verify document type
        self.assertEqual(result.document_type, DocumentType.INVOICE)
        
        # Verify extracted fields
        self.assertEqual(result.fields["invoice_number"], "INV-2023-001")
        self.assertEqual(result.fields["issue_date"], "2023-01-15")
        self.assertEqual(result.fields["due_date"], "2023-02-15")
        self.assertEqual(result.fields["total_amount"], 1000.0)
        self.assertEqual(result.fields["seller_name"], "ABC Company")
        self.assertEqual(result.fields["buyer_name"], "XYZ Corp")
        
        # Verify confidence (should be average of field confidences)
        expected_confidence = (0.9 + 0.8 + 0.7 + 0.85 + 0.95 + 0.8) / 6
        self.assertAlmostEqual(result.confidence, expected_confidence)


class TestMLExtractorFactory(unittest.TestCase):
    """Test cases for the MLExtractorFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = MLExtractorFactory()
    
    def test_create_invoice_extractor(self):
        """Test creation of invoice extractor."""
        extractor = self.factory.create_extractor(DocumentType.INVOICE)
        self.assertIsInstance(extractor, MLInvoiceExtractor)
        self.assertEqual(extractor.document_type, DocumentType.INVOICE)
    
    def test_create_receipt_extractor(self):
        """Test creation of receipt extractor."""
        extractor = self.factory.create_extractor(DocumentType.RECEIPT)
        self.assertIsInstance(extractor, MLReceiptExtractor)
        self.assertEqual(extractor.document_type, DocumentType.RECEIPT)
    
    def test_create_with_string_type(self):
        """Test creation with string document type."""
        extractor = self.factory.create_extractor("invoice")
        self.assertIsInstance(extractor, MLInvoiceExtractor)
        self.assertEqual(extractor.document_type, DocumentType.INVOICE)
    
    def test_create_with_model_name(self):
        """Test creation with custom model name."""
        factory = MLExtractorFactory(model_name="custom-model")
        extractor = factory.create_extractor(DocumentType.INVOICE)
        
        # Check that the model name was passed to the extractor
        for field_extractor in extractor.field_extractors.values():
            self.assertEqual(field_extractor.model_name, "custom-model")
    
    def test_unsupported_type(self):
        """Test error handling for unsupported document type."""
        with self.assertRaises(ValueError):
            self.factory.create_extractor(DocumentType.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
