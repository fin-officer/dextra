"""
Unit tests for the regex extractor module.
"""

import unittest
from datetime import datetime

from dextra.base import DocumentType
from dextra.regex_extractor import (
    RegexFieldExtractor, 
    AmountExtractor, 
    DateExtractor,
    TextExtractor,
    InvoiceNumberExtractor,
    TaxIDExtractor,
    RegexInvoiceExtractor,
    RegexReceiptExtractor,
    RegexExtractorFactory
)


class TestRegexFieldExtractor(unittest.TestCase):
    """Test cases for the RegexFieldExtractor class."""
    
    def test_basic_extraction(self):
        """Test basic field extraction with simple patterns."""
        extractor = RegexFieldExtractor(
            field_name="test_field",
            patterns=[r"test:\s*(\w+)"]
        )
        
        # Test successful extraction
        value, confidence = extractor.extract_field("This is a test: value123")
        self.assertEqual(value, "value123")
        self.assertGreater(confidence, 0.0)
        
        # Test failed extraction
        value, confidence = extractor.extract_field("This has no match")
        self.assertIsNone(value)
        self.assertEqual(confidence, 0.0)
    
    def test_multiple_patterns(self):
        """Test extraction with multiple patterns."""
        extractor = RegexFieldExtractor(
            field_name="test_field",
            patterns=[
                r"pattern1:\s*(\w+)",
                r"pattern2:\s*(\w+)"
            ]
        )
        
        # Test first pattern
        value, confidence1 = extractor.extract_field("This is pattern1: value1")
        self.assertEqual(value, "value1")
        
        # Test second pattern
        value, confidence2 = extractor.extract_field("This is pattern2: value2")
        self.assertEqual(value, "value2")
        
        # Second pattern should have lower confidence
        self.assertLess(confidence2, confidence1)
    
    def test_preprocessing(self):
        """Test extraction with preprocessing function."""
        def preprocess(text):
            return text.lower()
        
        extractor = RegexFieldExtractor(
            field_name="test_field",
            patterns=[r"test:\s*(\w+)"],
            preprocess_func=preprocess
        )
        
        # Test with uppercase text
        value, confidence = extractor.extract_field("This is a TEST: value123")
        self.assertEqual(value, "value123")
    
    def test_postprocessing(self):
        """Test extraction with postprocessing function."""
        def postprocess(value):
            return value.upper()
        
        extractor = RegexFieldExtractor(
            field_name="test_field",
            patterns=[r"test:\s*(\w+)"],
            postprocess_func=postprocess
        )
        
        # Test with postprocessing
        value, confidence = extractor.extract_field("This is a test: value123")
        self.assertEqual(value, "VALUE123")


class TestAmountExtractor(unittest.TestCase):
    """Test cases for the AmountExtractor class."""
    
    def test_amount_extraction(self):
        """Test extraction of monetary amounts."""
        extractor = AmountExtractor(field_name="amount")
        
        # Test with currency symbol
        value, confidence = extractor.extract_field("Total amount: $123.45")
        self.assertEqual(value, 123.45)
        self.assertGreater(confidence, 0.0)
        
        # Test with different format
        value, confidence = extractor.extract_field("Amount is €1,234.56")
        self.assertEqual(value, 1234.56)
        
        # Test with comma as decimal separator
        value, confidence = extractor.extract_field("Price: 98,76€")
        self.assertEqual(value, 98.76)
    
    def test_specific_currency(self):
        """Test extraction with specific currency symbol."""
        extractor = AmountExtractor(field_name="amount", currency_symbol="€")
        
        # Test with matching currency
        value, confidence = extractor.extract_field("Total: € 123.45")
        self.assertEqual(value, 123.45)
        
        # Test with non-matching currency
        value, confidence = extractor.extract_field("Total: $ 123.45")
        self.assertIsNone(value)  # Should not match $ when looking for €


class TestDateExtractor(unittest.TestCase):
    """Test cases for the DateExtractor class."""
    
    def test_date_extraction(self):
        """Test extraction of dates."""
        extractor = DateExtractor(field_name="date")
        
        # Test common date format
        value, confidence = extractor.extract_field("Invoice date: 01/15/2023")
        self.assertIsInstance(value, datetime)
        self.assertEqual(value.year, 2023)
        self.assertEqual(value.month, 1)
        self.assertEqual(value.day, 15)
        
        # Test another format
        value, confidence = extractor.extract_field("Date: 2023-06-30")
        self.assertEqual(value.year, 2023)
        self.assertEqual(value.month, 6)
        self.assertEqual(value.day, 30)
        
        # Test text month format
        value, confidence = extractor.extract_field("Issued on 15 Jan 2023")
        self.assertEqual(value.year, 2023)
        self.assertEqual(value.month, 1)
        self.assertEqual(value.day, 15)


class TestInvoiceNumberExtractor(unittest.TestCase):
    """Test cases for the InvoiceNumberExtractor class."""
    
    def test_invoice_number_extraction(self):
        """Test extraction of invoice numbers."""
        extractor = InvoiceNumberExtractor()
        
        # Test basic invoice number
        value, confidence = extractor.extract_field("Invoice Number: INV-2023-001")
        self.assertEqual(value, "INV-2023-001")
        
        # Test with hash symbol
        value, confidence = extractor.extract_field("Invoice #A12345")
        self.assertEqual(value, "A12345")
        
        # Test abbreviated format
        value, confidence = extractor.extract_field("Inv No: XYZ/789")
        self.assertEqual(value, "XYZ/789")


class TestRegexInvoiceExtractor(unittest.TestCase):
    """Test cases for the RegexInvoiceExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = RegexInvoiceExtractor()
        self.sample_text = """
        INVOICE
        
        Invoice Number: INV-2023-001
        Issue Date: 01/15/2023
        Due Date: 02/15/2023
        
        Seller: ABC Company
        Seller Tax ID: VAT123456789
        
        Buyer: XYZ Corporation
        Buyer Tax ID: TAX987654321
        
        Net Amount: $900.00
        Tax Amount: $100.00
        Total Amount: $1,000.00
        """
    
    def test_document_type(self):
        """Test document type identification."""
        self.assertEqual(self.extractor.document_type, DocumentType.INVOICE)
    
    def test_extract_fields(self):
        """Test extraction of all invoice fields."""
        result = self.extractor.extract(self.sample_text)
        
        # Check result properties
        self.assertEqual(result.document_type, DocumentType.INVOICE)
        self.assertGreater(result.confidence, 0.0)
        
        # Check extracted fields
        self.assertEqual(result.fields["invoice_number"], "INV-2023-001")
        self.assertIsInstance(result.fields["issue_date"], datetime)
        self.assertEqual(result.fields["issue_date"].year, 2023)
        self.assertEqual(result.fields["issue_date"].month, 1)
        self.assertEqual(result.fields["issue_date"].day, 15)
        self.assertEqual(result.fields["total_amount"], 1000.00)
        self.assertEqual(result.fields["tax_amount"], 100.00)
        self.assertEqual(result.fields["net_amount"], 900.00)
        self.assertEqual(result.fields["seller_name"], "ABC Company")
        self.assertEqual(result.fields["buyer_name"], "XYZ Corporation")
        self.assertEqual(result.fields["seller_tax_id"], "VAT123456789")
        self.assertEqual(result.fields["buyer_tax_id"], "TAX987654321")


class TestRegexExtractorFactory(unittest.TestCase):
    """Test cases for the RegexExtractorFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = RegexExtractorFactory()
    
    def test_create_invoice_extractor(self):
        """Test creation of invoice extractor."""
        extractor = self.factory.create_extractor(DocumentType.INVOICE)
        self.assertIsInstance(extractor, RegexInvoiceExtractor)
        self.assertEqual(extractor.document_type, DocumentType.INVOICE)
    
    def test_create_receipt_extractor(self):
        """Test creation of receipt extractor."""
        extractor = self.factory.create_extractor(DocumentType.RECEIPT)
        self.assertIsInstance(extractor, RegexReceiptExtractor)
        self.assertEqual(extractor.document_type, DocumentType.RECEIPT)
    
    def test_create_with_string_type(self):
        """Test creation with string document type."""
        extractor = self.factory.create_extractor("invoice")
        self.assertIsInstance(extractor, RegexInvoiceExtractor)
        self.assertEqual(extractor.document_type, DocumentType.INVOICE)
    
    def test_unsupported_type(self):
        """Test error handling for unsupported document type."""
        with self.assertRaises(ValueError):
            self.factory.create_extractor(DocumentType.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
