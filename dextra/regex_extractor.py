"""
Regex-based extractors for financial document fields.

This module provides extractors that use regular expressions to extract
common fields from financial documents.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union

from invutil.date_utils import parse_date_multilingual
from invutil.numeric_utils import parse_amount

from .base import DocumentType, ExtractionResult, FieldExtractor, DocumentExtractor


class RegexFieldExtractor(FieldExtractor):
    """Field extractor using regular expressions."""
    
    def __init__(
        self,
        field_name: str,
        patterns: List[Union[str, Pattern]],
        group_idx: int = 1,
        preprocess_func: Optional[callable] = None,
        postprocess_func: Optional[callable] = None
    ):
        """
        Initialize regex field extractor.
        
        Args:
            field_name: Name of the field to extract
            patterns: List of regex patterns to try
            group_idx: Capture group index to extract (default: 1)
            preprocess_func: Optional function to preprocess text before matching
            postprocess_func: Optional function to postprocess extracted value
        """
        super().__init__(field_name)
        
        # Compile patterns if they are strings
        self.patterns = []
        for pattern in patterns:
            if isinstance(pattern, str):
                self.patterns.append(re.compile(pattern, re.IGNORECASE | re.MULTILINE))
            else:
                self.patterns.append(pattern)
        
        self.group_idx = group_idx
        self.preprocess_func = preprocess_func
        self.postprocess_func = postprocess_func
    
    def extract_field(self, text: str, **kwargs) -> Tuple[Any, float]:
        """
        Extract field using regex patterns.
        
        Args:
            text: Text to extract from
            **kwargs: Additional extraction parameters
            
        Returns:
            Tuple of (extracted value, confidence score)
        """
        if not text:
            return None, 0.0
        
        # Preprocess text if needed
        if self.preprocess_func:
            text = self.preprocess_func(text)
        
        # Try each pattern
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                value = match.group(self.group_idx).strip()
                
                # Postprocess value if needed
                if self.postprocess_func:
                    value = self.postprocess_func(value)
                
                # Calculate confidence based on match length and pattern index
                confidence = min(0.9, 0.5 + (len(value) / 100))
                
                # Reduce confidence for later patterns
                pattern_idx = self.patterns.index(pattern)
                if pattern_idx > 0:
                    confidence *= (1.0 - (pattern_idx * 0.1))
                
                return value, confidence
        
        return None, 0.0


class AmountExtractor(RegexFieldExtractor):
    """Extractor for monetary amounts."""
    
    def __init__(self, field_name: str, currency_symbol: Optional[str] = None):
        """
        Initialize amount extractor.
        
        Args:
            field_name: Name of the amount field
            currency_symbol: Optional currency symbol to include in patterns
        """
        # Define patterns for amounts
        patterns = []
        
        # With currency symbol if provided
        if currency_symbol:
            patterns.extend([
                rf"{currency_symbol}\s*(\d+[.,]\d+)",
                rf"{currency_symbol}\s*([\d,]+\.\d+)",
                rf"{currency_symbol}\s*([\d.]+,\d+)",
                rf"{currency_symbol}\s*(\d+)",
            ])
        
        # General amount patterns
        patterns.extend([
            r"(?:total|amount|sum|price)(?:\s*:|\s+is|\s+of|\s+)[\$€£]?\s*(\d+[.,]\d+)",
            r"(?:total|amount|sum|price)(?:\s*:|\s+is|\s+of|\s+)[\$€£]?\s*([\d,]+\.\d+)",
            r"(?:total|amount|sum|price)(?:\s*:|\s+is|\s+of|\s+)[\$€£]?\s*([\d.]+,\d+)",
            r"(?:total|amount|sum|price)(?:\s*:|\s+is|\s+of|\s+)[\$€£]?\s*(\d+)",
            r"[\$€£]\s*(\d+[.,]\d+)",
            r"[\$€£]\s*([\d,]+\.\d+)",
            r"[\$€£]\s*([\d.]+,\d+)",
            r"(\d+[.,]\d+)(?:\s*[\$€£]|\s+(?:dollars|euros|pounds))",
            r"([\d,]+\.\d+)(?:\s*[\$€£]|\s+(?:dollars|euros|pounds))",
            r"([\d.]+,\d+)(?:\s*[\$€£]|\s+(?:dollars|euros|pounds))",
        ])
        
        super().__init__(
            field_name=field_name,
            patterns=patterns,
            postprocess_func=parse_amount
        )


class DateExtractor(RegexFieldExtractor):
    """Extractor for dates."""
    
    def __init__(self, field_name: str, languages: Optional[List[str]] = None):
        """
        Initialize date extractor.
        
        Args:
            field_name: Name of the date field
            languages: List of language codes for date parsing
        """
        # Define patterns for dates
        patterns = [
            # Date patterns with labels
            r"(?:date|issued|invoice date)(?:\s*:|\s+is|\s+of|\s+)(?:\s*)([\d/.-]+)",
            r"(?:date|issued|invoice date)(?:\s*:|\s+is|\s+of|\s+)(?:\s*)(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})",
            r"(?:date|issued|invoice date)(?:\s*:|\s+is|\s+of|\s+)(?:\s*)(\d{4}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2})",
            
            # Common date formats
            r"\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\b",
            r"\b(\d{4}[/.-]\d{1,2}[/.-]\d{1,2})\b",
            r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
            r"\b(\d{4}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2})\b",
        ]
        
        self.languages = languages or ["en", "de", "fr", "pl", "es"]
        
        super().__init__(
            field_name=field_name,
            patterns=patterns,
            postprocess_func=self._parse_date
        )
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string using multilingual parsing.
        
        Args:
            date_str: String representation of a date
            
        Returns:
            Parsed datetime object or None if parsing fails
        """
        return parse_date_multilingual(date_str, self.languages)


class TextExtractor(RegexFieldExtractor):
    """Extractor for text fields."""
    
    def __init__(self, field_name: str, label: str, max_words: int = 10):
        """
        Initialize text extractor.
        
        Args:
            field_name: Name of the text field
            label: Label text to search for
            max_words: Maximum number of words to extract
        """
        # Define patterns for text fields
        word_pattern = r"[\w\d]+"
        words_pattern = rf"({word_pattern}(?:\s+{word_pattern}){{{max_words-1}}})"
        
        patterns = [
            rf"{re.escape(label)}\s*:\s*{words_pattern}",
            rf"{re.escape(label)}\s+{words_pattern}",
            rf"{re.escape(label.upper())}\s*:\s*{words_pattern}",
            rf"{re.escape(label.upper())}\s+{words_pattern}",
        ]
        
        super().__init__(
            field_name=field_name,
            patterns=patterns
        )


class InvoiceNumberExtractor(RegexFieldExtractor):
    """Specialized extractor for invoice numbers."""
    
    def __init__(self):
        """Initialize invoice number extractor."""
        patterns = [
            r"(?:invoice|inv|invoice number|inv number|invoice no|inv no|invoice #|inv #)(?:\s*:|\s+is|\s+)(?:\s*)([A-Za-z0-9-_/]+)",
            r"(?:invoice|inv|invoice number|inv number|invoice no|inv no|invoice #|inv #)(?:\s*:|\s+is|\s+)(?:\s*)#?\s*([A-Za-z0-9-_/]+)",
            r"#\s*([A-Za-z0-9-_/]+)(?:\s+invoice|\s+inv)",
            r"(?:invoice|inv)[:\s#]+([A-Za-z0-9-_/]+)",
        ]
        
        super().__init__(
            field_name="invoice_number",
            patterns=patterns
        )


class TaxIDExtractor(RegexFieldExtractor):
    """Specialized extractor for tax identification numbers."""
    
    def __init__(self, field_name: str = "tax_id"):
        """
        Initialize tax ID extractor.
        
        Args:
            field_name: Name of the tax ID field
        """
        patterns = [
            # VAT ID patterns
            r"(?:vat|vat number|vat id|vat identification number)(?:\s*:|\s+is|\s+)(?:\s*)([A-Z]{2}[0-9A-Za-z]{2,12})",
            r"(?:vat|vat number|vat id|vat identification number)(?:\s*:|\s+is|\s+)(?:\s*)([0-9]{2,15})",
            
            # Tax ID patterns
            r"(?:tax id|tax number|tax identification number)(?:\s*:|\s+is|\s+)(?:\s*)([A-Z0-9-]{5,20})",
            r"(?:tax id|tax number|tax identification number)(?:\s*:|\s+is|\s+)(?:\s*)([0-9]{2,15})",
            
            # NIP (Poland)
            r"(?:nip)(?:\s*:|\s+is|\s+)(?:\s*)([0-9]{10})",
            r"(?:nip)(?:\s*:|\s+is|\s+)(?:\s*)([0-9]{3}-[0-9]{3}-[0-9]{2}-[0-9]{2})",
            r"(?:nip)(?:\s*:|\s+is|\s+)(?:\s*)([0-9]{3}-[0-9]{2}-[0-9]{2}-[0-9]{3})",
        ]
        
        super().__init__(
            field_name=field_name,
            patterns=patterns
        )


class RegexInvoiceExtractor(DocumentExtractor):
    """Document extractor for invoices using regex-based field extractors."""
    
    def __init__(self):
        """Initialize regex-based invoice extractor."""
        # Create field extractors
        field_extractors = {
            "invoice_number": InvoiceNumberExtractor(),
            "issue_date": DateExtractor("issue_date"),
            "due_date": DateExtractor("due_date"),
            "total_amount": AmountExtractor("total_amount"),
            "tax_amount": AmountExtractor("tax_amount"),
            "net_amount": AmountExtractor("net_amount"),
            "seller_name": TextExtractor("seller_name", "seller"),
            "buyer_name": TextExtractor("buyer_name", "buyer"),
            "seller_tax_id": TaxIDExtractor("seller_tax_id"),
            "buyer_tax_id": TaxIDExtractor("buyer_tax_id"),
        }
        
        super().__init__(
            document_type=DocumentType.INVOICE,
            field_extractors=field_extractors
        )


class RegexReceiptExtractor(DocumentExtractor):
    """Document extractor for receipts using regex-based field extractors."""
    
    def __init__(self):
        """Initialize regex-based receipt extractor."""
        # Create field extractors
        field_extractors = {
            "receipt_number": RegexFieldExtractor(
                "receipt_number",
                [
                    r"(?:receipt|rcpt|receipt number|rcpt number|receipt no|rcpt no|receipt #|rcpt #)(?:\s*:|\s+is|\s+)(?:\s*)([A-Za-z0-9-_/]+)",
                    r"(?:receipt|rcpt|receipt number|rcpt number|receipt no|rcpt no|receipt #|rcpt #)(?:\s*:|\s+is|\s+)(?:\s*)#?\s*([A-Za-z0-9-_/]+)",
                    r"#\s*([A-Za-z0-9-_/]+)(?:\s+receipt|\s+rcpt)",
                    r"(?:receipt|rcpt)[:\s#]+([A-Za-z0-9-_/]+)",
                ]
            ),
            "date": DateExtractor("date"),
            "total_amount": AmountExtractor("total_amount"),
            "tax_amount": AmountExtractor("tax_amount"),
            "merchant_name": TextExtractor("merchant_name", "merchant"),
        }
        
        super().__init__(
            document_type=DocumentType.RECEIPT,
            field_extractors=field_extractors
        )


class RegexExtractorFactory:
    """Factory for creating regex-based extractors."""
    
    def create_extractor(
        self, 
        document_type: Union[DocumentType, str],
        **kwargs
    ) -> DocumentExtractor:
        """
        Create an appropriate regex-based extractor for the given document type.
        
        Args:
            document_type: Type of document to extract
            **kwargs: Additional parameters for the extractor
            
        Returns:
            A document extractor instance
            
        Raises:
            ValueError: If no extractor is available for the document type
        """
        if isinstance(document_type, str):
            try:
                document_type = DocumentType(document_type.lower())
            except ValueError:
                document_type = DocumentType.UNKNOWN
        
        if document_type == DocumentType.INVOICE:
            return RegexInvoiceExtractor()
        elif document_type == DocumentType.RECEIPT:
            return RegexReceiptExtractor()
        else:
            raise ValueError(f"No regex extractor available for document type: {document_type}")
