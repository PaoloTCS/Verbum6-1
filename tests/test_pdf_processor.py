import unittest
from pathlib import Path
from pdf_processor import PDFProcessor

class TestPDFProcessor(unittest.TestCase):
    def setUp(self):
        self.pdf_processor = PDFProcessor()
        self.test_pdf = Path("InputDocs/math/SetTheory/set_theory.pdf")
        
    def test_pdf_text_extraction(self):
        """Test basic PDF text extraction"""
        text = self.pdf_processor.extract_text(self.test_pdf)
        self.assertIsNotNone(text)
        self.assertTrue(len(text) > 0)