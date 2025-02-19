"""
app/core/document_processor.py
Core document processing functionality for Verbum6.
"""

import os
import logging
from typing import Dict, List, Optional, Any
import fitz  # PyMuPDF
from dataclasses import dataclass
from pathlib import Path
import openai
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and hierarchy building."""
    
    def __init__(self, base_path: str):
        """
        Initialize the document processor.
        
        Args:
            base_path (str): Path to the root documents directory
        """
        self.base_path = base_path
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            self.client = openai.OpenAI(api_key=self.openai_api_key)

    def get_top_level_folders(self):
        """Get all top-level folders in InputDocs."""
        return [d for d in os.listdir(self.base_path) 
                if os.path.isdir(os.path.join(self.base_path, d))
                and not d.startswith('.')]

    def get_folder_contents(self, folder_path):
        """Get contents of a folder with hierarchical structure."""
        full_path = os.path.join(self.base_path, folder_path)
        contents = []
        
        try:
            for item in sorted(os.listdir(full_path)):
                if item.startswith('.'):
                    continue
                    
                item_path = os.path.join(full_path, item)
                rel_path = os.path.join(folder_path, item)
                
                if os.path.isdir(item_path):
                    contents.append({
                        "name": item,
                        "type": "folder",
                        "path": rel_path,
                        "children": self.get_folder_contents(rel_path)
                    })
                else:
                    contents.append({
                        "name": item,
                        "type": "document",
                        "path": rel_path
                    })
            
            return contents
            
        except Exception as e:
            logger.error(f"Error processing {folder_path}: {str(e)}")
            return []

    def process_document_query(self, doc_path: str, query: str) -> str:
        """Process a query about a specific document."""
        if not self.openai_api_key:
            return "OpenAI API key not configured"

        try:
            # Extract text from PDF
            full_path = os.path.join(self.base_path, doc_path)
            pdf = PdfReader(full_path)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

            # Create OpenAI query with context using new client
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant explaining concepts from documents."},
                    {"role": "user", "content": f"Based on this document content:\n\n{text[:4000]}...\n\nQuestion: {query}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error processing query: {str(e)}"