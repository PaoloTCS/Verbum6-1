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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # Add this import
import tiktoken

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
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = 4000  # Safe limit for GPT-4

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

    def process_document_query(self, doc_path: str, query: str) -> Dict[str, Any]:
        """
        Process a query about a specific document.
        
        Args:
            doc_path: Path to the document relative to base_path
            query: Question to answer about the document
            
        Returns:
            Dict containing answer and confidence score
        """
        if not self.openai_api_key:
            return "OpenAI API key not configured"

        try:
            # Extract text from PDF
            full_path = os.path.join(self.base_path, doc_path)
            text = self.process_document(full_path)
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Get embeddings for chunks and query
            chunk_embeddings = self.embed_text(chunks)
            query_embedding = self.embed_text([query])[0]
            
            # Find most relevant chunks using cosine similarity
            similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
            top_chunk_indices = similarities.argsort()[-3:][::-1]  # Get top 3 chunks
            
            relevant_text = "\n\n".join([chunks[i] for i in top_chunk_indices])
            
            # Create OpenAI query with relevant context
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant explaining concepts from documents."},
                    {"role": "user", "content": f"Based on this relevant document content:\n\n{relevant_text}\n\nQuestion: {query}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"

    def process_document(self, doc_path: str) -> str:
        """Extract text from document"""
        try:
            if not os.path.exists(doc_path):
                raise FileNotFoundError(f"Document not found: {doc_path}")
            
            if doc_path.lower().endswith('.pdf'):
                with fitz.open(doc_path) as doc:
                    return " ".join(page.get_text() for page in doc)
            else:
                raise ValueError(f"Unsupported document type: {doc_path}")
                
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    def chunk_text(self, text: str) -> List[str]:
        """Split text into smaller chunks based on token count"""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in text.split('\n\n'):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Count tokens in this paragraph
            para_tokens = len(self.tokenizer.encode(paragraph))
            
            if current_tokens + para_tokens <= self.max_tokens:
                current_chunk += paragraph + '\n\n'
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
                current_tokens = para_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Take only first chunk to stay within limits
        return chunks[:1] if chunks else []

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for text chunks"""
        if not self.client:
            raise ValueError("OpenAI client not initialized - check API key")
        
        embeddings = []
        try:
            for text in texts:
                if not text.strip():
                    continue
                
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                embeddings.append(response.data[0].embedding)
            
            if not embeddings:
                raise ValueError("No valid text chunks to embed")
            
            return np.array(embeddings)
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise