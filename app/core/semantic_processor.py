"""
app/core/semantic_processor.py
"""

import os
import numpy as np
import logging
from openai import OpenAI
from PyPDF2 import PdfReader
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticProcessor:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found in environment variables")
        self.client = OpenAI(api_key=self.openai_api_key)
        self.embeddings_cache = {}
    
    def compute_level_0_distances(self) -> Dict[Tuple[str, str], float]:
        """Compute semantic distances between top-level folders."""
        try:
            distances = {}
            top_folders = self._get_top_level_folders()
            logger.info(f"Processing {len(top_folders)} top-level folders")
            
            # Get embeddings for each top-level folder
            folder_embeddings = {}
            for folder in top_folders:
                embedding = self._get_folder_embedding(folder)
                if embedding is not None:
                    folder_embeddings[folder] = embedding
                else:
                    logger.warning(f"Could not generate embedding for folder: {folder}")
            
            # Compute distances between all folder pairs
            for i, folder1 in enumerate(top_folders):
                for folder2 in top_folders[i+1:]:
                    if folder1 in folder_embeddings and folder2 in folder_embeddings:
                        distance = self._compute_distance(
                            folder_embeddings[folder1],
                            folder_embeddings[folder2]
                        )
                        distances[(folder1, folder2)] = distance
            
            return distances
            
        except Exception as e:
            logger.error(f"Error computing level 0 distances: {str(e)}")
            return {}
    
    def _get_top_level_folders(self) -> List[str]:
        """Get all top-level folders in the base path."""
        try:
            return [d for d in os.listdir(self.base_path) 
                    if os.path.isdir(os.path.join(self.base_path, d))
                    and not d.startswith('.')]
        except Exception as e:
            logger.error(f"Error getting top-level folders: {str(e)}")
            return []
    
    def _get_folder_embedding(self, folder_path: str) -> Optional[np.ndarray]:
        """Compute aggregate embedding for a folder based on its contents."""
        try:
            if folder_path in self.embeddings_cache:
                return self.embeddings_cache[folder_path]
                
            folder_summary = self._generate_folder_summary(folder_path)
            if not folder_summary:
                return None
                
            embedding = self._get_text_embedding(folder_summary)
            self.embeddings_cache[folder_path] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for {folder_path}: {str(e)}")
            return None
    
    def _generate_folder_summary(self, folder_path: str) -> str:
        """Generate a summary of folder contents for embedding."""
        try:
            content_summary = []
            full_path = os.path.join(self.base_path, folder_path)
            
            # Add folder name with domain context
            content_summary.append(f"Knowledge domain: {folder_path}")
            
            # Add subfolder names as subdomains
            subfolders = [
                d for d in os.listdir(full_path)
                if os.path.isdir(os.path.join(full_path, d))
                and not d.startswith('.')
            ]
            if subfolders:
                content_summary.append(f"Subdomains: {', '.join(subfolders)}")
            
            # Sample document titles for topic inference
            docs = []
            for root, _, files in os.walk(full_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        # Clean and format document names
                        doc_name = os.path.splitext(file)[0]
                        doc_name = doc_name.replace('_', ' ').replace('-', ' ')
                        docs.append(doc_name)
                    if len(docs) >= 5:  # Limit to 5 representative documents
                        break
            if docs:
                content_summary.append(f"Representative topics: {', '.join(docs)}")
            
            return ' '.join(content_summary)
            
        except Exception as e:
            logger.error(f"Error generating summary for {folder_path}: {str(e)}")
            return ""
    
    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text using OpenAI's API."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text[:8191]  # API token limit
            )
            return np.array(response.data[0].embedding)
            
        except Exception as e:
            logger.error(f"Error getting embedding from OpenAI: {str(e)}")
            return None
    
    def _compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute semantic distance between two embeddings."""
        try:
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return 1 - similarity
            
        except Exception as e:
            logger.error(f"Error computing distance: {str(e)}")
            return 1.0  # Maximum distance on error