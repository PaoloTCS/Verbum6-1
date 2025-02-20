import json
from pathlib import Path
from typing import Dict, Optional
from pypdf import PdfReader

class PDFProcessor:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache: Dict[str, str] = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        """Load cached extractions from disk"""
        cache_file = self.cache_dir / "pdf_extractions.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        cache_file = self.cache_dir / "pdf_extractions.json"
        with open(cache_file, 'w') as f:
            json.dump(self.cache, f)

    def extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF, using cache if available"""
        cache_key = str(pdf_path.absolute())
        
        # Check cache first
        if cache_key in self.cache:
            print(f"Using cached extraction for {pdf_path.name}")
            return self.cache[cache_key]

        # Extract if not in cache
        print(f"Extracting text from {pdf_path.name}")
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
                
        # Cache the result
        self.cache[cache_key] = text
        self._save_cache()
        
        return text