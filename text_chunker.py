from typing import List, Tuple
import time

class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 100, max_chunks: int = 1000, 
                 batch_size: int = 50):
        """Initialize TextChunker with optimized default parameters
        
        Args:
            chunk_size (int): Target size for each chunk
            overlap (int): Number of characters to overlap between chunks
            max_chunks (int): Maximum number of chunks to create
            batch_size (int): Number of chunks to process in each batch
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_chunks = max_chunks
        self.batch_size = batch_size
        self._stats = {
            'total_chars': 0,
            'total_chunks': 0,
            'processing_time': 0
        }

    def get_stats(self) -> dict:
        """Return processing statistics"""
        return self._stats

    def find_boundary(self, text: str, end: int, start: int) -> int:
        """Find the nearest sentence boundary"""
        for i in range(end, max(end - 50, start), -1):
            if text[i-1] in '.!?\n':
                return i
        return end

    def process_batch(self, text: str, start_positions: List[int], 
                     text_length: int) -> List[Tuple[str, int]]:
        """Process a batch of chunks efficiently"""
        results = []
        
        # Pre-calculate the end position for the batch
        batch_end = min(max(start_positions) + self.chunk_size, text_length)
        text_slice = text[:batch_end]  # Get relevant text slice once
        
        for start in start_positions:
            if start >= text_length:
                continue
                
            end = min(start + self.chunk_size, text_length)
            if end < text_length:
                end = self.find_boundary(text_slice, end, start)
                
            chunk = text_slice[start:end].strip()
            if chunk and len(chunk) >= 100:
                results.append((chunk, end))
        
        return results

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks using batch processing"""
        chunks = []
        text_length = len(text)
        start_time = time.time()
        
        # Calculate batch positions
        starts = list(range(0, text_length, self.chunk_size - self.overlap))
        total_batches = (len(starts) + self.batch_size - 1) // self.batch_size
        
        print(f"Starting chunking (length: {text_length}, estimated batches: {total_batches})")
        
        # Process in batches
        for batch_idx in range(0, len(starts), self.batch_size):
            batch_starts = starts[batch_idx:batch_idx + self.batch_size]
            batch_results = self.process_batch(text, batch_starts, text_length)
            
            chunks.extend(chunk for chunk, _ in batch_results)
            
            if len(chunks) >= self.max_chunks:
                print("Reached maximum chunk limit")
                break
                
            elapsed = time.time() - start_time
            progress = (batch_idx + self.batch_size) / len(starts) * 100
            print(f"Processed batch {batch_idx//self.batch_size + 1}/{total_batches} "
                  f"({progress:.1f}%) in {elapsed:.2f}s - {len(chunks)} chunks")
            
        print(f"Chunking completed. Created {len(chunks)} chunks in {time.time()-start_time:.2f}s")
        return chunks[:self.max_chunks]