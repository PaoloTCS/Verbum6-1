import openai
from typing import Dict, List, Any
import tiktoken

class QuestionAnswerer:
    def __init__(self, model_name: str = "gpt-4"):
        self.client = openai.OpenAI()
        self.model = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def answer_question(self, question: str, chunks: List[str]) -> Dict[str, Any]:
        try:
            # Use only first chunk and ensure we're within limits
            context = chunks[0] if chunks else ""
            prompt = f"Based on this content:\n\n{context}\n\nQuestion: {question}"
            
            # Check total tokens
            total_tokens = len(self.tokenizer.encode(prompt))
            if total_tokens > 4000:
                # Truncate context if needed
                context = context[:int(len(context) * (4000 / total_tokens))]
                prompt = f"Based on this content:\n\n{context}\n\nQuestion: {question}"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant explaining concepts from documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return {
                "answer": response.choices[0].message.content,
                "confidence": 0.9
            }
            
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")