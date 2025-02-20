import os
from flask import Blueprint, jsonify, request, current_app
from app.core.document_processor import DocumentProcessor
from app.core.question_answerer import QuestionAnswerer

# Create Blueprint for Q&A routes
qa_bp = Blueprint('qa', __name__)

@qa_bp.route('/ask', methods=['POST'])
def ask_question():
    """Handle document Q&A requests"""
    try:
        data = request.get_json()
        current_app.logger.info(f"Received Q&A request: {data}")
        
        if not data or 'question' not in data or 'document' not in data:
            return jsonify({'error': 'Missing required fields'}), 400

        # Get base path and initialize processors
        base_path = current_app.config['UPLOAD_FOLDER']
        doc_path = os.path.join(base_path, data['document'])
        question = data['question']
        
        current_app.logger.info(f"Processing document: {doc_path}")
        
        # Initialize processors
        doc_processor = DocumentProcessor(base_path=base_path)
        qa = QuestionAnswerer()

        try:
            # Extract and process text
            text = doc_processor.process_document(doc_path)
            chunks = doc_processor.chunk_text(text)
            
            # Get answer
            result = qa.answer_question(
                question=question,
                chunks=chunks
            )
            
            current_app.logger.info(f"Answer generated successfully")
            return jsonify(result)

        except FileNotFoundError:
            return jsonify({
                'error': f'Document not found: {data["document"]}'
            }), 404
            
        except Exception as e:
            current_app.logger.error(f"Processing error: {str(e)}")
            return jsonify({
                'error': f'Error processing document: {str(e)}'
            }), 500

    except Exception as e:
        current_app.logger.error(f"Q&A error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Additional routes can be added here as needed
