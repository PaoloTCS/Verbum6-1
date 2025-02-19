"""
app/api/routes.py
Routes for the Verbum6 application.
"""

from flask import Blueprint, jsonify, current_app, render_template, send_file, request, send_from_directory
import os
import fitz  # PyMuPDF for PDF processing
from app.core.document_processor import DocumentProcessor
from app.core.semantic_processor import SemanticProcessor

# Create blueprint with template folder specified
api_bp = Blueprint('api', __name__, template_folder='../templates', static_folder='../static')

@api_bp.route('/')
def index():
    """Serve the main visualization page."""
    return render_template('index.html')

@api_bp.route('/api/hierarchy')
def get_hierarchy():
    """Get the document hierarchy for visualization."""
    try:
        processor = DocumentProcessor(current_app.config['UPLOAD_FOLDER'])
        hierarchy = {
            "hierarchy": {
                "name": "root",
                "type": "folder",
                "children": [
                    {
                        "name": folder,
                        "type": "folder",
                        "path": folder,
                        "children": processor.get_folder_contents(folder)
                    } for folder in processor.get_top_level_folders()
                ]
            }
        }
        return jsonify(hierarchy)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/document/<path:filepath>')
def get_document(filepath):
    """Serve document content."""
    try:
        full_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filepath)
        if filepath.lower().endswith('.pdf'):
            return send_file(
                full_path,
                mimetype='application/pdf'
            )
        else:
            # For text-based documents
            with open(full_path, 'r') as f:
                content = f.read()
            return jsonify({
                'content': content,
                'filename': os.path.basename(filepath)
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/document/query', methods=['POST'])
def query_document():
    """Process a query about a specific document."""
    try:
        data = request.json
        doc_path = data.get('path')
        query = data.get('query')
        
        processor = DocumentProcessor(current_app.config['UPLOAD_FOLDER'])
        response = processor.process_document_query(doc_path, query)
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/semantic-distances/level-0')
def get_level_0_distances():
    """Get semantic distances between top-level folders."""
    try:
        processor = SemanticProcessor(current_app.config['UPLOAD_FOLDER'])
        distances = processor.compute_level_0_distances()
        
        # Convert distances to a format suitable for visualization
        distance_data = {
            'nodes': processor._get_top_level_folders(),
            'distances': {f"{k[0]}|{k[1]}": v for k, v in distances.items()}
        }
        return jsonify(distance_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/document/<path:doc_path>')
def serve_document(doc_path):
    """Serve a document file."""
    try:
        full_path = os.path.join(current_app.config['UPLOAD_FOLDER'], doc_path)
        current_app.logger.info(f"Attempting to serve: {full_path}")
        
        if not os.path.exists(full_path):
            current_app.logger.error(f"File not found: {full_path}")
            return jsonify({'error': 'File not found'}), 404
            
        # Force download for debugging
        response = send_file(
            full_path,
            mimetype='application/pdf',
            as_attachment=True,  # Changed to True for testing
            download_name=os.path.basename(doc_path)
        )
        
        # Add debugging headers
        response.headers.update({
            'Content-Type': 'application/pdf',
            'Content-Disposition': f'attachment; filename="{os.path.basename(doc_path)}"',
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        })
        
        return response
        
    except Exception as e:
        current_app.logger.error(f"Error serving document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(current_app.root_path, 'static'),
        'favicon.ico', 
        mimetype='image/vnd.microsoft.icon'
    )