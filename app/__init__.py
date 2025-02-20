"""
app/__init__.py
Application factory for Verbum6.
"""

import os
from flask import Flask
from flask_cors import CORS
from app.api.routes import api_bp
from dotenv import load_dotenv

load_dotenv()

def create_app(env):
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    CORS(app)
    
    # Configuration
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-key-change-in-production'),
        UPLOAD_FOLDER=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'InputDocs'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        TEMPLATES_AUTO_RELOAD=True if env == 'development' else False,
        MODEL_NAME=os.getenv('MODEL_NAME', 'distilbert-base-cased-distilled-squad'),
        DEVICE=os.getenv('DEVICE', 'mps' if os.path.exists('/dev/mps0') else 'cpu'),
        MAX_CHUNK_SIZE=int(os.getenv('MAX_CHUNK_SIZE', 512)),
        MIN_CONFIDENCE=float(os.getenv('MIN_CONFIDENCE', 0.3))
    )
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/')
    
    # Import and register Q&A routes
    from app.routes import qa_bp
    app.register_blueprint(qa_bp, url_prefix='/api')
    
    return app