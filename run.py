"""
run.py
Entry point for the Verbum6 Knowledge Landscape application.
"""

import os
import sys
import logging
import argparse
from app import create_app

# Configure basic logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_port():
    """Get and validate port number from environment."""
    try:
        port = int(os.getenv('PORT', 5001))
        if not (1024 <= port <= 65535):
            logger.warning(f"Port {port} out of range, using default 5001")
            return 5001
        return port
    except ValueError:
        logger.error("Invalid PORT environment variable")
        return 5001

def get_environment():
    """Get and validate environment configuration."""
    env = os.getenv('FLASK_ENV', 'development')
    valid_environments = ['development', 'testing', 'production']
    if env not in valid_environments:
        logger.warning(f"Invalid environment '{env}' specified, defaulting to development")
        return 'development'
    return env

def main():
    # Get environment and port settings
    env = get_environment()
    port = get_port()
    
    # Create and configure the application instance
    app = create_app(env)
    app.run(host='0.0.0.0', port=port, debug=(env == 'development'))

if __name__ == '__main__':
    main()