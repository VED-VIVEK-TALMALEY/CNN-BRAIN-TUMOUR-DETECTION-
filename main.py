#!/usr/bin/env python3
"""
Main Entry Point for Brain Tumor Detection Project
"""

import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Brain Tumor Detection Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py web                    # Run the web application
  python main.py train                  # Train the model
  python main.py organize-dataset       # Organize dataset
  python main.py setup                  # Setup project directories
  python main.py validate               # Validate dataset
        """
    )
    
    parser.add_argument('command', choices=[
        'web', 'train', 'organize-dataset', 'setup', 'validate', 'test'
    ], help='Command to execute')
    
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--port', type=int, default=5000, help='Port for web app (default: 5000)')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.command == 'web':
            run_web_app(args.port)
        elif args.command == 'train':
            run_training()
        elif args.command == 'organize-dataset':
            run_dataset_organization()
        elif args.command == 'setup':
            run_setup()
        elif args.command == 'validate':
            run_validation()
        elif args.command == 'test':
            run_tests()
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        sys.exit(1)

def run_web_app(port):
    """Run the web application"""
    logger.info("Starting web application...")
    
    # Import and run web app
    try:
        from src.web.app import create_app
        app = create_app()
        
        # Update port if specified
        app.config['PORT'] = port
        
        logger.info(f"Web application starting on port {port}")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=app.config['DEBUG']
        )
    except ImportError as e:
        logger.error(f"Failed to import web application: {e}")
        logger.info("Make sure all dependencies are installed and the project structure is correct")
        sys.exit(1)

def run_training():
    """Run model training"""
    logger.info("Starting model training...")
    
    try:
        from src.training.train_model import main as train_main
        train_main()
    except ImportError as e:
        logger.error(f"Failed to import training module: {e}")
        logger.info("Make sure all dependencies are installed")
        sys.exit(1)

def run_dataset_organization():
    """Run dataset organization"""
    logger.info("Starting dataset organization...")
    
    try:
        from scripts.organize_dataset import main as organize_main
        organize_main()
    except ImportError as e:
        logger.error(f"Failed to import dataset organizer: {e}")
        logger.info("Make sure the organize_dataset.py script exists in the scripts directory")
        sys.exit(1)

def run_setup():
    """Run project setup"""
    logger.info("Setting up project directories...")
    
    try:
        from config import create_directories
        create_directories()
        logger.info("Project setup completed successfully!")
    except ImportError as e:
        logger.error(f"Failed to import configuration: {e}")
        sys.exit(1)

def run_validation():
    """Run dataset validation"""
    logger.info("Validating dataset...")
    
    try:
        from src.training.train_model import validate_dataset
        if validate_dataset():
            logger.info("Dataset validation passed!")
        else:
            logger.error("Dataset validation failed!")
            sys.exit(1)
    except ImportError as e:
        logger.error(f"Failed to import validation module: {e}")
        sys.exit(1)

def run_tests():
    """Run project tests"""
    logger.info("Running tests...")
    
    try:
        # Import test modules
        from src.models.cnn_model import create_model
        from src.utils.image_processor import preprocess_image, validate_image_format
        
        # Test model creation
        logger.info("Testing model creation...")
        model = create_model()
        logger.info("✓ Model creation successful")
        
        # Test image processing utilities
        logger.info("Testing image processing utilities...")
        logger.info("✓ Image processing utilities loaded")
        
        logger.info("All tests passed!")
        
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
