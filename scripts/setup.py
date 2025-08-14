#!/usr/bin/env python3
"""
Setup script for Brain Tumor Detection Project
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    logger.info(f"Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = [
        'tensorflow',
        'flask',
        'PIL',
        'numpy',
        'cv2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            logger.info(f"✓ {package} available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} not available")
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        return False
    
    return True

def install_dependencies():
    """Install project dependencies"""
    logger.info("Installing project dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        logger.info("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create project directory structure"""
    logger.info("Creating project directory structure...")
    
    try:
        from config import create_directories
        create_directories()
        logger.info("✓ Project directories created")
        return True
    except ImportError as e:
        logger.error(f"Failed to import configuration: {e}")
        return False

def setup_virtual_environment():
    """Set up virtual environment"""
    logger.info("Setting up virtual environment...")
    
    try:
        # Check if virtual environment exists
        if os.path.exists('venv'):
            logger.info("Virtual environment already exists")
            return True
        
        # Create virtual environment
        subprocess.check_call([
            sys.executable, '-m', 'venv', 'venv'
        ])
        
        logger.info("✓ Virtual environment created")
        logger.info("To activate:")
        if os.name == 'nt':  # Windows
            logger.info("  venv\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            logger.info("  source venv/bin/activate")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        return False

def run_tests():
    """Run basic tests to verify setup"""
    logger.info("Running basic tests...")
    
    try:
        # Test imports
        from src.models.cnn_model import create_model
        from src.utils.image_processor import preprocess_image
        from config import validate_config
        
        # Test model creation
        model = create_model()
        logger.info("✓ Model creation successful")
        
        # Test configuration
        validate_config()
        logger.info("✓ Configuration validation successful")
        
        logger.info("✓ All tests passed")
        return True
        
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🧠 Brain Tumor Detection Project Setup")
    logger.info("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        logger.info("Installing missing dependencies...")
        if not install_dependencies():
            logger.error("Failed to install dependencies")
            sys.exit(1)
    
    # Create directories
    if not create_directories():
        logger.error("Failed to create project directories")
        sys.exit(1)
    
    # Setup virtual environment
    if not setup_virtual_environment():
        logger.error("Failed to setup virtual environment")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        logger.error("Tests failed")
        sys.exit(1)
    
    logger.info("\n" + "="*50)
    logger.info("🎉 SETUP COMPLETED SUCCESSFULLY!")
    logger.info("="*50)
    logger.info("\nNext steps:")
    logger.info("1. Activate virtual environment:")
    if os.name == 'nt':  # Windows
        logger.info("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        logger.info("   source venv/bin/activate")
    
    logger.info("2. Organize your dataset:")
    logger.info("   python main.py organize-dataset")
    
    logger.info("3. Train the model:")
    logger.info("   python main.py train")
    
    logger.info("4. Run the web application:")
    logger.info("   python main.py web")
    
    logger.info("\nFor help:")
    logger.info("   python main.py --help")

if __name__ == "__main__":
    main()
