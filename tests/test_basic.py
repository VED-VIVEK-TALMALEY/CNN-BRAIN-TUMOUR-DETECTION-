"""
Basic tests for Brain Tumor Detection Project
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class TestProjectStructure(unittest.TestCase):
    """Test basic project structure and imports"""
    
    def test_config_import(self):
        """Test that configuration can be imported"""
        try:
            from config import PROJECT_ROOT, WEB_CONFIG, TRAINING_CONFIG
            self.assertTrue(PROJECT_ROOT.exists())
            self.assertIsInstance(WEB_CONFIG, dict)
            self.assertIsInstance(TRAINING_CONFIG, dict)
        except ImportError as e:
            self.fail(f"Failed to import config: {e}")
    
    def test_models_import(self):
        """Test that models can be imported"""
        try:
            from src.models.cnn_model import create_model
            model = create_model()
            self.assertIsNotNone(model)
        except ImportError as e:
            self.fail(f"Failed to import models: {e}")
    
    def test_utils_import(self):
        """Test that utilities can be imported"""
        try:
            from src.utils.image_processor import preprocess_image, validate_image_format
            self.assertIsNotNone(preprocess_image)
            self.assertIsNotNone(validate_image_format)
        except ImportError as e:
            self.fail(f"Failed to import utilities: {e}")
    
    def test_web_import(self):
        """Test that web application can be imported"""
        try:
            from src.web.app import create_app
            app = create_app()
            self.assertIsNotNone(app)
        except ImportError as e:
            self.fail(f"Failed to import web application: {e}")
    
    def test_training_import(self):
        """Test that training module can be imported"""
        try:
            from src.training.train_model import validate_dataset
            self.assertIsNotNone(validate_dataset)
        except ImportError as e:
            self.fail(f"Failed to import training module: {e}")

class TestConfiguration(unittest.TestCase):
    """Test configuration settings"""
    
    def test_project_paths(self):
        """Test that project paths are valid"""
        from config import PROJECT_ROOT, DATA_DIR, SRC_DIR
        
        self.assertTrue(PROJECT_ROOT.exists())
        self.assertTrue(DATA_DIR.exists())
        self.assertTrue(SRC_DIR.exists())
    
    def test_web_config(self):
        """Test web configuration"""
        from config import WEB_CONFIG
        
        required_keys = ['HOST', 'PORT', 'DEBUG', 'SECRET_KEY']
        for key in required_keys:
            self.assertIn(key, WEB_CONFIG)
    
    def test_training_config(self):
        """Test training configuration"""
        from config import TRAINING_CONFIG
        
        required_keys = ['INPUT_SHAPE', 'BATCH_SIZE', 'EPOCHS', 'LEARNING_RATE']
        for key in required_keys:
            self.assertIn(key, TRAINING_CONFIG)

class TestModelCreation(unittest.TestCase):
    """Test model creation and basic functionality"""
    
    def test_model_creation(self):
        """Test that model can be created"""
        from src.models.cnn_model import create_model
        
        model = create_model()
        self.assertIsNotNone(model)
        
        # Test model input shape
        expected_input = (150, 150, 3)
        self.assertEqual(model.input_shape[1:], expected_input)
        
        # Test model output shape
        self.assertEqual(model.output_shape[1], 1)
    
    def test_model_compilation(self):
        """Test that model is properly compiled"""
        from src.models.cnn_model import create_model
        
        model = create_model()
        
        # Check if model has optimizer
        self.assertIsNotNone(model.optimizer)
        
        # Check if model has loss function
        self.assertIsNotNone(model.loss)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestProjectStructure))
    test_suite.addTest(unittest.makeSuite(TestConfiguration))
    test_suite.addTest(unittest.makeSuite(TestModelCreation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())
