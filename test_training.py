#!/usr/bin/env python3
"""
Test script for CLIP Browser training functionality.
This script tests the training components without actually running training.
"""

import os
import sys
from pathlib import Path
import pandas as pd

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    modules = [
        ("training", "Training module"),
        ("text_augment", "Text augmentation"),
        ("manual_captions", "Manual captioning"),
        ("inference", "Inference module"),
        ("config", "Config module")
    ]
    
    all_success = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError as e:
            print(f"✗ {display_name}: {e}")
            all_success = False
    
    return all_success

def test_training_manager():
    """Test the TrainingManager class."""
    print("\nTesting TrainingManager...")
    
    try:
        from training import TrainingManager
        
        # Create instance
        manager = TrainingManager()
        print("✓ TrainingManager created successfully")
        
        # Test status
        status = manager.get_training_status()
        expected_keys = ["is_training", "progress", "status", "current_epoch", "total_epochs"]
        for key in expected_keys:
            if key in status:
                print(f"✓ Status key '{key}' present")
            else:
                print(f"✗ Status key '{key}' missing")
                return False
        
        print("✓ TrainingManager status test passed")
        return True
        
    except Exception as e:
        print(f"✗ TrainingManager test failed: {e}")
        return False

def test_data_preparation():
    """Test data preparation functionality."""
    print("\nTesting data preparation...")
    
    try:
        # Create test data
        test_data = [
            {"filepath": "image1.jpg", "caption": "A beautiful sunset"},
            {"filepath": "image2.jpg", "caption": "A cute cat"},
            {"filepath": "image3.jpg", "caption": "A red car"},
            {"filepath": "image4.jpg", "caption": "A green tree"},
            {"filepath": "image5.jpg", "caption": "A blue sky"}
        ]
        
        # Create test directory
        test_dir = Path("test_dataset")
        test_dir.mkdir(exist_ok=True)
        
        # Save test data
        df = pd.DataFrame(test_data)
        index_csv = test_dir / "index.csv"
        df.to_csv(index_csv, index=False)
        print(f"✓ Created test data: {index_csv}")
        
        # Test data preparation
        from training import TrainingManager
        manager = TrainingManager()
        
        result = manager.prepare_training_data(str(test_dir), test_size=0.2)
        
        if result["success"]:
            print(f"✓ Data preparation successful: {result['train_count']} train, {result['val_count']} val")
            
            # Check if files were created
            train_file = test_dir / "train_original.csv"
            val_file = test_dir / "val.csv"
            
            if train_file.exists() and val_file.exists():
                print("✓ Train/val files created successfully")
                return True
            else:
                print("✗ Train/val files not created")
                return False
        else:
            print(f"✗ Data preparation failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
        return False
    finally:
        # Cleanup
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "training.py",
        "text_augment.py", 
        "manual_captions.py",
        "inference.py",
        "config.py",
        "app.py",
        "gui.py",
        "requirements-training.txt",
        "setup_training.py",
        "TRAINING_README.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("=== CLIP Browser Training Test Suite ===\n")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("TrainingManager", test_training_manager),
        ("Data Preparation", test_data_preparation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n=== Test Results ===")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Training functionality is ready.")
        print("\nNext steps:")
        print("1. Run: python setup_training.py")
        print("2. Start the application: python run.py")
        print("3. Follow the training guide in TRAINING_README.md")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 