#!/usr/bin/env python3
"""
Setup script for CLIP Browser training functionality.
This script helps install the required dependencies for training custom CLIP models.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_training_dependencies():
    """Install training dependencies."""
    print("\n=== Installing Training Dependencies ===")
    
    # Install basic training dependencies
    if not run_command(
        f"{sys.executable} -m pip install -r requirements-training.txt",
        "Installing training dependencies"
    ):
        return False
    
    # Install open-clip-torch training dependencies
    if not run_command(
        f"{sys.executable} -m pip install open-clip-torch[training]",
        "Installing open-clip-torch training dependencies"
    ):
        return False
    
    return True

def check_cuda_availability():
    """Check if CUDA is available for training."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available - training will use CPU (slower)")
            return True
    except ImportError:
        print("⚠ PyTorch not installed - CUDA check skipped")
        return True

def create_training_directories():
    """Create necessary directories for training."""
    print("\n=== Creating Training Directories ===")
    
    directories = [
        "models_finetuned",
        ".thumbnails",
        ".indexes"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def test_imports():
    """Test that all required modules can be imported."""
    print("\n=== Testing Imports ===")
    
    modules_to_test = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("open_clip", "OpenCLIP"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("umap", "UMAP"),
        ("faiss", "FAISS"),
        ("tensorboard", "TensorBoard")
    ]
    
    all_success = True
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {display_name} imported successfully")
        except ImportError as e:
            print(f"✗ {display_name} import failed: {e}")
            all_success = False
    
    return all_success

def main():
    """Main setup function."""
    print("=== CLIP Browser Training Setup ===")
    print("This script will install dependencies for training custom CLIP models.")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_training_dependencies():
        print("\n✗ Failed to install dependencies. Please check the error messages above.")
        sys.exit(1)
    
    # Check CUDA
    check_cuda_availability()
    
    # Create directories
    if not create_training_directories():
        print("\n✗ Failed to create directories.")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n✗ Some modules failed to import. Please check the installation.")
        sys.exit(1)
    
    print("\n=== Setup Complete! ===")
    print("✓ Training environment is ready")
    print("\nNext steps:")
    print("1. Start the CLIP Browser application")
    print("2. Select a dataset root directory")
    print("3. Go to the Training tab")
    print("4. Follow the step-by-step process to train your custom CLIP model")
    print("\nNote: For LLM-based text augmentation, the first run will download")
    print("a local language model (~2GB) which may take some time.")

if __name__ == "__main__":
    main() 