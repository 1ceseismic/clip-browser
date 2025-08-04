import os
import sys
import subprocess
import pandas as pd
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
from sklearn.model_selection import train_test_split
import text_augment
import manual_captions
import config

class TrainingManager:
    def __init__(self):
        self.is_training = False
        self.training_progress = 0.0
        self.training_status = "idle"
        self.training_log = []
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0
        self.loss = 0.0
        self.learning_rate = 0.0
        self._progress_callback = None
        self._status_callback = None
        self._log_callback = None
        self._has_output_received = False
        
    def set_callbacks(self, progress_callback: Callable, status_callback: Callable, log_callback: Callable):
        """Set callbacks for updating the GUI during training."""
        self._progress_callback = progress_callback
        self._status_callback = status_callback
        self._log_callback = log_callback
    
    def _update_progress(self, progress: float):
        self.training_progress = progress
        if self._progress_callback:
            self._progress_callback(progress)
    
    def _update_status(self, status: str):
        self.training_status = status
        if self._status_callback:
            self._status_callback(status)
    
    def _add_log(self, message: str):
        self.training_log.append(message)
        self._has_output_received = True
        if self._log_callback:
            self._log_callback(message)
    
    def _has_received_output(self) -> bool:
        """Check if we've received any output from the training process."""
        return self._has_output_received
    
    def prepare_training_data(self, dataset_root: str, test_size: float = 0.2) -> Dict[str, any]:
        """Prepare training data from the dataset root."""
        try:
            self._update_status("Preparing training data...")
            self._add_log("Starting data preparation...")
            
            # Look for index.csv (manual captions) in the dataset root
            index_csv_path = Path(dataset_root) / "index.csv"
            if not index_csv_path.exists():
                return {
                    "success": False,
                    "error": f"No index.csv found in {dataset_root}. Please run manual captioning first."
                }
            
            # Load the data
            df = pd.read_csv(index_csv_path)
            self._add_log(f"Loaded {len(df)} samples from index.csv")
            
            # Split into train/val
            train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
            self._add_log(f"Split data: {len(train_df)} train, {len(val_df)} validation")
            
            # Save the splits
            train_original_path = Path(dataset_root) / "train_original.csv"
            val_path = Path(dataset_root) / "val.csv"
            
            train_df.to_csv(train_original_path, index=False)
            val_df.to_csv(val_path, index=False)
            
            self._add_log(f"Saved train/val splits to {train_original_path} and {val_path}")
            self._update_status("Data preparation completed successfully")
            
            return {
                "success": True,
                "train_count": len(train_df),
                "val_count": len(val_df),
                "train_path": str(train_original_path),
                "val_path": str(val_path)
            }
            
        except Exception as e:
            self._add_log(f"Error preparing training data: {e}")
            return {"success": False, "error": str(e)}
    
    def run_text_augmentation(self, dataset_root: str, method: str = "llm", num_aug: int = 3) -> Dict[str, any]:
        """Run text augmentation on the training data."""
        try:
            self._update_status("Running text augmentation...")
            self._add_log(f"Starting {method} text augmentation...")
            
            # Get the path to text_augment.py in the main application directory
            script_dir = Path(__file__).parent
            text_augment_script = script_dir / "text_augment.py"
            
            if not text_augment_script.exists():
                self._add_log(f"Error: text_augment.py not found at {text_augment_script}")
                return {"success": False, "error": f"text_augment.py not found at {text_augment_script}"}
            
            # Run text augmentation
            cmd = [
                sys.executable, str(text_augment_script),
                "--train_original_csv", "train_original.csv",
                "--train_augmented_csv", "train.csv",
                "--augmentation_method", method,
                "--num_aug_per_original", str(num_aug)
            ]
            
            self._add_log(f"Running command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=dataset_root  # Run from dataset directory but use script from app directory
            )
            
            # Monitor the process
            while process.poll() is None:
                output = process.stdout.readline()
                if output:
                    self._add_log(output.strip())
                time.sleep(0.1)
            
            # Get final output
            stdout, stderr = process.communicate()
            if stdout:
                self._add_log(stdout)
            if stderr:
                self._add_log(f"STDERR: {stderr}")
            
            if process.returncode == 0:
                # Check if train.csv was created
                train_augmented_path = Path(dataset_root) / "train.csv"
                if train_augmented_path.exists():
                    df = pd.read_csv(train_augmented_path)
                    self._add_log(f"Text augmentation complete. Generated {len(df)} training samples.")
                    self._update_status("Text augmentation completed successfully")
                    return {
                        "success": True,
                        "augmented_count": len(df),
                        "train_path": str(train_augmented_path)
                    }
                else:
                    self._update_status("Text augmentation failed - train.csv not created")
                    return {"success": False, "error": "train.csv was not created"}
            else:
                self._update_status("Text augmentation failed")
                error_msg = f"Text augmentation failed with return code {process.returncode}"
                if stderr:
                    error_msg += f". Error: {stderr}"
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            self._add_log(f"Error during text augmentation: {e}")
            return {"success": False, "error": str(e)}
    
    def start_training(self, dataset_root: str, model_name: str = "ViT-B-32", 
                      pretrained: str = "openai", epochs: int = 10, batch_size: int = 32,
                      learning_rate: float = 1e-4, warmup_steps: int = 10000) -> Dict[str, any]:
        """Start CLIP fine-tuning training."""
        if self.is_training:
            return {"success": False, "error": "Training already in progress"}
        
        try:
            self.is_training = True
            self.training_progress = 0.0
            self.current_epoch = 0
            self.total_epochs = epochs
            self.current_step = 0
            self.total_steps = 0
            self.loss = 0.0
            self.learning_rate = learning_rate
            
            self._update_status("Starting training...")
            self._add_log(f"Starting CLIP fine-tuning with {model_name}/{pretrained}")
            self._add_log(f"Training for {epochs} epochs with batch size {batch_size}")
            
            # Check for required files
            train_csv = Path(dataset_root) / "train.csv"
            val_csv = Path(dataset_root) / "val.csv"
            
            if not train_csv.exists():
                return {"success": False, "error": "train.csv not found. Run text augmentation first."}
            if not val_csv.exists():
                return {"success": False, "error": "val.csv not found. Run data preparation first."}
            
            # Test if open_clip_train is available
            try:
                import subprocess
                test_result = subprocess.run([sys.executable, "-c", "import open_clip_train"], 
                                           capture_output=True, text=True, timeout=10)
                if test_result.returncode != 0:
                    self._add_log(f"WARNING: open_clip_train import test failed: {test_result.stderr}")
                    self._add_log("This might cause training to fail.")
            except Exception as e:
                self._add_log(f"WARNING: Could not test open_clip_train import: {e}")
            
            # Validate dataset files
            self._add_log("Validating dataset files...")
            try:
                train_df = pd.read_csv(train_csv)
                val_df = pd.read_csv(val_csv)
                self._add_log(f"Train CSV: {len(train_df)} samples, columns: {list(train_df.columns)}")
                self._add_log(f"Val CSV: {len(val_df)} samples, columns: {list(val_df.columns)}")
                
                # Check for required columns
                required_columns = ['filepath', 'caption']
                for col in required_columns:
                    if col not in train_df.columns:
                        return {"success": False, "error": f"Missing required column '{col}' in train.csv"}
                    if col not in val_df.columns:
                        return {"success": False, "error": f"Missing required column '{col}' in val.csv"}
                
                # Check if image files exist
                sample_paths = train_df['filepath'].head(5).tolist()
                for path in sample_paths:
                    full_path = Path(dataset_root) / path
                    if not full_path.exists():
                        self._add_log(f"WARNING: Image file not found: {full_path}")
                        return {"success": False, "error": f"Image file not found: {full_path}"}
                
                self._add_log("Dataset validation passed!")
                
            except Exception as e:
                self._add_log(f"Dataset validation failed: {e}")
                return {"success": False, "error": f"Dataset validation failed: {e}"}
            
            # Prepare training command
            aug_cfg_args = [
                "scale=(0.4,1.0)",
                "ratio=(0.75,1.3333333333333333)",
                "color_jitter=(0.4,0.4,0.4,0.1)",
                "color_jitter_prob=0.8",
                "re_prob=0.25",
                "re_count=1",
                "gray_scale_prob=0.2"
            ]
            
            cmd = [
                "python", "-m", "open_clip_train.main",
                "--train-data", str(train_csv),
                "--val-data", str(val_csv),
                "--csv-img-key", "filepath",
                "--csv-caption-key", "caption",
                "--csv-separator", ",",
                "--model", model_name,
                "--pretrained", pretrained,
                "--report-to", "tensorboard",
                "--log-every-n-steps", "50",
                "--batch-size", str(batch_size),
                "--lr", str(learning_rate),
                "--epochs", str(epochs),
                "--warmup", str(warmup_steps),
                "--workers", "4",
                "--device", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                "--aug-cfg"
            ] + aug_cfg_args
            
            self._add_log(f"Training command: {' '.join(cmd)}")
            
            # Change to dataset directory
            original_cwd = os.getcwd()
            os.chdir(dataset_root)
            
            # Create models_finetuned directory if it doesn't exist
            models_dir = Path("models_finetuned")
            models_dir.mkdir(exist_ok=True)
            
            # Start training process
            try:
                self._add_log(f"Starting process with command: {' '.join(cmd)}")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                self._add_log(f"Process started with PID: {process.pid}")
            except FileNotFoundError:
                self._add_log("Error: open_clip_train.main not found. Please install open-clip-torch[training]")
                return {"success": False, "error": "open_clip_train.main not found. Please install open-clip-torch[training]"}
            except Exception as e:
                self._add_log(f"Error starting process: {e}")
                return {"success": False, "error": f"Failed to start training process: {e}"}
            
            # Monitor training progress
            self._add_log("Starting to monitor training process...")
            start_time = time.time()
            
            while process.poll() is None:
                # Read stdout
                output = process.stdout.readline()
                if output:
                    self._add_log(f"STDOUT: {output.strip()}")
                    # Parse progress information
                    self._parse_training_output(output)
                
                # Read stderr
                error_output = process.stderr.readline()
                if error_output:
                    self._add_log(f"STDERR: {error_output.strip()}")
                
                # Check if process has been running too long without output
                if time.time() - start_time > 30 and not self._has_received_output():
                    self._add_log("WARNING: No output received for 30 seconds. Process may be hanging.")
                    self._add_log("Checking if process is still alive...")
                    if process.poll() is None:
                        self._add_log("Process is still running but no output. This might indicate:")
                        self._add_log("1. Missing dependencies (open_clip_train not installed)")
                        self._add_log("2. CUDA/GPU issues")
                        self._add_log("3. Dataset loading problems")
                        self._add_log("4. Memory issues")
                        self._add_log("5. Dataset file format issues")
                        self._add_log("6. Permission issues")
                        
                        # Kill the hanging process after 60 seconds total
                        if time.time() - start_time > 60:
                            self._add_log("Killing hanging process...")
                            try:
                                process.terminate()
                                process.wait(timeout=5)
                            except:
                                process.kill()
                            return {"success": False, "error": "Training process hung during startup and was terminated"}
                
                time.sleep(0.1)
            
            # Get final output
            stdout, stderr = process.communicate()
            if stdout:
                self._add_log(stdout)
            if stderr:
                self._add_log(f"STDERR: {stderr}")
            
            # Restore original directory
            os.chdir(original_cwd)
            
            if process.returncode == 0:
                self._update_status("Training completed successfully")
                self._update_progress(1.0)
                return {"success": True, "message": "Training completed successfully"}
            else:
                self._update_status("Training failed")
                return {"success": False, "error": f"Training failed with return code {process.returncode}"}
                
        except Exception as e:
            self._add_log(f"Error during training: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.is_training = False
    
    def _parse_training_output(self, output: str):
        """Parse training output to extract progress information."""
        try:
            # Look for epoch information
            if "epoch" in output.lower():
                # Extract epoch number
                import re
                epoch_match = re.search(r'epoch\s+(\d+)/(\d+)', output.lower())
                if epoch_match:
                    self.current_epoch = int(epoch_match.group(1))
                    self.total_epochs = int(epoch_match.group(2))
                    progress = self.current_epoch / self.total_epochs
                    self._update_progress(progress)
            
            # Look for step information
            if "step" in output.lower():
                step_match = re.search(r'step\s+(\d+)/(\d+)', output.lower())
                if step_match:
                    self.current_step = int(step_match.group(1))
                    self.total_steps = int(step_match.group(2))
            
            # Look for loss information
            loss_match = re.search(r'loss:\s*([\d.]+)', output.lower())
            if loss_match:
                self.loss = float(loss_match.group(1))
            
            # Look for learning rate information
            lr_match = re.search(r'lr:\s*([\d.e+-]+)', output.lower())
            if lr_match:
                self.learning_rate = float(lr_match.group(1))
                
        except Exception as e:
            # Ignore parsing errors
            pass
    
    def stop_training(self):
        """Stop the current training process."""
        if self.is_training:
            self.is_training = False
            self._update_status("Training stopped")
            self._add_log("Training stopped by user")
    
    def get_training_status(self) -> Dict[str, any]:
        """Get current training status."""
        return {
            "is_training": self.is_training,
            "progress": self.training_progress,
            "status": self.training_status,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "recent_logs": self.training_log[-50:]  # Last 50 log entries
        }
    
    def validate_training_ready(self, dataset_root: str) -> Dict[str, any]:
        """Validate that all required files exist for training."""
        try:
            dataset_path = Path(dataset_root)
            
            # Check for required files
            required_files = {
                "index.csv": "Manual captions file (run Step 1)",
                "train_original.csv": "Training data split (run Step 2)", 
                "val.csv": "Validation data split (run Step 2)",
                "train.csv": "Augmented training data (run Step 3)"
            }
            
            missing_files = []
            for filename, description in required_files.items():
                file_path = dataset_path / filename
                if not file_path.exists():
                    missing_files.append(f"{filename} - {description}")
            
            if missing_files:
                return {
                    "ready": False,
                    "missing_files": missing_files,
                    "message": "Missing required files. Please complete all previous steps."
                }
            
            # Validate CSV structure
            try:
                train_df = pd.read_csv(dataset_path / "train.csv")
                val_df = pd.read_csv(dataset_path / "val.csv")
                
                required_columns = ['filepath', 'caption']
                for col in required_columns:
                    if col not in train_df.columns or col not in val_df.columns:
                        return {
                            "ready": False,
                            "message": f"Missing required column '{col}' in CSV files"
                        }
                
                return {
                    "ready": True,
                    "train_samples": len(train_df),
                    "val_samples": len(val_df),
                    "message": "All files ready for training"
                }
                
            except Exception as e:
                return {
                    "ready": False,
                    "message": f"Error reading CSV files: {e}"
                }
                
        except Exception as e:
            return {
                "ready": False,
                "message": f"Validation error: {e}"
            }
    
    def run_manual_captioning(self, dataset_root: str) -> Dict[str, any]:
        """Run the manual captioning tool."""
        try:
            self._update_status("Starting manual captioning...")
            self._add_log("Launching manual captioning interface...")
            
            # Check if dataset folder exists
            dataset_path = Path(dataset_root)
            if not dataset_path.exists():
                return {"success": False, "error": f"Dataset path {dataset_root} does not exist"}
            
            # Change to dataset directory
            original_cwd = os.getcwd()
            os.chdir(dataset_root)
            
            # Run manual captioning in a separate process
            try:
                # Get the path to manual_captions.py in the main application directory
                script_dir = Path(__file__).parent
                manual_captions_script = script_dir / "manual_captions.py"
                
                if not manual_captions_script.exists():
                    self._add_log(f"Error: manual_captions.py not found at {manual_captions_script}")
                    return {"success": False, "error": f"manual_captions.py not found at {manual_captions_script}"}
                
                cmd = [sys.executable, str(manual_captions_script)]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=dataset_root  # Run from dataset directory but use script from app directory
                )
                
                # Wait for the process to complete
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    self._add_log("Manual captioning completed successfully")
                    return {"success": True, "message": "Manual captioning completed"}
                else:
                    self._add_log(f"Manual captioning failed: {stderr}")
                    return {"success": False, "error": f"Manual captioning failed: {stderr}"}
                    
            except Exception as e:
                self._add_log(f"Error during manual captioning: {e}")
                return {"success": False, "error": str(e)}
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            self._add_log(f"Error starting manual captioning: {e}")
            return {"success": False, "error": str(e)}

# Global training manager instance
training_manager = TrainingManager() 