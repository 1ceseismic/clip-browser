# CLIP Browser Training Guide

This guide explains how to use the training functionality in CLIP Browser to fine-tune CLIP models on your custom datasets.

## Overview

The training functionality allows you to:

1. **Manual Captioning**: Create captions for your images using a GUI interface
2. **Data Preparation**: Split your data into training and validation sets
3. **Text Augmentation**: Generate additional training samples through text variations
4. **Model Training**: Fine-tune CLIP models on your custom dataset

## Prerequisites

### 1. Install Training Dependencies

Run the setup script to install all required dependencies:

```bash
python setup_training.py
```

This will install:

- PyTorch and TorchVision
- OpenCLIP with training support
- Pandas, Scikit-learn, OpenCV
- UMAP, FAISS for clustering
- TensorBoard for training visualization
- Local LLM for text augmentation

### 2. Prepare Your Dataset

Organize your images in a directory structure like this:

```
your_dataset/
├── image1.jpg
├── image2.png
├── image3.jpeg
└── ...
```

## Step-by-Step Training Process

### Step 1: Manual Captioning

1. Start CLIP Browser and select your dataset root directory
2. Go to the **Training** tab
3. Click **"Start Manual Captioning"**
4. A GUI window will open showing your images one by one
5. Type a descriptive caption for each image and press Enter
6. Continue until all images are captioned
7. This creates an `index.csv` file in your dataset directory

**Controls:**

- **Type**: Enter your caption
- **Enter**: Save caption and move to next image
- **Backspace**: Delete last character
- **ESC**: Quit (saves progress)

### Step 2: Data Preparation

1. Click **"Prepare Training Data"**
2. This splits your captioned data into training (80%) and validation (20%) sets
3. Creates `train_original.csv` and `val.csv` files

### Step 3: Text Augmentation

1. Choose augmentation method:
   - **LLM**: Uses a local language model to generate paraphrases (recommended)
   - **Manual**: Uses predefined synonym mappings
2. Set the number of augmentations per sample (default: 3)
3. Click **"Run Text Augmentation"**
4. This creates `train.csv` with augmented training data

**Note**: LLM augmentation downloads a ~2GB model on first use.

### Step 4: Model Training

1. Configure training parameters:

   - **Model**: Choose CLIP architecture (ViT-B-32 recommended for most cases)
   - **Pretrained**: Choose pretrained weights (openai recommended)
   - **Epochs**: Number of training epochs (10-20 recommended)
   - **Batch Size**: Training batch size (32 recommended, reduce if out of memory)
   - **Learning Rate**: Training learning rate (1e-4 recommended)
   - **Warmup Steps**: Learning rate warmup steps (10000 recommended)

2. Click **"Start Training"**
3. Monitor progress in the Training Progress section
4. View detailed logs in the Training Log section
5. Use **"Stop Training"** to halt training early

## Training Output

### Model Checkpoints

Trained models are saved in the `models_finetuned/` directory with timestamps.

### TensorBoard Logs

Training logs are saved for TensorBoard visualization:

```bash
tensorboard --logdir runs/
```

### Training Metrics

Monitor these metrics during training:

- **Loss**: Should decrease over time
- **Learning Rate**: Follows warmup schedule
- **Epoch Progress**: Overall training progress

## Using Your Trained Model

After training completes:

1. The trained model will be available in the model selector
2. Select your custom model from the dropdown
3. Build an index on your dataset using the trained model
4. Perform semantic search with your domain-specific model

## Tips for Better Training

### Data Quality

- **Diverse Captions**: Use varied, descriptive captions
- **Consistent Style**: Maintain consistent captioning style
- **Relevant Content**: Ensure captions accurately describe images

### Training Parameters

- **Start Small**: Begin with fewer epochs and smaller models
- **Monitor Loss**: Stop training if loss plateaus or increases
- **Validation**: Use validation set to prevent overfitting

### Hardware Considerations

- **GPU**: Training is much faster with CUDA-capable GPU
- **Memory**: Reduce batch size if you encounter out-of-memory errors
- **Storage**: Ensure sufficient disk space for model checkpoints

## Troubleshooting

### Common Issues

**"No index.csv found"**

- Run manual captioning first to create captions

**"CUDA out of memory"**

- Reduce batch size
- Use smaller model architecture
- Close other GPU applications

**"Training failed to start"**

- Check that all dependencies are installed
- Ensure dataset directory is accessible
- Verify CSV files are properly formatted

**"LLM augmentation fails"**

- Check internet connection for model download
- Ensure sufficient disk space (~2GB)
- Try manual augmentation as fallback

### Getting Help

If you encounter issues:

1. Check the training log for error messages
2. Verify all dependencies are installed correctly
3. Ensure your dataset structure is correct
4. Try with a smaller subset of data first

## Advanced Features

### Custom Training Scripts

You can also run training manually using the command line:

```bash
python -m open_clip_train.main --train-data train.csv --val-data val.csv --model ViT-B-32 --pretrained openai --epochs 10
```

### Custom Augmentation

Modify `manual_synonym_map.json` to add custom synonym mappings for manual augmentation.

### Model Architectures

Available CLIP architectures:

- **ViT-B-32**: Fast, good for most use cases
- **ViT-L-14**: Larger, potentially better performance
- **ViT-H-14**: Largest, best performance but slower

## Performance Expectations

### Training Time (with GPU)

- **ViT-B-32**: ~1-2 hours for 10 epochs on 1000 images
- **ViT-L-14**: ~3-4 hours for 10 epochs on 1000 images
- **ViT-H-14**: ~6-8 hours for 10 epochs on 1000 images

### Memory Requirements

- **ViT-B-32**: ~4GB GPU memory
- **ViT-L-14**: ~8GB GPU memory
- **ViT-H-14**: ~16GB GPU memory

### Dataset Size Recommendations

- **Minimum**: 100 images for basic fine-tuning
- **Recommended**: 1000+ images for good results
- **Optimal**: 5000+ images for robust performance
