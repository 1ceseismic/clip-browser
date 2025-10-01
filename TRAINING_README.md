
## training 

1. create captions for your images using a GUI interface
2. generate additional training samples
3. finetune clip models on your dataset(s)


```bash
python setup_training.py
```
will install:

- PyTorch, TorchVision
- OpenCLIP with training support
- Pandas, Scikit-learn, OpenCV
- UMAP, FAISS for clustering
- TensorBoard for training visualization
- Local LLM for text augmentation



put images in a dir like this:

```
your_dataset/
    image1.jpg
    image2.png
    image3.jpeg
    ...
```


1. Start CLIP Browser and select your dataset root directory
2. Go to the Training tab
3. Click "Start Manual Captioning"
5. Type caption for each image
this creates an `index.csv` file in your dataset dir

Controls:
- Enter: Save caption and move to next image
- ESC: Quit + save progress


1. Click "Prepare Training Data"
2. This splits your captioned data into training (80%) and validation (20%) sets
3. Creates `train_original.csv` and `val.csv` files


1. Choose augmentation method:
   - LLM: Uses a local language model to generate paraphrases 
   - Manual: Uses predefined synonym mappings
2. Set the number of augmentations per sample (default: 3)
3. Click "Run Text Augmentation"
4. This creates `train.csv` with augmented training data

**Note**: LLM augmentation download ~2GB model on first use

### training

1. Configure training parameters:

   - Model: Choose CLIP architecture (ViT-B-32 recommended for most cases)
   - Pretrained: Choose pretrained weights (openai recommended)
   - Epochs: Number of training epochs (10-20 recommended)
   - Batch Size**: Training batch size (32 recommended, reduce if out of memory)
   - Learning Rate: Training learning rate (1e-4 recommended)
   - Warmup Steps: Learning rate warmup steps (10000 recommended)

2. press **"Start Training"**
logs in the Training Log+progress sections
Use **"Stop Training"** to halt early


Trained models are saved in `models_finetuned/`

logs:

```bash
tensorboard --logdir runs/
```
### Using trained model

After training completes:

1. The trained model will be available in the model selector
2. Select your custom model from the dropdown
3. Build an index on your dataset using the trained model
4. Perform semantic search with your domain-specific model


### custom train scripts

You can also run training manually using the command line:

```bash
python -m open_clip_train.main --train-data train.csv --val-data val.csv --model ViT-B-32 --pretrained openai --epochs 10
```

Modify `manual_synonym_map.json` to add custom synonym mappings for manual augmentation.

CLIP architectures:

- **ViT-B-32**: Fast, good generally (~4gb)
- **ViT-L-14**: Larger, better performance  (~8gb)
- **ViT-H-14**: Largest, best performance but slower (~16gb)

minimum 100 images for basic fine-tuning
