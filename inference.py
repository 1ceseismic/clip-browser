import torch
from PIL import Image
import open_clip

print(torch.cuda.is_available())
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

img_path = "sad dog.jpg"
image = preprocess(Image.open(img_path)).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])


with torch.no_grad(), torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)