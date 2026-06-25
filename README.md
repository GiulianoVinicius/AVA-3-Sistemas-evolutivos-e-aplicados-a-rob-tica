import requests
import torch
import time
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

url = "https://images.pexels.com/photos/10781027/pexels-photo-10781027.png"

image = Image.open(requests.get(url, stream = True).raw).convert("RGB")

model = ViTForImageClassification.from_pretrained ("google/vit-large-patch16-224")
feature_extractor = ViTImageProcessor.from_pretrained ("google/vit-large-patch16-224")

inputs = feature_extractor(images=image, return_tensors="pt")

ti=time.time()

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    # Get top 5 predicted probabilities and indices
    top5_prob, top5_indices = torch.topk(probs, 5)

tf=time.time()
tt=tf-ti

print(f"Tempo de inferência: {tt:.4f} segundos")
print("Top 5 Classes Preditas:")
for i in range(top5_prob.size(1)):
    label = model.config.id2label[top5_indices[0][i].item()]
    probability = top5_prob[0][i].item()
    print(f"- {label}: {probability:.4f}")
