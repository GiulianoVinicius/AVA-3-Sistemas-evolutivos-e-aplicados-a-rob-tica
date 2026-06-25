# Classificação de Imagens com Vision Transformer (ViT)

## 📋 Descrição
Este código realiza **classificação de imagens usando um modelo Vision Transformer (ViT)** pré-treinado do Google. O modelo identifica objetos/cenas em uma imagem e retorna os 5 resultados mais prováveis com suas respectivas confiabilidades.

## 🔍 Explicação Detalhada do Código

### 1. **Importações**
```python
import requests          # Para baixar imagens da web
import torch            # Framework de deep learning
import time             # Para medir tempo de execução
from PIL import Image   # Para manipular imagens
from transformers import ViTForImageClassification, ViTImageProcessor
# Modelos pré-treinados do Hugging Face
```

### 2. **Carregamento da Imagem**
```python
url = "https://images.pexels.com/photos/10781027/pexels-photo-10781027.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
```
- Baixa uma imagem da internet usando `requests`
- Converte para formato RGB (3 canais de cor)

### 3. **Carregamento do Modelo e Processador**
```python
model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
```
- Carrega um modelo **Vision Transformer** pré-treinado do Google
- Carrega o processador que prepara a imagem no formato esperado pelo modelo

### 4. **Pré-processamento da Imagem**
```python
inputs = feature_extractor(images=image, return_tensors="pt")
```
- Redimensiona a imagem para 224x224 pixels
- Normaliza os valores dos pixels
- Converte em tensor PyTorch para processamento

### 5. **Inferência com Cronometragem**
```python
ti = time.time()
with torch.no_grad():  # Desativa cálculo de gradientes (mais rápido)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    top5_prob, top5_indices = torch.topk(probs, 5)
tf = time.time()
tt = tf - ti
```
- **`time.time()`**: Marca o início e fim da inferência
- **`torch.no_grad()`**: Desativa gradientes para economizar memória e acelerar
- **`softmax`**: Converte os logits em probabilidades (0 a 1)
- **`torch.topk()`**: Extrai os 5 resultados mais prováveis

### 6. **Exibição dos Resultados**
```python
print(f"Tempo de inferência: {tt:.4f} segundos")
print("Top 5 Classes Preditas:")
for i in range(top5_prob.size(1)):
    label = model.config.id2label[top5_indices[0][i].item()]
    probability = top5_prob[0][i].item()
    print(f"- {label}: {probability:.4f}")
```
- Mostra o tempo total de processamento
- Lista as 5 classes mais prováveis com suas porcentagens de confiança

## 🚀 Resumo
O código identifica **qual objeto/cena está na imagem** usando IA, mostrando os 5 resultados mais prováveis com suas respectivas confiabilidades em formato de porcentagem.

## 📦 Dependências
- `requests` - HTTP requests
- `torch` - Deep Learning Framework
- `Pillow (PIL)` - Processamento de imagens
- `transformers` - Modelos pré-treinados do Hugging Face

## 💡 Aplicações Práticas
- Classificação automática de imagens
- Análise de conteúdo visual
- Sistemas de recomendação baseados em imagens
- Robótica e visão computacional

---

**Código Original:**
```python
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
```
