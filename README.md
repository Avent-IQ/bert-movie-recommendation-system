# Book Recommendation System with Bert

## 📌 Overview

This repository hosts the quantized version of the bert-base-cased model fine-tuned for movie reccommendation tasks. The model has been trained on the wykonos/movies dataset from Hugging Face. The model is quantized to Float16 (FP16) to optimize inference speed and efficiency while maintaining high performance.

## 🏗 Model Details

- **Model Architecture:** bert-base-cased
- **Task:** Book Recommendation System  
- **Dataset:** Hugging Face's `wykonos/movies`  
- **Quantization:** Float16 (FP16) for optimized inference  
- **Fine-tuning Framework:** Hugging Face Transformers  

## 🚀 Usage

### Installation

```bash
pip install transformers torch
```

### Loading the Model

```python
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
```

### Question Answer Example

```python
model_name = "AventIQ-AI/bert-movie-recommendation-system"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)

genre_to_label = {
    "Action": 0, "Adventure": 1, "Animation": 2, "Comedy": 3, "Crime": 4,
    "Documentary": 5, "Drama": 6, "Family": 7, "Fantasy": 8, "History": 9,
    "Horror": 10, "Music": 11, "Mystery": 12, "Romance": 13, "Science Fiction": 14,
    "TV Movie": 15, "Thriller": 16, "War": 17, "Western": 18
}

def recommend_movies(genre, top_n=10):
    """Return a list of movies for a given genre."""
    if genre not in genre_to_label:
        return "Unknown Genre"
    # Filter dataset for movies in the requested genre
    genre_movies = df[df["genres"].str.contains(genre, case=False, na=False)]["title"].tolist()
 
    # Return top N movies (or all if fewer exist)
    return genre_movies[:top_n]

genres_to_test = ["Horror", "Comedy", "Drama"]
for genre in genres_to_test:
    recommended_movies = recommend_movies(genre)
    print(f"Genre: {genre} -> Recommended Movies: {recommended_movies}")
```

## ⚡ Quantization Details

Post-training quantization was applied using PyTorch's built-in quantization framework. The model was quantized to Float16 (FP16) to reduce model size and improve inference efficiency while balancing accuracy.

## Evaluation Metrics: NDCG

NDCG → If close to 1, the ranking matches expected relevance. Our model's NDCG score is 0.84

## 🔧 Fine-Tuning Details

### Dataset
The **wykonos/movies** dataset was used for training and evaluation. The dataset consists of **texts**.

### Training Configuration
- **Number of epochs**: 5
- **Batch size**: 8
- **Evaluation strategy**: epochs


## 📂 Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

## ⚠️ Limitations

- The model may struggle for out of scope tasks.
- Quantization may lead to slight degradation in accuracy compared to full-precision models.
- Performance may vary across different writing styles and sentence structures.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
