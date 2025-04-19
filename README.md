# Transformer Exploration for News Summarization :newspaper:

This project explores Transformer-based models inspired by the paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). We compare three approaches for **news article summarization** using the [Kaggle News Summarization dataset](https://www.kaggle.com/datasets/sbhatti/news-summarization):

1. :zap: A **Transformer** model (as in *Attention Is All You Need*)  

---

## 📁 Structure du projet

- `src/` : Contient l'ensemble des scripts Python nécessaires à l'entraînement, au pré-traitement, etc.
- `train_test.py` *(à la racine)* : Permet d'entraîner le modèle.
- `app/api.py` *(à la racine)* : Lance une API pour interagir avec le modèle entraîné.

## 📦 Données et poids du modèle

- **Données** : Hébergées publiquement sur un serveur **S3** (données brutes et données nettoyées).
- **Poids du modèle** : Disponibles sur **HuggingFace** et automatiquement récupérés lors du lancement de l’API.

---

## Results and Experiments

The models were trained on an NVIDIA A100 GPU with 40 GB of high-bandwidth memory :computer:

| Model                        | ROUGE-1 | ROUGE-2 | ROUGE-L | Train Time (ep)  | Params  |
|------------------------------|---------|---------|---------|------------------|---------|
| Transformer                  | 0.20    | 0.04    | 0.15    | ~ $1.6 \times 10^4$ s (25) | $1.25 \times 10^7$  |





