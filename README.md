# Fake News Detection using Deep Learning

This repository contains the implementation of various deep learning models to detect fake news using textual data and metadata. The models include BiLSTM, CNN-BiLSTM, Transformer, and a fine-tuned multimodal DistilBERT. Each model is trained and evaluated on a labeled dataset to classify news as either real or fake.

## Project Overview

Fake news can spread rapidly and mislead the public. This project aims to build reliable classifiers that can detect such misinformation by leveraging natural language processing (NLP) and deep learning techniques.

---

## Dataset

* **Source:** [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)
* **Samples:**

  * Real News: 21,417
  * Fake News: 23,481
* **Fields Used:** `title`, `text`, `date`, `label`
* Date fields were converted into structured components (day, month, year) for metadata integration.

---

## Models Implemented

### **BiLSTM**

* Tokenizes and pads sequences (maxlen=200)
* Embedding → BiLSTM (forward & backward) → Dense → Sigmoid
* Used EarlyStopping (patience=3)

### **CNN + BiLSTM**

* CNN extracts n-gram features, followed by BiLSTM to capture sequence
* Dropout to reduce overfitting
* Dense + Sigmoid for binary classification

### **Transformer (Custom)**

* 2-layer Transformer Encoder with self-attention
* Positional encoding used to maintain token order
* Mean pooling → Dense Layer → Binary Output

### **Fine-tuned Multimodal DistilBERT**

* Uses DistilBERT with structured metadata (year, category)
* Combines \[CLS] token output with metadata
* Trained using binary cross-entropy loss and AdamW optimizer

---

## Results Summary

| Model              | Strengths                                 | Weaknesses                                        |
| ------------------ | ----------------------------------------- | ------------------------------------------------- |
| BiLSTM             | Captures long-term dependencies           | Slower training, lacks parallelism                |
| CNN + BiLSTM       | Combines local and global context         | More hyperparameters to tune                      |
| Transformer        | Great for long dependencies & parallelism | Needs more data, less effective on local features |
| DistilBERT (Multi) | Fuses text + metadata, robust performance | Requires careful tensor handling and high compute |

---

## Future Work

* Explore lightweight Transformer architectures for efficiency.
* Use domain-specific pretraining.
* Add Explainable AI tools for model transparency.

---

## Getting Started

```bash
git clone https://github.com/Keerthana-1024/Fake-News-Detection.git
cd Fake-News-Detection
pip install -r requirements.txt
jupyter notebook fork-of-fakenews2.ipynb
```

---

## Dependencies

* Python 3.x
* PyTorch
* Transformers (HuggingFace)
* NLTK
* NumPy, Pandas, Matplotlib, Scikit-learn

---

## References

* [PMC Article on Transformer Pruning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10800750/)
* [ResearchGate: Fast Transformer Pruning](https://www.researchgate.net/publication/382939584_A_novel_iteration_scheme_with_conjugate_gradient_for_faster_pruning_on_transformer_models)
* [Papers with Code - BiLSTM](https://paperswithcode.com/method/bilstm)

---

