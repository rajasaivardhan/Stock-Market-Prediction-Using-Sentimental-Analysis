# 🧠 Financial Sentiment Analysis using Hybrid RoBERTa–FinBERT Model
A deep learning–based hybrid transformer architecture for sentiment classification on noisy, slang-heavy financial text from Reddit’s r/WallStreetBets (WSB).  
This model fuses **RoBERTa-Large** (general contextual understanding) and **FinBERT** (finance-specific sentiment understanding) to achieve **state-of-the-art accuracy**.

---

## 🚀 Project Overview
Online financial communities such as **Reddit WallStreetBets** generate massive volumes of emotional, slang-filled, and highly unstructured text.  
Traditional sentiment analysis methods fail to understand:

- Slang, memes, sarcasm  
- Financial terminology  
- Noisy and irregular text  
- Abbreviations and emojis  

This project introduces a **Hybrid Transformer Model** that combines:

- **RoBERTa-Large** → understands slang, memes, sarcasm  
- **FinBERT** → understands financial terminology and sentiment  
- **BiGRU Layer (optional)** → for sequential pattern representation  
- **Attention-based Fusion Layer** → merges embeddings effectively  

The model achieves superior performance on noisy Reddit WSB data.

---

## 📊 Dataset
A custom dataset of **32,470 posts** was collected from:

- subreddit: r/WallStreetBets  
- fields used: *title* + *body* (merged into one text field)

### Sentiment Labels
Since the dataset had no labels, it was **auto-labeled using FinBERT**:

- **0 → Negative**
- **1 → Neutral**
- **2 → Positive**

### Final Label Distribution
| Sentiment | Count |
|----------|--------|
| Positive | 27,336 |
| Neutral  | 4,189  |
| Negative | 945    |

---

## 🧹 Preprocessing Pipeline
Text preprocessing includes:

- Merging *title + body*
- Lowercasing
- Removing URLs, markdown, special characters
- Replacing emojis with words (`🚀 → "rocket"`, `💎 → "diamond hands"`)
- Preserving financial stock tickers (`GME`, `TSLA`, `AMC`)
- Removing extra punctuation and whitespace
- Dual tokenization (RoBERTa + FinBERT)

---

## 🏗️ Model Architecture

**Input Text**  
→ **RoBERTa-Large Encoder**  
→ **RoBERTa CLS Embedding (1024-d)**  

**Input Text**  
→ **FinBERT Encoder**  
→ **FinBERT CLS Embedding (768-d)**  

**Fusion Layer**  
→ Concatenation (1024 + 768 = **1792-d vector**)  
→ Attention mechanism  

**Classifier**  
→ Dense(512) → ReLU → Dropout(0.3) → Dense(3) → Softmax  

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM | 0.65 | 0.62 | 0.60 | 0.61 |
| FinBERT | 0.74 | 0.72 | 0.70 | 0.71 |
| RoBERTa-base | 0.78 | 0.75 | 0.74 | 0.74 |
| RoBERTa-large | 0.80 | 0.78 | 0.77 | 0.77 |
| **Hybrid (RoBERTa + FinBERT)** | **0.84** | **0.79** | **0.78** | **0.80** |

🔥 The **Hybrid Model** achieves the **highest accuracy (84%)**, outperforming all baseline models.

---

## 🧠 Hybrid Model Code (Showcase)

```python
class HybridRoBERTaFinBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("roberta-large")
        self.finbert = AutoModel.from_pretrained("yiyanghkust/finbert-tone")

        self.classifier = nn.Sequential(
            nn.Linear(1024 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3)
        )

    def forward(self, text):
        # Dual tokenization
        rob_tok = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        fin_tok = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Encoder outputs
        r_cls = self.roberta(**rob_tok).last_hidden_state[:, 0, :]
        f_cls = self.finbert(**fin_tok).last_hidden_state[:, 0, :]

        # Fusion
        fusion = torch.cat([r_cls, f_cls], dim=1)

        return self.classifier(fusion)
