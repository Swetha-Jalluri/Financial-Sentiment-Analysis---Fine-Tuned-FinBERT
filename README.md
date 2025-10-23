# Financial Sentiment Analysis with Fine-Tuned FinBERT

Fine-tuned FinBERT model for financial news sentiment classification, achieving **88.98% accuracy** on the Financial PhraseBank dataset.

---

## 🎯 Key Results

| Model | Baseline | Fine-Tuned | Improvement |
|-------|----------|------------|-------------|
| **FinBERT** | 4.96% | **88.98%** | **+84.02 pp** |
| DistilBERT | 55.51% | 85.67% | +30.17 pp |

**Best Configuration:**
- Accuracy: 88.98%
- F1 Score: 0.8891
- Inference: ~50ms per text

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```python
from inference_pipeline import FinancialSentimentAnalyzer

analyzer = FinancialSentimentAnalyzer()
result = analyzer.predict("Apple reported strong earnings this quarter.")
print(f"{result['sentiment']}: {result['confidence']:.1%}")
```

---

## 📁 Project Structure
```
├── Financial_Sentiment_FineTuning.ipynb    # Main notebook
├── inference_pipeline.py                    # Production inference
├── requirements.txt                         # Dependencies
├── baseline_results.json                    # Pre-training results
├── hyperparameter_optimization_results.json # 3 configs tested
├── error_analysis_report.json              # Error patterns
├── train_data.csv / val_data.csv / test_data.csv
└── models/finbert_finetuned_config1/       # Fine-tuned weights
```

---

## 🔬 What Was Done

✅ **Dual-Model Comparison**: FinBERT (domain-specific) vs DistilBERT (general)  
✅ **Baseline Evaluation**: Measured pre-fine-tuning performance  
✅ **Hyperparameter Search**: 3 configurations tested systematically  
✅ **Error Analysis**: Multi-dimensional analysis (confidence, linguistic features)  
✅ **Production Pipeline**: Batch processing, Gradio UI, standalone deployment  

---

## 📊 Dataset

- **Source**: [Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- **Size**: 4,840 sentences
- **Classes**: Negative, Neutral, Positive
- **Split**: 70/15/15 train/val/test (stratified)

---

## 🛠️ Tech Stack

- **Framework**: Hugging Face Transformers
- **Model**: ProsusAI/finbert
- **Training**: Google Colab Pro (T4 GPU)
- **Training Time**: ~45 minutes

---

## 📈 Performance Details

**Hyperparameter Configurations Tested:**
1. Config 1: LR=2e-5, BS=16, Epochs=3 → **88.98% (Best)**
2. Config 2: LR=1e-5, BS=16, Epochs=5
3. Config 3: LR=5e-5, BS=8, Epochs=4

**Error Analysis Findings:**
- Model is 15-20% less confident on errors
- Negations and contrasting language increase errors
- Recommended confidence threshold: >65%

---

## 📝 Files Included

- `baseline_results.json` - Pre-fine-tuning metrics
- `hyperparameter_optimization_results.json` - All configs
- `error_analysis_report.json` - Error patterns & suggestions
- `finetuning_results_config1.json` - Best model results
- `misclassified_examples.csv` - Error cases for analysis

---

## 🎓 Course Assignment

**Assignment**: Fine-Tuning Large Language Models  
**Date**: 10/23/2025


---

## 📄 License

Academic project for educational purposes.

---

## 📚 References

- Financial PhraseBank: Malo et al. (2014)
- FinBERT: Araci (2019)
- Transformers: Wolf et al. (2020)
