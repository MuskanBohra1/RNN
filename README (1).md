

#  Airline Sentiment Analysis Using RNN (PyTorch)

A deep learning project that performs sentiment analysis on airline-related tweets using a Recurrent Neural Network (RNN) implemented in PyTorch. The model classifies tweets as **positive**, **neutral**, or **negative**, helping airlines understand public opinion and improve customer service.

---

##  Table of Contents

- [About the Project](#about-the-project)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Limitations & Future Scope](#limitations--future-scope)
- [Installation & Usage](#installation--usage)
- [License](#license)

---

## About the Project

This project builds a text classification model using an RNN to detect sentiment in airline tweets. It involves:

- Text cleaning & preprocessing
- Vocabulary creation & tokenization
- Sequence padding
- Model training & evaluation using PyTorch

The end goal is to predict if a tweet expresses a **positive**, **negative**, or **neutral** sentiment towards an airline.

---

##  Technologies Used

- Python
- Pandas, NumPy
- PyTorch
- Scikit-learn
- Matplotlib, Seaborn
- WordCloud
- Google Colab

---

##  Dataset

- **Source**: [Kaggle Airline Twitter Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Samples**: ~14,640 tweets
- **Labels**:
  - `negative`
  - `neutral`
  - `positive`
- **Attributes**:
  - `airline_sentiment` (target)
  - `text` (tweet content)
  - `airline`
  - `negativereason` (if negative)

---

##  Project Pipeline

1. **Import Libraries**
2. **Load and Explore Data**
3. **Preprocessing**:
   - Lowercasing
   - URL & punctuation removal
   - Tokenization
4. **Vocabulary Creation** with `Counter`
5. **Numerical Encoding** of tokens
6. **Padding** sequences
7. **Train-Test Split**
8. **Custom Dataset and DataLoader**
9. **RNN Model Implementation**
10. **Model Training**
11. **Evaluation & Visualization**

---

##  Model Architecture

```python
Embedding Layer → Simple RNN Layer → Fully Connected Layer → LogSoftmax
```

- **Embedding**: Converts words to dense vectors
- **RNN Layer**: Captures sequential dependencies
- **FC Layer**: Maps hidden state to 3 output classes
- **Activation**: `LogSoftmax` for multi-class classification

---

##  Results

- The model achieves reasonable accuracy for a basic RNN.
- Performance is visualized using:
  - Confusion matrix
  - Training loss graph
  - WordClouds for each sentiment
- Class imbalance slightly affected accuracy, especially between `neutral` and `negative`.

---

##  Limitations & Future Scope

### Limitations:
- Basic RNN struggles with long-term dependencies.
- No pretrained embeddings (GloVe/BERT).
- Class imbalance impacts performance.

### Future Scope:
- Replace RNN with **LSTM/GRU**
- Add **pretrained embeddings**
- Use **transformers (e.g., BERT)**
- Apply **attention mechanisms**
- Improve data preprocessing (lemmatization, POS tagging)

---

##  Installation & Usage

###  Requirements:
```bash
pip install pandas numpy torch scikit-learn wordcloud matplotlib seaborn
```

### ▶Run the Project:

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/airline-sentiment-rnn.git
   cd airline-sentiment-rnn
   ```

2. Launch the notebook:
   - Open `RNN_PROJECT.ipynb` in Jupyter or Google Colab
   - Run all cells step by step

---

##  License

This project is licensed under the **MIT License**. Feel free to use and adapt it for academic or non-commercial purposes.

---

