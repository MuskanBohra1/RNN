# Airline Tweet Sentiment Analysis Using RNN (PyTorch)

This project demonstrates a Natural Language Processing (NLP) application for sentiment classification of tweets related to US airlines using a Recurrent Neural Network (RNN). Implemented in **PyTorch**, the model classifies each tweet as **positive**, **neutral**, or **negative**, providing insights that can assist airline companies in understanding public opinion and enhancing customer service.

---

## Project Summary

- **Objective**: To develop a deep learning model that classifies sentiments in tweets directed at US airlines.
- **Model**: A custom RNN built in PyTorch.
- **Dataset**: Kaggle's Twitter US Airline Sentiment Dataset (~14,640 labeled tweets).
- **Output**: Sentiment classification with visualization and analysis of results.

---

##  Technologies Used

| Area              | Tools & Libraries                          |
|-------------------|--------------------------------------------|
| Programming       | Python                                     |
| Data Manipulation | pandas, numpy                              |
| Deep Learning     | PyTorch                                    |
| NLP               | re, collections (Counter), tokenization    |
| Evaluation        | scikit-learn                               |
| Visualization     | matplotlib, seaborn, WordCloud             |


---

##  Dataset Description

- **Source**: [Kaggle: Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Attributes Used**:
  - `text`: Raw tweet content
  - `airline_sentiment`: Target sentiment label
  - `airline`: Airline mentioned in the tweet
  - `negativereason`: Reason provided for negative sentiments (if applicable)
- **Label Distribution**:
  - `Negative`: Majority
  - `Neutral`: Moderate
  - `Positive`: Minority

---

##  Project Workflow

1. **Data Loading & Exploration**
2. **Text Preprocessing**:
   - Lowercasing
   - Removing URLs, mentions, special characters
   - Tokenization
3. **Vocabulary Creation & Encoding**
4. **Sequence Padding**
5. **Train-Test Split**
6. **Custom Dataset Class & DataLoader Setup**
7. **Model Definition**:
   - Embedding Layer
   - RNN Layer
   - Fully Connected Output Layer
   - LogSoftmax Activation
8. **Model Training & Validation**
9. **Performance Evaluation & Visualization**

---

##  Results

- Achieved satisfactory classification accuracy for a basic RNN.
- Confusion matrix analysis revealed overlap between `neutral` and `negative` sentiments due to linguistic similarity.
- WordClouds and sentiment-wise analysis highlight commonly used words in each class.
- Loss curves demonstrate consistent learning during training.

---

##  Future Enhancements

- Upgrade to **LSTM** or **GRU** architectures for improved memory handling.
- Incorporate **pretrained embeddings** to improve semantic understanding.
- Experiment with **transformer-based models** such as BERT.
- Apply **attention mechanisms** to focus on key parts of each tweet.
- Implement techniques to handle **class imbalance**, such as oversampling or weighted loss.



