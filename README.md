#  Airline Tweet Sentiment Analysis Using RNN (PyTorch)

This project applies deep learning techniques to analyze sentiments in airline-related tweets using a **Recurrent Neural Network (RNN)** built in **PyTorch**. By classifying tweets into **positive**, **neutral**, or **negative** sentiments, the model helps uncover public opinions toward airline services—enabling data-driven decisions for brand improvement and customer engagement.

---

##  Project Summary

In the age of digital communication, social media has become a primary channel for customer feedback—especially for service-based industries like aviation. Airlines receive thousands of mentions daily, many of which include valuable feedback hidden within informal, unstructured text.

This project presents a complete Natural Language Processing (NLP) pipeline to **automatically classify the sentiment of such tweets**, using a custom-built RNN model. The workflow includes:

- Cleaning and preprocessing noisy Twitter data,
- Tokenization, vocabulary generation, and sequence encoding,
- Padding and batching of input sequences,
- Custom RNN implementation using PyTorch,
- Model training and evaluation using standard metrics,
- Visual analysis via confusion matrices and word clouds.

This solution is intended to bridge the gap between unstructured social media feedback and actionable insights, offering a strong foundation for sentiment monitoring tools in real-world applications.

---

##  Dataset Details

- **Source**: [Kaggle – Twitter US Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Total Records**: 14,640 labeled tweets
- **Classes**:
  - `positive`
  - `neutral`
  - `negative`
- **Key Features**:
  - `text`: Raw tweet content
  - `airline_sentiment`: Sentiment label (target)
  - `airline`: Mentioned airline
  - `negativereason`: Reason for dissatisfaction (if applicable)

---

##  Project Workflow

1. **Data Loading & Inspection**
2. **Text Cleaning**: Lowercasing, removing URLs, mentions, punctuation
3. **Tokenization & Vocabulary Building**
4. **Sequence Encoding & Padding**
5. **Train-Test Split**
6. **Custom PyTorch Dataset Class**
7. **RNN Model Construction**
8. **Model Training & Optimization**
9. **Evaluation Metrics & Visualization**

---

##  Future Enhancements

To improve performance and expand real-world usability, the following enhancements are planned:

### 1. **Model Improvements**
- Upgrade to **LSTM** or **GRU** for better handling of long-term dependencies.
- Incorporate **attention mechanisms** to focus on impactful words.

### 2. **Language Understanding**
- Use **pretrained embeddings** like GloVe or FastText to bring semantic depth.
- Transition to **transformer-based models** (e.g., BERT) for superior NLP performance.

### 3. **Data Challenges**
- Address **class imbalance** using oversampling, data augmentation, or weighted loss functions.

### 4. **Deployment & Scaling**
- Integrate the model into a **real-time dashboard** using the Twitter API to track public sentiment live.
- Extend to multiple industries (e.g., banking, telecom) for generalized feedback analysis.
