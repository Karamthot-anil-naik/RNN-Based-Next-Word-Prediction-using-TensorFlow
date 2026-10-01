# RNN-Based Next Word Prediction using TensorFlow

A simple **Natural Language Processing (NLP)** project that uses a **TensorFlow SimpleRNN model** to predict the next word from a given text sequence.

## 📌 Overview

The model learns word patterns from a small training text and predicts the most likely next word based on the input phrase.

Example:

```text
Input:
i like machine

Prediction:
learning
```

## 🚀 Features

* Text preprocessing using TensorFlow
* TextVectorization and vocabulary creation
* Word embeddings
* SimpleRNN neural network
* Next-word prediction
* Interactive Streamlit interface
* Real-time model inference

## 🔄 Working Process

```text
Training Text
     ↓
TextVectorization
     ↓
Tokenization
     ↓
Create Word Sequences
     ↓
Embedding Layer
     ↓
SimpleRNN
     ↓
Dense + Softmax
     ↓
Next Word Prediction
```

## 🧠 Model Architecture

```text
Input Sequence
      ↓
Embedding Layer
      ↓
SimpleRNN (64 units)
      ↓
Dense Layer
      ↓
Softmax
      ↓
Predicted Next Word
```

### Model Components

* **TextVectorization** — Converts text into integer token IDs.
* **Embedding** — Converts token IDs into dense vector representations.
* **SimpleRNN** — Learns sequential relationships between words.
* **Dense + Softmax** — Predicts the most likely next word.

## 📝 Training Text

The model is trained on example sentences containing patterns related to machine learning and NLP:

```text
i like machine learning
machine learning is useful
i like natural language processing
natural language processing is interesting
machine learning uses data
data helps machine learning
I like machine learning
I like deep learning
I love machine learning
machine learning is interesting
deep learning is powerful
```

## 🔤 Example Predictions

| Input                     | Predicted Word           |
| ------------------------- | ------------------------ |
| `i like machine`          | `learning`               |
| `machine learning is`     | `useful` / `interesting` |
| `i like natural language` | `processing`             |
| `i love machine`          | `learning`               |

> Predictions depend on the trained model and input sequence.

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Natural Language Processing
* SimpleRNN
* NumPy
* Streamlit

## 📂 Project Structure

```text
rnn-next-word-prediction/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Karamthot-anil-naik/rnn-next-word-prediction.git
```

Navigate to the project:

```bash
cd rnn-next-word-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application provides an interface where you can enter a phrase and generate the predicted next word.

## 🎯 Learning Outcomes

This project demonstrates practical understanding of:

* NLP text preprocessing
* Tokenization
* Vocabulary creation
* Word embeddings
* Sequence generation
* Recurrent Neural Networks
* SimpleRNN
* Softmax classification
* Model training
* Model inference
* Streamlit deployment

## 🔮 Future Improvements

* Train on a larger text corpus
* Use LSTM or GRU
* Add multiple-word text generation
* Improve tokenization using subword techniques
* Add model evaluation metrics
* Experiment with different sequence lengths
* Deploy the application online

## 👨‍💻 Author

**Karamthot Anil Naik**

B.Tech Computer Science Engineering

GitHub: **Karamthot-anil-naik**

---

⭐ **A beginner-friendly NLP project demonstrating next-word prediction using a TensorFlow SimpleRNN model.**
