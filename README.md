# SpamEmailDetector 🛡️
A high-performance Recurrent Neural Network (RNN) using LSTM (Long Short-Term Memory) architecture to classify emails/messages as Ham (Secure) or Spam.

This project leverages a large-scale Turkish dataset (190,000+ entries) and is optimized for Apple Silicon (M4) hardware using TensorFlow Metal.

## 🚀 Features
Deep Learning Architecture: Utilizes LSTM layers to capture long-term dependencies in text.

Massive Dataset: Trained on 190K+ rows to handle complex linguistic patterns and slang/agglutinative structure.

Hardware Accelerated: Optimized for M4 GPU (Metal) for significantly faster training epochs.

Advanced Preprocessing: Custom tokenization and padding pipeline for handling characters and varying message lengths.

## 🛠️ Tech Stack
Core: Python 3.12

Deep Learning: TensorFlow / Keras

Hardware Acceleration: tensorflow-metal (MPS)

Data Processing: Pandas, NumPy, Scikit-learn

Serialization: Pickle (for Tokenizer export)

## 📊 Model Architecture
The model uses a sequential bottleneck design to compress semantic information into a final probability score:

Embedding Layer: Maps 20,000+ vocabulary tokens into a 128-dimensional vector space.

LSTM Layers: (Optional Bidirectional) 128-unit LSTM to process sequence context.

Dropout: 0.2 - 0.5 rate to prevent overfitting on the large dataset.

Dense Output: 1-unit Sigmoid layer producing a probability between 0.0 and 1.0.

📥 Installation
Bash
## Clone the repository
git clone https://github.com/yourusername/SpamEmailDetector.git

## Install dependencies optimized for Apple Silicon
`pip install tensorflow-metal`

`pip install -r requirements.txt`

## 🖥️ Usage
### 1. Training

Ensure your dataset is cleaned and labels are encoded. Run the training script to generate the .keras model and .pickle tokenizer.

`model.fit(X, labels, epochs=6, batch_size=1024, validation_split=0.2)
`
### 2.Inference (Prediction)

Input any message to check its status:

`preds = model.predict(new_pad)`

`state = ("KATIKSIZ SPAM" if prob > 0.75 else "SPAM") if prob > 0.5 else "SECURE"`


## 📈 Performance on M4
Dataset Size: 190,000+ samples.

Training Speed: ~4 minutes per epoch (using Metal GPU).
20-30 minutes to train.

Inference Latency: <10ms per message.