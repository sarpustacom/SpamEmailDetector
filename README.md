# SpamEmailDetector 🛡️

A high-performance neural network architecture designed to classify Turkish emails/messages as **Ham (Secure)** or **Spam**. This project is specifically optimized for **Apple Silicon (M4)** hardware using TensorFlow Metal (MPS).

## 🚀 Key Features
- **Hybrid Learning:** Utilizes advanced TF-IDF vectorization with N-gram (1,2) support to capture both word frequency and context.
- **Hardware Accelerated:** Full **M4 GPU** integration via `tensorflow-metal`, reducing training time to under 2 minutes for 190K+ entries.
- **Dynamic Feedback Loop:** Built-in Streamlit feature to add new, real-world examples directly into the dataset with a **two-step confirmation** mechanism.
- **Spam Nuance Detection:** Categorizes results into four levels: *KATIKSIZ SPAM, SPAM, GÜVENLİ, and TEMİZ*.

## 📊 Model Architecture & Optimization
The model is engineered for high precision even with aggressive Turkish slang:
- **Dense Bottleneck:** 256 -> 128 -> 64 unit design.
- **L2 Regularization:** ($l2=0.0001$) to prevent overfitting on dominant spam keywords.
- **Batch Normalization:** Ensures stable training across deep layers.
- **LeakyReLU Activation:** Prevents "Dead Neuron" syndrome, improving detection of subtle spam patterns.

## 🛠️ Tech Stack
- **Core:** Python 3.12
- **Deep Learning:** TensorFlow / Keras (Sequential API)
- **Acceleration:** `tensorflow-metal` (Metal Performance Shaders)
- **UI/UX:** Streamlit
- **Data:** Scikit-learn (TF-IDF & Stratified Splitting), Pandas, NumPy

## 📥 Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/SpamEmailDetector.git](https://github.com/yourusername/SpamEmailDetector.git)

# Install Apple Silicon optimized dependencies
pip install tensorflow-metal streamlit scikit-learn pandas
```

## 🖥️ Usage
1. **Train the Model**: Ensure your dataset is in data/tr_email_spam.csv and run:
```bash
python train_super_model.py
```

2. **Launch the UI**: Use the Streamlit application to analyze messages and expand your dataset:
```bash
streamlit run app_super.py
```

## 📈 Performance on Apple M4
Training Speed: ~10s per epoch.

Inference Latency: <8ms (Real-time).

Validation Accuracy: Optimized for low False Positives (Secure messages scoring <20%).

Developed and optimized for Apple M4 Performance.