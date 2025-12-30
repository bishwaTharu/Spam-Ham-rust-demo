# Spam-Ham Classifier (Rust)

A high-performance machine learning pipeline built in Rust for classifying SMS messages as **Spam** or **Ham** (Legitimate). This project implements the entire workflow from raw data preprocessing to a serving API.

## 🚀 Features

- **Data Processing**: Leverages `Polars` for fast CSV handling and `Regex` for text cleaning.
- **ML Engine**: Uses `Linfa` (a Rust ML framework) to implement a Multinomial Naive Bayes classifier.
- **Vectorization**: Custom TF-IDF implementation with automated stopword removal.
- **API Serving**: A high-performance REST API built with `Axum` and `Tokio`.
- **Serialization**: Efficient model persistence using `Bincode` and `Serde`.

## 📦 Getting Started

### Prerequisites
- [Rust](https://rustup.rs/) (2024 edition)
- Cargo

### Installation
```bash
git clone <repository-url>
cd ml_rust
```

### Usage

#### 1. Train the Model
The training pipeline processes the raw data, fits the model, and saves the artifacts (`model.bin` and `vectorizer.json`).
```bash
cargo run --bin train
```

#### 2. Run the API Server
Start the Axum server to serve predictions.
```bash
cargo run --bin api
```



