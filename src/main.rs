mod preprocessing;
mod training;
mod inference;
mod api;

use std::error::Error;
use std::env;
use ndarray::Array1;
use crate::preprocessing::{load_data, preprocess_dataframe, TfIdfVectorizer};
use crate::training::{split_data, train_model, evaluate_model, save_model, save_vectorizer};
use crate::inference::predict_message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && args[1] == "serve" {
        println!("Starting API server...");
        api::start_server().await?;
        return Ok(());
    }

    if args.len() > 1 && args[1] == "predict" {
        if args.len() < 3 {
            println!("Usage: cargo run predict \"your message here\"");
            return Ok(());
        }
        let message = &args[2];
        println!("Predicting for message: \"{}\"", message);
        let result = predict_message(message)?;
        println!("Prediction: {}", result);
        return Ok(());
    }

    // Default to Train mode
    println!("--- Spam Classification Training Pipeline ---");

    // 1. Data Loading
    println!("Loading data...");
    let df = load_data("data/spam.csv")?;
    println!("Loaded {} rows.", df.height());

    // 2. Preprocessing
    println!("Preprocessing data...");
    let (messages, labels) = preprocess_dataframe(&df)?;
    let y = Array1::from_vec(labels);

    // 3. Vectorization
    println!("Vectorizing text (TF-IDF)...");
    let vectorizer = TfIdfVectorizer::fit(&messages);
    let x = vectorizer.transform(&messages);
    println!("Feature matrix shape: {:?}", x.dim());

    // 4. Data Split
    println!("Splitting data into train/test sets...");
    let (x_train, y_train, x_test, y_test) = split_data(&x, &y, 0.8);

    // 5. Training
    println!("Training Multinomial Naive Bayes model...");
    let model = train_model(&x_train, &y_train);

    // 6. Evaluation
    println!("Evaluating model...");
    let accuracy = evaluate_model(&model, &x_test, &y_test);
    println!("Model Accuracy: {:.2}%", accuracy * 100.0);

    // 7. Save Model and Vectorizer
    println!("Saving model and vectorizer...");
    save_model(&model, "model.bin")?;
    save_vectorizer(&vectorizer, "vectorizer.json")?;
    println!("Model saved to model.bin and vectorizer to vectorizer.json");

    println!("--- Pipeline Finished ---");
    Ok(())
}
