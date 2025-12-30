use linfa::prelude::*;
use linfa_bayes::MultinomialNb;
use ndarray::{Array2, Array1};
use std::fs::File;
use std::io::{Write, Read};
use crate::preprocessing::TfIdfVectorizer;

pub fn train_model(x_train: &Array2<f32>, y_train: &Array1<usize>) -> MultinomialNb<f32, usize> {
    let dataset = Dataset::new(x_train.clone(), y_train.clone());
    
    MultinomialNb::params()
        .fit(&dataset)
        .expect("Model fitting failed")
}

pub fn evaluate_model(model: &MultinomialNb<f32, usize>, x_test: &Array2<f32>, y_test: &Array1<usize>) -> f32 {
    let dataset = Dataset::new(x_test.clone(), y_test.clone());
    let prediction = model.predict(&dataset);
    
    let cm = prediction.confusion_matrix(&dataset).expect("Failed to create confusion matrix");
    cm.accuracy()
}

pub fn split_data(
    x: &Array2<f32>, 
    y: &Array1<usize>, 
    ratio: f32
) -> (Array2<f32>, Array1<usize>, Array2<f32>, Array1<usize>) {
    let n = x.nrows();
    let n_train = (n as f32 * ratio) as usize;
    
    let x_train = x.slice(ndarray::s![0..n_train, ..]).to_owned();
    let y_train = y.slice(ndarray::s![0..n_train]).to_owned();
    let x_test = x.slice(ndarray::s![n_train..n, ..]).to_owned();
    let y_test = y.slice(ndarray::s![n_train..n]).to_owned();
    
    (x_train, y_train, x_test, y_test)
}

pub fn save_model(model: &MultinomialNb<f32, usize>, path: &str) -> bincode::Result<()> {
    let mut file = File::create(path)?;
    let encoded: Vec<u8> = bincode::serialize(model)?;
    file.write_all(&encoded)?;
    Ok(())
}

pub fn load_model(path: &str) -> bincode::Result<MultinomialNb<f32, usize>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let decoded: MultinomialNb<f32, usize> = bincode::deserialize(&buffer)?;
    Ok(decoded)
}

pub fn save_vectorizer(v: &TfIdfVectorizer, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    serde_json::to_writer(file, v)?;
    Ok(())
}

pub fn load_vectorizer(path: &str) -> Result<TfIdfVectorizer, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let v = serde_json::from_reader(file)?;
    Ok(v)
}
