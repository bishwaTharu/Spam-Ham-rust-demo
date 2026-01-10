use crate::preprocessing::{clean_text, TfIdfVectorizer};
use crate::training::{load_model, load_vectorizer};
use linfa::prelude::*;
use ndarray::Array1;
use linfa_bayes::MultinomialNb;

pub struct SpamClassifier {
    model: MultinomialNb<f32, usize>,
    vectorizer: TfIdfVectorizer,
}

impl SpamClassifier {
    pub fn new(model_path: &str, vectorizer_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model = load_model(model_path)?;
        let vectorizer = load_vectorizer(vectorizer_path)?;
        Ok(Self { model, vectorizer })
    }

    pub fn predict(&self, message: &str) -> Result<String, Box<dyn std::error::Error>> {
        let cleaned = clean_text(message);
        let messages = vec![cleaned];
        let x = self.vectorizer.transform(&messages);
        let dataset = Dataset::new(x, Array1::from_elem(1, 0usize));
        let prediction = self.model.predict(&dataset);

        if prediction[0] == 1 {
            Ok("SPAM".to_string())
        } else {
            Ok("HAM".to_string())
        }
    }
}

pub fn predict_message(message: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Legacy function for one-off prediction (reloads model every time)
    // Kept for compatibility if needed, or better yet, use the new struct
    let classifier = SpamClassifier::new("model.bin", "vectorizer.json")?;
    classifier.predict(message)
}
