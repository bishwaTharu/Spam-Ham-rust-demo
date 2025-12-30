use crate::preprocessing::clean_text;
use crate::training::{load_model, load_vectorizer};
use linfa::prelude::*;
use ndarray::Array1;

pub fn predict_message(message: &str) -> Result<String, Box<dyn std::error::Error>> {
    // 1. Load Model and Vectorizer
    let model = load_model("model.bin")?;
    let vectorizer = load_vectorizer("vectorizer.json")?;

    // 2. Preprocess Message
    let cleaned = clean_text(message);
    let messages = vec![cleaned];

    // 3. Vectorize
    let x = vectorizer.transform(&messages);

    // 4. Predict
    let dataset = Dataset::new(x, Array1::from_elem(1, 0usize));
    let prediction = model.predict(&dataset);

    // 5. Interpret Result
    let result = if prediction[0] == 1 {
        "SPAM".to_string()
    } else {
        "HAM".to_string()
    };

    Ok(result)
}
