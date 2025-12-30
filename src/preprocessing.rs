use polars::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use ndarray::Array2;

pub fn load_data(path: &str) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(Some(10000))
        .with_ignore_errors(true)
        .try_into_reader_with_file_path(Some(path.into()))?
        .finish()
}

pub fn clean_text(text: &str) -> String {
    let re = Regex::new(r"[^a-zA-Z\s]").unwrap();
    let cleaned = re.replace_all(text, "");
    cleaned.to_lowercase()
}

pub fn preprocess_dataframe(df: &DataFrame) -> PolarsResult<(Vec<String>, Vec<usize>)> {
    let v1 = df.column("v1")?.str()?;
    let v2 = df.column("v2")?.str()?;

    let labels: Vec<usize> = v1.into_iter()
        .map(|opt_v| {
            match opt_v {
                Some("ham") => 0,
                Some("spam") => 1,
                _ => 0,
            }
        })
        .collect();

    let messages: Vec<String> = v2.into_iter()
        .map(|opt_v| {
            clean_text(opt_v.unwrap_or(""))
        })
        .collect();

    Ok((messages, labels))
}

#[derive(Serialize, Deserialize)]
pub struct TfIdfVectorizer {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f64>,
}

impl TfIdfVectorizer {
    pub fn fit(messages: &[String]) -> Self {
        let mut vocabulary = HashMap::new();
        let mut doc_counts = HashMap::new();
        let n_docs = messages.len() as f64;

        for msg in messages {
            let words: Vec<_> = msg.split_whitespace().collect();
            let unique_words: std::collections::HashSet<_> = words.iter().collect();
            for word in unique_words {
                let count = doc_counts.entry(word.to_string()).or_insert(0.0);
                *count += 1.0;
            }
            for word in words {
                if !vocabulary.contains_key(word) {
                    let next_idx = vocabulary.len();
                    vocabulary.insert(word.to_string(), next_idx);
                }
            }
        }

        let mut idf = vec![0.0; vocabulary.len()];
        for (word, count) in doc_counts {
            if let Some(&idx) = vocabulary.get(&word) {
                idf[idx] = (n_docs / (1.0 + count)).ln();
            }
        }

        TfIdfVectorizer { vocabulary, idf }
    }

    pub fn transform(&self, messages: &[String]) -> Array2<f32> {
        let mut data = Array2::zeros((messages.len(), self.vocabulary.len()));
        for (i, msg) in messages.iter().enumerate() {
            let words: Vec<_> = msg.split_whitespace().collect();
            let mut counts = HashMap::new();
            for word in words {
                let count = counts.entry(word).or_insert(0.0);
                *count += 1.0;
            }

            for (word, count) in counts {
                if let Some(&idx) = self.vocabulary.get(word) {
                    data[[i, idx]] = (count as f32) * (self.idf[idx] as f32);
                }
            }
        }
        data
    }
}
