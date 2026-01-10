use std::env;
use log::{info, error, debug};
use ml_rust::inference::SpamClassifier;
use futures::stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the logger.
    // Initialize the logger.
    // Use "info" as default if RUST_LOG is not set.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // 1. Initialize Classifier
    info!("Loading model and vectorizer...");
    let classifier = match SpamClassifier::new("model.bin", "vectorizer.json") {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to load model: {}", e);
            return Err(e);
        }
    };
    info!("Model loaded successfully.");

    // Parse Redis URL: Env Var -> CLI Arg -> Default
    let redis_url = env::var("REDIS_URL")
        .ok()
        .or_else(|| env::args().nth(1))
        .unwrap_or_else(|| "redis://127.0.0.1:6379/".to_string());

    info!("Connecting to Redis at {}", redis_url);

    let client = redis::Client::open(redis_url)?;
    let mut pubsub = client.get_async_connection().await?.into_pubsub();

    // Subscribe to BullMQ event patterns
    let pattern = "bull:*:*";
    pubsub.psubscribe(pattern).await?;
    info!("Subscribed to pattern: {}", pattern);

    let mut msg_stream = pubsub.on_message();

    loop {
        if let Some(msg) = msg_stream.next().await {
            let channel: String = msg.get_channel_name().to_string();
            // Using debug! because payload might be large/noisy
            let payload: String = match msg.get_payload() {
                Ok(p) => p,
                Err(e) => {
                    error!("Failed to get payload: {}", e);
                    continue;
                }
            };
            
            info!("Received event on channel '{}'", channel);
            debug!("Payload: {}", payload);
            
            // Assume payload is the message content for "active" jobs or just try to classify it
            // Realistically, for "active" or "completed", we might want to check the content.
            // For this demo, let's classify the payload itself.
            
            match classifier.predict(&payload) {
                Ok(prediction) => {
                    info!("Prediction for event payload: {}", prediction);
                },
                Err(e) => {
                    error!("Failed to predict: {}", e);
                }
            }

            // Basic parsing to identify event type based on channel suffix
            if channel.ends_with(":completed") {
                info!(" -> Job completed!");
            } else if channel.ends_with(":failed") {
                error!(" -> Job failed!");
            } else if channel.ends_with(":active") {
                info!(" -> Job started!");
            } else if channel.ends_with(":waiting") {
                 info!(" -> Job waiting!");
            }
        }
    }
}
