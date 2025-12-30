use axum::{
    routing::post,
    Json, Router,
};
use tower_http::cors::{Any, CorsLayer};
use serde::{Deserialize, Serialize};
use crate::inference::predict_message;
use std::error::Error;
use std::net::SocketAddr;
use axum::http::Method;

#[derive(Deserialize)]
pub struct PredictionRequest {
    pub message: String,
}

#[derive(Serialize)]
pub struct PredictionResponse {
    pub message: String,
    pub prediction: String,
}

async fn predict_handler(Json(payload): Json<PredictionRequest>) -> Json<PredictionResponse> {
    let prediction = predict_message(&payload.message).unwrap_or_else(|e| format!("Error: {}", e));
    
    Json(PredictionResponse {
        message: payload.message,
        prediction,
    })
}

pub async fn start_server() -> Result<(), Box<dyn Error>> {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    let app = Router::new()
        .route("/predict", post(predict_handler))
        .layer(cors);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("API server listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
