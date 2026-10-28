//! Axum 0.7 REST & WebSocket Dual-Protocol Routes (OpenAI & Gemini Live)

use crate::batcher::{ContinuousLLMBatcher, GenerationRequest, TokenDelta};
use crate::tools::{BuiltinToolRouter, OpenAITool};
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::Value;
use std::convert::Infallible;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub batcher: Arc<ContinuousLLMBatcher>,
    pub tool_router: Arc<BuiltinToolRouter>,
    pub model_id: String,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/ws/live", get(ws_live_handler))
        .route("/cache/stats", get(cache_stats))
        .with_state(state)
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "engine": "omni-llm",
        "scheduler": "iteration-level-continuous-batching",
        "cache": "two-tier-w-tinylfu",
        "protocols": ["openai-v1-chat-completions", "gemini-live-websocket"]
    }))
}

async fn cache_stats(State(state): State<AppState>) -> impl IntoResponse {
    let snapshot = state.batcher.cache.metrics.snapshot();
    Json(snapshot)
}

// --- OpenAI Format Schemas ---

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<usize>,
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let req_id = Uuid::new_v4().to_string();

    let mut system_prompt = String::new();
    let mut user_query = String::new();

    for msg in &payload.messages {
        if msg.role == "system" {
            system_prompt = msg.content.clone();
        } else if msg.role == "user" {
            user_query = msg.content.clone();
        }
    }

    let tools_schema = payload.tools
        .as_ref()
        .map(|t| serde_json::to_string(t).unwrap_or_default())
        .unwrap_or_default();

    let max_tokens = payload.max_tokens.unwrap_or(32);
    let cancellation_token = Arc::new(AtomicBool::new(false));
    let (delta_tx, mut delta_rx) = mpsc::channel::<TokenDelta>(128);

    let gen_req = GenerationRequest {
        id: req_id.clone(),
        system_prompt,
        tools_schema,
        user_query,
        max_new_tokens: max_tokens,
        cancellation_token,
        delta_sender: delta_tx,
    };

    let _ = state.batcher.submit(gen_req).await;

    if payload.stream {
        // SSE Streaming
        let stream = async_stream::stream! {
            while let Some(delta) = delta_rx.recv().await {
                let finish_reason = if delta.is_terminal {
                    if delta.tool_call.is_some() {
                        Some("tool_calls")
                    } else {
                        Some("stop")
                    }
                } else {
                    None
                };

                let chunk_json = if let Some(tc) = delta.tool_call {
                    serde_json::json!({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.arguments.to_string()
                                    }
                                }]
                            },
                            "finish_reason": finish_reason
                        }]
                    })
                } else {
                    serde_json::json!({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": delta.text
                            },
                            "finish_reason": finish_reason
                        }]
                    })
                };

                yield Ok::<Event, Infallible>(Event::default().data(chunk_json.to_string()));
                if delta.is_terminal {
                    break;
                }
            }
            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
        };

        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        // Non-streaming completion
        let mut full_text = String::new();
        let mut tool_calls = Vec::new();

        while let Some(delta) = delta_rx.recv().await {
            full_text.push_str(&delta.text);
            if let Some(tc) = delta.tool_call {
                tool_calls.push(serde_json::json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments.to_string()
                    }
                }));
            }
            if delta.is_terminal {
                break;
            }
        }

        let resp = if !tool_calls.is_empty() {
            serde_json::json!({
                "id": req_id,
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": tool_calls
                    },
                    "finish_reason": "tool_calls"
                }]
            })
        } else {
            serde_json::json!({
                "id": req_id,
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_text
                    },
                    "finish_reason": "stop"
                }]
            })
        };

        Json(resp).into_response()
    }
}

// --- Gemini Live WebSocket Protocol Handler ---

async fn ws_live_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_gemini_live_socket(socket, state))
}

async fn handle_gemini_live_socket(socket: WebSocket, state: AppState) {
    let (mut ws_sender, mut ws_receiver) = socket.split();
    let mut system_prompt = String::new();
    let mut registered_tools = String::new();

    while let Some(Ok(msg)) = ws_receiver.next().await {
        if let Message::Text(text) = msg {
            if let Ok(v) = serde_json::from_str::<Value>(&text) {
                // 1. Session Setup
                if let Some(setup) = v.get("setup") {
                    if let Some(tools) = setup.get("tools") {
                        registered_tools = tools.to_string();
                    }
                    system_prompt = setup.get("systemInstruction")
                        .and_then(|s| s.get("parts"))
                        .and_then(|p| p.get(0))
                        .and_then(|t| t.get("text"))
                        .and_then(|txt| txt.as_str())
                        .unwrap_or("You are an agentic clinical AI assistant.")
                        .to_string();

                    // Pre-warm Tier 2 Prompt Cache!
                    let prefix_key = crate::cache::TwoTierPromptCache::compute_prefix_key(
                        "omni-llm",
                        &system_prompt,
                        &registered_tools,
                    );
                    let kv = crate::model::SequenceKVCache::new(8);
                    state.batcher.cache.insert_prefix_kv(prefix_key, kv).await;

                    // Send Setup Acknowledgment
                    let ack = serde_json::json!({
                        "serverContent": {
                            "interrupted": false,
                            "modelTurn": {
                                "parts": [{ "text": "Session initialized with prompt cache." }]
                            }
                        }
                    });
                    let _ = ws_sender.send(Message::Text(ack.to_string())).await;
                    continue;
                }

                // 2. Client User Content Turn
                if let Some(client_content) = v.get("clientContent") {
                    let user_text = client_content.get("turns")
                        .and_then(|t| t.get(0))
                        .and_then(|turn| turn.get("parts"))
                        .and_then(|p| p.get(0))
                        .and_then(|part| part.get("text"))
                        .and_then(|txt| txt.as_str())
                        .unwrap_or("What is the trial dosage?");

                    let (delta_tx, mut delta_rx) = mpsc::channel::<TokenDelta>(64);
                    let req_id = Uuid::new_v4().to_string();
                    let cancellation_token = Arc::new(AtomicBool::new(false));

                    let gen_req = GenerationRequest {
                        id: req_id.clone(),
                        system_prompt: system_prompt.clone(),
                        tools_schema: registered_tools.clone(),
                        user_query: user_text.to_string(),
                        max_new_tokens: 32,
                        cancellation_token,
                        delta_sender: delta_tx,
                    };

                    let _ = state.batcher.submit(gen_req).await;

                    while let Some(delta) = delta_rx.recv().await {
                        if let Some(tc) = delta.tool_call {
                            // Stream Gemini Live functionCall event
                            let live_call = serde_json::json!({
                                "serverContent": {
                                    "modelTurn": {
                                        "parts": [{
                                            "functionCall": {
                                                "name": tc.name,
                                                "id": tc.id,
                                                "args": tc.arguments
                                            }
                                        }]
                                    }
                                }
                            });
                            let _ = ws_sender.send(Message::Text(live_call.to_string())).await;
                        } else {
                            let live_text = serde_json::json!({
                                "serverContent": {
                                    "modelTurn": {
                                        "parts": [{ "text": delta.text }]
                                    }
                                }
                            });
                            let _ = ws_sender.send(Message::Text(live_text.to_string())).await;
                        }

                        if delta.is_terminal {
                            break;
                        }
                    }
                }
            }
        }
    }
}
