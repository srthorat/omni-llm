//! Omni-LLM: Ultra-High-Throughput Native Rust Inference Engine for Agentic LLMs
//!
//! Features:
//! - Iteration-Level Continuous Dynamic Batching (TGI/vLLM style with cu_seqlens)
//! - Two-Tiered W-TinyLFU Prompt & Response Cache (Exact response + Prefix KV tensor caching)
//! - Modern LLaMA/Gemma Decoder (Pre-RMSNorm, RoPE, SwiGLU, and GQA)
//! - Dual Agentic Tool-Calling Router (OpenAI Function Calling + Gemini Live WebSocket)
//! - Native Pipecat Framework compatibility for real-time voice and agentic RAG

pub mod batcher;
pub mod cache;
pub mod model;
pub mod ragged;
pub mod routes;
pub mod tools;

pub use batcher::{ContinuousLLMBatcher, GenerationRequest, TokenDelta};
pub use cache::{CacheMetricsSnapshot, TwoTierPromptCache};
pub use model::{LlamaConfig, LlamaDecoderLayer, RotaryEmbedding, SequenceKVCache};
pub use ragged::RaggedPromptBatch;
pub use routes::{create_router, AppState};
pub use tools::{BuiltinToolRouter, DetectedToolCall, OpenAITool};
