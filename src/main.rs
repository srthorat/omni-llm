//! Omni-LLM CLI High-Performance Server

use clap::Parser;
use omni_llm::batcher::ContinuousLLMBatcher;
use omni_llm::cache::TwoTierPromptCache;
use omni_llm::model::LlamaConfig;
use omni_llm::routes::{create_router, AppState};
use omni_llm::tools::BuiltinToolRouter;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(
    name = "omni-llm",
    about = "High-Throughput Native Rust LLM Inference Server with Continuous Ragged Batching, Two-Tier Prompt Cache, and Dual Tool-Calling Protocols (OpenAI & Gemini Live)"
)]
struct Args {
    #[arg(short, long, env = "PORT", default_value_t = 8001)]
    port: u16,

    #[arg(long, env = "HOST", default_value = "0.0.0.0")]
    host: String,

    #[arg(short, long, env = "MODEL_ID", default_value = "omni-llm-1b")]
    model_id: String,

    #[arg(long, env = "MAX_BATCH_SIZE", default_value_t = 64)]
    max_batch_size: usize,

    #[arg(long, env = "MAX_BATCH_TOKENS", default_value_t = 8192)]
    max_batch_tokens: usize,

    #[arg(long, env = "CACHE_SIZE", default_value_t = 100000)]
    cache_size: u64,

    #[arg(long, env = "EMBEDDINGS_URL", default_value = "http://localhost:8000")]
    embeddings_url: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "omni_llm=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    tracing::info!("Initializing Omni-LLM Inference Server...");
    tracing::info!("Model ID: {}", args.model_id);
    tracing::info!("Max Batch Size: {}", args.max_batch_size);
    tracing::info!("Max Batch Tokens: {}", args.max_batch_tokens);
    tracing::info!("Prompt Cache Capacity: {} entries", args.cache_size);
    tracing::info!("Embeddings Backend URL: {}", args.embeddings_url);

    // Initialize Two-Tiered W-TinyLFU Cache
    let cache = TwoTierPromptCache::new(args.cache_size, args.cache_size / 2);

    // Initialize LLaMA Model Configuration
    let config = LlamaConfig::default();

    // Initialize Iteration-Level Continuous Batcher
    let batcher = Arc::new(ContinuousLLMBatcher::new(
        args.max_batch_size,
        args.max_batch_tokens,
        cache,
        config,
    ));

    // Initialize Built-in Tool Router
    let tool_router = Arc::new(BuiltinToolRouter::new(args.embeddings_url));

    let state = AppState {
        batcher,
        tool_router,
        model_id: args.model_id,
    };

    let router = create_router(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    tracing::info!("🚀 Omni-LLM listening on http://{}", addr);
    tracing::info!("👉 OpenAI Compatible API: POST http://{}/v1/chat/completions", addr);
    tracing::info!("👉 Gemini Live WebSocket API: ws://{}/ws/live", addr);
    tracing::info!("👉 Cache Analytics: GET http://{}/cache/stats", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
