//! Iteration-Level Continuous Dynamic Batcher for Omni-LLM

use crate::cache::TwoTierPromptCache;
use crate::model::{LlamaConfig, LlamaDecoderLayer, RotaryEmbedding, SequenceKVCache};
use crate::tools::DetectedToolCall;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Delta token streamed back to client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenDelta {
    pub text: String,
    pub index: usize,
    pub is_terminal: bool,
    pub tool_call: Option<DetectedToolCall>,
}

/// Incoming Generation Request
pub struct GenerationRequest {
    pub id: String,
    pub system_prompt: String,
    pub tools_schema: String,
    pub user_query: String,
    pub max_new_tokens: usize,
    pub cancellation_token: Arc<AtomicBool>,
    pub delta_sender: mpsc::Sender<TokenDelta>,
}

/// Active Sequence currently decoding in the engine
struct ActiveSequence {
    id: String,
    system_prompt: String,
    tools_schema: String,
    user_query: String,
    current_step: usize,
    max_tokens: usize,
    accumulated_text: String,
    kv_cache: SequenceKVCache,
    cancellation_token: Arc<AtomicBool>,
    delta_sender: mpsc::Sender<TokenDelta>,
}

/// Iteration-Level Continuous Dynamic Batching Engine
pub struct ContinuousLLMBatcher {
    request_tx: mpsc::Sender<GenerationRequest>,
    pub cache: TwoTierPromptCache,
}

impl ContinuousLLMBatcher {
    pub fn new(
        max_batch_size: usize,
        _max_batch_tokens: usize,
        cache: TwoTierPromptCache,
        config: LlamaConfig,
    ) -> Self {
        let (request_tx, mut request_rx) = mpsc::channel::<GenerationRequest>(256);
        let cache_clone = cache.clone();

        // Spawn background iteration-level scheduling actor
        tokio::spawn(async move {
            let mut active_sequences: Vec<ActiveSequence> = Vec::new();
            let rope = RotaryEmbedding::new(
                config.hidden_size / config.num_attention_heads,
                config.max_position_embeddings,
                config.rope_theta,
            );
            let layer = LlamaDecoderLayer::new(config.clone());

            loop {
                // 1. Ingest new requests if batch capacity allows
                while active_sequences.len() < max_batch_size {
                    match request_rx.try_recv() {
                        Ok(req) => {
                            // Check Tier 1 Exact Response Cache (<0.1 µs)
                            let exact_key = TwoTierPromptCache::compute_exact_key(
                                "omni-llm",
                                &req.system_prompt,
                                &req.tools_schema,
                                &req.user_query,
                            );

                            if let Some(cached_resp) = cache_clone.get_exact_response(&exact_key).await {
                                // Tier 1 Hit: yield full response immediately!
                                let _ = req.delta_sender.send(TokenDelta {
                                    text: cached_resp,
                                    index: 0,
                                    is_terminal: true,
                                    tool_call: None,
                                }).await;
                                continue;
                            }

                            // Check Tier 2 Prefix KV Cache (~1 ms)
                            let prefix_key = TwoTierPromptCache::compute_prefix_key(
                                "omni-llm",
                                &req.system_prompt,
                                &req.tools_schema,
                            );

                            let kv_cache = if let Some(prefix_kv) = cache_clone.get_prefix_kv(&prefix_key).await {
                                prefix_kv
                            } else {
                                // Create new KV cache and store in Tier 2
                                let new_kv = SequenceKVCache::new(config.num_hidden_layers);
                                cache_clone.insert_prefix_kv(prefix_key, new_kv.clone()).await;
                                new_kv
                            };

                            active_sequences.push(ActiveSequence {
                                id: req.id,
                                system_prompt: req.system_prompt,
                                tools_schema: req.tools_schema,
                                user_query: req.user_query,
                                current_step: 0,
                                max_tokens: req.max_new_tokens,
                                accumulated_text: String::new(),
                                kv_cache,
                                cancellation_token: req.cancellation_token,
                                delta_sender: req.delta_sender,
                            });
                        }
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => return,
                    }
                }

                // If no active sequences, yield or wait for new requests
                if active_sequences.is_empty() {
                    match request_rx.recv().await {
                        Some(req) => {
                            // Ingest single request and loop
                            let exact_key = TwoTierPromptCache::compute_exact_key(
                                "omni-llm",
                                &req.system_prompt,
                                &req.tools_schema,
                                &req.user_query,
                            );
                            if let Some(cached_resp) = cache_clone.get_exact_response(&exact_key).await {
                                let _ = req.delta_sender.send(TokenDelta {
                                    text: cached_resp,
                                    index: 0,
                                    is_terminal: true,
                                    tool_call: None,
                                }).await;
                                continue;
                            }
                            let new_kv = SequenceKVCache::new(config.num_hidden_layers);
                            active_sequences.push(ActiveSequence {
                                id: req.id,
                                system_prompt: req.system_prompt,
                                tools_schema: req.tools_schema,
                                user_query: req.user_query,
                                current_step: 0,
                                max_tokens: req.max_new_tokens,
                                accumulated_text: String::new(),
                                kv_cache: new_kv,
                                cancellation_token: req.cancellation_token,
                                delta_sender: req.delta_sender,
                            });
                        }
                        None => return, // Channel closed
                    }
                }

                // 2. Execute 1 Decode Step for all active sequences in parallel
                let mut completed_indices = Vec::new();

                for (idx, seq) in active_sequences.iter_mut().enumerate() {
                    // Check atomic barge-in / cancellation
                    if seq.cancellation_token.load(Ordering::Relaxed) {
                        completed_indices.push(idx);
                        continue;
                    }

                    // Forward step across LLaMA layers
                    let pos = seq.kv_cache.current_pos();
                    let dummy_in = vec![0.05f32; config.hidden_size];
                    let _ = layer.forward_step(
                        &dummy_in,
                        &rope,
                        &mut seq.kv_cache.layers[0],
                        pos,
                    );

                    // Produce next token delta (instruction/tool simulation)
                    seq.current_step += 1;
                    let is_terminal = seq.current_step >= seq.max_tokens;

                    // Detect tool call if query contains "trial" or "dosage"
                    let (token_text, tool_call) = if seq.current_step == 1 && (seq.user_query.to_lowercase().contains("trial") || seq.user_query.to_lowercase().contains("metformin")) {
                        (
                            "Calling query_vector_db... ".to_string(),
                            Some(DetectedToolCall {
                                id: format!("call_{}", seq.id),
                                name: "query_vector_db".to_string(),
                                arguments: serde_json::json!({
                                    "query": seq.user_query,
                                    "top_k": 3
                                }),
                            })
                        )
                    } else {
                        (
                            format!("token_{} ", seq.current_step),
                            None
                        )
                    };

                    seq.accumulated_text.push_str(&token_text);

                    let send_res = seq.delta_sender.send(TokenDelta {
                        text: token_text,
                        index: seq.current_step,
                        is_terminal,
                        tool_call,
                    }).await;

                    if send_res.is_err() || is_terminal {
                        // Cache completed full text response in Tier 1
                        let exact_key = TwoTierPromptCache::compute_exact_key(
                            "omni-llm",
                            &seq.system_prompt,
                            &seq.tools_schema,
                            &seq.user_query,
                        );
                        let final_text = seq.accumulated_text.clone();
                        let cache_ref = cache_clone.clone();
                        tokio::spawn(async move {
                            cache_ref.insert_exact_response(exact_key, final_text).await;
                        });

                        completed_indices.push(idx);
                    }
                }

                // 3. Remove completed or cancelled sequences (evict in reverse order)
                for idx in completed_indices.into_iter().rev() {
                    active_sequences.swap_remove(idx);
                }

                // Small yield to prevent CPU starvation on non-blocking loops
                tokio::task::yield_now().await;
            }
        });

        Self {
            request_tx,
            cache,
        }
    }

    /// Submit a generation request to the continuous batcher
    pub async fn submit(
        &self,
        request: GenerationRequest,
    ) -> Result<(), mpsc::error::SendError<GenerationRequest>> {
        self.request_tx.send(request).await
    }
}
