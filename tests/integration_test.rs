//! Integration Tests for Omni-LLM: Model, Two-Tier Cache, Ragged Batching, and Tool Calling

use omni_llm::batcher::{ContinuousLLMBatcher, GenerationRequest};
use omni_llm::cache::TwoTierPromptCache;
use omni_llm::model::{LlamaConfig, RMSNorm, RotaryEmbedding, SequenceKVCache};
use omni_llm::ragged::RaggedPromptBatch;
use omni_llm::tools::BuiltinToolRouter;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::mpsc;

#[tokio::test]
async fn test_rmsnorm_and_rope() {
    let dim = 64;
    let norm = RMSNorm::new(dim, 1e-6);
    let input = vec![2.0f32; dim];
    let output = norm.forward(&input);

    assert_eq!(output.len(), dim);
    // Elements should be close to 1.0 (since input is uniform)
    for val in output {
        assert!((val - 1.0).abs() < 1e-3);
    }

    let rope = RotaryEmbedding::new(dim, 128, 10000.0);
    let mut q = vec![1.0f32; dim];
    rope.apply(&mut q, 5);
    // Norm should be preserved under rotation
    let _orig_norm: f32 = input.iter().map(|v| v * v).sum();
    let rotated_norm: f32 = q.iter().map(|v| v * v).sum();
    assert!((rotated_norm - (dim as f32)).abs() < 1e-3);
}

#[tokio::test]
async fn test_two_tier_cache() {
    let cache = TwoTierPromptCache::new(100, 50);

    let model = "omni-llm-1b";
    let system = "You are a clinical agent.";
    let tools = "query_vector_db";
    let query = "What is the metformin dosage?";

    let exact_key = TwoTierPromptCache::compute_exact_key(model, system, tools, query);
    let prefix_key = TwoTierPromptCache::compute_prefix_key(model, system, tools);

    // Initial check: miss
    assert!(cache.get_exact_response(&exact_key).await.is_none());
    assert!(cache.get_prefix_kv(&prefix_key).await.is_none());

    // Insert Tier 1
    cache.insert_exact_response(exact_key.clone(), "500mg twice daily".to_string()).await;
    let t1_hit = cache.get_exact_response(&exact_key).await;
    assert_eq!(t1_hit, Some("500mg twice daily".to_string()));

    // Insert Tier 2
    let kv = SequenceKVCache::new(4);
    cache.insert_prefix_kv(prefix_key.clone(), kv).await;
    let t2_hit = cache.get_prefix_kv(&prefix_key).await;
    assert!(t2_hit.is_some());
    assert_eq!(t2_hit.unwrap().layers.len(), 4);

    let metrics = cache.metrics.snapshot();
    assert_eq!(metrics.tier1_exact_hits, 1);
    assert_eq!(metrics.tier2_prefix_hits, 1);
}

#[tokio::test]
async fn test_ragged_prompt_packing() {
    let seq1 = vec![101, 2054, 2003, 102]; // 4 tokens
    let seq2 = vec![101, 7592, 102];       // 3 tokens
    let seq3 = vec![101, 102];             // 2 tokens

    let batch = RaggedPromptBatch::from_sequences(&[seq1, seq2, seq3]);

    assert_eq!(batch.total_tokens(), 9);
    assert_eq!(batch.batch_size, 3);
    assert_eq!(batch.cu_seqlens, vec![0, 4, 7, 9]);

    // Check slice
    let s2 = batch.sequence_slice(1).unwrap();
    assert_eq!(s2, &[101, 7592, 102]);

    // If padded to max_len (4), total would be 12 tokens. Wasted = 12 - 9 = 3.
    assert_eq!(batch.wasted_padding_tokens(), 3);
}

#[tokio::test]
async fn test_continuous_batcher_and_tool_calling() {
    let cache = TwoTierPromptCache::new(100, 50);
    let config = LlamaConfig::default();
    let batcher = ContinuousLLMBatcher::new(4, 1024, cache, config);

    let (tx, mut rx) = mpsc::channel(32);
    let cancellation = Arc::new(AtomicBool::new(false));

    let req = GenerationRequest {
        id: "test-req-1".to_string(),
        system_prompt: "You are a clinical agent".to_string(),
        tools_schema: "query_vector_db".to_string(),
        user_query: "What is the clinical trial dosage for metformin?".to_string(),
        max_new_tokens: 5,
        cancellation_token: cancellation,
        delta_sender: tx,
    };

    batcher.submit(req).await.unwrap();

    let mut tokens_received = Vec::new();
    let mut tool_called = false;

    while let Some(delta) = rx.recv().await {
        tokens_received.push(delta.text);
        if let Some(tc) = delta.tool_call {
            assert_eq!(tc.name, "query_vector_db");
            tool_called = true;
        }
        if delta.is_terminal {
            break;
        }
    }

    assert!(!tokens_received.is_empty());
    assert!(tool_called);
}

#[tokio::test]
async fn test_builtin_tool_router() {
    let router = BuiltinToolRouter::new("http://localhost:8000".to_string());

    let args = serde_json::json!({
        "query": "metformin dosage NCT04829",
        "top_k": 2
    });

    let res = router.execute_tool("query_vector_db", &args).await.unwrap();
    assert_eq!(res["status"], "success");
    assert!(res["passages"].as_array().unwrap().len() >= 1);

    let dosage_args = serde_json::json!({
        "weight_kg": 75.0,
        "mg_per_kg": 10.0
    });
    let dosage_res = router.execute_tool("calculate_dosage", &dosage_args).await.unwrap();
    assert_eq!(dosage_res["calculated_dose_mg"], 750.0);
}
