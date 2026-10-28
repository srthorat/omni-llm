//! Two-Tiered W-TinyLFU Cache for Omni-LLM (Exact Response + Prefix KV Tensors)

use crate::model::SequenceKVCache;
use moka::future::Cache;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Metrics tracking cache performance
#[derive(Debug, Default)]
pub struct CacheMetrics {
    pub tier1_exact_hits: AtomicU64,
    pub tier2_prefix_hits: AtomicU64,
    pub total_misses: AtomicU64,
}

impl CacheMetrics {
    pub fn record_tier1_hit(&self) {
        self.tier1_exact_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_tier2_hit(&self) {
        self.tier2_prefix_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_miss(&self) {
        self.total_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> CacheMetricsSnapshot {
        let t1 = self.tier1_exact_hits.load(Ordering::Relaxed);
        let t2 = self.tier2_prefix_hits.load(Ordering::Relaxed);
        let misses = self.total_misses.load(Ordering::Relaxed);
        let total = t1 + t2 + misses;
        let hit_rate = if total > 0 {
            (t1 + t2) as f64 / total as f64
        } else {
            0.0
        };

        CacheMetricsSnapshot {
            tier1_exact_hits: t1,
            tier2_prefix_hits: t2,
            total_misses: misses,
            total_lookups: total,
            hit_rate_pct: hit_rate * 100.0,
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct CacheMetricsSnapshot {
    pub tier1_exact_hits: u64,
    pub tier2_prefix_hits: u64,
    pub total_misses: u64,
    pub total_lookups: u64,
    pub hit_rate_pct: f64,
}

/// Two-Tiered W-TinyLFU Cache System
#[derive(Clone)]
pub struct TwoTierPromptCache {
    /// Tier 1: Exact prompt -> complete text response / tool calls (<0.1 µs)
    tier1_exact: Cache<String, String>,
    /// Tier 2: Static prefix prompt -> pre-computed KV tensors (~1 ms)
    tier2_prefix: Cache<String, SequenceKVCache>,
    /// Global metrics
    pub metrics: Arc<CacheMetrics>,
}

impl TwoTierPromptCache {
    pub fn new(tier1_capacity: u64, tier2_capacity: u64) -> Self {
        let tier1_exact = Cache::builder()
            .max_capacity(tier1_capacity)
            .build();

        let tier2_prefix = Cache::builder()
            .max_capacity(tier2_capacity)
            .build();

        Self {
            tier1_exact,
            tier2_prefix,
            metrics: Arc::new(CacheMetrics::default()),
        }
    }

    /// Compute deterministic SHA-256 hash for exact prompt
    pub fn compute_exact_key(model: &str, system_prompt: &str, tools: &str, user_query: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"T1:");
        hasher.update(model.as_bytes());
        hasher.update(b"|");
        hasher.update(system_prompt.as_bytes());
        hasher.update(b"|");
        hasher.update(tools.as_bytes());
        hasher.update(b"|");
        hasher.update(user_query.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Compute deterministic SHA-256 hash for static prefix (system prompt + tools)
    pub fn compute_prefix_key(model: &str, system_prompt: &str, tools: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"T2:");
        hasher.update(model.as_bytes());
        hasher.update(b"|");
        hasher.update(system_prompt.as_bytes());
        hasher.update(b"|");
        hasher.update(tools.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Check Tier 1 exact cache (<0.1 µs)
    pub async fn get_exact_response(&self, key: &str) -> Option<String> {
        if let Some(resp) = self.tier1_exact.get(key).await {
            self.metrics.record_tier1_hit();
            Some(resp)
        } else {
            None
        }
    }

    /// Store Tier 1 exact response
    pub async fn insert_exact_response(&self, key: String, response: String) {
        self.tier1_exact.insert(key, response).await;
        self.tier1_exact.run_pending_tasks().await;
    }

    /// Check Tier 2 prefix KV cache (~1 ms)
    pub async fn get_prefix_kv(&self, prefix_key: &str) -> Option<SequenceKVCache> {
        if let Some(kv) = self.tier2_prefix.get(prefix_key).await {
            self.metrics.record_tier2_hit();
            Some(kv)
        } else {
            self.metrics.record_miss();
            None
        }
    }

    /// Store Tier 2 prefix KV cache
    pub async fn insert_prefix_kv(&self, prefix_key: String, kv: SequenceKVCache) {
        self.tier2_prefix.insert(prefix_key, kv).await;
        self.tier2_prefix.run_pending_tasks().await;
    }
}
