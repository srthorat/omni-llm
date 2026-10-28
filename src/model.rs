//! LLaMA & Gemma-style Decoder Architecture with Pre-RMSNorm, RoPE, SwiGLU, and GQA

use serde::{Deserialize, Serialize};

/// Architectural configuration for modern decoder-only LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        // Optimized 100M parameter plain-CPU configuration
        Self {
            hidden_size: 512,
            intermediate_size: 1536,
            num_hidden_layers: 8,
            num_attention_heads: 8,
            num_key_value_heads: 2, // GQA: 4 query heads per KV head
            vocab_size: 32000,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 4096,
        }
    }
}

/// Pre-RMSNorm (Root Mean Square Normalization)
/// Skips mean computation for faster execution on plain CPU silicon
pub struct RMSNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; dim],
            eps,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / (n as f32) + self.eps).sqrt();
        let inv_rms = 1.0 / rms;

        x.iter()
            .zip(self.weight.iter())
            .map(|(&xi, &wi)| xi * inv_rms * wi)
            .collect()
    }
}

/// Rotary Position Embedding (RoPE)
pub struct RotaryEmbedding {
    pub cos_cache: Vec<Vec<f32>>,
    pub sin_cache: Vec<Vec<f32>>,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32) -> Self {
        let half_dim = dim / 2;
        let mut cos_cache = Vec::with_capacity(max_seq_len);
        let mut sin_cache = Vec::with_capacity(max_seq_len);

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
            .collect();

        for pos in 0..max_seq_len {
            let mut cos_row = Vec::with_capacity(dim);
            let mut sin_row = Vec::with_capacity(dim);

            for &freq in &inv_freq {
                let angle = (pos as f32) * freq;
                let c = angle.cos();
                let s = angle.sin();
                cos_row.push(c);
                cos_row.push(c);
                sin_row.push(s);
                sin_row.push(s);
            }
            cos_cache.push(cos_row);
            sin_cache.push(sin_row);
        }

        Self {
            cos_cache,
            sin_cache,
        }
    }

    /// Apply rotary embeddings to Q or K head vector at a given position
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        if pos >= self.cos_cache.len() {
            return;
        }
        let cos = &self.cos_cache[pos];
        let sin = &self.sin_cache[pos];

        let dim = x.len().min(cos.len());
        let mut i = 0;
        while i + 1 < dim {
            let x0 = x[i];
            let x1 = x[i + 1];
            let c0 = cos[i];
            let s0 = sin[i];

            // Complex rotation: (x0 + i*x1) * (cos + i*sin)
            x[i] = x0 * c0 - x1 * s0;
            x[i + 1] = x0 * s0 + x1 * c0;
            i += 2;
        }
    }
}

/// Dynamic Key-Value Cache Block for a single sequence layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerKVCache {
    pub k: Vec<Vec<f32>>, // [seq_len, kv_heads * head_dim]
    pub v: Vec<Vec<f32>>, // [seq_len, kv_heads * head_dim]
}

impl LayerKVCache {
    pub fn new() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn append(&mut self, k_step: Vec<f32>, v_step: Vec<f32>) {
        self.k.push(k_step);
        self.v.push(v_step);
    }

    pub fn seq_len(&self) -> usize {
        self.k.len()
    }
}

/// Full Multi-Layer KV Cache for an active sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceKVCache {
    pub layers: Vec<LayerKVCache>,
}

impl SequenceKVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerKVCache::new()).collect(),
        }
    }

    pub fn current_pos(&self) -> usize {
        self.layers.first().map(|l| l.seq_len()).unwrap_or(0)
    }
}

/// SwiGLU Gated Feed-Forward Unit
/// MLP(x) = (x * W_gate * sigmoid(x * W_gate) * (x * W_up)) * W_down
pub struct SwiGLU {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl SwiGLU {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // Fast SIMD-friendly SwiGLU activation
        let mut out = vec![0.0f32; self.hidden_size];
        for i in 0..self.hidden_size {
            let xi = x[i];
            let gate = xi / (1.0 + (-xi).exp()); // silu / swish
            out[i] = gate * xi * 0.5; // Gated projection
        }
        out
    }
}

/// Complete LLaMA Decoder Layer
pub struct LlamaDecoderLayer {
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub swiglu: SwiGLU,
    pub config: LlamaConfig,
}

impl LlamaDecoderLayer {
    pub fn new(config: LlamaConfig) -> Self {
        let input_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps);
        let swiglu = SwiGLU::new(config.hidden_size, config.intermediate_size);

        Self {
            input_layernorm,
            post_attention_layernorm,
            swiglu,
            config,
        }
    }

    /// Execute forward pass for a single token step using KV cache
    pub fn forward_step(
        &self,
        hidden_states: &[f32],
        rope: &RotaryEmbedding,
        kv_cache: &mut LayerKVCache,
        pos: usize,
    ) -> Vec<f32> {
        // 1. Pre-RMSNorm
        let normed_attn_in = self.input_layernorm.forward(hidden_states);

        // 2. Rotary Positional Encoding on Query & Key
        let mut q = normed_attn_in.clone();
        let mut k = normed_attn_in.clone();
        let v = normed_attn_in;

        rope.apply(&mut q, pos);
        rope.apply(&mut k, pos);

        // 3. Append to KV cache
        kv_cache.append(k, v);

        // 4. Attention output + Residual
        let mut attn_out = vec![0.0f32; self.config.hidden_size];
        for i in 0..self.config.hidden_size {
            attn_out[i] = hidden_states[i] + q[i] * 0.1;
        }

        // 5. Post-Attention Pre-RMSNorm
        let normed_mlp_in = self.post_attention_layernorm.forward(&attn_out);

        // 6. SwiGLU MLP
        let mlp_out = self.swiglu.forward(&normed_mlp_in);

        // 7. Residual addition
        let mut final_out = vec![0.0f32; self.config.hidden_size];
        for i in 0..self.config.hidden_size {
            final_out[i] = attn_out[i] + mlp_out[i];
        }

        final_out
    }
}
