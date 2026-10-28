//! Zero-Padding Ragged Tensor Packing for LLM Prompt Prefill

/// Represents a batch of variable-length prompt sequences concatenated into a single
/// continuous 1D token array with cumulative sequence offsets (cu_seqlens).
/// Eliminates zero-padding computation on plain CPU silicon.
#[derive(Debug, Clone)]
pub struct RaggedPromptBatch {
    /// Contiguous 1D flattened token stream
    pub tokens: Vec<u32>,
    /// Cumulative sequence boundaries: [0, L1, L1+L2, ..., sum(Li)]
    pub cu_seqlens: Vec<usize>,
    /// Number of sequences in this ragged batch
    pub batch_size: usize,
}

impl RaggedPromptBatch {
    /// Construct a packed ragged batch from a list of tokenized sequences
    pub fn from_sequences(sequences: &[Vec<u32>]) -> Self {
        let batch_size = sequences.len();
        let total_tokens: usize = sequences.iter().map(|s| s.len()).sum();
        let mut tokens = Vec::with_capacity(total_tokens);
        let mut cu_seqlens = Vec::with_capacity(batch_size + 1);

        cu_seqlens.push(0);
        let mut cumulative = 0;

        for seq in sequences {
            tokens.extend_from_slice(seq);
            cumulative += seq.len();
            cu_seqlens.push(cumulative);
        }

        Self {
            tokens,
            cu_seqlens,
            batch_size,
        }
    }

    /// Total token count in the ragged batch
    pub fn total_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Slice tokens for sequence at index i
    pub fn sequence_slice(&self, idx: usize) -> Option<&[u32]> {
        if idx >= self.batch_size {
            return None;
        }
        let start = self.cu_seqlens[idx];
        let end = self.cu_seqlens[idx + 1];
        Some(&self.tokens[start..end])
    }

    /// Calculate wasted tokens if this batch were padded with standard 2D padding
    pub fn wasted_padding_tokens(&self) -> usize {
        if self.batch_size == 0 {
            return 0;
        }
        let max_len = (0..self.batch_size)
            .map(|i| self.cu_seqlens[i + 1] - self.cu_seqlens[i])
            .max()
            .unwrap_or(0);
        (max_len * self.batch_size).saturating_sub(self.tokens.len())
    }
}
