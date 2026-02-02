"""
Transformer model for autoregressive climb path generation.

Architecture:
- Embedding layer for tokens
- Positional encoding
- Transformer decoder layers
- Output projection to vocabulary
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ClimbPathTransformer(nn.Module):
    """
    Autoregressive transformer for climb path generation.
    
    Generates climb paths token-by-token conditioned on grade.
    """
    
    def __init__(
        self,
        vocab_size: int = 45,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        """
        Initialize the transformer model.
        
        Args:
            vocab_size: Size of token vocabulary
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate causal mask for autoregressive generation.
        
        Args:
            sz: Sequence length
            
        Returns:
            Mask tensor of shape [sz, sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Input token IDs [batch_size, seq_len]
            src_mask: Attention mask [seq_len, seq_len]
            src_key_padding_mask: Padding mask [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # src: [batch_size, seq_len]
        batch_size, seq_len = src.shape
        
        # Generate causal mask if not provided
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # Embed tokens and scale
        src_emb = self.embedding(src) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        
        # Transpose for transformer: [seq_len, batch_size, d_model]
        src_emb = src_emb.transpose(0, 1)
        
        # Add positional encoding
        src_emb = self.pos_encoder(src_emb)
        
        # Pass through transformer decoder
        # Using decoder as encoder-decoder with memory=src_emb (self-attention only)
        output = self.transformer_decoder(
            tgt=src_emb,
            memory=src_emb,
            tgt_mask=src_mask,
            tgt_key_padding_mask=src_key_padding_mask,
        )  # [seq_len, batch_size, d_model]
        
        # Transpose back: [batch_size, seq_len, d_model]
        output = output.transpose(0, 1)
        
        # Project to vocabulary
        logits = self.output_projection(output)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        grade_token: int,
        tokenizer,
        max_length: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """
        Generate a climb path autoregressively.
        
        Args:
            grade_token: Grade token ID to condition on
            tokenizer: ClimbPathTokenizer instance
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            device: Device to run on
            
        Returns:
            Generated token sequence
        """
        self.eval()
        
        # Start with BOS and grade token
        generated = torch.tensor([[tokenizer.BOS_TOKEN, grade_token]], dtype=torch.long, device=device)
        
        for _ in range(max_length - 2):  # -2 for BOS and grade
            # Forward pass
            logits = self.forward(generated)  # [1, seq_len, vocab_size]
            
            # Get logits for next token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS generated
            if next_token.item() == tokenizer.EOS_TOKEN:
                break
        
        return generated[0]


class ClimbPathTransformerWithGeneration(ClimbPathTransformer):
    """
    Extended transformer with HuggingFace-style generation using logits processors.
    """
    
    @torch.no_grad()
    def generate_with_processors(
        self,
        grade_token: int,
        tokenizer,
        logits_processors=None,
        max_length: int = 64,
        temperature: float = 1.0,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """
        Generate with custom logits processors for constraint enforcement.
        
        Args:
            grade_token: Grade token ID
            tokenizer: ClimbPathTokenizer instance
            logits_processors: List of logits processors
            max_length: Maximum sequence length
            temperature: Sampling temperature
            device: Device to run on
            
        Returns:
            Generated token sequence
        """
        self.eval()
        
        # Start with BOS and grade token
        generated = torch.tensor([[tokenizer.BOS_TOKEN, grade_token]], dtype=torch.long, device=device)
        
        for _ in range(max_length - 2):
            # Forward pass
            logits = self.forward(generated)  # [1, seq_len, vocab_size]
            
            # Get logits for next token
            next_token_logits = logits[:, -1, :] / temperature  # [1, vocab_size]
            
            # Apply logits processors
            if logits_processors is not None:
                for processor in logits_processors:
                    next_token_logits = processor(generated, next_token_logits)
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs[0], num_samples=1).unsqueeze(0)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS generated
            if next_token.item() == tokenizer.EOS_TOKEN:
                break
        
        return generated[0]
