import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion network for multimodal medical data (histology and radiology).
    
    Implements self-attention and cross-attention mechanisms to fuse
    histology and radiology embeddings for survival prediction.
    
    Attributes:
        embedding_dim (int): Dimension of input embeddings
        num_layers (int): Number of attention layers
        cross_attention_layers (nn.ModuleList): Stack of attention modules
        
    Args:
        embedding_dim (int): Input embedding dimension
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
    """
    def __init__(self, embedding_dim, dropout=0.1, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = 2
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                # Self-attention for each modality
                'hist_self_attn': nn.MultiheadAttention(embedding_dim, num_heads, dropout, batch_first=True),
                'rad_self_attn': nn.MultiheadAttention(embedding_dim, num_heads, dropout, batch_first=True),

                # Cross attention
                'hist_to_rad': nn.MultiheadAttention(embedding_dim, num_heads, dropout, batch_first=True),
                'rad_to_hist': nn.MultiheadAttention(embedding_dim, num_heads, dropout, batch_first=True),

                # Normalisation layers
                'norm_hist': nn.LayerNorm(embedding_dim),
                'norm_rad': nn.LayerNorm(embedding_dim)
            }) for _ in range(self.num_layers)
        ])

        # Bilinear and fusion layers
        self.BilinearLayer = nn.Bilinear(embedding_dim, embedding_dim, embedding_dim)
        self.fusion_MLP = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim * 2),              
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        

    def forward(self, h, r):
        hist_features = h
        rad_features = r
        for i, layer in enumerate(self.cross_attention_layers):
            # Prepare for attention (add sequence dim)
            hist_seq = hist_features.unsqueeze(1)
            rad_seq = rad_features.unsqueeze(1)

            # Get attention modules for this layer
            hist_self_attn = layer['hist_self_attn']
            rad_self_attn = layer['rad_self_attn']
            hist_to_rad_attn = layer['hist_to_rad']
            rad_to_hist_attn = layer['rad_to_hist']
            norm_hist = layer['norm_hist']
            norm_rad = layer['norm_rad']

            # Self attention
            hist_seq, _ = hist_self_attn(hist_seq, hist_seq, hist_seq)
            rad_seq, _ = rad_self_attn(rad_seq, rad_seq, rad_seq)

            # Cross-attention   
            hist_cross_attended, _ = hist_to_rad_attn(hist_seq, rad_seq, rad_seq)
            rad_cross_attended, _ = rad_to_hist_attn(rad_seq, hist_seq, hist_seq)

            # Remove sequence dimension
            hist_cross_attended = hist_cross_attended.squeeze(1)
            rad_cross_attended = rad_cross_attended.squeeze(1)

            # Residual + norm
            hist_features = norm_hist(hist_features + hist_cross_attended)
            rad_features = norm_rad(rad_features + rad_cross_attended)

        # Final fusion after all layers
        fused_embedding = self.BilinearLayer(hist_features, rad_features)
        return self.fusion_MLP(fused_embedding)