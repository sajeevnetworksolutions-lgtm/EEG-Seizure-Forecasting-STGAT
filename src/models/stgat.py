"""
Spatio-Temporal Graph Attention Network (ST-GAT) for Real-Time Epileptic Seizure Forecasting
Pure PyTorch implementation with multi-head dynamic graph attention and temporal GRU propagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import MODEL_PARAMS

class TemporalWaveletConv(nn.Module):
    """
    1D Multi-Scale Temporal Wavelet Filter.
    Extracts neurological frequency representations from raw EEG electrode epochs (256 samples).
    Transforms (B, 19, 256) -> (B, 19, 64).
    """
    def __init__(self, in_samples=256, out_dim=64):
        super(TemporalWaveletConv, self).__init__()
        # Parallel multi-scale 1D convolutions representing Morlet/Wavelet frequency bands
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3)  # Fast dynamics (Gamma/Beta)
        self.conv2 = nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7) # Alpha/Theta dynamics
        self.conv3 = nn.Conv1d(1, 16, kernel_size=31, stride=2, padding=15) # Delta/Slow waves
        self.conv4 = nn.Conv1d(1, 16, kernel_size=63, stride=2, padding=31) # Ultra-slow background
        
        # Linear projection to target feature dimension
        self.fc = nn.Sequential(
            nn.Linear(64 * (in_samples // 2), out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        """
        Args:
            x: (B, num_nodes=19, in_samples=256) or (B*Seq, 19, 256)
        Returns:
            out: (B, 19, out_dim=64)
        """
        orig_shape = x.shape
        # Flatten nodes into batch dimension for 1D conv: (B*19, 1, 256)
        x_reshaped = x.view(-1, 1, orig_shape[-1])
        
        c1 = F.elu(self.conv1(x_reshaped))
        c2 = F.elu(self.conv2(x_reshaped))
        c3 = F.elu(self.conv3(x_reshaped))
        c4 = F.elu(self.conv4(x_reshaped))
        
        # Concat multi-scale filters along channel dim: (B*19, 64, 128)
        c_all = torch.cat([c1, c2, c3, c4], dim=1)
        c_flat = c_all.view(c_all.size(0), -1)
        
        proj = self.fc(c_flat) # (B*19, out_dim)
        out = proj.view(orig_shape[0], orig_shape[1], -1) # (B, 19, out_dim)
        return out


class SpatialGATLayer(nn.Module):
    """
    Multi-Head Dynamic Spatial Graph Attention Layer.
    Computes dynamic attention coefficients between 19 EEG electrode nodes
    modulated by the time-resolved Phase Locking Value (PLV) dynamic adjacency matrix A^t.
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.3, leak_slope=0.2, concat=True):
        super(SpatialGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.leak_slope = leak_slope
        
        # Each head computes a projection
        self.head_dim = out_features // num_heads if concat else out_features
        
        self.W = nn.Linear(in_features, self.head_dim * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        
        self.leaky_relu = nn.LeakyReLU(self.leak_slope)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
        
        # Residual connection if dimensions match
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj=None):
        """
        Args:
            h: Node feature tensor of shape (B, num_nodes=19, in_features)
            adj: Dynamic PLV adjacency matrix of shape (B, 19, 19) or None
        Returns:
            h_out: (B, 19, out_features)
            attn_weights: (B, num_heads, 19, 19) attention coefficients
        """
        B, N, _ = h.shape
        
        # Linear projection: (B, N, num_heads, head_dim)
        h_proj = self.W(h).view(B, N, self.num_heads, self.head_dim)
        h_proj = h_proj.permute(0, 2, 1, 3) # (B, num_heads, N, head_dim)
        
        # Compute self and target attention scores: (B, num_heads, N, 1)
        score_src = torch.einsum('bhnk,hk->bhn', h_proj, self.a_src).unsqueeze(-1)
        score_dst = torch.einsum('bhnk,hk->bhn', h_proj, self.a_dst).unsqueeze(-2)
        
        # Pairwise attention scores: (B, num_heads, N, N)
        attn_logits = self.leaky_relu(score_src + score_dst)
        
        # Modulate with Dynamic PLV Adjacency Matrix A^t if provided
        if adj is not None:
            # Expand adj to (B, 1, N, N)
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B, -1, -1)
            adj_mask = adj.unsqueeze(1) # (B, 1, N, N)
            
            # Mask out zero/disconnected edges
            mask = (adj_mask <= 1e-4)
            attn_logits = attn_logits.masked_fill(mask, -1e9)
            # Add PLV edge synchrony weight as additive/multiplicative bias
            attn_logits = attn_logits + torch.log(torch.clamp(adj_mask, min=1e-4, max=1.0))
            
        # Softmax over neighborhood nodes
        attn_weights = F.softmax(attn_logits, dim=-1) # (B, num_heads, N, N)
        attn_weights = self.dropout(attn_weights)
        
        # Message aggregation: (B, num_heads, N, head_dim)
        h_agg = torch.matmul(attn_weights, h_proj)
        
        if self.concat:
            # Concatenate heads: (B, N, num_heads * head_dim) = (B, N, out_features)
            h_out = h_agg.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        else:
            # Average heads: (B, N, out_features)
            h_out = h_agg.mean(dim=1)
            
        # Residual connection + LayerNorm + ELU activation
        res = self.residual(h)
        h_out = self.layer_norm(F.elu(h_out + res))
        return h_out, attn_weights


class STGATSeizureForecaster(nn.Module):
    """
    Complete Spatio-Temporal Graph Attention Network (ST-GAT) Seizure Forecasting Architecture.
    
    Pipeline:
    1. Temporal Wavelet Filter: (B, Seq, 19, 256) -> (B*Seq, 19, 64)
    2. Spatial GAT Layer 1: (B*Seq, 19, 64) -> (B*Seq, 19, 128) with 4 Attention Heads
    3. Spatial GAT Layer 2: (B*Seq, 19, 128) -> (B*Seq, 19, 128) with 4 Attention Heads
    4. Global Spatial Pooling & Temporal GRU: (B, Seq, 19, 128) -> (B, 128)
    5. Risk Classifier Head: (B, 128) -> (B, 1) Calibrated Pre-Ictal Probability
    """
    def __init__(self,
                 num_nodes=MODEL_PARAMS["num_nodes"],
                 in_features=MODEL_PARAMS["in_features"],
                 wavelet_out_dim=MODEL_PARAMS["wavelet_out_dim"],
                 gat_hidden_dim=MODEL_PARAMS["gat_hidden_dim"],
                 num_gat_heads=MODEL_PARAMS["num_gat_heads"],
                 gru_hidden_dim=MODEL_PARAMS["gru_hidden_dim"],
                 dropout=MODEL_PARAMS["dropout"]):
        super(STGATSeizureForecaster, self).__init__()
        
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.gat_hidden_dim = gat_hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        
        # 1. Temporal Wavelet Conv
        self.wavelet_filter = TemporalWaveletConv(in_samples=in_features, out_dim=wavelet_out_dim)
        
        # 2. Spatial GAT Layers (Dynamic Graph Attention)
        self.gat1 = SpatialGATLayer(
            in_features=wavelet_out_dim,
            out_features=gat_hidden_dim,
            num_heads=num_gat_heads,
            dropout=dropout,
            concat=True
        )
        self.gat2 = SpatialGATLayer(
            in_features=gat_hidden_dim,
            out_features=gat_hidden_dim,
            num_heads=num_gat_heads,
            dropout=dropout,
            concat=True
        )
        
        # 3. Temporal Sequence Aggregation (GRU)
        # Node readout: combines mean + max spatial features across all 19 channels
        self.node_readout = nn.Sequential(
            nn.Linear(gat_hidden_dim * 2, gru_hidden_dim),
            nn.LayerNorm(gru_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            input_size=gru_hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # 4. Pre-Ictal Risk Classifier Head
        self.risk_classifier = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, adj=None, return_attention=False):
        """
        Forward pass.
        Args:
            x: EEG tensor of shape (B, Seq, 19, 256) or (B, 19, 256)
            adj: Dynamic PLV Adjacency matrix of shape (B, Seq, 19, 19) or (B, 19, 19) or None
            return_attention: If True, returns attention weight matrices for dashboard explainability
        Returns:
            logits: (B, 1) raw logit output
            probs: (B, 1) pre-ictal probability in [0, 1]
            attn_dict: (Optional) Dict containing GAT attention coefficients
        """
        is_single_epoch = (x.dim() == 3)
        if is_single_epoch:
            # Convert (B, 19, 256) -> (B, 1, 19, 256)
            x = x.unsqueeze(1)
            if adj is not None and adj.dim() == 3:
                adj = adj.unsqueeze(1)
                
        B, Seq, N, T = x.shape
        
        # Flatten Batch and Seq dimensions for spatial GAT operations
        x_flat = x.view(B * Seq, N, T) # (B*Seq, 19, 256)
        
        adj_flat = None
        if adj is not None:
            adj_flat = adj.view(B * Seq, N, N) # (B*Seq, 19, 19)
            
        # Step 1: Temporal Wavelet Filter
        h_wavelet = self.wavelet_filter(x_flat) # (B*Seq, 19, 64)
        
        # Step 2: Spatial Graph Attention Layer 1
        h_gat1, attn1 = self.gat1(h_wavelet, adj_flat) # (B*Seq, 19, 128)
        
        # Step 3: Spatial Graph Attention Layer 2
        h_gat2, attn2 = self.gat2(h_gat1, adj_flat) # (B*Seq, 19, 128)
        
        # Step 4: Spatial Global Pooling (Mean + Max pooling across 19 electrode nodes)
        h_mean = torch.mean(h_gat2, dim=1) # (B*Seq, 128)
        h_max = torch.max(h_gat2, dim=1)[0] # (B*Seq, 128)
        h_spatial = torch.cat([h_mean, h_max], dim=-1) # (B*Seq, 256)
        h_spatial = self.node_readout(h_spatial) # (B*Seq, 128)
        
        # Reshape back to temporal sequence: (B, Seq, 128)
        h_seq = h_spatial.view(B, Seq, -1)
        
        # Step 5: Temporal GRU State Propagation
        gru_out, h_last = self.gru(h_seq) # gru_out: (B, Seq, 128), h_last: (1, B, 128)
        h_context = gru_out[:, -1, :] # Last sequence context: (B, 128)
        
        # Step 6: Risk Classifier Head
        logits = self.risk_classifier(h_context) # (B, 1)
        probs = torch.sigmoid(logits) # (B, 1)
        
        if return_attention:
            attn_dict = {
                "gat1_weights": attn1.view(B, Seq, -1, N, N)[:, -1], # Most recent epoch attention
                "gat2_weights": attn2.view(B, Seq, -1, N, N)[:, -1],
                "node_embeddings": h_gat2.view(B, Seq, N, -1)[:, -1]
            }
            return logits, probs, attn_dict
            
        return logits, probs
