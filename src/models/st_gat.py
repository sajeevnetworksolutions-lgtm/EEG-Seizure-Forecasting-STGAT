# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Spatio-Temporal Graph Attention Network (ST-GAT) for Real-Time Epileptic Seizure Forecasting
Pure PyTorch implementation with multi-head dynamic graph attention and temporal GRU sequence aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalWaveletEncoder(nn.Module):
    """
    1D Multi-Scale Temporal Feature Encoder.
    Extracts multi-frequency neurological wave dynamics from raw EEG electrode signals.
    Transforms (B, N=19, T=256) -> (B, N=19, D=64).
    """
    def __init__(self, in_samples: int = 256, out_dim: int = 64):
        super(TemporalWaveletEncoder, self).__init__()
        # Parallel multi-scale 1D convolutions representing Morlet/Wavelet frequency bands
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3)  # Fast dynamics (Gamma/Beta)
        self.conv2 = nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7) # Alpha/Theta dynamics
        self.conv3 = nn.Conv1d(1, 16, kernel_size=31, stride=2, padding=15) # Delta/Slow waves
        self.conv4 = nn.Conv1d(1, 16, kernel_size=63, stride=2, padding=31) # Ultra-slow background
        
        # Projection to node feature embedding dimension
        self.fc = nn.Sequential(
            nn.Linear(64 * (in_samples // 2), out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, N, T) or (B*L, N, T)
            
        Returns
        -------
        out : torch.Tensor
            Shape (B, N, out_dim)
        """
        orig_shape = x.shape
        x_flat = x.view(-1, 1, orig_shape[-1])
        
        c1 = F.elu(self.conv1(x_flat))
        c2 = F.elu(self.conv2(x_flat))
        c3 = F.elu(self.conv3(x_flat))
        c4 = F.elu(self.conv4(x_flat))
        
        c_all = torch.cat([c1, c2, c3, c4], dim=1)
        c_flat = c_all.view(c_all.size(0), -1)
        
        proj = self.fc(c_flat)
        out = proj.view(orig_shape[0], orig_shape[1], -1)
        return out


class SpatialGraphAttentionLayer(nn.Module):
    """
    Multi-Head Dynamic Graph Attention Layer.
    Computes dynamic attention coefficients between 19 EEG electrode nodes
    modulated by time-resolved Phase Locking Value (PLV) dynamic adjacency matrix A^t.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 negative_slope: float = 0.2,
                 concat: bool = True):
        super(SpatialGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope
        
        self.head_dim = out_features // num_heads if concat else out_features
        
        self.W = nn.Linear(in_features, self.head_dim * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
        
        # Residual projection
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: torch.Tensor, adj: torch.Tensor = None):
        """
        Parameters
        ----------
        h : torch.Tensor
            Node features (B, N=19, in_features)
        adj : torch.Tensor, optional
            Dynamic PLV Adjacency matrix (B, N=19, N=19)
            
        Returns
        -------
        h_out : torch.Tensor
            Updated node features (B, N=19, out_features)
        attn_weights : torch.Tensor
            Attention coefficients (B, num_heads, N, N)
        """
        B, N, _ = h.shape
        
        # Linear projection: (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        h_proj = self.W(h).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention scores
        score_src = torch.einsum('bhnk,hk->bhn', h_proj, self.a_src).unsqueeze(-1) # (B, H, N, 1)
        score_dst = torch.einsum('bhnk,hk->bhn', h_proj, self.a_dst).unsqueeze(-2) # (B, H, 1, N)
        
        attn_logits = self.leaky_relu(score_src + score_dst) # (B, H, N, N)
        
        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B, -1, -1)
            adj_mask = adj.unsqueeze(1) # (B, 1, N, N)
            
            # Mask disconnected edges
            mask = (adj_mask <= 1e-5)
            attn_logits = attn_logits.masked_fill(mask, -1e9)
            # Add continuous PLV connection weight
            attn_logits = attn_logits + torch.log(torch.clamp(adj_mask, min=1e-5, max=1.0))
            
        attn_weights = F.softmax(attn_logits, dim=-1) # (B, H, N, N)
        attn_weights = self.dropout(attn_weights)
        
        # Message aggregation: (B, H, N, head_dim)
        h_agg = torch.matmul(attn_weights, h_proj)
        
        if self.concat:
            h_out = h_agg.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        else:
            h_out = h_agg.mean(dim=1)
            
        res = self.residual(h)
        h_out = self.layer_norm(F.elu(h_out + res))
        return h_out, attn_weights


class STGATSeizureForecaster(nn.Module):
    """
    Complete Spatio-Temporal Graph Attention Network Architecture.
    
    Dataflow:
    1. Temporal 1D Wavelet Filter: (B, L, 19, 256) -> (B*L, 19, 64)
    2. Spatial GAT Layer 1 (4 Heads): (B*L, 19, 64) -> (B*L, 19, 128)
    3. Spatial GAT Layer 2 (4 Heads): (B*L, 19, 128) -> (B*L, 19, 128)
    4. Global Mean+Max Spatial Node Pooling: (B*L, 19, 128) -> (B*L, 128)
    5. Temporal GRU: (B, L, 128) -> (B, 128)
    6. Pre-Ictal Risk Classifier Head: (B, 128) -> (B, 1) logit / risk score
    """
    def __init__(self,
                 num_nodes: int = 19,
                 in_features: int = 256,
                 wavelet_out_dim: int = 64,
                 gat_hidden_dim: int = 128,
                 num_heads: int = 4,
                 gru_hidden_dim: int = 128,
                 dropout: float = 0.2):
        super(STGATSeizureForecaster, self).__init__()
        
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.gat_hidden_dim = gat_hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        
        # 1. Temporal Wavelet Filter
        self.wavelet_filter = TemporalWaveletEncoder(in_samples=in_features, out_dim=wavelet_out_dim)
        
        # 2. Dynamic Spatial GAT Layers
        self.gat1 = SpatialGraphAttentionLayer(
            in_features=wavelet_out_dim,
            out_features=gat_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            concat=True
        )
        self.gat2 = SpatialGraphAttentionLayer(
            in_features=gat_hidden_dim,
            out_features=gat_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        # 3. Spatial Readout & Temporal Sequence GRU
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
        
        # 4. Pre-Ictal Risk Classifier
        self.risk_classifier = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None, return_attention: bool = False):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, L, 19, 256) or (B, 19, 256)
        adj : torch.Tensor, optional
            Shape (B, L, 19, 19) or (B, 19, 19)
        return_attention : bool
            Whether to return attention weights for clinical dashboard visualization.
            
        Returns
        -------
        logits : torch.Tensor
            Shape (B, 1) raw unnormalized logits for Focal Loss
        probs : torch.Tensor
            Shape (B, 1) sigmoid risk probabilities in [0, 1]
        attn_dict : dict (optional)
        """
        is_single = (x.dim() == 3)
        if is_single:
            x = x.unsqueeze(1)
            if adj is not None and adj.dim() == 3:
                adj = adj.unsqueeze(1)
                
        B, L, N, T = x.shape
        x_flat = x.view(B * L, N, T)
        adj_flat = adj.view(B * L, N, N) if adj is not None else None
        
        # 1. Temporal Wavelet Filter
        h_w = self.wavelet_filter(x_flat) # (B*L, 19, 64)
        
        # 2. Spatial GAT 1 & 2
        h_g1, attn1 = self.gat1(h_w, adj_flat)  # (B*L, 19, 128)
        h_g2, attn2 = self.gat2(h_g1, adj_flat) # (B*L, 19, 128)
        
        # 3. Spatial Pooling (Mean + Max across 19 electrodes)
        h_mean = torch.mean(h_g2, dim=1) # (B*L, 128)
        h_max = torch.max(h_g2, dim=1)[0] # (B*L, 128)
        h_spatial = self.node_readout(torch.cat([h_mean, h_max], dim=-1)) # (B*L, 128)
        
        # 4. Temporal Sequence GRU
        h_seq = h_spatial.view(B, L, -1) # (B, L, 128)
        gru_out, _ = self.gru(h_seq)     # (B, L, 128)
        h_last = gru_out[:, -1, :]        # Context from most recent time window (B, 128)
        
        # 5. Classifier Head
        logits = self.risk_classifier(h_last)  # (B, 1)
        probs = torch.sigmoid(logits)    # (B, 1)
        
        if return_attention:
            attn_dict = {
                "gat1_weights": attn1.view(B, L, -1, N, N)[:, -1],
                "gat2_weights": attn2.view(B, L, -1, N, N)[:, -1],
                "node_embeddings": h_g2.view(B, L, N, -1)[:, -1]
            }
            return logits, probs, attn_dict
            
        return logits, probs
