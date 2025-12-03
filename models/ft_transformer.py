import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProteinFTTransformer(nn.Module):
    """
    Enhanced FT-Transformer for multiclass protein classification
    """
    def __init__(self, input_dim, num_classes, d_token=128, n_head=8, 
                 n_layers=6, dropout=0.2, ff_dim_factor=4):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # Feature Tokenizer
        self.feature_tokenizer = nn.Sequential(
            nn.Linear(1, d_token // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_token // 2, d_token),
            nn.LayerNorm(d_token)
        )
        
        # Feature-specific bias
        self.feature_bias = nn.Parameter(torch.randn(1, input_dim, d_token))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, input_dim + 1, d_token))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_head,
            dim_feedforward=d_token * ff_dim_factor,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_token)
        
        # Multi-head classification (ensemble within model)
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_token, d_token * 2),
                nn.LayerNorm(d_token * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_token * 2, d_token),
                nn.LayerNorm(d_token),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(d_token, num_classes)
            ) for _ in range(3)
        ])
        
        # Feature importance
        self.importance_proj = nn.Linear(d_token, 1)
        
        # Output combiner
        self.output_combiner = nn.Sequential(
            nn.Linear(num_classes * 3, num_classes * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_classes * 2, num_classes)
        )
        
        # Temperature scaling for class imbalance
        self.temperature = nn.Parameter(torch.ones(num_classes))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                if 'attention' in name:
                    nn.init.xavier_uniform_(param)
                elif 'tokenizer' in name or 'classifier' in name:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize special parameters
        nn.init.normal_(self.feature_bias, mean=0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0, std=0.02)
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)
        nn.init.ones_(self.temperature)
    
    def forward(self, x, return_importance=False):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, input_dim]
            return_importance: Whether to return feature importance
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
            importance: Feature importance scores [batch_size, input_dim] (optional)
        """
        batch_size = x.shape[0]
        
        # Tokenize features
        x_reshaped = x.unsqueeze(-1)  # [batch, input_dim, 1]
        tokens = self.feature_tokenizer(x_reshaped)  # [batch, input_dim, d_token]
        
        # Add feature bias
        tokens = tokens + self.feature_bias
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        all_tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # Add positional encoding
        all_tokens = all_tokens + self.positional_encoding
        
        # Apply transformer
        all_tokens = self.transformer(all_tokens)
        all_tokens = self.norm(all_tokens)
        
        # Extract CLS token
        cls_output = all_tokens[:, 0, :]
        
        # Multiple classification heads
        head_outputs = []
        for head in self.classification_heads:
            head_out = head(cls_output)
            head_outputs.append(head_out)
        
        # Combine outputs
        combined = torch.cat(head_outputs, dim=-1)
        logits = self.output_combiner(combined)
        
        # Apply temperature scaling
        logits = logits / self.temperature.unsqueeze(0)
        
        if return_importance:
            # Calculate feature importance
            feature_tokens = all_tokens[:, 1:, :]
            importance = self.importance_proj(feature_tokens).squeeze(-1)
            return logits, importance
        
        return logits
    
    def get_feature_importance(self, x):
        """Get feature importance scores for input"""
        self.eval()
        with torch.no_grad():
            _, importance = self(x, return_importance=True)
        return importance
    
    def save(self, path):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'num_classes': self.num_classes,
                'd_token': self.d_token if hasattr(self, 'd_token') else 128,
                'n_head': self.n_head if hasattr(self, 'n_head') else 8,
                'n_layers': self.n_layers if hasattr(self, 'n_layers') else 6,
                'dropout': self.dropout if hasattr(self, 'dropout') else 0.2,
            }
        }, path)
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
