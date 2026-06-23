import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        output_dim=1,
        hidden_dims=[512, 256, 128],
        dropout=0.1,
        activation='relu',
        use_norm=True
    ):
        super(MLPHead, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self.activation)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        assert x.dim() == 2, f"Hope 2 dimensions input, but get {x.dim()} dimensions"
        assert x.size(1) == self.input_dim, f"Dimension doesn't match: {x.size(1)} vs {self.input_dim}"
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return self.mlp(x)


def create_mlp(mlp_type='default', dropout=0.1):
    if mlp_type == 'simple':
        return nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    else:
        return MLPHead(1024, 1, [512, 256, 128], dropout)