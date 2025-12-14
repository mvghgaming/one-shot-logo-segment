# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast
from efficientnet_pytorch import EfficientNet

class LogoEncoder(nn.Module):
    def __init__(self, efficientnet_weights_path, embedding_dim=512, dropout_rate=0.6):
        super().__init__()
        # Load EfficientNet backbone from local weights for deployment flexibility
        self.backbone = EfficientNet.from_pretrained(
            "efficientnet-b4",
            weights_path=efficientnet_weights_path
        )
        
        # Get the number of features from the last fully connected layer
        in_features = self.backbone._fc.in_features
        # Remove the original classifier layer (cast to satisfy type checker)
        self.backbone._fc = cast(nn.Linear, nn.Identity())

        # Define the embedding head with two dropout layers and batch normalization
        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5), # Second dropout with half rate
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x, mask=None):
        # Extract features from the backbone
        features = self.backbone.extract_features(x)
        
        if mask is not None:
            # === Attention pooling logic using mask ===
            # Resize mask to match feature map size
            mask = F.interpolate(mask, size=features.shape[2:], mode='nearest')
            mask = mask.clamp(0, 1)
            
            # Compute attention weights from feature map
            attention = torch.sigmoid(features.mean(dim=1, keepdim=True))
            # Combine attention and mask, background weight is 0.1
            weight = attention * mask + (1 - mask) * 0.1
            
            # Weighted sum pooling
            weight_sum = weight.sum(dim=[2,3], keepdim=True) + 1e-6
            pooled = (features * weight).sum(dim=[2,3], keepdim=True) / weight_sum
            pooled = pooled.squeeze(-1).squeeze(-1)
        else:
            # If no mask, use standard global average pooling
            pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
            
        # Pass pooled features through embedding head
        embeddings = self.embedding(pooled)
        # Normalize embeddings to unit length
        return F.normalize(embeddings, dim=1)